# train_cawn_dtdg.py
import os
import sys
import torch
import numpy as np
import math
import random
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
import timeit
from collections import defaultdict
from models.CAWN.module import CAWN
from models.CAWN.neighborfinder import NeighborFinder


def update_adj_list(adj_list, srcs, dsts, edge_idxs, ts):
    """
    增量更新 adj_list。
    """
    for src, dst, eid, t in zip(srcs, dsts, edge_idxs, ts):
        adj_list[src].append((dst, eid, t))
        adj_list[dst].append((src, eid, t))
    return adj_list

def test_cawn_dtdg(test_data, model, base_adj_list, neg_sampler, evaluator, metric, split_mode='val'):
    model.eval()
    perf_list = []
    
    # 创建一个可修改的副本，用于在评估期间逐步添加边
    adj_list_eval = [neighbors[:] for neighbors in base_adj_list]
    
    test_snapshots = test_data['edge_index']
    test_edges_orig = test_data['original_edges']
    
    for snapshot_idx in sorted(test_snapshots.keys()):
        # 1. 为当前快照的预测创建 NeighborFinder (基于历史)
        ngh_finder = NeighborFinder(adj_list_eval)
        model.update_ngh_finder(ngh_finder)
        
        # 2. 进行预测
        pos_edges = test_edges_orig[snapshot_idx]
        pos_srcs, pos_dsts = pos_edges[0], pos_edges[1]
        
        neg_batch_list = neg_sampler.query_batch(pos_srcs, pos_dsts, np.array([snapshot_idx]*len(pos_srcs)), split_mode=split_mode)
        
        with torch.no_grad():
            for i in range(len(pos_srcs)):
                src, dst, t = pos_srcs[i], pos_dsts[i], snapshot_idx
                neg_nodes = neg_batch_list[i]
                
                pos_score = model.forward(np.array([src]), np.array([dst]), np.array([t])).sigmoid().cpu().item()
                neg_scores = [model.forward(np.array([src]), np.array([neg]), np.array([t])).sigmoid().cpu().item() for neg in neg_nodes]

                input_dict = {"y_pred_pos": np.array([pos_score]), "y_pred_neg": np.array(neg_scores), "eval_metric": [metric]}
                perf_list.append(evaluator.eval(input_dict)[metric])

        # 3. 将当前快照的边加入 adj_list_eval，为下一个快照做准备
        srcs, dsts = test_snapshots[snapshot_idx][0], test_snapshots[snapshot_idx][1]
        eids = np.zeros(len(srcs), dtype=np.int32)
        ts = np.full(len(srcs), snapshot_idx, dtype=np.float32)
        adj_list_eval = update_adj_list(adj_list_eval, srcs, dsts, eids, ts)

    return float(np.mean(np.array(perf_list)))


if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader
    from utils.logger import logger
    
    # --- 日志设置 ---
    log_directory = args.output_folder
    log_filename = f"cawn_dtdg_{args.dataset}_{args.lr}.log"
    log_file_path = os.path.join(log_directory, log_filename)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    print(f"INFO: Logging to {log_file_path}")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = logger(original_stdout, log_file)
    sys.stderr = logger(original_stderr, log_file)

    # --- 参数和种子设置 ---
    set_random(args.seed)
    num_epochs = args.max_epoch
    lr = args.lr
    NUM_NEIGHBORS = [20, 20]
    NUM_LAYERS = len(NUM_NEIGHBORS)

    # --- 数据加载 ---
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data, val_data, test_data = data['train_data'], data['val_data'], data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1

    # --- 特征准备 ---
    node_feat = np.random.rand(num_nodes, 128)
    edge_feat = np.random.rand(1, 128) # DTDG通常没有边特征，创建一个虚拟的
    node_feat[0, :] = 0

    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print(f"INFO: Start a fresh run with random seed: {seed}")
        
        model = CAWN(n_feat=node_feat, e_feat=edge_feat,
                     num_layers=NUM_LAYERS, num_neighbors=NUM_NEIGHBORS, drop_out=0.1,
                     agg='tree', pos_dim=100).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        
        best_val, best_test, best_epoch = 0, 0, 0

        for epoch in range(num_epochs):
            print ("-"*100)
            train_start_time = timeit.default_timer()
            model.train()
            total_loss = 0
            
            # 每个epoch都从一个空的图开始，逐步构建
            adj_list_train = [[] for _ in range(num_nodes)]
            
            snapshot_list = train_data['edge_index']
            for snapshot_idx in range(train_data['time_length']):
                optimizer.zero_grad()
                
                # 1. 为当前预测创建 NeighborFinder (基于 t-1 及之前的快照)
                ngh_finder = NeighborFinder(adj_list_train)
                model.update_ngh_finder(ngh_finder)
                
                # 2. 预测当前快照的边
                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.cpu().numpy()
                src, dst = pos_index[0], pos_index[1]
                t = np.full_like(src, snapshot_idx, dtype=np.float32)
                assert isinstance(src, np.ndarray), f"src is not a numpy array, but {type(src)}"
                assert isinstance(dst, np.ndarray), f"dst is not a numpy array, but {type(dst)}"
                bgd_nodes = np.random.randint(0, num_nodes, len(src))

                pos_prob, neg_prob = model.contrast(src, dst, bgd_nodes, t)
                
                loss = criterion(pos_prob, torch.ones_like(pos_prob)) + criterion(neg_prob, torch.zeros_like(neg_prob))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # 3. 将当前快照的边加入 adj_list，为下一个快照做准备
                eids = np.zeros(len(src), dtype=np.int32)
                adj_list_train = update_adj_list(adj_list_train, src, dst, eids, t)

            train_time = timeit.default_timer() - train_start_time
            print(f'INFO: Epoch {epoch}/{num_epochs}, Loss: {total_loss / train_data["time_length"]}')

            # --- VALIDATION ---
            evaluator = Evaluator(name="tgbl-wiki")
            metric = "mrr"
            neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")
            neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_val_ns.pkl", split_mode="val")
            
            # 验证时，基础图结构包含所有训练快照
            val_start_time = timeit.default_timer()
            val_metric = test_cawn_dtdg(val_data, model, adj_list_train, neg_sampler, evaluator, metric, split_mode='val')
            print(f"INFO: Validation {metric}: {val_metric}")
            
            # --- TEST ---
            if val_metric > best_val:
                best_val, best_epoch = val_metric, epoch
                neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_test_ns.pkl", split_mode="test")
                
                # 测试时，基础图结构包含所有训练+验证快照
                adj_list_test = [neighbors[:] for neighbors in adj_list_train]
                for snapshot_idx in sorted(val_data['edge_index'].keys()):
                    srcs, dsts = val_data['edge_index'][snapshot_idx]
                    eids = np.zeros(len(srcs), dtype=np.int32)
                    ts = np.full(len(srcs), snapshot_idx, dtype=np.float32)
                    adj_list_test = update_adj_list(adj_list_test, srcs, dsts, eids, ts)

                test_start_time = timeit.default_timer()
                test_metric = test_cawn_dtdg(test_data, model, adj_list_test, neg_sampler, evaluator, metric, split_mode='test')
                best_test = test_metric
                print(f"INFO: Test {metric}: {test_metric}")

            if epoch - best_epoch > args.patience:
                print("Early stopping...")
                break
        
        print(f"INFO: Run {seed} finished. Best epoch: {best_epoch}, Best val: {best_val}, Best test: {best_test}")
        print("-" * 100)

    sys.stdout, sys.stderr = original_stdout, original_stderr
    log_file.close()

#python dtdg_cawn.py --model=CAWN --dataset=uci -t weekly --lr 0.001 --max_epoch 200 --patience 40 --seed 1 --num_runs 5 --device_id 1
