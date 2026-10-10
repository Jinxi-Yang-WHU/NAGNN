# train_hgnnlp_ctdg.py
import os
import sys
import torch
import numpy as np
import timeit
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import Data

# 确保模型文件在可导入的路径中
from models.hawkes import HGNNLP


def test_tgb_for_hgnnlp(test_loader, 
                        test_snapshots, 
                        ts_map, 
                        node_feat,
                        model,
                        initial_graph_edges,
                        neg_sampler, 
                        evaluator,
                        metric, 
                        split_mode='val'):
    """
    A new test function specifically designed for HGNNLP on CTDG datasets.
    It updates node embeddings at discrete snapshot boundaries and uses them
    to predict events in the continuous-time stream.
    """
    model.eval()
    perf_list = []
    device = node_feat.device

    # --- Initialization ---
    # The initial graph state is from the end of the previous phase (e.g., training)
    current_graph_edges = initial_graph_edges.to(device)
    
    # Pre-compute the initial node embeddings based on the initial graph
    with torch.no_grad():
        edge_attr = torch.zeros(current_graph_edges.size(1), 1, device=device)
        initial_data = Data(x=node_feat, edge_index=current_graph_edges, edge_attr=edge_attr)
        h = model(initial_data) # Initial node embeddings

    # Initialize snapshot tracking
    snapshot_keys = sorted(ts_map.keys())
    current_snapshot_idx_ptr = 0
    
    # --- Process Event Stream ---
    for batch in test_loader:
        batch.to(device)
        
        # --- CRUCIAL: Update embeddings if the batch crosses a snapshot boundary ---
        last_event_t = batch.t[-1].item()
        
        # Check if we need to advance to the next snapshot
        while (current_snapshot_idx_ptr < len(snapshot_keys) and 
               last_event_t > ts_map[snapshot_keys[current_snapshot_idx_ptr]]):
            
            # Use the graph of the snapshot we are crossing
            snapshot_key_to_use = snapshot_keys[current_snapshot_idx_ptr]
            #print(f"INFO: Crossing snapshot boundary {snapshot_key_to_use} at t={ts_map[snapshot_key_to_use]}. Updating embeddings.")
            
            current_graph_edges = test_snapshots[snapshot_key_to_use].long().to(device)
            
            # Re-compute node embeddings with the new graph structure
            with torch.no_grad():
                edge_attr = torch.zeros(current_graph_edges.size(1), 1, device=device)
                graph_data = Data(x=node_feat, edge_index=current_graph_edges, edge_attr=edge_attr)
                h = model(graph_data)
            
            current_snapshot_idx_ptr += 1

        # --- Prediction using the current embeddings `h` ---
        pos_src, pos_dst, pos_t = batch.src, batch.dst, batch.t
        
        # Query negative samples for the batch
        neg_batch_list = neg_sampler.query_batch(pos_src.cpu().numpy(), pos_dst.cpu().numpy(), pos_t.cpu().numpy(), split_mode=split_mode)
        
        # Iterate over each event in the batch for evaluation
        for i, neg_batch in enumerate(neg_batch_list):
            source_node = pos_src[i].item()
            target_node = pos_dst[i].item()
            
            query_src = torch.tensor([source_node] * (1 + len(neg_batch)), device=device)
            query_dst = torch.cat([
                torch.tensor([target_node], device=device),
                torch.tensor(neg_batch).long().to(device)
            ], dim=0)

            with torch.no_grad():
                pred = model.predictor(h[query_src], h[query_dst]).squeeze(-1)
            
            pos_pred = pred[0]
            neg_pred = pred[1:]

            input_dict = {
                "y_pred_pos": np.array([pos_pred.cpu()]),
                "y_pred_neg": neg_pred.cpu().numpy(),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

    mean_metric = float(np.mean(np.array(perf_list)))

    # The state to pass to the next phase is the graph of the very last snapshot
    final_snapshot_key = snapshot_keys[-1]
    final_graph_edges = test_snapshots[final_snapshot_key].long().to(device)
    
    return mean_metric, final_graph_edges


if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader
    from utils.logger import logger

    log_directory = args.output_folder
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_filename = f"hgnnlp_{args.model_name}_{args.dataset}_{args.lr}_{args.seed}.log"
    log_file_path = os.path.join(log_directory, log_filename)

    print(f"INFO: Logging to {log_file_path}")
    log_file = open(log_file_path, 'w', encoding='utf-8')

    original_stdout = sys.stdout 
    sys.stdout = logger(original_stdout, log_file)
    original_stderr = sys.stderr
    sys.stderr = logger(original_stderr, log_file)
    
    set_random(args.seed)

    # --- Data Loading ---
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets")
    train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask
    full_data = dataset.get_TemporalData().to(args.device)
    train_edges, val_edges, test_edges = full_data[train_mask], full_data[val_mask], full_data[test_mask]

    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)

    # Load discretized snapshot data for training
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data, val_data, test_data = data['train_data'], data['val_data'], data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    print(f"INFO: Number of nodes: {num_nodes}")

    # Set up node features
    node_feat_dim = 128
    node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)
    edge_feat_dim = 1

    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print(f"INFO: Start a fresh run with random seed: {seed}")
        
        # --- Model Definition ---
        model = HGNNLP(n_node=num_nodes, n_feat=node_feat_dim, n_edge=edge_feat_dim, 
                       n_hidden=node_feat_dim, dropout=args.dropout, name=args.model_name, 
                       layers=args.layers, heads=args.heads, layer_skip=args.layer_skip,
                       initial_skip=args.initial_skip, norm=args.norm).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_val, best_test, best_epoch = 0, 0, 0

        for epoch in range(args.max_epoch):
            print ("-"*100)
            train_start_time = timeit.default_timer()
            
            model.train()
            total_loss = 0
            snapshot_list = train_data['edge_index']

            # --- Training Loop (t-1 -> t) ---
            for snapshot_idx in range(train_data['time_length']):
                pos_edge = snapshot_list[snapshot_idx].long().to(args.device)
                
                if snapshot_idx == 0:
                    input_edge_index = pos_edge
                else:
                    input_edge_index = snapshot_list[snapshot_idx - 1].long().to(args.device)

                neg_dst = torch.randint(0, num_nodes, (pos_edge.size(1),), dtype=torch.long, device=args.device)
                neg_edge = torch.stack([pos_edge[0], neg_dst], dim=0)

                edge_attr = torch.zeros(input_edge_index.size(1), edge_feat_dim, device=args.device)
                graph_data = Data(x=node_feat, edge_index=input_edge_index, edge_attr=edge_attr)
                
                optimizer.zero_grad()
                loss = model.train_step(graph_data, pos_edge, neg_edge)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_time = timeit.default_timer() - train_start_time
            print(f'INFO: Epoch {epoch}/{args.max_epoch}, Loss: {total_loss / train_data["time_length"]:.4f}')
            print(f"INFO: Train time: {train_time:.4f}s")
            
            # --- VALIDATION ---
            # Get the graph from the last training snapshot to initialize validation
            train_snapshot_keys = sorted(train_data['edge_index'].keys())
            last_train_snapshot_edges = train_data['edge_index'][train_snapshot_keys[-1]].long()

            val_loader = TemporalDataLoader(val_edges, batch_size=args.batch_size)
            dataset.load_val_ns()

            start_epoch_val = timeit.default_timer()
            val_metric, last_val_snapshot_edges = test_tgb_for_hgnnlp(val_loader, 
                                                                    val_data['edge_index'], val_data['ts_map'], node_feat, 
                                                                    model, last_train_snapshot_edges, 
                                                                    neg_sampler, evaluator,
                                                                    metric, split_mode='val')
            
            val_time = timeit.default_timer() - start_epoch_val
            print(f"INFO: Validation {metric}: {val_metric:.4f}")
            print(f"INFO: Validation time: {val_time:.4f}s")

            # --- TEST ---
            if val_metric > best_val:
                best_val, best_epoch = val_metric, epoch
                
                test_loader = TemporalDataLoader(test_edges, batch_size=args.batch_size)
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                test_metric, _ = test_tgb_for_hgnnlp(test_loader, 
                                                     test_data['edge_index'], test_data['ts_map'], node_feat, 
                                                     model, last_val_snapshot_edges, # Pass last val snapshot
                                                     neg_sampler, evaluator,
                                                     metric, split_mode='test')
                
                test_time = timeit.default_timer() - test_start_time
                best_test = test_metric

                print(f"INFO: >>> New Best Found <<< Test {metric}: {best_test:.4f}")
                print(f"INFO: Test time: {test_time:.4f}s")
                print ("-"*100)
                
        print(f"INFO: Run {seed} finished.")
        print(f"INFO: Best epoch: {best_epoch}")
        print(f"INFO: Best val performance: {best_val:.4f}")
        print(f"INFO: Best test performance: {best_test:.4f}")
        print ("-"*100)


#python ctdg_hgnnlp.py --dataset tgbl-wiki --model_name hgat --lr 0.0002 --max_epoch 40 --patience 20 --dropout 0.1 --seed 1 --num_runs 5 --heads 2 --layers 2 --initial_skip --batch_size 200 --output_folder ./logs_ctdg/