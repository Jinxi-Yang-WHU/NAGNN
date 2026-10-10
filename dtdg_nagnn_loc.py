import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
import numpy as np
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
import timeit
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from models.NAGNN import ReSASGNN2 

def test_tgb(test_data, node_feat,
             encoder,
             current_states,
             neg_sampler, evaluator,
             metric, split_mode='val'):
    
    test_snapshots = test_data['edge_index'] 
    test_edges = test_data['original_edges']
    
    encoder.eval()

    perf_list = []

    for snapshot_idx in test_snapshots.keys():
        pos_index = torch.from_numpy(test_edges[snapshot_idx])
        pos_index = pos_index.long().to(args.device)

        for i in range(pos_index.shape[1]):
            pos_src = pos_index[0][i].item()
            pos_dst = pos_index[1][i].item()

            pos_t = snapshot_idx
            neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode=split_mode)

            for idx, neg_batch in enumerate(neg_batch_list):
                query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                query_src = torch.from_numpy(query_src).long().to(args.device)
                query_dst = torch.from_numpy(query_dst).long().to(args.device)
                edge_index = torch.stack((query_src, query_dst), dim=0)

                with torch.no_grad():
                    final_embeddings = torch.cat([current_states['h2'], current_states['r_pool1'], current_states['r_pool2']], dim=-1)
                    # final_embeddings = torch.cat([current_states['h2'], current_states['n_state'], current_states['n_state_two']], dim=-1)
                    y_pred = encoder.decode((query_src, query_dst), final_embeddings)
                y_pred = y_pred.squeeze(dim=-1).detach()


            input_dict = {
                        "y_pred_pos": np.array([y_pred[0].cpu()]),
                        "y_pred_neg": np.array(y_pred[1:].cpu()),
                        "eval_metric": [metric],
                        }
            perf_list.append(evaluator.eval(input_dict)[metric])

        last_states = current_states
        
        #* update the model now if the prediction batch has moved to next snapshot
        cur_index = test_snapshots[snapshot_idx]
        cur_index = cur_index.long().to(args.device)
        _, current_states = encoder(node_feat, cur_index, states=last_states)

    test_metric = float(np.mean(np.array(perf_list)))

    return test_metric, last_states


if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader
    from utils.logger import logger
    
    if args.wandb == 0:
        log_directory = args.output_folder
        log_filename = f"result_{args.lr}_{args.patience}_{args.max_epoch}_{args.agg_mode}_{args.res_mode}.log"

        log_file_path = os.path.join(log_directory, log_filename)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        print(f"INFO: Logging to {log_file_path}")
        log_file = open(log_file_path, 'w', encoding='utf-8')

        original_stdout = sys.stdout 
        sys.stdout = logger(original_stdout, log_file)  # Redirect stdout to both console and log file
        original_stderr = sys.stderr
        sys.stderr = logger(original_stderr, log_file)

    # set the random seed
    set_random(args.seed)

    # set up some parameters
    batch_size = args.batch_size
    num_epochs = args.max_epoch
    lr = args.lr

    # Model-related parameters
    model_dim = {
        "h": 256,
        "e_joint":128,
        "reservoir":128
    }

    node_feat_dim = 128
    pos_dim = args.pos_dim # MODIFIED: 我们的新模型不直接使用 pos_dim，但为保持兼容性，可以保留
    output_dim = 1
    link_pred_num_layers = 2

    # Data Loading...
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    print(f"INFO: Number of nodes: {num_nodes}")

    # Initialize node features (randomly, as no real features are available)
    node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)

    for run in range(args.num_runs):
        seed = args.seed + run

        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="utg",
                name=f"seed-{seed}",
                group=f"ReSASGNN_{args.dataset}",
                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "architecture":"ReSASGNN",
                "dataset": args.dataset,
                "time granularity": args.time_scale,
                "patience": args.patience,
                "max_epoch": args.max_epoch,
                "roland_update": args.roland_update,
                "neighbor_update": args.neighbor_update,
                }
            )

        set_random(seed)
        print(f"INFO: Start Run {run + 1}/{args.num_runs} with random seed: {seed}")
                
        # MODIFIED: 实例化新的模型
        encoder = ReSASGNN2(
            input_dim=node_feat_dim,
            model_dim=model_dim,
            num_nodes=num_nodes,
            reservoir_size=args.reservoir_size, # 假设这些参数在 args 中定义
            num_buckets=args.num_buckets,
            device=args.device
        ).to(args.device)

        all_params = set(encoder.parameters())
        high_lr_params_set = set(encoder.predictor.parameters()) | \
                     set(encoder.pool_mlp.parameters()) | \
                     set(encoder.attention_layer.parameters())
        # high_lr_params_set = set(encoder.predictor.parameters())

        low_lr_params_set = all_params - high_lr_params_set
        param_groups1 = [
            {'params': list(high_lr_params_set), 'lr': lr}, 
            {'params': list(low_lr_params_set), 'lr': lr / 100}
        ]
        param_groups2 = [
            {'params': list(high_lr_params_set), 'lr': lr}, 
        ]

        optimizer = torch.optim.Adam(param_groups1)

        criterion = torch.nn.BCELoss() 

        best_val = 0
        best_test = 0
        best_epoch = 0
        patience_counter = 0


        for epoch in range(num_epochs):
            # if epoch < 60:
            #     optimizer = torch.optim.Adam(param_groups1)
            # else:
            #     optimizer = torch.optim.Adam(param_groups2)


            print ("-"*100)
            train_start_time = timeit.default_timer()
            total_loss = 0

            encoder.train()

            snapshot_list = train_data['edge_index']
            total_loss = 0

            for snapshot_idx in range(train_data['time_length']):

                # neg_edges = negative_sampling(pos_index, num_nodes=num_nodes, num_neg_samples=(pos_index.size(1)*1), force_undirected = True)
                if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                    cur_index = snapshot_list[snapshot_idx]
                    cur_index = cur_index.long().to(args.device)
                    last_states = {
                        "h1": None,
                        "h2": None,
                        "n_state": None,
                        "n_state_two": None,
                        "res1": None,
                        "res2": None,
                        "r_pool1": None,
                        "r_pool2": None
                    }
                    edge_index = cur_index

                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    edge_index = prev_index

                # pass through the encoder
                current_embeddings, current_states = encoder(node_feat, edge_index, states=last_states)

                # update last-embeddings for the next round
                last_states = current_states

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)

                neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_index.shape[1],),
                        dtype=torch.long,
                        device=args.device,
                    )

                # print(current_embeddings)
                pos_pred = encoder.decode((pos_index[0], pos_index[1]), current_embeddings)
                neg_pred = encoder.decode((pos_index[0], neg_dst), current_embeddings)

                # if snapshot_idx%10  == 0:
                #     print(f"INFO: Epoch {epoch}, Snapshot {snapshot_idx}, pos_pred: {pos_pred.mean().item()}, neg_pred: {neg_pred.mean().item()}")
                
                loss = criterion(pos_pred, torch.ones_like(pos_pred))
                loss += criterion(neg_pred, torch.zeros_like(neg_pred))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() / pos_index.shape[1]

            train_time = timeit.default_timer() - train_start_time
            print (f'INFO: Epoch {epoch}/{num_epochs}, Loss: {total_loss}')
            print ("INFO: Train time: ", train_time)

            # VALIDATION
            evaluator = Evaluator(name="tgbl-wiki") #reuse MRR evaluator from TGB
            metric = "mrr"
            neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")

            #* load the val negative samples
            neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_val_ns.pkl", split_mode="val")
            # neg_sampler.load_eval_set(fname=f"{args.dataset}_val_ns.pkl", split_mode="val")

            val_start_time = timeit.default_timer()
            test_tgb(val_data, node_feat,
             encoder,
             current_states,
             neg_sampler, evaluator,
             metric, split_mode='val')


            val_metric, val_last_states = test_tgb(val_data, node_feat, 
                                                    encoder,
                                                    current_states, 
                                                    neg_sampler, evaluator,
                                                    metric, split_mode='val')

            val_time = timeit.default_timer() - val_start_time
            print(f"INFO: Validation {metric}: {val_metric}")
            print (f"INFO: Validation time: {val_time}")

            if args.wandb:
                wandb.log({"train_loss": avg_loss, f"val_{metric}": val_mrr, "epoch": epoch})
        

            # TEST
            # report test results when validation improves
            if (val_metric > best_val):
                patience_counter = 0
                best_val = val_metric
                neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_test_ns.pkl", split_mode="test",)
                # neg_sampler.load_eval_set(fname=f"{args.dataset}_test_ns.pkl", split_mode="test",)

                test_start_time = timeit.default_timer()

                test_metric, test_last_states = test_tgb(test_data, node_feat, 
                                                        encoder,
                                                        val_last_states, 
                                                        neg_sampler, evaluator,
                                                        metric, split_mode='test')



                test_time = timeit.default_timer() - test_start_time
                best_val = val_metric
                best_test = test_metric
                best_epoch = epoch

                print (f"INFO: Test metric: {test_metric}")
                print (f"INFO: Test time: {test_time}")
                print ("-"*100)
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience and epoch >= 100:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation MRR.")
                break

        if args.wandb:
            wandb.finish()

        print(f"INFO: Run {run + 1} finished.")
        print(f"INFO: Best Epoch: {best_epoch}")
        print(f"INFO: Best Validation MRR: {best_val}")
        print(f"INFO: Final Test MRR: {best_test}")
        print("-" * 100)
