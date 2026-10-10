import os
import sys
import torch
import numpy as np
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch_geometric.loader import TemporalDataLoader

import timeit

from models.NAGNN import ReSASGNNSim 

def test_tgb(test_loader, test_snapshots, ts_list, node_feat,
             encoder, current_states, 
             current_embeddings, num_previous_edges,
             neg_sampler, evaluator,
             metric, split_mode='val'):
    
    encoder.eval()

    perf_list = []
    ts_idx = min(list(ts_list.keys()))
    max_ts_idx = max(list(ts_list.keys()))

    for batch in test_loader:
        pos_src, pos_dst, pos_t, pos_msg = (batch.src, batch.dst, batch.t, batch.msg,)
        neg_batch_list = neg_sampler.query_batch(np.array(pos_src.cpu()), np.array(pos_dst.cpu()), np.array(pos_t.cpu()), split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.full((1 + len(neg_batch),), pos_src[idx], device=args.device)
            query_dst = torch.tensor(
                        np.concatenate(
                            ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                            axis=0,
                        ),
                        device=args.device,
                    )
            with torch.no_grad():
                y_pred = encoder.decode((query_src, query_dst), current_embeddings)

            y_pred = y_pred.squeeze(dim=-1).detach()

            input_dict = {
            "y_pred_pos": np.array([y_pred[0].cpu()]),
            "y_pred_neg": np.array(y_pred[1:].cpu()),
            "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # these are the previous snapshot parameters, update before a new prediction
        # computer new number of edges
        num_current_edges = len(pos_src)
        num_previous_edges = num_previous_edges + num_current_edges
        # update last-embeddings for the next round
        last_embeddings = current_embeddings
        last_states = current_states
        
        #* update the model now if the prediction batch has moved to next snapshot
        while (pos_t[-1] > ts_list[ts_idx] and ts_idx < max_ts_idx):
            with torch.no_grad():
                cur_index = test_snapshots[ts_idx]
                cur_index = cur_index.long().to(args.device)
                current_embeddings, current_states = encoder(node_feat, cur_index, states=last_states)
    
            ts_idx += 1

    #* update to the final snapshot
    with torch.no_grad():
        cur_index = test_snapshots[max_ts_idx]
        cur_index = cur_index.long().to(args.device)
        num_current_edges = cur_index.shape[1]
        current_embeddings, current_states = encoder(node_feat, cur_index, states=last_states)
        
        # computer new number of edges
        num_previous_edges = num_previous_edges + num_current_edges
        # update last-embeddings for the next round
        last_embeddings = current_embeddings
        last_states = current_states

    test_metric = float(np.mean(np.array(perf_list)))

    return test_metric, last_states, last_embeddings



if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader
    from utils.logger import logger

    if args.wandb == 0:
        log_directory = args.output_folder
        log_filename = f"result_{args.lr}_{args.patience}_{args.max_epoch}.log"

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
    hidden_dim = 128
    node_feat_dim = 128
    pos_dim = args.pos_dim
    output_dim = 1
    link_pred_num_layers = 2

    model_dim = {
        "h": 32,
        "e_joint":128,
        "reservoir":128
    }

    # CTDG dataset
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets")
    full_data = dataset.get_TemporalData()
    full_data = full_data.to(args.device)
    # get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    train_edges = full_data[train_mask]
    val_edges = full_data[val_mask]
    test_edges = full_data[test_mask]

    #* set up TGB queries, this is only for val and test
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    min_dst_idx, max_dst_idx = int(full_data.dst.min()), int(full_data.dst.max())

    #* load the discretized version
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    print(f"INFO: Number of nodes: {num_nodes}")

    # set up node features: TODO: there is no node features for any dataset
    node_feat = dataset.node_feat
    if (node_feat is not None):
        print(f"DEBUG: node_feat is not None!")
        node_feat = node_feat.to(args.device)
        node_feat_dim = node_feat.size(1)
    else:
        print(f"DEBUG: node_feat is originally None! It is set to random features.")
        # node_feat_dim = node_feat_dim
        # node_feat = torch.randn((full_data.num_nodes, node_feat_dim)).to(args.device)
        node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)


    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print(f"INFO: Start a fresh run with random seed: {seed}")

        # define model
        encoder = ReSASGNNSim(
            input_dim=node_feat_dim,
            model_dim=model_dim,
            num_nodes=num_nodes,
            reservoir_size=args.reservoir_size,
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
        
        # criterion
        criterion = torch.nn.BCELoss()
        # criterion = torch.nn.MSELoss()

        best_val = 0
        best_test = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print ("-"*100)
            train_start_time = timeit.default_timer()
            total_loss = 0
            last_h = None
            last_reservoir = None # Reset reservoir state at the beginning of each epoch
            last_e_joint = None

            encoder.train()

            snapshot_list = train_data['edge_index']
            total_loss = 0
            for snapshot_idx in range(train_data['time_length']):
                if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                    cur_index = snapshot_list[snapshot_idx]
                    cur_index = cur_index.long().to(args.device)
                    # current_embeddings = encoder(node_feat, cur_index)
                    num_previous_edges = 0
                    edge_index = cur_index
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

                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    edge_index = prev_index
                    # current_embeddings = encoder(node_feat, prev_index, \
                    #     last_embeddings, None, num_previous_edges)

                # pass through encoder
                num_current_edges = edge_index.shape[1]
                current_embeddings, current_states = encoder(node_feat, edge_index, states=last_states)

                
                # computer new number of edges
                num_previous_edges = num_previous_edges + num_current_edges
                # update last-embeddings for the next round
                last_embeddings = current_embeddings
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
                
                pos_pred = encoder.decode((pos_index[0], pos_index[1]), current_embeddings)
                neg_pred = encoder.decode((pos_index[0], neg_dst), current_embeddings)


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
            val_snapshots = data['val_data']['edge_index']
            ts_list = data['val_data']['ts_map']
            val_loader = TemporalDataLoader(val_edges, batch_size=batch_size)
            evaluator = Evaluator(name=args.dataset)
            neg_sampler = dataset.negative_sampler
            dataset.load_val_ns()

            start_epoch_val = timeit.default_timer()

            val_metric, val_states, val_last_embeddings = test_tgb(val_loader, 
                                            val_snapshots, ts_list, node_feat, 
                                            encoder, last_states, last_embeddings,
                                            num_previous_edges, 
                                            neg_sampler, evaluator,
                                            metric, split_mode='val')
            
            val_time = timeit.default_timer() - start_epoch_val
            print(f"INFO: Validation {metric}: {val_metric}")
            print (f"INFO: Validation time: {val_time}")

            # TEST
            # report test results when validation improves
            if (val_metric > best_val):
                dataset.load_test_ns()
                test_snapshots = data['test_data']['edge_index']
                ts_list = data['test_data']['ts_map']
                test_loader = TemporalDataLoader(test_edges, batch_size=batch_size)
                neg_sampler = dataset.negative_sampler
                dataset.load_test_ns()

                test_start_time = timeit.default_timer()
                
                test_metric, _, _ = test_tgb(test_loader, 
                                              test_snapshots, ts_list, node_feat, 
                                              encoder,
                                              val_states, val_last_embeddings,
                                              num_previous_edges, 
                                              neg_sampler, evaluator,
                                              metric, split_mode='test')

                test_time = timeit.default_timer() - test_start_time
                best_val = val_metric
                best_test = test_metric

                print (f"INFO: Test metric: {test_metric}")
                print (f"INFO: Test time: {test_time}")
                print ("-"*100)

                if ((epoch - best_epoch) >= args.patience and epoch > 1):
                    best_epoch = epoch
                    break
                best_epoch = epoch
                
        print (f"INFO: Run {seed} finished.")
        print (f"INFO: Best epoch: {best_epoch}")
        print (f"INFO: Best val performance: {best_val}")
        print (f"INFO: Best test performance: {best_test}")
        print ("-"*100)

#python -u ctdg_nagnn_sim.py --model="NAGNN_SIM" --dataset="tgbl-wiki" -t "hourly" --lr "2e-4" --max_epoch "40" --num_runs "5" --patience "20" --device_id 3
#python -u ctdg_nagnn_sim.py --model="NAGNN_SIM" --dataset="tgbl-review" -t "monthly" --lr "2e-4" --max_epoch "40" --num_runs "5" --patience "20" --device_id 5 