# train_hgnnlp.py
import os
import sys
import torch
import numpy as np
import timeit
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from torch_geometric.data import Data

# 确保模型文件在可导入的路径中
from models.hawkes import HGNNLP

def test_tgb(test_data, 
             node_feat, 
             model,
             prev_snapshot_edges, 
             neg_sampler, 
             evaluator,
             metric, 
             split_mode='val'):
    """
    Corrected test function for HGNNLP, following the t -> t+1 prediction logic.

    Args:
        test_data (dict): The data dictionary for validation or testing.
        node_feat (torch.Tensor): Static node features.
        model (torch.nn.Module): The HGNNLP model.
        prev_snapshot_edges (torch.Tensor): Edge index from the last snapshot of the PREVIOUS phase 
                                            (e.g., last training snapshot for validation).
        neg_sampler (NegativeEdgeSampler): The TGB negative sampler.
        evaluator (Evaluator): The TGB evaluator.
        metric (str): The evaluation metric (e.g., 'mrr').
        split_mode (str): The split mode for the negative sampler ('val' or 'test').
    
    Returns:
        tuple[float, torch.Tensor]: A tuple containing the mean evaluation metric and the edge index 
                                    of the last snapshot in this phase.
    """
    model.eval()

    test_snapshots = test_data['edge_index'] 
    test_original_edges = test_data['original_edges']
    
    perf_list = []
    device = node_feat.device

    # Ensure chronological order for testing by sorting the dictionary keys
    sorted_snapshot_keys = sorted(test_snapshots.keys())
    
    # Initialize the graph structure with the last snapshot from the previous phase
    # This `current_input_edges` represents the graph at time t-1
    current_input_edges = prev_snapshot_edges.to(device)

    # Loop through each snapshot in the current phase (validation or test)
    for snapshot_idx in sorted_snapshot_keys:
        # =================================================================================
        # Step 1: Generate node embeddings using the graph structure from the PREVIOUS snapshot (t-1)
        # =================================================================================
        
        # HGNNLP expects an edge_attr tensor for edge_age. For snapshot-based models,
        # we can assume the age is 0 within the static graph of the previous snapshot.
        edge_attr = torch.zeros(current_input_edges.size(1), 1, device=device)
        
        # Create the PyG Data object based on the t-1 graph
        graph_data = Data(x=node_feat, edge_index=current_input_edges, edge_attr=edge_attr)
        
        with torch.no_grad():
            # h contains node embeddings computed from the t-1 graph
            h = model(graph_data)

        # =================================================================================
        # Step 2: Define the prediction TARGET as the edges from the CURRENT snapshot (t)
        # =================================================================================
        pos_edges = torch.from_numpy(test_original_edges[snapshot_idx]).long().to(device)

        # =================================================================================
        # Step 3: Iterate through each positive edge at time t and its sampled negatives to evaluate
        # =================================================================================
        for i in range(pos_edges.shape[1]):
            pos_src = pos_edges[0, i].item()
            pos_dst = pos_edges[1, i].item()
            pos_t = snapshot_idx # The timestamp for querying negative samples is the current snapshot index

            # Query the pre-sampled negative edges for this specific positive edge
            neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode=split_mode)

            for neg_batch in neg_batch_list:
                # Create a batch of edges: 1 positive + N negatives
                query_src = torch.tensor([pos_src] * (len(neg_batch) + 1), device=device)
                query_dst = torch.cat([
                    torch.tensor([pos_dst], device=device), 
                    torch.tensor(neg_batch).long().to(device)
                ], dim=0)
                
                # Predict scores for this batch using the embeddings 'h' from the t-1 graph
                with torch.no_grad():
                    pred = model.predictor(h[query_src], h[query_dst]).squeeze(-1)

                # Separate the scores for the positive edge and the negative edges
                pos_pred = pred[0]
                neg_pred = pred[1:]

                # Evaluate the performance for this single query
                input_dict = {
                    "y_pred_pos": np.array([pos_pred.cpu()]),
                    "y_pred_neg": neg_pred.cpu().numpy(),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])
    
        # =================================================================================
        # Step 4: <<< CRUCIAL UPDATE >>> Update the input graph for the NEXT iteration.
        # The graph structure of the current snapshot (t) becomes the input for predicting the next one (t+1).
        # =================================================================================
        current_input_edges = test_snapshots[snapshot_idx].long().to(device)

    # Calculate the final mean metric over all queries in this phase
    mean_metric = float(np.mean(np.array(perf_list)))
    
    # Return the final metric and the edge index of the very last snapshot in this phase.
    # This last edge index will be used as the starting point for the subsequent phase (e.g., testing).
    return mean_metric, current_input_edges


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

    # Data Loading...
    data = loader(dataset=args.dataset, time_scale=args.time_scale)

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    print(f"INFO: Number of nodes: {num_nodes}")

    # Set up node features (using random features as in the original script)
    node_feat_dim = 128
    node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)

    # Edge features are not used by HGNNLP's GAT/GCN layers in this setup, 
    # but the dimension is needed for the `Data` object if it existed.
    # We will generate a placeholder for `edge_age` later.
    edge_feat_dim = 1 

    for run in range(args.num_runs):
        seed = args.seed + run
        set_random(seed)
        print(f"INFO: Start a fresh run with random seed: {seed}")
        
        # Define HGNNLP model
        model = HGNNLP(n_node=num_nodes, 
                       n_feat=node_feat_dim, 
                       n_edge=edge_feat_dim, 
                       n_hidden=node_feat_dim, # Hidden dimension same as input
                       dropout=args.dropout, 
                       name=args.model_name, 
                       layers=args.layers, 
                       heads=args.heads,
                       layer_skip=args.layer_skip,
                       initial_skip=args.initial_skip,
                       norm=args.norm,
                       jodie=False # Using standard LinkPredictor
                      ).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_val = 0
        best_test = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(args.max_epoch):
            print ("-"*100)
            train_start_time = timeit.default_timer()
            
            model.train()
            
            total_loss = 0
            snapshot_list = train_data['edge_index']

            for snapshot_idx in range(train_data['time_length']):
                if snapshot_idx == 0:
                    pos_edge = snapshot_list[snapshot_idx].long().to(args.device)
                else:
                    pos_edge = snapshot_list[snapshot_idx-1].long().to(args.device)
                
                # Simple random negative sampling for training
                neg_dst = torch.randint(0, num_nodes, (pos_edge.size(1),), dtype=torch.long, device=args.device)
                neg_edge = torch.stack([pos_edge[0], neg_dst], dim=0)

                # Create Data object for the model
                # Assume edge_age is 0 for all edges in the snapshot
                edge_attr = torch.zeros(pos_edge.size(1), edge_feat_dim, device=args.device)
                graph_data = Data(x=node_feat, edge_index=pos_edge, edge_attr=edge_attr)
                
                optimizer.zero_grad()
                # Use the built-in train_step method from BaseLPModel
                loss = model.train_step(graph_data, pos_edge, neg_edge)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            train_time = timeit.default_timer() - train_start_time
            avg_loss = total_loss / train_data['time_length']
            print(f'INFO: Epoch {epoch}/{args.max_epoch}, Average Loss: {avg_loss:.4f}')
            print(f"INFO: Train time: {train_time:.4f}s")


            # VALIDATION
            evaluator = Evaluator(name="tgbl-wiki") # Reuse MRR evaluator from TGB
            metric = "mrr"
            neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")
            neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_val_ns.pkl", split_mode="val")
            train_snapshot_keys = sorted(train_data['edge_index'].keys())
            last_train_snapshot_key = train_snapshot_keys[-1]
            last_train_snapshot = train_data['edge_index'][last_train_snapshot_key]

            val_start_time = timeit.default_timer()
            val_metric, last_val_snapshot = test_tgb(
                test_data=val_data, 
                node_feat=node_feat, 
                model=model, 
                prev_snapshot_edges=last_train_snapshot, # Pass the last train snapshot
                neg_sampler=neg_sampler, 
                evaluator=evaluator,
                metric=metric, 
                split_mode='val'
            )               
            val_time = timeit.default_timer() - val_start_time
            print(f"INFO: Validation {metric}: {val_metric:.4f}")
            print(f"INFO: Validation time: {val_time:.4f}s")
                
            # TEST
            if val_metric > best_val:
                best_val = val_metric
                best_epoch = epoch
                patience_counter = 0
                
                neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_test_ns.pkl", split_mode="test")
                test_start_time = timeit.default_timer()
                test_metric, _ = test_tgb(
                    test_data=test_data, 
                    node_feat=node_feat, 
                    model=model,
                    prev_snapshot_edges=last_val_snapshot, # Pass the last validation snapshot
                    neg_sampler=neg_sampler, 
                    evaluator=evaluator,
                    metric=metric, 
                    split_mode='test'
                )

                test_time = timeit.default_timer() - test_start_time
                
                best_test = test_metric
                print(f"INFO: >>> New Best Found <<< Test {metric}: {best_test:.4f}")
                print(f"INFO: Test time: {test_time:.4f}s")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"INFO: Early stopping at epoch {epoch} due to no improvement in validation metric.")
                break

        print ("-"*100)
        print(f"INFO: Run {run+1} finished.")
        print(f"INFO: Best epoch: {best_epoch}")
        print(f"INFO: Best validation performance: {best_val:.4f}")
        print(f"INFO: Corresponding test performance: {best_test:.4f}")
        print ("-"*100)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


#python dtdg_hgnnlp.py --dataset uci -t weekly --model_name hgat --lr 0.001 --max_epoch 200 --patience 40 --dropout 0.1 --seed 1 --heads 2 --layers 2  --initial_skip --output_folder ./logs/
#python dtdg_hgnnlp.py  --dataset enron -t monthly --model_name hgat --lr 0.001 --max_epoch 200 --patience 40 --dropout 0.1 --seed 1 --num_runs 5 --heads 2 --layers 2  --initial_skip --output_folder ./logs/
#python dtdg_hgnnlp.py --dataset mooc -t daily --model_name hgat --lr 0.001 --max_epoch 200 --patience 40 --dropout 0.1 --seed 1 --num_runs 5 --heads 2 --layers 2  --initial_skip --output_folder ./logs/ --device_id 3
#python dtdg_hgnnlp.py --dataset social_evo -t daily --model_name hgat --lr 0.001 --max_epoch 200 --patience 40 --dropout 0.1 --seed 1 --num_runs 5 --heads 2 --layers 2  --initial_skip --output_folder ./logs/ --device_id 0
#python dtdg_hgnnlp.py --dataset contacts -t hourly --model_name hgat --lr 0.001 --max_epoch 200 --patience 40 --dropout 0.1 --seed 1 --num_runs 5 --heads 2 --layers 2  --initial_skip --output_folder ./logs/ --device_id 3
