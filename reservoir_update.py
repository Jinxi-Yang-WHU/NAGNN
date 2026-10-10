import torch
import torch.nn.functional as F
import math
from typing import Tuple
from models.NAGNN import ReSASGNN, ReservoirManager


# -------------------------
# Demo usage
# -------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = 8
    d = 16
    R = 4
    S = 6
    manager = ReservoirManager(N=N, d=d, R=R, S=S, seed=42, device=device, agg_mode='mean')

    # synthetic node features and synthetic snapshot edges
    node_feats = torch.randn((N, d), device=device)
    # make a small random graph
    num_edges = 20
    src = torch.randint(0, N, (num_edges,), device=device)
    dst = torch.randint(0, N, (num_edges,), device=device)
    edge_index = torch.stack([src, dst], dim=0)
    print(edge_index)
    t = 1
    manager.update_snapshot(node_feats=node_feats, edge_index=edge_index, t=t)

    keys, vals, last = manager.get_reservoir()
    print("Reservoir keys shape:", keys.shape)   # (N, R)
    print("Reservoir vals shape:", vals.shape)   # (N, R, d)
    print("Last update shape:", last.shape)
    print("Reservoir keys sample:\n", keys)
