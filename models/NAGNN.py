import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell
from torch_geometric.nn import GATConv, MessagePassing
from torch_scatter import scatter_mean, scatter
from torch_geometric.utils import degree
from torch_sparse import spspmm

def build_padded_neighbors(edge_index: torch.LongTensor, 
                                              num_nodes, 
                                              cap=32, 
                                              device='cuda'):
    """
    A fully vectorized version to build padded neighbor lists with a capacity.
    """
    src, dst = edge_index
    E = src.size(0)
    
    if E == 0:
        return (torch.full((num_nodes, cap), -1, dtype=torch.long, device=device),
                torch.zeros((num_nodes,), dtype=torch.long, device=device))

    # 1. Assign a random priority to each edge
    priorities = torch.rand(E, device=device)

    # 2. Stable sort by source node ID, then by priority to group neighbors
    # We sort by priority first to break ties randomly, then stable sort by src
    perm_priority = priorities.argsort()
    src, dst = src[perm_priority], dst[perm_priority]

    perm_src = src.argsort(stable=True)
    src_s, dst_s = src[perm_src], dst[perm_src]

    # 3. Compute local rank of each neighbor within its source group
    is_new_group = torch.cat([torch.tensor([True], device=device), src_s[1:] != src_s[:-1]])
    group_starts = torch.where(is_new_group)[0]
    group_ids = torch.zeros_like(src_s)
    group_ids.scatter_(0, group_starts, 1)
    group_ids = group_ids.cumsum(0) - 1
    local_ranks = torch.arange(E, device=device) - group_starts[group_ids]

    # 4. Keep only neighbors with a local rank less than the capacity
    keep_mask = local_ranks < cap
    src_kept = src_s[keep_mask]
    dst_kept = dst_s[keep_mask]
    rank_kept = local_ranks[keep_mask]
    
    # 5. Build the final padded matrix
    linear_idx = src_kept * cap + rank_kept
    neigh_idx_flat = torch.full((num_nodes * cap,), -1, dtype=torch.long, device=device)
    neigh_idx_flat[linear_idx] = dst_kept
    neigh_idx = neigh_idx_flat.view(num_nodes, cap)

    # 6. Compute final degrees
    degs = torch.bincount(src, minlength=num_nodes)
    final_degs = torch.minimum(degs, torch.tensor(cap, device=device, dtype=degs.dtype))

    return neigh_idx, final_degs

# -------------------------
# RandomProjectionHasher: maps features -> slot index via signed random projections
# -------------------------
class RandomProjectionHasher:
    def __init__(self, d: int, k: int, R: int, seed: int = 0, device: torch.device = None):
        """
        d: feature dim
        k: number of bits in projection
        R: number of slots (reservoir capacity)
        """
        self.d = d
        self.k = k
        self.R = R
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        # gaussian projection matrix
        self.P = torch.randn((d, k), device=self.device)  # fixed random projections

        # precompute bit weights (1<<i) as int64
        self.bit_weights = (1 << torch.arange(k, dtype=torch.long, device=self.device)).unsqueeze(0)  # (1,k)

    def features_to_slot(self, feats: torch.Tensor) -> torch.LongTensor:
        """
        feats: (..., d) float tensor
        returns ints in [0, R-1] of shape (...,)
        """
        original_shape = feats.shape[:-1]
        x = feats.view(-1, self.d)   # (M, d)
        # x = F.normalize(x, p=2, dim=-1)
        z = x @ self.P               # (M, k)
        bits = (z > 0).to(torch.long)  # (M, k)
        # convert bit vector to integer
        bit_int = (bits * self.bit_weights).sum(dim=1)  # (M,)
        slot = (bit_int % self.R).to(torch.long)
        return slot.view(*original_shape)

class Reservoir(nn.Module):
    def __init__(self, N: int, feat_dim: int, R: int = 16, kbits: int = None,
                 seed: int = 0, agg_mode: str = 'mean', device: torch.device = None):
        """
        N: number of nodes
        feat_dim: hidden dim
        R: buckets per node
        agg_mode: 'mean' or 'sum'
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.N = N
        self.feat_dim = feat_dim
        self.R = R
        if kbits is None:
            kbits = max(8, math.ceil(1.5 * math.log2(R + 1)))
        self.hasher = RandomProjectionHasher(feat_dim, kbits, R, seed=seed, device=self.device)
        self.agg_mode = agg_mode
        # GRUCell for per-slot temporal refinement applied to flattened slots
        self.grucell1 = nn.GRUCell(feat_dim, feat_dim)
        self.grucell2 = nn.GRUCell(feat_dim, feat_dim)

        # buffers for reservoirs (initialized zeros)
        self.register_buffer('res1', torch.zeros((N, R, feat_dim), dtype=torch.float32, device=self.device), persistent=False)
        self.register_buffer('res2', torch.zeros((N, R, feat_dim), dtype=torch.float32, device=self.device), persistent=False)

    def _aggregate_to_buckets(self, src_nodes: torch.LongTensor, cand_nodes: torch.LongTensor,
                              node_feats: torch.Tensor, total_slots: int):
        """
        Aggregate candidate nodes into per-(src_node, bucket) slots.
        src_nodes: (M,) int source node ids for each candidate (u in (u->w))
        cand_nodes: (M,) int candidate target node ids (w)
        node_feats: (N, d)
        total_slots = N * R
        Returns aggregated_vals_flat (total_slots, d) and counts_flat (total_slots,)
        """
        device = self.device
        # get candidate feat vectors (handle -1 indices)
        valid_mask = cand_nodes >= 0
        if valid_mask.sum() == 0:
            # empty
            agg = torch.zeros((total_slots, self.feat_dim), device=device)
            counts = torch.zeros((total_slots,), dtype=torch.long, device=device)
            return agg, counts

        cand_nodes_valid = cand_nodes[valid_mask]
        src_nodes_valid = src_nodes[valid_mask]
        feats_valid = node_feats[cand_nodes_valid]  # (M_valid, d)

        # compute slot per candidate
        slots_valid = self.hasher.features_to_slot(feats_valid)                 # (M_valid,)
        flat_slot = src_nodes_valid * self.R + slots_valid                  # (M_valid,)

        # scatter_add sums into flat_slot indices
        total_slots = total_slots
        agg = torch.zeros((total_slots, self.feat_dim), device=device)
        idx_expand = flat_slot.unsqueeze(-1).expand(-1, self.feat_dim)
        agg = agg.scatter_add(0, idx_expand, feats_valid)

        counts = torch.zeros((total_slots,), dtype=torch.long, device=device)
        counts = counts.scatter_add(0, flat_slot, torch.ones_like(flat_slot, dtype=torch.long))

        return agg, counts

    def update(self, node_feats: torch.Tensor, edge_index: torch.LongTensor):
        """
        Update both 1-hop and 2-hop reservoirs given a snapshot.
        node_feats: (N, d)
        edge_index: (2, E) edges (src -> dst)
        """
        device = self.device
        N, d = node_feats.shape
        assert N == self.N and d == self.feat_dim

        src, dst = edge_index
        E = src.size(0)
        total_slots = self.N * self.R

        # 1) Build padded neighbor lists: neigh_idx (N, D1), degs
        neigh_idx_1, deg1 = build_padded_neighbors(edge_index, self.N, cap=32, device=device)  # (N, D1)
        D1 = neigh_idx_1.size(1)

        # --- 1-hop: for each edge (u->v) we already have candidate (u, v) by using src,dst arrays ---
        if E == 0:
            # no edges: tilde zeros
            tilde1_flat = torch.zeros((total_slots, d), device=device)
        else:
            # candidates for 1-hop are simply (src, dst)
            src_flat_1 = src  # (E,)
            cand_flat_1 = dst  # (E,)
            agg1, counts1 = self._aggregate_to_buckets(src_flat_1, cand_flat_1, node_feats, total_slots)
            if self.agg_mode == 'mean':
                counts1_f = counts1.to(torch.float32).clamp(min=1.0)
                tilde1_flat = (agg1 / counts1_f.unsqueeze(-1))
            else:
                tilde1_flat = agg1  # sum

        # 2) 2-hop: need all pairs (u -> v -> w) -> candidate (u,w)
        # We have neigh_idx_1: (N, D1) padded with -1
        # For each u, its neighbors are neigh_idx_1[u, :deg_u]; for each neighbor v we take neigh_idx_1[v, :deg_v]
        # We'll form pairs by: for each u, take neighbor list A (len D1), then gather neighbors-of-neighbors B = neigh_idx_1[A] -> shape (D1, D1)
        # Vectorized strategy: safe_idx = neigh_idx_1.clamp(min=0), then two_hop_matrix = neigh_idx_1[safe_idx] -> (N, D1, D1)
        # Mask invalid entries where neigh_idx_1 == -1
        safe_idx = neigh_idx_1.clamp(min=0)  # (N, D1)
        # take neighbors-of-neighbors: (N, D1, D1)
        two_hop = neigh_idx_1[safe_idx]  # fancy indexing: safe_idx flattened used to index rows; result shape (N*D1, D1) then reshape
        # Note: above uses advanced indexing; to ensure shape correctness, we'll do explicit:
        safe_flat = safe_idx.view(-1)  # (N*D1,)
        two_hop_flat = neigh_idx_1[safe_flat]  # (N*D1, D1)
        two_hop = two_hop_flat.view(N, D1, D1)  # (N, D1, D1)

        # Prepare src indices repeated: for each u, we have D1*D1 candidate positions (some padded)
        u_ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(1).unsqueeze(2).expand(N, D1, D1).reshape(-1)  # (N*D1*D1,)
        cand_w = two_hop.reshape(-1)  # (N*D1*D1,)
        # mask out invalid (where original neighbor missing or second hop missing)
        mask_first = (neigh_idx_1.view(-1) != -1).unsqueeze(1).expand(-1, D1).reshape(-1)  # whether first hop exists for that slot
        mask_second = cand_w != -1
        valid_mask = mask_first & mask_second
        if valid_mask.sum() == 0:
            tilde2_flat = torch.zeros((total_slots, d), device=device)
        else:
            src_valid = u_ids[valid_mask]
            cand_valid = cand_w[valid_mask]
            agg2, counts2 = self._aggregate_to_buckets(src_valid, cand_valid, node_feats, total_slots)
            if self.agg_mode == 'mean':
                counts2_f = counts2.to(torch.float32).clamp(min=1.0)
                tilde2_flat = (agg2 / counts2_f.unsqueeze(-1))
            else:
                tilde2_flat = agg2

        # reshape tilde flats into (N, R, d)
        tilde1 = tilde1_flat.view(N, self.R, d)
        tilde2 = tilde2_flat.view(N, self.R, d)

        # Zero-out empty slots where counts==0 to avoid artifacts for mean==sum
        # For tilde1: counts1 per flatslot available if E>0
        if E > 0:
            counts1_mat = counts1.view(N, self.R)
            if self.agg_mode == 'mean':
                tilde1[counts1_mat == 0] = 0.0
        else:
            tilde1 = torch.zeros_like(self.res1)

        # For tilde2: counts2 mat available if any valid; else we already zeroed
        # counts2 variable may not be defined if valid==0; handle:
        try:
            counts2_mat = counts2.view(N, self.R)
            if self.agg_mode == 'mean':
                tilde2[counts2_mat == 0] = 0.0
        except Exception:
            tilde2 = torch.zeros_like(self.res2)

        # --- Temporal refinement via GRU per-slot ---
        prev1 = self.res1.view(-1, d)
        tilde1_flat2 = tilde1.view(-1, d)
        updated1_flat = self.grucell1(tilde1_flat2, prev1)#.detach()
        new_res1 = updated1_flat.view(N, self.R, d)

        prev2 = self.res2.view(-1, d)
        tilde2_flat2 = tilde2.view(-1, d)
        updated2_flat = self.grucell2(tilde2_flat2, prev2)#.detach()
        new_res2 = updated2_flat.view(N, self.R, d)

        # write back
        self.res1 = new_res1.detach()
        self.res2 = new_res2.detach()

        return new_res1, new_res2

    def get_reservoirs(self):
        return self.res1, self.res2


class SimpleGNNLayer(MessagePassing):
    """
    一个简单的 GNN 层，聚合邻居特征的均值。
    这等价于执行 D^(-1) * A * X 的传播规则。
    """
    def __init__(self, mode='sum'):
        # 'mean' 聚合会自动处理求和与除以度数的操作
        super().__init__(aggr=mode)

    def forward(self, x, edge_index):
        # propagate 方法会调用 message 和 aggregate
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j 代表每条边上源节点(j)的特征
        # 我们直接传递这个特征作为消息
        return x_j

class NeighborhoodAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, mode='sum'):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim*2, hidden_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*2, hidden_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*2, hidden_dim*2)
        # )
        self.gnn_layer = SimpleGNNLayer(mode=mode)
        # GRU_n 用于时间平滑
        self.gru_n = GRUCell(hidden_dim*2, hidden_dim*2)

    def forward(self, h_i, edge_index, prev_n_state, prev_n_state_two):
        row, col = edge_index
        deg1 = torch.bincount(row, minlength=h_i.size(0)).float().clamp(min=1) # clamp(min=1)防止除以0
        deg1_inv = deg1.pow(-1).view(-1, 1)
        deg1 = deg1.view(-1,1)
        num_nodes = h_i.size(0)
        # --- 1. 一阶邻居聚合 ---
        n_i_1hop_fir = self.gnn_layer(h_i, edge_index)
        n_i_1hop_sec = self.gnn_layer(h_i*h_i, edge_index)
        # mu1 = n_i_1hop_fir*deg1_inv
        # sqr1 = n_i_1hop_sec*deg1_inv
        # var1 = sqr1 - mu1.pow(2)
        # var1 = torch.relu(var1)
        # sigma1 = torch.sqrt(var1)
        # epsilon1 = torch.randn_like(sigma1)
        # n_i_1hop = n_i_1hop_fir + deg1*sigma1*epsilon1

        n_i_1hop = torch.cat([n_i_1hop_fir, n_i_1hop_sec], dim=-1)
        # n_i_1hop = self.mlp(n_i_1hop)

        if prev_n_state is None:
            prev_n_state = torch.zeros_like(n_i_1hop)
        tilde_n_i = self.gru_n(n_i_1hop, prev_n_state)#.detach()

        # --- 2. 二阶邻居聚合 ---
        deg2 = self.gnn_layer(deg1.view(-1, 1), edge_index).view(-1)
        deg2 = deg2.clamp(min=1) # 防止除以0
        deg2_inv = deg2.pow(-1).view(-1, 1)
        deg2 = deg2.view(-1, 1)

        n_i_2hop_fir = self.gnn_layer(n_i_1hop_fir, edge_index)
        n_i_2hop_sec = self.gnn_layer(n_i_1hop_sec, edge_index)
        # mu2 = n_i_2hop_fir*deg2_inv
        # sqr2 = n_i_2hop_sec*deg2_inv
        # var2 = sqr2 - mu2.pow(2)
        # var2 = torch.relu(var2)
        # sigma2 = torch.sqrt(var2)
        # epsilon2 = torch.randn_like(sigma2)
        # n_i_2hop = n_i_2hop_fir + deg2*sigma2*epsilon2   
        
        n_i_2hop = torch.cat([n_i_2hop_fir, n_i_2hop_sec], dim=-1)
        # n_i_2hop = self.mlp(n_i_2hop)

        if prev_n_state_two is None:
            prev_n_state_two = torch.zeros_like(n_i_2hop)

        tilde_n_i_2hop = self.gru_n(n_i_2hop, prev_n_state_two)#.detach()

        
        return tilde_n_i, tilde_n_i_2hop

class ReSASGNN(nn.Module):
    """
    ReSAS-GNN integrated with snapshot reservoir manager.
    - neighbor_agg: expected to produce (n_state, n_state_two) from (h2, edge_index, prev_n_state, prev_n_state_two)
      If your existing neighbor_agg can't accept extra fusion, we fuse externally via `fusion_gru`.
    """
    def __init__(self, input_dim, model_dim, num_nodes,
                 reservoir_size=16, num_buckets=128, agg_mode="sum", res_mode="mean", device="cuda"):
        super().__init__()
        self.device = torch.device(device if isinstance(device, str) else device)
        self.num_nodes = num_nodes
        self.model_dim = model_dim  # dict-like with 'h'
        self.h = model_dim['h']
        # GNN layers (your original setup)
        self.conv1 = GATConv(input_dim, self.h // 2, heads=2)
        self.conv2 = GATConv(self.h, self.h // 2, heads=2)

        self.h1_gru = GRUCell(self.h, self.h)
        self.h2_gru = GRUCell(self.h, self.h)

        # neighbor aggregator (user-provided)
        self.neighbor_agg = NeighborhoodAggregator(input_dim=self.h, hidden_dim=self.h, mode=agg_mode)  # keep as-is

        # reservoir managers for 1-hop and 2-hop
        kbits = max(32, math.ceil(1.5 * math.log2(reservoir_size + 1)))
        self.resman = Reservoir(N=num_nodes, feat_dim=self.h, R=reservoir_size,
                                   kbits=kbits, seed=1234, agg_mode=res_mode, device=self.device)

        #a small MLP to pool reservoir slot features into fixed-dim r_pool
        self.pool_mlp = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h)
        )

        # predictor: keep your original predictor shape and dropout etc.
        self.predictor = nn.Sequential(
            nn.Linear(self.h * 7, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, 1)
        )

        self.attention_layer = nn.Linear(self.h, 1)
        # internal snapshot counter (if user doesn't supply t)
        self._snapshot_ptr = 0

    def forward(self, x, edge_index, states, snapshot_idx: int = None):
        """
        x: input node features (N, input_dim) — you'll likely pass GNN input
        edge_index: (2,E)
        states: dict containing previous states (h1, h2, n_state, n_state_two) and optionally reservoirs
        snapshot_idx: optional integer index for this snapshot; if None uses internal counter
        """

        if snapshot_idx is None:
            t = self._snapshot_ptr
            self._snapshot_ptr += 1
        else:
            t = int(snapshot_idx)
            self._snapshot_ptr = t + 1

        # --- unpack states ---
        prev_h1 = states.get('h1')
        prev_h2 = states.get('h2')
        prev_n_state = states.get('n_state')
        prev_n_state_two = states.get('n_state_two')
        prev_res1 = states.get('res1') 
        prev_res2 = states.get('res2')
        # reservoirs may be stored externally; but ReservoirManager keeps internal state

        # initialize if None
        h = self.model_dim['h']
        if prev_h1 is None:
            prev_h1 = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_h2 is None:
            prev_h2 = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state is None:
        #     prev_n_state = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state_two is None:
        #     prev_n_state_two = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_res1 is not None:
            self.resman.res1 = prev_res1.to(self.device)
        if prev_res2 is not None:
            self.resman.res2 = prev_res2.to(self.device)

        # --- GNN layers (you may replace Identity with your GATConv results) ---
        # Here we assume x already is processed features for convs; if not, apply convs.
        h1_gnn = F.relu(self.conv1(x, edge_index))
        h1 = self.h1_gru(h1_gnn, prev_h1)#.detach()
        h2_gnn = F.relu(self.conv2(h1, edge_index))
        h2 = self.h2_gru(h2_gnn, prev_h2)#.detach()

        # --- Reservoir update for this snapshot (uses h2 as candidate feature sources) ---
        # Note: ReservoirManager.update_snapshot expects `node_feats` to be of dim d=h.
        res1, res2 = self.resman.update(node_feats=h2, edge_index=edge_index)  # each (N,R,h)
        N, R, d = res1.shape
        transformed_vals1 = self.pool_mlp(res1)
        attention_scores1 = self.attention_layer(transformed_vals1).squeeze(-1)
        attention_weights1 = torch.softmax(attention_scores1, dim=1)
        r_pool1 = torch.sum(transformed_vals1 * attention_weights1.unsqueeze(-1), dim=1)

        transformed_vals2 = self.pool_mlp(res2)
        attention_scores2 = self.attention_layer(transformed_vals2).squeeze(-1)
        attention_weights2 = torch.softmax(attention_scores2, dim=1)
        r_pool2 = torch.sum(transformed_vals2 * attention_weights2.unsqueeze(-1), dim=1)

        r_pool1 = self.pool_mlp(res1.view(-1, d)).view(N, R, d).sum(dim=1) 
        r_pool2 = self.pool_mlp(res2.view(-1, d)).view(N, R, d).sum(dim=1)
        # --- neighbor aggregator: produce base neighbor states (unchanged API) ---
        n_state, n_state_two = self.neighbor_agg(h2, edge_index, prev_n_state, prev_n_state_two)

        # --- final node embedding ---
        final_node_embeddings = torch.cat([h2, n_state, n_state_two, r_pool1, r_pool2], dim=-1) 
        # final_node_embeddings = torch.cat([h2, n_state, n_state_two], dim=-1) 

        # --- prepare next_states and include reservoir metadata (optional) ---
        next_states = {
            "h1": h1.detach(),
            "h2": h2.detach(),
            "n_state": n_state.detach(),
            "n_state_two": n_state_two.detach(),
            # optionally we can export reservoir buffers into states for checkpointing
            'res1': res1.detach(),
            'res2': res2.detach(),
            'r_pool1': r_pool1.detach(),
            'r_pool2': r_pool2.detach( )
        }

        return final_node_embeddings, next_states

    def decode(self, edges, h):
        src, dst = edges
        feats_src = h[src]
        feats_dst = h[dst]
        feats = feats_src * feats_dst
        return torch.sigmoid(self.predictor(feats))

class ReSASGNN1(nn.Module):
    """
    ReSAS-GNN integrated with snapshot reservoir manager.
    - neighbor_agg: expected to produce (n_state, n_state_two) from (h2, edge_index, prev_n_state, prev_n_state_two)
      If your existing neighbor_agg can't accept extra fusion, we fuse externally via `fusion_gru`.
    """
    def __init__(self, input_dim, model_dim, num_nodes,
                 reservoir_size=16, num_buckets=128, agg_mode="sum", res_mode="mean", device="cuda"):
        super().__init__()
        self.device = torch.device(device if isinstance(device, str) else device)
        self.num_nodes = num_nodes
        self.model_dim = model_dim  # dict-like with 'h'
        self.h = model_dim['h']
        # GNN layers (your original setup)
        self.conv1 = GATConv(input_dim, self.h // 2, heads=2)
        self.conv2 = GATConv(self.h, self.h // 2, heads=2)

        self.h1_gru = GRUCell(self.h, self.h)
        self.h2_gru = GRUCell(self.h, self.h)

        # neighbor aggregator (user-provided)
        self.neighbor_agg = NeighborhoodAggregator(input_dim=self.h, hidden_dim=self.h, mode=agg_mode)  # keep as-is

        # reservoir managers for 1-hop and 2-hop
        kbits = max(32, math.ceil(1.5 * math.log2(reservoir_size + 1)))
        self.resman = Reservoir(N=num_nodes, feat_dim=self.h, R=reservoir_size,
                                   kbits=kbits, seed=1234, agg_mode=res_mode, device=self.device)

        #a small MLP to pool reservoir slot features into fixed-dim r_pool
        self.pool_mlp = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h)
        )

        # predictor: keep your original predictor shape and dropout etc.
        self.predictor = nn.Sequential(
            nn.Linear(self.h * 5, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, 1)
        )

        self.attention_layer = nn.Linear(self.h, 1)
        # internal snapshot counter (if user doesn't supply t)
        self._snapshot_ptr = 0

    def forward(self, x, edge_index, states, snapshot_idx: int = None):
        """
        x: input node features (N, input_dim) — you'll likely pass GNN input
        edge_index: (2,E)
        states: dict containing previous states (h1, h2, n_state, n_state_two) and optionally reservoirs
        snapshot_idx: optional integer index for this snapshot; if None uses internal counter
        """

        if snapshot_idx is None:
            t = self._snapshot_ptr
            self._snapshot_ptr += 1
        else:
            t = int(snapshot_idx)
            self._snapshot_ptr = t + 1

        # --- unpack states ---
        prev_h1 = states.get('h1')
        prev_h2 = states.get('h2')
        prev_n_state = states.get('n_state')
        prev_n_state_two = states.get('n_state_two')
        prev_res1 = states.get('res1') 
        prev_res2 = states.get('res2')
        # reservoirs may be stored externally; but ReservoirManager keeps internal state

        # initialize if None
        h = self.model_dim['h']
        if prev_h1 is None:
            prev_h1 = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_h2 is None:
            prev_h2 = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state is None:
        #     prev_n_state = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state_two is None:
        #     prev_n_state_two = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_res1 is not None:
            self.resman.res1 = prev_res1.to(self.device)
        if prev_res2 is not None:
            self.resman.res2 = prev_res2.to(self.device)

        # --- GNN layers (you may replace Identity with your GATConv results) ---
        # Here we assume x already is processed features for convs; if not, apply convs.
        h1_gnn = F.relu(self.conv1(x, edge_index))
        h1 = self.h1_gru(h1_gnn, prev_h1)#.detach()
        h2_gnn = F.relu(self.conv2(h1, edge_index))
        h2 = self.h2_gru(h2_gnn, prev_h2)#.detach()

        # --- neighbor aggregator: produce base neighbor states (unchanged API) ---
        n_state, n_state_two = self.neighbor_agg(h2, edge_index, prev_n_state, prev_n_state_two)

        # --- final node embedding ---
        final_node_embeddings = torch.cat([h2, n_state, n_state_two], dim=-1) 
        # final_node_embeddings = torch.cat([h2, n_state, n_state_two], dim=-1) 

        # --- prepare next_states and include reservoir metadata (optional) ---
        next_states = {
            "h1": h1.detach(),
            "h2": h2.detach(),
            "n_state": n_state.detach(),
            "n_state_two": n_state_two.detach()
        }

        return final_node_embeddings, next_states

    def decode(self, edges, h):
        src, dst = edges
        feats_src = h[src]
        feats_dst = h[dst]
        feats = feats_src * feats_dst
        return torch.sigmoid(self.predictor(feats))


class ReSASGNN2(nn.Module):
    """
    ReSAS-GNN integrated with snapshot reservoir manager.
    - neighbor_agg: expected to produce (n_state, n_state_two) from (h2, edge_index, prev_n_state, prev_n_state_two)
      If your existing neighbor_agg can't accept extra fusion, we fuse externally via `fusion_gru`.
    """
    def __init__(self, input_dim, model_dim, num_nodes,
                 reservoir_size=16, num_buckets=128, agg_mode="sum", res_mode="mean", device="cuda"):
        super().__init__()
        self.device = torch.device(device if isinstance(device, str) else device)
        self.num_nodes = num_nodes
        self.model_dim = model_dim  # dict-like with 'h'
        self.h = model_dim['h']
        # GNN layers (your original setup)
        self.conv1 = GATConv(input_dim, self.h // 2, heads=2)
        self.conv2 = GATConv(self.h, self.h // 2, heads=2)

        self.h1_gru = GRUCell(self.h, self.h)
        self.h2_gru = GRUCell(self.h, self.h)

        # reservoir managers for 1-hop and 2-hop
        kbits = max(32, math.ceil(1.5 * math.log2(reservoir_size + 1)))
        self.resman = Reservoir(N=num_nodes, feat_dim=self.h, R=reservoir_size,
                                   kbits=kbits, seed=1234, agg_mode=res_mode, device=self.device)

        #a small MLP to pool reservoir slot features into fixed-dim r_pool
        self.pool_mlp = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h)
        )

        # predictor: keep your original predictor shape and dropout etc.
        self.predictor = nn.Sequential(
            nn.Linear(self.h * 3, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, 1)
        )

        self.attention_layer = nn.Linear(self.h, 1)
        # internal snapshot counter (if user doesn't supply t)
        self._snapshot_ptr = 0

    def forward(self, x, edge_index, states, snapshot_idx: int = None):
        """
        x: input node features (N, input_dim) — you'll likely pass GNN input
        edge_index: (2,E)
        states: dict containing previous states (h1, h2, n_state, n_state_two) and optionally reservoirs
        snapshot_idx: optional integer index for this snapshot; if None uses internal counter
        """

        if snapshot_idx is None:
            t = self._snapshot_ptr
            self._snapshot_ptr += 1
        else:
            t = int(snapshot_idx)
            self._snapshot_ptr = t + 1

        # --- unpack states ---
        prev_h1 = states.get('h1')
        prev_h2 = states.get('h2')
        prev_n_state = states.get('n_state')
        prev_n_state_two = states.get('n_state_two')
        prev_res1 = states.get('res1') 
        prev_res2 = states.get('res2')
        # reservoirs may be stored externally; but ReservoirManager keeps internal state

        # initialize if None
        h = self.model_dim['h']
        if prev_h1 is None:
            prev_h1 = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_h2 is None:
            prev_h2 = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state is None:
        #     prev_n_state = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state_two is None:
        #     prev_n_state_two = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_res1 is not None:
            self.resman.res1 = prev_res1.to(self.device)
        if prev_res2 is not None:
            self.resman.res2 = prev_res2.to(self.device)

        # --- GNN layers (you may replace Identity with your GATConv results) ---
        # Here we assume x already is processed features for convs; if not, apply convs.
        h1_gnn = F.relu(self.conv1(x, edge_index))
        h1 = self.h1_gru(h1_gnn, prev_h1)#.detach()
        h2_gnn = F.relu(self.conv2(h1, edge_index))
        h2 = self.h2_gru(h2_gnn, prev_h2)#.detach()

        # --- Reservoir update for this snapshot (uses h2 as candidate feature sources) ---
        # Note: ReservoirManager.update_snapshot expects `node_feats` to be of dim d=h.
        res1, res2 = self.resman.update(node_feats=h2, edge_index=edge_index)  # each (N,R,h)
        N, R, d = res1.shape
        transformed_vals1 = self.pool_mlp(res1)
        attention_scores1 = self.attention_layer(transformed_vals1).squeeze(-1)
        attention_weights1 = torch.softmax(attention_scores1, dim=1)
        r_pool1 = torch.sum(transformed_vals1 * attention_weights1.unsqueeze(-1), dim=1)

        transformed_vals2 = self.pool_mlp(res2)
        attention_scores2 = self.attention_layer(transformed_vals2).squeeze(-1)
        attention_weights2 = torch.softmax(attention_scores2, dim=1)
        r_pool2 = torch.sum(transformed_vals2 * attention_weights2.unsqueeze(-1), dim=1)

        r_pool1 = self.pool_mlp(res1.view(-1, d)).view(N, R, d).sum(dim=1) 
        r_pool2 = self.pool_mlp(res2.view(-1, d)).view(N, R, d).sum(dim=1)
        # --- neighbor aggregator: produce base neighbor states (unchanged API) ---

        # --- final node embedding ---
        final_node_embeddings = torch.cat([h2, r_pool1, r_pool2], dim=-1) 
        # final_node_embeddings = torch.cat([h2, n_state, n_state_two], dim=-1) 

        # --- prepare next_states and include reservoir metadata (optional) ---
        next_states = {
            "h1": h1.detach(),
            "h2": h2.detach(),
            'res1': res1.detach(),
            'res2': res2.detach(),
            'r_pool1': r_pool1.detach(),
            'r_pool2': r_pool2.detach( )
        }

        return final_node_embeddings, next_states

    def decode(self, edges, h):
        src, dst = edges
        feats_src = h[src]
        feats_dst = h[dst]
        feats = feats_src * feats_dst
        return torch.sigmoid(self.predictor(feats))


class ReservoirSim(nn.Module):
    def __init__(self, N: int, feat_dim: int, R: int = 16, kbits: int = None,
                 seed: int = 0, agg_mode: str = 'mean', device: torch.device = None):
        """
        [Ablation Version: Uniform Random Sampling]
        Instead of LSH sketching, this version randomly samples R neighbors.
        
        N: number of nodes
        feat_dim: hidden dim
        R: buckets per node (Here, treated as sample size)
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.N = N
        self.feat_dim = feat_dim
        self.R = R
        self.agg_mode = agg_mode # Keep for API compatibility, though aggregation is now identity for sampled nodes
        
        # NOTE: Hasher is removed in this ablation
        # self.hasher = RandomProjectionHasher(...) 

        # GRUCell for per-slot temporal refinement
        # We keep this to ensure parameter count remains identical to the original model
        self.grucell1 = nn.GRUCell(feat_dim, feat_dim)
        self.grucell2 = nn.GRUCell(feat_dim, feat_dim)

        # buffers for reservoirs
        self.register_buffer('res1', torch.zeros((N, R, feat_dim), dtype=torch.float32, device=self.device), persistent=False)
        self.register_buffer('res2', torch.zeros((N, R, feat_dim), dtype=torch.float32, device=self.device), persistent=False)

    def _get_random_samples(self, candidates: torch.LongTensor, node_feats: torch.Tensor):
        """
        Helper to extract features from candidate indices.
        candidates: (N, K) indices, -1 denotes padding
        Returns: (N, R, d) features
        """
        N, K = candidates.shape
        device = self.device
        
        # If we have more candidates than R, we need to select R randomly
        # If candidates are already randomized (which build_padded_neighbors does), 
        # we can just take the first R.
        
        # However, to be strictly "Uniform Random" even if K > R:
        # build_padded_neighbors already returns the top-K based on random priority.
        # So taking columns 0..R is equivalent to random sampling without replacement.
        
        if K > self.R:
            # Take top R (since they are already randomly sorted candidates)
            selected_idx = candidates[:, :self.R]
        elif K < self.R:
            # Pad with -1
            pad_size = self.R - K
            padding = torch.full((N, pad_size), -1, dtype=torch.long, device=device)
            selected_idx = torch.cat([candidates, padding], dim=1)
        else:
            selected_idx = candidates

        # Gather features
        # Mask -1 indices to avoid indexing error (map -1 to 0 temporarily)
        mask = selected_idx != -1
        safe_idx = selected_idx.clone()
        safe_idx[~mask] = 0
        
        feats = node_feats[safe_idx] # (N, R, d)
        
        # Zero out features for invalid (padding) slots
        feats = feats * mask.unsqueeze(-1).float()
        
        return feats

    def update(self, node_feats: torch.Tensor, edge_index: torch.LongTensor):
        """
        Update 1-hop and 2-hop reservoirs using Uniform Random Sampling.
        """
        device = self.device
        N, d = node_feats.shape
        assert N == self.N and d == self.feat_dim

        # --- 1-hop Sampling ---
        # build_padded_neighbors uses random priorities internally.
        # By setting cap=self.R, we effectively perform uniform random sampling 
        # of R neighbors for each node.
        # neigh_idx_1: (N, R)
        neigh_idx_1, deg1 = build_padded_neighbors(edge_index, self.N, cap=self.R, device=device)
        
        # Construct 1-hop features
        tilde1 = self._get_random_samples(neigh_idx_1, node_feats) # (N, R, d)

        # --- 2-hop Sampling ---
        # Strategy: 
        # 1. Expand the already sampled 1-hop neighbors to find 2-hop candidates.
        # 2. Flatten the candidates and randomly pick R.
        
        # 1-hop indices (N, R) -> mask invalid
        safe_idx_1 = neigh_idx_1.clamp(min=0)
        
        # Look up neighbors of neighbors using the same neigh_idx_1 table
        # (This implies we sample 2-hop paths through the 1-hop samples)
        # Result shape: (N, R, R) -> (N, R*R)
        two_hop_candidates = neigh_idx_1[safe_idx_1].view(N, -1)
        
        # Mask: Valid 2-hop depends on valid 1-hop AND valid 2nd step
        mask_1 = (neigh_idx_1 != -1).unsqueeze(-1).expand(-1, -1, self.R).reshape(N, -1)
        mask_2 = (two_hop_candidates != -1)
        valid_mask = mask_1 & mask_2
        
        # Apply mask (set invalid to -1)
        two_hop_candidates[~valid_mask] = -1
        
        # Now we need to select R items from the R*R pool for each node.
        # Since the structure might be biased (first column corresponds to first neighbor),
        # we must shuffle to ensure true uniform sampling from the 2-hop pool.
        
        # Generate random priorities for the selection
        rand_priority = torch.rand_like(two_hop_candidates, dtype=torch.float)
        # Set priority of invalid entries to -infinity so they drift to the end
        rand_priority[two_hop_candidates == -1] = -1.0
        
        # Sort by random priority
        _, perm = torch.sort(rand_priority, dim=1, descending=True)
        
        # Gather the sorted candidates
        sorted_candidates = torch.gather(two_hop_candidates, 1, perm)
        
        # Select top R
        tilde2 = self._get_random_samples(sorted_candidates[:, :self.R], node_feats) # (N, R, d)

        # --- Temporal Refinement (Keep GRU Unchanged) ---
        # Even though the "slots" effectively change meaning every step (random neighbors),
        # we keep the GRU to maintain model architecture/parameter consistency.
        
        prev1 = self.res1.view(-1, d)
        tilde1_flat = tilde1.view(-1, d)
        updated1_flat = self.grucell1(tilde1_flat, prev1)
        new_res1 = updated1_flat.view(N, self.R, d)

        prev2 = self.res2.view(-1, d)
        tilde2_flat = tilde2.view(-1, d)
        updated2_flat = self.grucell2(tilde2_flat, prev2)
        new_res2 = updated2_flat.view(N, self.R, d)

        # write back
        self.res1 = new_res1.detach()
        self.res2 = new_res2.detach()

        return new_res1, new_res2

    def get_reservoirs(self):
        return self.res1, self.res2


class ReSASGNNSim(nn.Module):
    """
    ReSAS-GNN integrated with snapshot reservoir manager.
    - neighbor_agg: expected to produce (n_state, n_state_two) from (h2, edge_index, prev_n_state, prev_n_state_two)
      If your existing neighbor_agg can't accept extra fusion, we fuse externally via `fusion_gru`.
    """
    def __init__(self, input_dim, model_dim, num_nodes,
                 reservoir_size=16, num_buckets=128, agg_mode="sum", res_mode="mean", device="cuda"):
        super().__init__()
        self.device = torch.device(device if isinstance(device, str) else device)
        self.num_nodes = num_nodes
        self.model_dim = model_dim  # dict-like with 'h'
        self.h = model_dim['h']
        # GNN layers (your original setup)
        self.conv1 = GATConv(input_dim, self.h // 2, heads=2)
        self.conv2 = GATConv(self.h, self.h // 2, heads=2)

        self.h1_gru = GRUCell(self.h, self.h)
        self.h2_gru = GRUCell(self.h, self.h)

        # neighbor aggregator (user-provided)
        self.neighbor_agg = NeighborhoodAggregator(input_dim=self.h, hidden_dim=self.h, mode=agg_mode)  # keep as-is

        # reservoir managers for 1-hop and 2-hop
        kbits = max(32, math.ceil(1.5 * math.log2(reservoir_size + 1)))
        self.resman = ReservoirSim(N=num_nodes, feat_dim=self.h, R=reservoir_size,
                                   kbits=kbits, seed=1234, agg_mode=res_mode, device=self.device)

        #a small MLP to pool reservoir slot features into fixed-dim r_pool
        self.pool_mlp = nn.Sequential(
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Linear(self.h, self.h)
        )

        # predictor: keep your original predictor shape and dropout etc.
        self.predictor = nn.Sequential(
            nn.Linear(self.h * 7, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, self.h),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.h, 1)
        )

        self.attention_layer = nn.Linear(self.h, 1)
        # internal snapshot counter (if user doesn't supply t)
        self._snapshot_ptr = 0

    def forward(self, x, edge_index, states, snapshot_idx: int = None):
        """
        x: input node features (N, input_dim) — you'll likely pass GNN input
        edge_index: (2,E)
        states: dict containing previous states (h1, h2, n_state, n_state_two) and optionally reservoirs
        snapshot_idx: optional integer index for this snapshot; if None uses internal counter
        """

        if snapshot_idx is None:
            t = self._snapshot_ptr
            self._snapshot_ptr += 1
        else:
            t = int(snapshot_idx)
            self._snapshot_ptr = t + 1

        # --- unpack states ---
        prev_h1 = states.get('h1')
        prev_h2 = states.get('h2')
        prev_n_state = states.get('n_state')
        prev_n_state_two = states.get('n_state_two')
        prev_res1 = states.get('res1') 
        prev_res2 = states.get('res2')
        # reservoirs may be stored externally; but ReservoirManager keeps internal state

        # initialize if None
        h = self.model_dim['h']
        if prev_h1 is None:
            prev_h1 = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_h2 is None:
            prev_h2 = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state is None:
        #     prev_n_state = torch.zeros(self.num_nodes, h, device=self.device)
        # if prev_n_state_two is None:
        #     prev_n_state_two = torch.zeros(self.num_nodes, h, device=self.device)
        if prev_res1 is not None:
            self.resman.res1 = prev_res1.to(self.device)
        if prev_res2 is not None:
            self.resman.res2 = prev_res2.to(self.device)

        # --- GNN layers (you may replace Identity with your GATConv results) ---
        # Here we assume x already is processed features for convs; if not, apply convs.
        h1_gnn = F.relu(self.conv1(x, edge_index))
        h1 = self.h1_gru(h1_gnn, prev_h1)#.detach()
        h2_gnn = F.relu(self.conv2(h1, edge_index))
        h2 = self.h2_gru(h2_gnn, prev_h2)#.detach()

        # --- Reservoir update for this snapshot (uses h2 as candidate feature sources) ---
        # Note: ReservoirManager.update_snapshot expects `node_feats` to be of dim d=h.
        res1, res2 = self.resman.update(node_feats=h2, edge_index=edge_index)  # each (N,R,h)
        N, R, d = res1.shape
        transformed_vals1 = self.pool_mlp(res1)
        attention_scores1 = self.attention_layer(transformed_vals1).squeeze(-1)
        attention_weights1 = torch.softmax(attention_scores1, dim=1)
        r_pool1 = torch.sum(transformed_vals1 * attention_weights1.unsqueeze(-1), dim=1)

        transformed_vals2 = self.pool_mlp(res2)
        attention_scores2 = self.attention_layer(transformed_vals2).squeeze(-1)
        attention_weights2 = torch.softmax(attention_scores2, dim=1)
        r_pool2 = torch.sum(transformed_vals2 * attention_weights2.unsqueeze(-1), dim=1)

        r_pool1 = self.pool_mlp(res1.view(-1, d)).view(N, R, d).sum(dim=1) 
        r_pool2 = self.pool_mlp(res2.view(-1, d)).view(N, R, d).sum(dim=1)
        # --- neighbor aggregator: produce base neighbor states (unchanged API) ---
        n_state, n_state_two = self.neighbor_agg(h2, edge_index, prev_n_state, prev_n_state_two)

        # --- final node embedding ---
        final_node_embeddings = torch.cat([h2, n_state, n_state_two, r_pool1, r_pool2], dim=-1) 
        # final_node_embeddings = torch.cat([h2, n_state, n_state_two], dim=-1) 

        # --- prepare next_states and include reservoir metadata (optional) ---
        next_states = {
            "h1": h1.detach(),
            "h2": h2.detach(),
            "n_state": n_state.detach(),
            "n_state_two": n_state_two.detach(),
            # optionally we can export reservoir buffers into states for checkpointing
            'res1': res1.detach(),
            'res2': res2.detach(),
            'r_pool1': r_pool1.detach(),
            'r_pool2': r_pool2.detach( )
        }

        return final_node_embeddings, next_states

    def decode(self, edges, h):
        src, dst = edges
        feats_src = h[src]
        feats_dst = h[dst]
        feats = feats_src * feats_dst
        return torch.sigmoid(self.predictor(feats))