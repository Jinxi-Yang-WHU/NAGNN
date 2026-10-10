import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell
from torch_geometric.nn import GATConv, GCNConv
from torch_scatter import scatter_mean, scatter
from torch_geometric.utils import degree

class NeighborhoodAggregator(nn.Module):
    """
    实现 ReSAS-GNN 第 3 节: 邻居聚合与时间平滑
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # MLP_a 用于计算注意力权重
        self.mlp_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # GRU_n 用于时间平滑
        self.gru_n = GRUCell(hidden_dim, hidden_dim)

    def forward(self, h_i, edge_index, prev_n_state):
        src, dst = edge_index
        
        # 1. 计算注意力权重 e_ij
        h_src = h_i[src]
        h_dst = h_i[dst]
        mlp_input = torch.cat([h_src, h_dst], dim=-1)
        e = self.mlp_a(mlp_input).squeeze(-1) # [E]
        
        # 2. 计算 softmax 得到 a_ij (使用 torch_geometric 的 scatter)
        from torch_geometric.utils import softmax
        a = softmax(e, src, num_nodes=h_i.size(0)) # [E]
        a = torch.ones_like(a)
        # 3. 聚合邻居表示得到 n_i^t
        n_i = torch.zeros_like(h_i) # [N, d]
        n_i.index_add_(0, src, a.unsqueeze(-1) * h_dst)
        
        # 4. GRU 细化
        if prev_n_state is None:
            prev_n_state = torch.zeros_like(h_i)
        
        tilde_n_i = self.gru_n(n_i, prev_n_state)
        
        return tilde_n_i, tilde_n_i.detach()


class ReSASGNN(nn.Module):
    """
    完整实现的 ReSAS-GNN 模型
    """
    def __init__(self, input_dim, hidden_dim, num_nodes,
                 reservoir_size=16, num_buckets=128, device="cuda"):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # GNN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # GRUs for temporal update
        self.h1_gru = GRUCell(hidden_dim, hidden_dim)
        self.h2_gru = GRUCell(hidden_dim, hidden_dim)

        # 解码器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim , hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, states):
        
        # --- 解包上一时刻的状态 ---
        prev_h1 = states.get('h1')
        prev_h2 = states.get('h2')
        prev_n_state = states.get('n_state')

        # --- 初始化状态 (仅在第一个时间步) ---
        if prev_h1 is None: prev_h1 = torch.zeros(self.num_nodes, self.hidden_dim, device=self.device)
        if prev_h2 is None: prev_h2 = torch.zeros(self.num_nodes, self.hidden_dim, device=self.device)

        # --- GNN Layer 1 + Temporal Update ---
        h1_gnn = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1_gnn, p=0.1, training=self.training)
        h1 = self.h1_gru(h1_gnn, prev_h1).detach()
        # --- GNN Layer 2 + Temporal Update ---
        h2_gnn = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2_gnn, p=0.1, training=self.training)
        h2 = self.h2_gru(h2_gnn, prev_h2).detach()

        final_node_embeddings = h2

        # --- 准备下一时间步的状态 ---
        next_states = {
            "h1": h1.clone(),
            "h2": h2.clone(),
        }
        
        return final_node_embeddings, next_states

    def decode(self, edges, h, states):
        """
        解码器在内部按需计算 r_pool。
        """
        src, dst = edges
        
        feats = h[src] * h[dst]
        return torch.sigmoid(self.predictor(feats))