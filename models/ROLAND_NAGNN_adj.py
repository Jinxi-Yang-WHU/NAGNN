"""
PyTorch implementation of ROLAND

reference: https://github.com/manuel-dileo/dynamic-gnn
"""


import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import copy


from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit,NormalizeFeatures, Constant
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, Linear

import torch
import networkx as nx
import numpy as np

import json

def edge_index_to_adj_matrix(edge_index, num_nodes=None):
    """
    将 edge_index 转换为带权重的邻接矩阵。
    邻接矩阵中 (u, v) 的值表示从 u 到 v 的边的数量。

    Args:
        edge_index (torch.Tensor): 形状为 (2, M) 的张量，其中 M 是边的数量。
                                   edge_index[0] 是源节点，edge_index[1] 是目标节点。
        num_nodes (int, optional): 图中的节点总数。如果为 None，则根据 edge_index 中的
                                   最大节点索引自动推断。默认为 None。

    Returns:
        torch.Tensor: 形状为 (num_nodes, num_nodes) 的稠密邻接矩阵。
    """
    # 如果未指定节点数，则从 edge_index 中自动推断
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    # 为每条边创建一个值为 1 的权重
    # 如果 (u, v) 出现 3 次，我们就有三个 1 对应这个位置
    edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float).to(edge_index.device)
    # 创建一个稀疏的 COO (Coordinate) 格式的张量
    # PyTorch 会自动将具有相同索引 (u, v) 的值相加
    # 这就高效地统计了平行边的数量
    sparse_adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weights,
        size=torch.Size((num_nodes, num_nodes))
    )
    # 将稀疏矩阵转换为稠密矩阵
    # .to_dense() 是一个高效的底层操作
    dense_adj = sparse_adj.to_dense()

    return dense_adj


def squash( tensor ):
    return 1.0/(1.0+tensor)

class MatrixMLP(torch.nn.Module):
    """
    一个带隐藏层的逐元素MLP，将 (N, N, k) 映射为 (N, N)。
    """
    def __init__(self, input_dim, output_dim, hidden_dim=2):
        super(MatrixMLP, self).__init__()
        # 定义网络层
        self.fc1 = Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

class ElementwiseGRUCell(torch.nn.Module):
    """
    一个逐元素的 GRU Cell，用于对矩阵进行高效的、并行的单步更新。
    所有 (i, j) 位置的更新都共享同一套 GRU 参数。

    它接收当前的状态矩阵 h_t 和新的输入矩阵 x_t，并返回更新后的状态矩阵 h_{t+1}。
    """
    def __init__(self, input_size, hidden_size):
        """
        初始化函数。

        Args:
            input_size (int): 每个元素的输入特征维度。
            hidden_size (int): GRU 的隐藏状态维度，也是每个元素的输出维度。
        """
        super(ElementwiseGRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # 我们使用 GRUCell，它本身就是为单步计算设计的。
        # 它将高效地处理我们展平后的矩阵数据。
        self.gru_cell = GRUCell(input_size, hidden_size, bias=False)

    def forward(self, x_t, h_t):
        """
        前向传播函数，执行单步的矩阵更新。

        Args:
            x_t (torch.Tensor): 新的输入矩阵，形状为 (N, N, input_size)。
            h_t (torch.Tensor): 当前的隐藏状态矩阵，形状为 (N, N, hidden_size)。

        Returns:
            torch.Tensor: 更新后的隐藏状态矩阵，形状为 (N, N, hidden_size)。
        """
        # 获取矩阵维度 N
        N = x_t.shape[0]
        
        # --- 核心操作：展平以进行并行计算 ---
        # 这一步是效率的关键。我们没有在 Python 中写 N*N 的循环，
        # 而是将问题转化为一个大的批处理任务。

        # 将 (N, N, ...) 的矩阵展平为 (N*N, ...) 的向量
        # x_t: (N, N, input_size)  -> (N*N, input_size)
        # h_t: (N, N, hidden_size) -> (N*N, hidden_size)
        if x_t.dim() == 2:
            # (N, N) -> (N, N, 1)
            x_t = x_t.unsqueeze(-1)

        # 同样地，检查 h_t 的维度
        if h_t.dim() == 2:
            # (N, N) -> (N, N, 1)
            h_t = h_t.unsqueeze(-1)
        
        # 经过处理后，我们保证 x_t 和 h_t 都是三维的
        # 这使得后续代码更加安全和可预测

        # 获取矩阵维度 N 和输入特征维度
        N, _, input_size = x_t.shape
        
        # 将 (N, N, D) 的矩阵展平为 (N*N, D) 的向量
        x_t_flat = x_t.reshape(-1, input_size)
        h_t_flat = h_t.reshape(-1, self.hidden_size)
        # 将展平后的数据送入 GRUCell
        # GRUCell 将这 N*N 个元素视为一个批次，并行处理所有计算
        h_t_next_flat = self.gru_cell(x_t_flat, h_t_flat)
        
        # 将计算出的新隐藏状态恢复成矩阵形状
        # (N*N, hidden_size) -> (N, N, hidden_size)
        h_t_next = h_t_next_flat.reshape(N, N, self.hidden_size)
        
        if self.hidden_size == 1:
            return h_t_next.squeeze(-1)
        else:
            # 如果 hidden_size > 1，保持原样返回
            return h_t_next


class ROLANDNAGNN(torch.nn.Module):
    def __init__(self, model_dim, 
                 num_nodes, dropout=0.0, update='moving', neighbor_update='moving'):
        
        super(ROLANDNAGNN, self).__init__()
        
        self.conv1 = GCNConv(model_dim["input_dim"], model_dim["hidden_conv_1"])
        self.conv2 = GCNConv(model_dim["hidden_conv_1"], model_dim["hidden_conv_2"])  

        self.dropout = dropout
        self.update = update
        self.neighbor_update = neighbor_update
        if update=='moving':
            self.tau = torch.Tensor([0])
        elif update=='learnable':
            self.tau = torch.nn.Parameter(torch.Tensor([0]))
        elif update=='gru':
            self.gru1 = GRUCell(model_dim["hidden_conv_1"], model_dim["hidden_conv_1"]) #input_size, hidden_size
            self.gru2 = GRUCell(model_dim["hidden_conv_2"], model_dim["hidden_conv_2"])
        elif update=='mlp':
            self.mlp1 = Linear(model_dim["hidden_conv_1"] * 2, model_dim["hidden_conv_1"])
            self.mlp2 = Linear(model_dim["hidden_conv_2"] * 2, model_dim["hidden_conv_2"])
        else:
            assert(update>=0 and update <=1)
            self.tau = torch.Tensor([update])

        # Initialize previous embeddings with zeros. Dimensions are hidden_conv_num * hidden_dim * num_nodes.
        self.previous_embeddings = [torch.Tensor([[0 for i in range(model_dim["hidden_conv_1"])] for j in range(num_nodes)]),\
                                    torch.Tensor([[0 for i in range(model_dim["hidden_conv_2"])] for j in range(num_nodes)])] 
        
        if neighbor_update=='learnable':
            self.alpha = torch.nn.Parameter(torch.Tensor([1]))
            self.beta = torch.nn.Parameter(torch.Tensor([0]))
        elif neighbor_update=='mlp':
            self.mlp = MatrixMLP(2, 1)
        elif neighbor_update=='gru':
            self.gru = ElementwiseGRUCell(1, 1)
        else:
            assert(update>=0 and update <=1)
            self.tau = torch.Tensor([update])
        self.previous_adjacent_matrix = torch.Tensor([[0 for i in range(num_nodes)] for j in range(num_nodes)])                            
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, previous_embeddings=None, previous_adjacent_matrix=None, num_current_edges=None, num_previous_edges=None):
        
        #You do not need all the parameters to be different to None in test phase
        #You can just use the saved previous embeddings and tau
        if previous_embeddings is not None: #None if test
            self.previous_embeddings = [previous_embeddings[0].clone(), previous_embeddings[1].clone()]
        if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
            #compute moving average parameter
            self.tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone() # tau -- past weight
        
        current_embeddings = [torch.Tensor([]), torch.Tensor([])]
        
        #GRAPHCONV
        #GraphConv1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        #Embedding Update after first layer
        if self.update=='gru':
            h = torch.Tensor(self.gru1(h, self.previous_embeddings[0].clone().to(h.device)).detach())  # .numpy()
        elif self.update=='mlp':
            hin = torch.cat((h, self.previous_embeddings[0].clone().to(h.device)), dim=1)
            h = torch.Tensor(self.mlp1(hin).detach()) # .numpy()
        else:
            h = torch.Tensor((self.tau * self.previous_embeddings[0].clone() + (1-self.tau) * h.clone()).detach())  # .numpy()
       
        current_embeddings[0] = h.clone()
        
        #GraphConv2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        #Embedding Update after second layer
        if self.update=='gru':
            h = torch.Tensor(self.gru2(h, self.previous_embeddings[1].clone().to(h.device)).detach()) # .numpy()
        elif self.update=='mlp':
            hin = torch.cat((h,self.previous_embeddings[1].clone().to(h.device)),dim=1)
            h = torch.Tensor(self.mlp2(hin).detach())  # .numpy()
        else:
            h = torch.Tensor((self.tau * self.previous_embeddings[1].clone() + (1-self.tau) * h.clone()).detach())  # .numpy()
        current_embeddings[1] = h.clone()

        #Adjacent Matrix Update
        if previous_adjacent_matrix is not None: #None if test
            self.previous_adjacent_matrix = previous_adjacent_matrix.clone()

        current_adjacent_matrix = edge_index_to_adj_matrix(edge_index, num_nodes=x.size(0))
        last_adjacent_matrix = self.previous_adjacent_matrix.clone().to(current_adjacent_matrix.device)
        current_adjacent_matrix = squash(current_adjacent_matrix)
        last_adjacent_matrix = squash(last_adjacent_matrix)
        interaction_adjacent_matrix = current_adjacent_matrix*last_adjacent_matrix
        mask = ((current_adjacent_matrix > 0) | (last_adjacent_matrix > 0)).float()
        #Embedding Update after second layer
        if self.neighbor_update=='gru':
            current_adjacent_matrix = torch.Tensor(self.gru(current_adjacent_matrix, last_adjacent_matrix).detach()) # .numpy()
        elif self.neighbor_update=='mlp':
            combined_adjacent_matrix = torch.stack((current_adjacent_matrix, last_adjacent_matrix), dim=-1)
            current_adjacent_matrix = torch.Tensor(self.mlp(combined_adjacent_matrix).detach())  # .numpy()
        else:
            current_adjacent_matrix = torch.Tensor((self.alpha * current_adjacent_matrix + self.beta*last_adjacent_matrix + (1-self.alpha-self.beta)).detach())  # .numpy()
        current_adjacent_matrix = squash(current_adjacent_matrix)
        # current_adjacent_matrix = current_adjacent_matrix * mask

        # NOTE: last GCNConv layer is considered as the embeddings
        return current_embeddings, current_adjacent_matrix

