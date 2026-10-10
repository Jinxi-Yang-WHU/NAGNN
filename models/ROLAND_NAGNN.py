"""
PyTorch implementation of ROLAND

reference: https://github.com/manuel-dileo/dynamic-gnn
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score,average_precision_score

import copy


from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit,NormalizeFeatures, Constant
from torch_geometric.utils import from_networkx, degree
from torch_geometric.nn import GCNConv, Linear, MessagePassing

import torch
import networkx as nx
import numpy as np

import json

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class NeighborAverage(MessagePassing):
    """
    Implements the D^(-1) * A propagation rule.
    - A is the adjacency matrix without self-loops.
    - D is the degree matrix for A.
    
    This is equivalent to taking the mean of the features of neighboring nodes.
    """
    def __init__(self):
        # 'mean' aggregation will sum the messages and divide by the degree.
        # This is exactly what D^(-1) * A does.
        super().__init__(aggr='sum')

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Node features of shape [N, in_channels]
            edge_index (LongTensor): Graph connectivity in COO format
                                     of shape [2, E]
        """
        # The propagate method will do all the work.
        # 1. It will gather neighbor features (message step).
        # 2. It will average them at each node (aggregate step).
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        The message function is trivial in this case. It just passes the
        feature vector of the source node (x_j) along the edge.
        
        Args:
            x_j (Tensor): Features of the source nodes for each edge.
                          Shape: [E, in_channels]
        """
        # We don't need to do anything to the message itself.
        # The aggregation ('mean') handles the D^(-1) part.
        return x_j

class Aligner(torch.nn.Module):
    """
    一个可以处理一批向量 (N, d 矩阵) 的 Aligner。
    它使用注意力机制对2个或3个输入矩阵进行加权平均。
    """
    def __init__(self, model_dim, second=False):
        super(Aligner, self).__init__()
        self.second = second
        self.model_dim = model_dim
        
        # 线性层保持不变，它们天然支持批处理
        self.linear1 = Linear(model_dim, model_dim)
        self.linear2 = Linear(model_dim, model_dim)
        self.linear_nei1 = Linear(model_dim, model_dim)
        self.linear_nei2 = Linear(model_dim, model_dim)
        
        # 可训练的注意力向量，形状为 (model_dim, 1) 以便进行矩阵乘法
        self.attention_vector = torch.nn.Parameter(torch.randn(model_dim, 1))
        
        if second:
            self.linear_sec1 = Linear(model_dim, model_dim)
            self.linear_sec2 = Linear(model_dim, model_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Xavier/Glorot 初始化注意力向量，这通常比 randn 更好
        torch.nn.init.xavier_uniform_(self.attention_vector)
        
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear_nei1.reset_parameters()
        self.linear_nei2.reset_parameters()
        if self.second:
            self.linear_sec1.reset_parameters()
            self.linear_sec2.reset_parameters()

    def forward(self, x, neighbor, second_neighbor=None):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 第一个输入矩阵，形状 (N, d)。
            neighbor (torch.Tensor): 第二个输入矩阵，形状 (N, d)。
            second_neighbor (torch.Tensor, optional): 第三个输入矩阵，形状 (N, d)。
                                                       仅在 self.second=True 时使用。

        Returns:
            torch.Tensor: 加权平均后的矩阵，形状 (N, d)。
        """
        # --- 1. 对每个输入矩阵进行特征变换 ---
        # 线性层和激活函数会自动在批次维度 N 上并行处理
        x_transformed = self.linear2(F.relu(self.linear1(x)))
        neighbor_transformed = self.linear_nei2(F.relu(self.linear_nei1(neighbor)))

        # --- 2. 计算每个输入的注意力分数 ---
        # x_transformed: (N, d), self.attention_vector: (d, 1)
        # alpha1: (N, d) @ (d, 1) -> (N, 1)
        alpha1 = torch.matmul(x_transformed, self.attention_vector)
        alpha2 = torch.matmul(neighbor_transformed, self.attention_vector)
        
        # 准备一个列表来收集所有的 alpha 分数和变换后的张量
        alphas = [alpha1, alpha2]
        tensors = [x_transformed, neighbor_transformed]
        
        if self.second:
            if second_neighbor is None:
                raise ValueError("second_neighbor 不能为空当 self.second=True 时")
            
            second_neighbor_transformed = self.linear_sec2(F.relu(self.linear_sec1(second_neighbor)))
            alpha3 = torch.matmul(second_neighbor_transformed, self.attention_vector)
            
            alphas.append(alpha3)
            tensors.append(second_neighbor_transformed)

        # --- 3. 计算 Softmax 权重 ---
        # 将 alpha 分数列表拼接成一个 (N, K) 的矩阵，K是输入的数量(2或3)
        # alpha_cat: (N, 1), (N, 1), ... -> (N, K)
        alpha_cat = torch.cat(alphas, dim=1)
        
        # 在最后一个维度 (K) 上计算 softmax，确保每一行的权重和为1
        # attention_weights: (N, K)
        attention_weights = F.softmax(alpha_cat, dim=-1)

        # --- 4. 计算加权平均 ---
        # 将输入张量列表堆叠成一个 (K, N, d) 的张量
        # tensors_stack: (K, N, d)
        tensors_stack = torch.stack(tensors, dim=0)

        # 为了进行加权，我们需要将权重的形状与堆叠张量的形状对齐
        # attention_weights: (N, K) -> (K, N, 1)
        # a. (N, K) -> (K, N)  (.t() or .T)
        # b. (K, N) -> (K, N, 1) (.unsqueeze(-1))
        reshaped_weights = attention_weights.t().unsqueeze(-1)

        # 使用广播机制进行加权
        # (K, N, 1) * (K, N, d) -> (K, N, d)
        weighted_tensors = reshaped_weights * tensors_stack
        
        # 沿着 K 维度求和，得到最终的加权平均结果
        # (K, N, d) -> (N, d)
        output = torch.sum(weighted_tensors, dim=0)
        
        return output


class ROLANDNAGNN(torch.nn.Module):
    def __init__(self, model_dim, 
                 num_nodes, dropout=0.0, update='moving', neighbor_update='moving', second=False, device='cuda'):
        
        super(ROLANDNAGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(model_dim["input_dim"], model_dim["hidden_conv_1"])
        self.conv2 = GCNConv(model_dim["hidden_conv_1"], model_dim["hidden_conv_2"])  

        self.device = device
        self.dropout = dropout
        self.update = update
        self.second = second
        self.neigbor_aggregation = NeighborAverage()
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
                                    
        if neighbor_update=='moving':
            self.nei_tau = torch.Tensor([0])
        elif neighbor_update=='learnable':
            self.nei_tau = torch.nn.Parameter(torch.Tensor([0]))
        elif neighbor_update=='gru':
            self.nei_gru = GRUCell(model_dim["hidden_conv_2"], model_dim["hidden_conv_2"])
        elif neighbor_update=='mlp':
            self.nei_mlp = Linear(model_dim["hidden_conv_2"] * 2, model_dim["hidden_conv_2"])
        else:
            assert(neighbor_update>=0 and neighbor_update <=1)
            self.nei_tau = torch.Tensor([neighbor_update])
        # Initialize previous embeddings with zeros. Dimensions are hidden_conv_num * hidden_dim * num_nodes.
        self.previous_neighbor_embeddings = torch.Tensor([[0 for i in range(model_dim["hidden_conv_2"])] for j in range(num_nodes)])        

        if self.second==True:
            if neighbor_update=='gru':
                self.gru_sec = GRUCell(model_dim["hidden_conv_2"], model_dim["hidden_conv_2"])
            elif neighbor_update=='mlp':
                self.mlp_sec = Linear(model_dim["hidden_conv_2"] * 2, model_dim["hidden_conv_2"])
            elif neighbor_update=='moving':
                self.tau_sec = torch.Tensor([0])
            elif neighbor_update=='learnable':
                self.tau_sec = torch.nn.Parameter(torch.Tensor([0]))
            else:
                assert(neighbor_update>=0 and neighbor_update <=1)
                self.tau_sec = torch.Tensor([neighbor_update])
            self.previous_second_neighbor_embeddings = torch.Tensor([[0 for i in range(model_dim["hidden_conv_2"])] for j in range(num_nodes)])        

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, previous_embeddings=None, previous_neighbor_embeddings=None, previous_second_neighbor_embeddings=None, num_current_edges=None, num_previous_edges=None):
        
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
            h = torch.Tensor((self.tau.to(h.device) * self.previous_embeddings[0].clone() + (1-self.tau).to(h.device) * h.clone()).detach())  # .numpy()
       
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
            h = torch.Tensor((self.tau.to(h.device) * self.previous_embeddings[1].clone() + (1-self.tau).to(h.device) * h.clone()).detach())  # .numpy()
        current_embeddings[1] = h.clone()

        # Neighborhood Aggregation
        if previous_neighbor_embeddings is not None: #None if test
            self.previous_neighbor_embeddings = previous_neighbor_embeddings.clone()
        if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
            #compute moving average parameter
            self.nei_tau = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone() # tau -- past weight

        current_neighbor_embeddings = torch.Tensor([])
        updated_neighbor_embeddings = self.neigbor_aggregation(current_embeddings[1], edge_index)
        #Embedding Update after second layer
        if self.neighbor_update=='gru':
            updated_neighbor_embeddings = torch.Tensor(self.nei_gru(updated_neighbor_embeddings, self.previous_neighbor_embeddings.clone().to(updated_neighbor_embeddings.device)).detach()) # .numpy()
        elif self.neighbor_update=='mlp':
            hin = torch.cat((updated_neighbor_embeddings,self.previous_neighbor_embeddings.clone().to(updated_neighbor_embeddings.device)),dim=1)
            updated_neighbor_embeddings = torch.Tensor(self.nei_mlp(hin).detach())  # .numpy()
        else:
            updated_neighbor_embeddings = torch.Tensor((self.nei_tau.to(updated_neighbor_embeddings.device) * self.previous_neighbor_embeddings.clone() + (1-self.nei_tau).to(updated_neighbor_embeddings.device) * updated_neighbor_embeddings.clone()).detach())  # .numpy()

        current_neighbor_embeddings = updated_neighbor_embeddings.clone()

        #Second Neighborhood Aggregation
        if self.second == True:
            if previous_second_neighbor_embeddings is not None: #None if test
                self.previous_second_neighbor_embeddings = previous_second_neighbor_embeddings.clone()
            if self.update=='moving' and num_current_edges is not None and num_previous_edges is not None: #None if test
                #compute moving average parameter
                self.tau_sec = torch.Tensor([num_previous_edges / (num_previous_edges + num_current_edges)]).clone() # tau -- past weight

            current_second_neighbor_embeddings = torch.Tensor([])
            updated_second_neighbor_embeddings = self.neigbor_aggregation(current_neighbor_embeddings, edge_index)
            #Embedding Update after second layer
            if self.neighbor_update=='gru':
                updated_second_neighbor_embeddings = torch.Tensor(self.gru_sec(updated_second_neighbor_embeddings, self.previous_second_neighbor_embeddings.clone().to(updated_second_neighbor_embeddings.device)).detach()) # .numpy()
            elif self.neighbor_update=='mlp':
                hin = torch.cat((updated_second_neighbor_embeddings,self.previous_second_neighbor_embeddings.clone().to(updated_second_neighbor_embeddings.device)),dim=1)
                updated_second_neighbor_embeddings = torch.Tensor(self.mlp_sec(hin).detach())  # .numpy()
            else:
                updated_second_neighbor_embeddings = torch.Tensor((self.tau_sec.to(updated_second_neighbor_embeddings.device) * self.previous_second_neighbor_embeddings.clone() + (1-self.tau_sec).to(updated_second_neighbor_embeddings.device) * updated_second_neighbor_embeddings.clone()).detach())  # .numpy()
            current_second_neighbor_embeddings = updated_second_neighbor_embeddings

        if self.second == True:
            return current_embeddings, current_neighbor_embeddings, current_second_neighbor_embeddings
        else:
            return current_embeddings, current_neighbor_embeddings
