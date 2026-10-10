from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch

from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,get_laplacian,remove_self_loops, to_dense_adj , degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from scipy.special import comb
import math

from typing import Optional

class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        swish_part = self.fc1(x) * torch.sigmoid(self.fc1(x))  
        gate = torch.sigmoid(self.fc2(x))  # Sigmoid
        return swish_part * gate 
    

class MLP(nn.Module):
    layers: nn.ModuleList
    fc: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, use_bn: bool, activation: str,
        dropout_prob: float, norm_type: str = "batch", NEW_BATCH_NORM=False, use_bias=True
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MLPLayer(in_dims, hidden_dims, use_bn, activation, dropout_prob, norm_type, NEW_BATCH_NORM)
            self.layers.append(layer)
            in_dims = hidden_dims

        self.fc = nn.Linear(hidden_dims, out_dims, bias=use_bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        for layer in self.layers:
            X = layer(X, mask=mask)      # [***, D_hid]
        X = self.fc(X)        # [***, D_out]
        X = self.dropout(X)   # [***, D_out]
        return X

    @property
    def out_dims(self) -> int:
        return self.fc.out_features


class MLPLayer(nn.Module):
    """
    Based on https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    """
    fc: nn.Linear
    bn: Optional[nn.BatchNorm1d]
    activation: nn.Module
    dropout: nn.Dropout

    def __init__(self, in_dims: int, out_dims: int, use_bn: bool, activation: str,
                 dropout_prob: float, norm_type: str = "batch", NEW_BATCH_NORM=False) -> None:
        super().__init__()
        # self.fc = nn.Linear(in_dims, out_dims, bias=not use_bn)
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        self.NEW_BATCH_NORM = NEW_BATCH_NORM
        if NEW_BATCH_NORM:
            self.bn = nn.BatchNorm1d(out_dims)
        elif use_bn:
            self.bn = nn.BatchNorm1d(out_dims) if norm_type == "batch" else nn.LayerNorm(out_dims)
        else:
            self.bn = None
        # self.bn = nn.BatchNorm1d(out_dims) if use_bn else None
        # self.ln = nn.LayerNorm(out_dims) if use_bn else None

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'none':
            self.activation = nn.Identity()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'swiglu':
            self.activation = SwiGLU(out_dims)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation!")
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        X = self.fc(X)                     # [***, D_out]
        if mask is not None:
            X[~mask] = 0
        if self.NEW_BATCH_NORM:
            if mask is None:
                X = X.transpose(0, 1)
                X = self.bn(X.transpose(1, 2)).transpose(1,2)
                X = X.transpose(0, 1)
            else:
                X[mask] = self.bn(X[mask])
        elif self.bn is not None:
            shape = X.size()
            X = X.reshape(-1, shape[-1])   # [prod(***), D_out]
            X = self.bn(X)                 # [prod(***), D_out]
            X = X.reshape(shape)           # [***, D_out]
        X = self.activation(X)             # [***, D_out]
        X = self.dropout(X)                # [***, D_out]
        return X

class GIN(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP],
            bn: bool = False, residual: bool = False, laplacian=None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(create_mlp(in_dims=in_dims, out_dims=hidden_dims))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))
        layer = GINLayer(create_mlp(hidden_dims, out_dims))
        self.layers.append(layer)
        self.laplacian=laplacian

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(X, edge_index, laplacian=self.laplacian, mask=mask)   # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            if mask is not None:
                X[~mask] = 0
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if mask is None:
                    if X.ndim == 3:
                        X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                    else:
                        X = self.batch_norms[i](X)
                else:
                    X[mask] = self.batch_norms[i](X[mask])
            if self.residual:
                X = X + X0
        return X                       # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(data=torch.randn(1), requires_grad=True) #torch.empty(1), requires_grad=True)
        self.mlp = mlp

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, laplacian=False, mask=None) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        if laplacian == 'L':
            edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=X.size(0))
            laplacian = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)  # [N_sum, N_sum]
            S = torch.einsum('ij,jkd->ikd', laplacian, X)   # [N_sum, ***, D_in]

            Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
            Z = Z + S                # [N_sum, ***, D_in]
            return self.mlp(Z, mask)  
        elif laplacian == 'RW':
            adj = to_dense_adj(edge_index).squeeze(0)  # [N_sum, N_sum]
            deg = degree(edge_index[0], num_nodes=X.size(0), dtype=torch.float)  # [N_sum]
            deg_inv = 1.0 / deg  # Inverse of the degree
            deg_inv[deg_inv == float('inf')] = 0  # Handle division by zero for isolated nodes
            deg_inv_diag = torch.diag(deg_inv)  # [N_sum, N_sum]
            random_walk = torch.matmul(adj, deg_inv_diag)  # [N_sum, N_sum]
            S = torch.einsum('ij,jkd->ikd', random_walk, X)  # [N_sum, *, D_in]
            Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
            Z = Z + S                # [N_sum, ***, D_in]
            return self.mlp(Z, mask)       # [N_sum, ***, D_out]

        S = self.propagate(edge_index, X=X)   # [N_sum, *** D_in]

        Z = (1 + self.eps) * X   # [N_sum, ***, D_in]
        Z = Z + S                # [N_sum, ***, D_in]
        return self.mlp(Z, mask)       # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j   # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims    
'''
    Basic graph filtering used in all experiments of our paper.
'''
def filter(S, W, k):
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k-1): 
        out = S @ out # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1) #NxMxK

'''
    Optional filter from BernNet: https://github.com/ivam-he/BernNet
'''
def bern_filter(S, W, k):
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(1, k): 
        L = (1/(2**k)) * math.comb(k, i) * torch.linalg.matrix_power(
                                    (2*(torch.eye(S.shape[0]).to(S.device)) - S), k) @ S
        out = L @ W # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1)

class PEARLPositionalEncoder(nn.Module):
    def __init__(self, sample_aggr: nn.Module, basis, k=16, mlp_nlayers=1, mlp_hid=16, pearl_act='relu', mlp_out=16) -> None:
        super().__init__()
        self.mlp_nlayers = mlp_nlayers
        if mlp_nlayers > 0:
            if mlp_nlayers == 1:
                assert(mlp_hid == mlp_out)
            self.mlp_nlayers = mlp_nlayers
            self.layers = nn.ModuleList([nn.Linear(k if i==0 else mlp_hid, 
                                        mlp_hid if i<mlp_nlayers-1 else mlp_out, bias=True) for i in range(mlp_nlayers)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(mlp_hid if i<mlp_nlayers-1 else mlp_out,track_running_stats=True) for i in range(mlp_nlayers)])
        if pearl_act == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif pearl_act == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = SwiGLU(mlp_hid) ## edit if you want more than 1 mlp layers!!
        self.sample_aggr = sample_aggr
        self.k = k
        self.basis = basis

    def forward(
        self, Lap, W, edge_index: torch.Tensor, batch: torch.Tensor, final=True
    ) -> torch.Tensor:
        """
        :param Lap: Laplacian
        :param W: B*[NxM] or BxNxN
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        W_list = []
        # for loop N times for each Nx1 e
        if isinstance(W[0], int):
            # split into N*B*[Nx1]
            j = 0
            for lap, w in zip(Lap, W):
                for i in range(w):
                    e_i = torch.zeros(w).to(device)
                    e_i[i] = 1
                    output = filter(lap, e_i, self.k)  #can also use bern_filter(lap, e_i, self.k) 
                    W_list.append(output)             # [NxMxK]*B
                if j == 0:
                    out = self.sample_aggr(W_list, edge_index, self.basis)
                else:
                    out += self.sample_aggr(W_list, edge_index, self.basis)
                j += 1
            return out
        else:
            for lap, w in zip(Lap, W):
                output = filter(lap, w, self.k)   # output [NxMxK]
                if self.mlp_nlayers > 0:
                    for layer, bn in zip(self.layers, self.norms):
                        output = output.transpose(0, 1)
                        output = layer(output)
                        output = bn(output.transpose(1,2)).transpose(1,2)
                        output = self.activation(output)
                        output = output.transpose(0, 1)
                W_list.append(output)             # [NxMxK]*B
            return self.sample_aggr(W_list, edge_index, self.basis, final=final)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.sample_aggr.out_dims


class GINSampleAggregator(nn.Module):
    gin: GIN

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP], bn: bool
    ) -> None:
        super().__init__()
        self.gin = GIN(n_layers, in_dims, hidden_dims, out_dims, create_mlp, bn, laplacian=None)
        self.mlp = create_mlp(out_dims, out_dims)
        self.running_sum = 0

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, basis, mean=False, final=True) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """ 
        if not basis:
            W = torch.cat(W_list, dim=0)  
            PE = self.gin(W, edge_index)  
            if mean:
                PE = (PE).mean(dim=1) # sum or mean along M dimension
            else:
                PE = (PE).sum(dim=1)
                self.running_sum += PE
            if final:
                PE = self.running_sum
                self.running_sum = 0
            return PE              
        else:
            n_max = max(W.size(0) for W in W_list)
            W_pad_list = []     # [N_i, N_max, M] * B
            mask = [] # node masking, [N_i, N_max] * B
            for W in W_list:
                zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
                W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
                W_pad_list.append(W_pad)
                mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]
            W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
            mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
            PE = self.gin(W, edge_index, mask=mask)       # [N_sum, N_max, D_pe]
            PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
            return PE

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims

def GetSampleAggregator(cfg, create_mlp: Callable[[int, int], MLP], device):
    if cfg.sample_aggr_model_name == 'gin':
        return GINSampleAggregator(cfg.n_sample_aggr_layers, cfg.pearl_mlp_out, cfg.sample_aggr_hidden_dims, cfg.pe_dims,
                                         create_mlp, cfg.batch_norm)
    else:
        raise Exception ("sample_aggr function not implemented!")
