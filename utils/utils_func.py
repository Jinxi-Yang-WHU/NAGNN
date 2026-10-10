import torch
from torch_geometric.data import TemporalData
import random
import os
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from typing import Optional, Dict, Any, Tuple
import math
import math


def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'INFO: fixed random seed: {random_seed}')


def list2csv(lst: list,
             fname: str,
             delimiter: str = ",",
             fmt: str = '%i'):
    out_list = np.array(lst)
    np.savetxt(fname, out_list, delimiter=delimiter,  fmt=fmt)


def remove_duplicate_edges(data):

    src = data.src.cpu().numpy()
    dst = data.dst.cpu().numpy()
    ts = data.t.cpu().numpy()
    msg = data.msg.cpu().numpy()
    y = data.y.cpu().numpy()

    query = np.stack([src, dst, ts], axis=0)
    uniq, idx = np.unique(query, axis=1, return_index=True)
    print("number of edges reduced from ",
          query.shape[1], " to ", uniq.shape[1])

    src = torch.from_numpy(src[idx])
    dst = torch.from_numpy(dst[idx])
    ts = torch.from_numpy(ts[idx])
    msg = torch.from_numpy(msg[idx])
    y = torch.from_numpy(y[idx])

    new_data = TemporalData(
        src=src,
        dst=dst,
        t=ts,
        msg=msg,
        y=y,
    )
    return new_data


def get_snapshot_batches(timestamps):
    r"""
    construct batches of edges based on timestamps 
    #! assume timestamp always start from 0, is sorted and no gap
    Parameters:
        timestamps: timestamps array
    Returns:
        index_dict: a dictionary of start and end indexes for each snapshot
    """
    index_dict = {}
    values, indices = np.unique(timestamps, return_index=True)
    for i in range(len(values)):
        if (i == len(values) - 1):
            index_dict[values[i]] = (indices[i], len(timestamps))
        else:
            index_dict[values[i]] = (indices[i], indices[i+1])
    return index_dict


def generate_splits(
    full_data: Dict[str, Any],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    r"""Generates train, validation, and test splits from the full dataset
    Args:
        full_data: dictionary containing the full dataset
        val_ratio: ratio of validation data
        test_ratio: ratio of test data
    Returns:
        train_data: dictionary containing the training dataset
        val_data: dictionary containing the validation dataset
        test_data: dictionary containing the test dataset
    """
    #! split by snapshot id instead of edge numbers
    ts_list = np.unique(full_data["timestamps"]).tolist()
    ts_list = sorted(ts_list)

    val_time, test_time = list(
        np.quantile(
            ts_list,
            [(1 - val_ratio - test_ratio), (1 - test_ratio)],
        )
    )

    #! changes added to ensure it works with integer correctly
    val_time = math.ceil(val_time)
    test_time = math.ceil(test_time)

    timestamps = full_data["timestamps"]

    train_mask = timestamps <= val_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    return train_mask, val_mask, test_mask


def convert2Torch(src,
                  dst,
                  ts):
    """
    convert numpy array to torch tensor
    Parameters:
        src: numpy array
        dst: numpy array
        ts: numpy array
    """
    src = torch.from_numpy(src)
    dst = torch.from_numpy(dst)
    ts = torch.from_numpy(ts)
    if src.dtype != torch.int64:
        src = src.long()

    # destination tensor must be of type int64
    if dst.dtype != torch.int64:
        dst = dst.long()

    # timestamp tensor must be of type int64
    if ts.dtype != torch.int64:
        ts = ts.long()
    return src, dst, ts


def mkdirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def convert_to_torch_extended(src, dst, ts, lbl, edge_idx):
    """
    convert numpy array to torch tensor
    NOTE: extended version is required, since NAT gets labels and edge_indices as well

    Parameters:
        src, dst, ts, lbl, edge_idx: numpy array
    """
    # convert to Torch tensors
    src = torch.from_numpy(src)
    dst = torch.from_numpy(dst)
    ts = torch.from_numpy(ts)
    lbl = torch.from_numpy(lbl)
    edge_idx = torch.from_numpy(edge_idx)

    # type checks
    if src.dtype != torch.int64:
        src = src.long()

    if dst.dtype != torch.int64:
        dst = dst.long()

    if ts.dtype != torch.int64:
        ts = ts.long()

    if lbl.dtype != torch.int64:
        lbl = lbl.long()

    if edge_idx.dtype != torch.int64:
        edge_idx = edge_idx.long()

    return src, dst, ts, lbl, edge_idx


def pca(data_matrix, k):
    """
    一个鲁棒的 PCA 实现，用于对输入矩阵的行向量进行降维。
    - 优先使用带正则化的特征值分解方法。
    - 如果失败，则自动回退到基于 SVD 的方法。
    - 包含输入检查和数值稳定性处理。

    Args:
        data_matrix (torch.Tensor): 输入的数据矩阵，形状为 (N, D)。
        k (int): 降维后的目标维度数。

    Returns:
        torch.Tensor: 降维后的数据表示，形状为 (N, k)。
                      如果两种方法都失败，则返回 None。
    """
    # 确保输入是浮点类型
    if not data_matrix.is_floating_point():
        data_matrix = data_matrix.float()
        
    # --- 1. 输入验证 ---
    if torch.isnan(data_matrix).any() or torch.isinf(data_matrix).any():
        Warning("输入矩阵中包含 NaN 或 Inf 值，PCA 结果可能无效。")
        # 可以选择返回或继续，这里我们选择继续，但发出警告
        data_matrix = torch.nan_to_num(data_matrix) # 将 NaN 替换为 0，Inf 替换为大数

    n_samples, n_features = data_matrix.shape
    if k > n_features:
        raise ValueError(f"目标维度 k={k} 不能大于原始特征维度 D={n_features}")

    mean = torch.mean(data_matrix, dim=0, keepdim=True)
    centered_matrix = data_matrix - mean
    covariance_matrix = torch.matmul(centered_matrix.T, centered_matrix) / (n_samples - 1)
    

    U, S, Vh = torch.linalg.svd(centered_matrix, full_matrices=False)
            
    principal_components = Vh[:k, :].T
            
    projected_data = torch.matmul(centered_matrix, principal_components)
    return projected_data

if __name__ == '__main__':
    # --- 使用示例 ---

    # 1. 创建一个与之前相同的、有社群结构的邻接矩阵 (N=10, D=10)
    num_nodes = 10
    adj = torch.zeros(num_nodes, num_nodes)
    # 社群 1: 节点 0-4
    adj[0:5, 0:5] = torch.rand(5, 5) > 0.5
    # 社群 2: 节点 5-9
    adj[5:10, 5:10] = torch.rand(5, 5) > 0.5
    # 增加一些跨社群的边
    adj[0, 8] = 1
    adj[7, 2] = 1
    # 使其对称（无向图）
    adj = torch.max(adj, adj.T)

    print("原始邻接矩阵 (N=10):\n", adj.int())

    # 2. 设定目标维度
    target_dim = 2

    # 3. 执行 PCA 降维
    low_dim_representation = pca(adj, k=target_dim)

    print(f"\n降维后的节点表示 (形状: {low_dim_representation.shape}):\n", low_dim_representation)

    # 4. (可选) 可视化结果来验证
    # 预期结果：属于同一社群的节点（行向量相似）在降维后的空间中应该聚集在一起。
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        # 绘制社群1的节点 (0-4)
        plt.scatter(
            low_dim_representation[:5, 0].numpy(), 
            low_dim_representation[:5, 1].numpy(), 
            c='blue', 
            label='Community 1 (Nodes 0-4)'
        )
        # 绘制社群2的节点 (5-9)
        plt.scatter(
            low_dim_representation[5:, 0].numpy(), 
            low_dim_representation[5:, 1].numpy(), 
            c='red', 
            label='Community 2 (Nodes 5-9)'
        )
        plt.title(f'PCA of Adjacency Matrix (降维到 {target_dim}D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.savefig("demo.png")
    except ImportError:
        print("\n请安装 matplotlib (`pip install matplotlib`) 以进行可视化。")