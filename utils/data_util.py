import os
import numpy as np
import json
import math
import tgx
import tgb
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.undirected import to_undirected

def ceiling_division(n, d):
    q, r = divmod(n, d)
    return q + bool(r)

def get_edges(edge_index_list: list) -> list:
    """
    obtain undirected edges and transpose the edge index
    #! appending snapshots to a long list is not speed efficient
    parameter: 
        edge_index_list: list of edge_index arrays, where each array is the edges in a given snapshot, can change to dictionary of edge index arrays
    output:
        undirected_edge_list: list of undirected edge_index arrays, where each array is the undirected edges in a given snapshot

    output usage syntax:
    data['edge_index_list'][snapshot_idx], this can just be a dictionary
    """
    undirected_edge_list = {}
    # idx = 0  #if there is time gap, the empty snapshots will be skipped
    # ! adapt to assume variable start index
    idx = min(list(edge_index_list.keys()))
    for i in range(idx, idx + len(edge_index_list)):
        if (i not in edge_index_list):
            raise Exception(
                "There are time gap in the dataset, encountered empty snapshot, please use coarser time granularity.")
            # undirected_edge_list[idx] = None
            # continue #empty snapshots
        else:
            edge_index, _ = remove_self_loops(
                torch.from_numpy(np.array(edge_index_list[i])))  # remove self-loop
            # convert to undirected/bi-directed edge_index
            undirected_edge_list[idx] = to_undirected(edge_index)
            idx += 1
    return undirected_edge_list


def load_dtdg(dataset_name: str,
              time_scale: str,
              verbose: bool = True):
    r"""
    load a DTDG dataset from built-in TGX datasets
    Parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
        verbose: print out the dataset information
    Output:
        dtdg: discretized temporal dynamic graph
        ts_list: list of timestamps
    """
    dataset_name = dataset_name.lower()
    if (dataset_name == "uci"):
        dataset = tgx.builtin.uci()
    elif (dataset_name == "canparl"):
        dataset = tgx.builtin.canparl()
    elif (dataset_name == "unvote"):
        dataset = tgx.builtin.unvote()
    elif (dataset_name == "uslegis"):
        dataset = tgx.builtin.uslegis()
    elif (dataset_name == "untrade"):
        dataset = tgx.builtin.untrade()
    elif (dataset_name == "enron"):
        dataset = tgx.builtin.enron()
    elif (dataset_name == "contacts"):
        dataset = tgx.builtin.contacts()
    elif (dataset_name == "social_evo"):
        dataset = tgx.builtin.social_evo()
    elif (dataset_name == "mooc"):
        dataset = tgx.builtin.mooc()
    else:
        raise ValueError("ERROR: unsupported dataset in TGX:  ", dataset_name)

    # data loading of dtdg dataset
    ctdg = tgx.Graph(dataset)
    time_scale = time_scale
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    if (verbose):
        print("processing dataset: ", dataset_name)
        print("discretize to ", time_scale)
        print("INFO: there is time gap: ", dtdg.check_time_gap())
    return dtdg, ts_list


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    print("INFO: log_folder:", log_folder)
    return log_folder


def process_edges(edge_index_list: dict,
                  num_nodes: int,
                  ts_map: list = None,
                  keep_original: bool = False):
    r"""
    return a Data object for a split of a TGB dataset
    Parameters:
        edge_index_list: dict, keys are integer timestamps, values are edge_index
        num_nodes: int, number of nodes in the graph
        ts_map: list, element are the corresponding unix timestamp of the snapshots
    """
    pos_undirected_edges = get_edges(edge_index_list)

    if (keep_original):
        data = {
            'edge_index': pos_undirected_edges,
            # total number of nodes across all split; this is the same value for each split
            'num_nodes': num_nodes,
            'time_length': len(pos_undirected_edges),
            'ts_map': ts_map,
            'original_edges': edge_index_list,
        }
    else:
        data = {
            'edge_index': pos_undirected_edges,
            # total number of nodes across all split; this is the same value for each split
            'num_nodes': num_nodes,
            'time_length': len(pos_undirected_edges),
            'ts_map': ts_map,
        }
    return data


def get_seconds_from_scale(time_scale: str) -> int:
    scale_map = {
        'hourly': 3600,
        'daily': 86400,
        'weekly': 604800,
        'monthly': 2592000,
    }
    
    time_scale_lower = time_scale.lower()
    
    if time_scale_lower in scale_map:
        return scale_map[time_scale_lower]
    else:
        try:
            if time_scale_lower.endswith('days'):
                num = int(time_scale_lower.replace('days', ''))
                return num * 86400
            elif time_scale_lower.endswith('hours'):
                num = int(time_scale_lower.replace('hours', ''))
                return num * 3600
        except ValueError:
            pass

    raise ValueError(f"Unsupported time_scale: '{time_scale}'.")

def convert_indices_to_timestamps(
    indices: list,
    start_timestamp: int,
    time_scale: str
) -> list:
    time_step_seconds = get_seconds_from_scale(time_scale)
    indices_array = np.array(indices)
    timestamps_array = start_timestamp + (indices_array * time_step_seconds)
    return timestamps_array.tolist()


#! use to load snapshots from TGB dataset
def TGB_data_discrete_processing(dataset_name: str,
                                 time_scale: str,
                                 split_mode: str = "train"):
    r"""
    process a TGB dataset with discretization
    parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
    Output:
    """

    # * for discrete models, we only use the discretized edges for training, the test phase will broadcast the predictions
    data_file = dataset_name + "_" + time_scale + "_tgx.pkl"

    if os.path.isfile(data_file):
        print("--------------------")
        print("Loading TGX graph", data_file)
        print("--------------------")
        dtdg = tgb.utils.utils.load_pkl(data_file)
    else:
        # * generate your own discrete timestamps
        # only keep the training snapshots
        tgx_dataset = tgx.tgb_data(dataset_name)
        if (split_mode == "train"):
            mask = tgx_dataset.train_mask
        elif (split_mode == "val"):
            mask = tgx_dataset.val_mask
        elif (split_mode == "test"):
            mask = tgx_dataset.test_mask
        # here only looking at the edges
        tgx_dataset.data = tgx_dataset.data[mask]
        ctdg = tgx.Graph(tgx_dataset)

        dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
        # dtdg.shift_time_to_zero()

    """
    the number of snapshots in ts_list is different than snapshots
    because snapshots skips over empty snapshots
    """
    snapshots = {}
    # dtdg.data format is {ts: {(u,v):1}}
    for ts in dtdg.data.keys():
        if isinstance(dtdg.data[ts], dict):
            edges = list(dtdg.data[ts].keys())
        else:
            edges = dtdg.data[ts]
        edges = np.array(edges).astype(int)
        # ! edges are in shape (num_edges,2) need to convert to (2, num_edges)
        edges = np.swapaxes(edges, 0, 1)
        snapshots[ts] = edges

    # num_nodes = dtdg.total_nodes() + 1 #this calculates the # of unique nodes
    # this calculates max node ID in the dataset
    num_nodes = int(dtdg.max_nid()) + 1

    ts_list = list(set(ts_list))
    ts_list.sort()
    return snapshots, num_nodes, ts_list


#! use to load snapshots from TGB dataset
def TGB_process_all(dataset_name: str,
                    time_scale: str,):
    # * generate your own discrete timestamps
    # only keep the training snapshots
    tgx_dataset = tgx.tgb_data(dataset_name)
    # tgx_dataset.data = tgx_dataset.data #here only looking at the edges
    ctdg = tgx.Graph(tgx_dataset)
    dtdg, _ = ctdg.discretize(time_scale=time_scale, store_unix=True)

    unique_ts_keys = sorted(list(ctdg.data.keys()))
    if not unique_ts_keys:
        return {}, 0
        
    start_time = unique_ts_keys[0]
    end_time = unique_ts_keys[-1]
    total_time = end_time - start_time
    
    if isinstance(time_scale, int):
        interval_size = total_time // time_scale if time_scale > 0 else total_time
    elif isinstance(time_scale, str):
        time_scale_dict = {
            "minutely": 60, "hourly": 3600, "daily": 86400, "semiweekly": 302400, "weekly": 604800,
            "monthly": 2628000, "yearly": 31536000, "biyearly": 63072000
        }
        interval_size = time_scale_dict.get(time_scale)
        if interval_size is None:
            raise TypeError(f"Invalid time interval string: {time_scale}")
    else:
        raise TypeError(f"Invalid time interval type: {type(time_scale)}")
        
    if interval_size == 0:
        interval_size = 1

    # b. 遍历原始连续时间数据，为每个分箱(bin)找到其真正的起始时间
    bin_to_true_start_ts = {}
    for ts in unique_ts_keys:
        # 使用与TGX完全相同的逻辑计算bin_id
        bin_id = ceiling_division(ts, interval_size)
        
        if bin_id not in bin_to_true_start_ts:
            bin_to_true_start_ts[bin_id] = ts
        else:
            bin_to_true_start_ts[bin_id] = min(bin_to_true_start_ts[bin_id], ts)


    """
    the number of snapshots in ts_list is different than snapshots
    because snapshots skips over empty snapshots
    """
    all_snapshots = {}
    sorted_bin_ids = sorted(dtdg.data.keys())
    # dtdg.data format is {ts: {(u,v):1}}
    for i, bin_id in enumerate(sorted_bin_ids):
        edges_data = dtdg.data[bin_id]
        if isinstance(edges_data, dict):
            edges = list(edges_data.keys())
        else:
            edges = edges_data
        
        if len(edges) == 0: continue
            
        edges = np.array(edges).astype(int)
        if edges.ndim == 2 and edges.shape[1] == 2:
            edges = np.swapaxes(edges, 0, 1)
        elif edges.ndim == 1 and len(edges) == 2:
             edges = np.array(edges).reshape(2, 1)
        else:
            edges = np.empty((2, 0), dtype=int)
            
        if bin_id not in bin_to_true_start_ts:
            print(f"Warning: bin_id {bin_id} from dtdg not found in reconstructed map. This may indicate empty snapshots. Skipping.")
            continue

        snapshot_unix_ts = bin_to_true_start_ts[bin_id]
        
        all_snapshots[i] = {
            "edges": edges,
            "ts": snapshot_unix_ts
        }

    num_nodes = int(dtdg.max_nid()) + 1

    return all_snapshots, num_nodes


def load_TGX_dataset(dataset_name: str,
                     time_scale: str,):
    r"""
    load a built in TGX dataset from the DGB paper
    parameters:
        dataset_name: name of the dataset
        time_scale: time scale for discretization
    Output:
        train_data: training snapshots
        val_data: validation snapshots
        test_data: testing snapshots
    """
    dataset_name = dataset_name.lower()
    if (dataset_name == "uci"):
        dataset = tgx.builtin.uci()
    elif (dataset_name == "canparl"):
        dataset = tgx.builtin.canparl()
    elif (dataset_name == "unvote"):
        dataset = tgx.builtin.unvote()
    elif (dataset_name == "uslegis"):
        dataset = tgx.builtin.uslegis()
    elif (dataset_name == "untrade"):
        dataset = tgx.builtin.untrade()
    elif (dataset_name == "enron"):
        dataset = tgx.builtin.enron()
    elif (dataset_name == "contacts"):
        dataset = tgx.builtin.contacts()
    elif (dataset_name == "social_evo"):
        dataset = tgx.builtin.social_evo()
    elif (dataset_name == "mooc"):
        dataset = tgx.builtin.mooc()
    else:
        raise ValueError("ERROR: unsupported dataset in TGX:  ", dataset_name)

    ctdg = tgx.Graph(dataset)
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    ts_list = list(dtdg.data.keys())
    ts_list.sort()

    val_ratio = 0.15
    test_ratio = 0.15
    # * generate val and test split
    val_time, test_time = list(
        np.quantile(
            ts_list,
            [(1 - val_ratio - test_ratio), (1 - test_ratio)],
        )
    )

    val_time = math.ceil(val_time)
    test_time = math.ceil(test_time)

    train_snapshots = {}
    val_snapshots = {}
    test_snapshots = {}
    # dtdg.data format is {ts: {(u,v):1}}

    for ts in ts_list:
        if isinstance(dtdg.data[ts], dict):
            edges = list(dtdg.data[ts].keys())
        else:
            edges = dtdg.data[ts]
        edges = np.array(edges).astype(int)
        # ! edges are in shape (num_edges,2) need to convert to (2, num_edges)
        edges = np.swapaxes(edges, 0, 1)
        assert edges.shape[0] == 2
        if (ts <= val_time):
            train_snapshots[ts] = edges
        elif (ts > val_time and ts <= test_time):
            val_snapshots[ts] = edges
        else:
            test_snapshots[ts] = edges

    # this calculates max node ID in the dataset
    num_nodes = int(dtdg.max_nid()) + 1
    print("there are ", dtdg.total_nodes(), " nodes in the dataset")
    print(" maximum node id is ", dtdg.max_nid())
    train_data = process_edges(train_snapshots, num_nodes, list(
        train_snapshots.keys()), keep_original=True)
    val_data = process_edges(val_snapshots, num_nodes, list(
        val_snapshots.keys()), keep_original=True)
    test_data = process_edges(test_snapshots, num_nodes, list(
        test_snapshots.keys()), keep_original=True)
    return train_data, val_data, test_data

class NumpyEncoder(json.JSONEncoder):
    """
    一个自定义的 JSON 编码器，用于处理 NumPy 数据类型。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


"""
loading a TGB dataset based on a given discretization
"""
def _filter_and_remap_snapshots(snapshots_dict, ts_dict):
    """
    过滤掉边列表为空的快照，并重新映射ID为连续整数。
    """
    # 筛选出非空的快照
    non_empty_snap_ids = [snap_id for snap_id, edges in snapshots_dict.items() if edges.shape[1] > 0]
    
    # 按时间顺序排序
    non_empty_snap_ids.sort()
    
    # 创建新的、ID连续的字典
    remapped_snapshots = {}
    remapped_ts = {}
    for new_id, old_id in enumerate(non_empty_snap_ids):
        remapped_snapshots[new_id] = snapshots_dict[old_id]
        remapped_ts[new_id] = ts_dict[old_id]
        
    return remapped_snapshots, remapped_ts

def load_TGB_dataset(dataset_name: str,
                     time_scale: str,
                     debug_path: str = "snapshot_manifest.json"):

    # 1. 加载数据并获取官方分割点 (此部分不变)
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
    full_data = dataset.get_TemporalData()
    num_nodes = full_data.num_nodes
    
    last_train_ts = full_data.t[dataset.train_mask].max().item()
    last_val_ts = full_data.t[dataset.val_mask].max().item()

    # ========================== MODIFICATION START ==========================
    # 核心改动：不再预先分割事件列表，而是统一处理所有事件，然后在每个时间仓内部进行精细分割。

    # 2. **第一步：将所有事件统一离散化到时间仓 (Snapshots)**
    all_events_with_ts = zip(
        full_data.src.numpy(),
        full_data.dst.numpy(),
        full_data.t.numpy()
    )

    # 计算时间间隔 (与原代码相同)
    all_timestamps_np = full_data.t.numpy()
    if len(all_timestamps_np) == 0: raise ValueError("Dataset is empty.")
    
    if isinstance(time_scale, int):
        total_time = all_timestamps_np.max() - all_timestamps_np.min()
        interval_size = total_time // time_scale if time_scale > 0 else total_time
    elif isinstance(time_scale, str):
        time_scale_dict = {
            "minutely": 60, "hourly": 3600, "daily": 86400, "semiweekly": 302400, "weekly": 604800,
            "monthly": 2628000, "yearly": 31536000, "biyearly": 63072000
        }
        interval_size = time_scale_dict.get(time_scale, 1)
    if interval_size <= 0: interval_size = 1
    
    print(f"Calculated interval size: {interval_size} seconds")

    # 3. **第二步：在每个时间仓内部，根据精确时间戳进行分割**
    # 这些 "raw" 字典的键是原始的 bin_id
    train_snapshots_raw, val_snapshots_raw, test_snapshots_raw = {}, {}, {}
    train_orig_ts_raw, val_orig_ts_raw, test_orig_ts_raw = {}, {}, {}

    for src, dst, t in all_events_with_ts:
        bin_id = ceiling_division(t, interval_size)
        
        # 根据事件的精确时间戳 t，将其分配到正确的集合中
        if t <= last_train_ts:
            target_snapshots = train_snapshots_raw
            target_ts = train_orig_ts_raw
        elif t <= last_val_ts:
            target_snapshots = val_snapshots_raw
            target_ts = val_orig_ts_raw
        else:
            target_snapshots = test_snapshots_raw
            target_ts = test_orig_ts_raw

        # 将事件添加到对应的 bin 中
        if bin_id not in target_snapshots:
            target_snapshots[bin_id] = []
            target_ts[bin_id] = []
        target_snapshots[bin_id].append((src, dst))
        target_ts[bin_id].append(t)
    
    # =========================== MODIFICATION END ===========================

    # --- 4. 最终的、 foolproof 的全局重映射 (此部分逻辑不变，但现在输入是正确的) ---
    # a. 获取每个分割的 bin_id 集合，以便快速查找
    # 注意：现在同一个 bin_id 可能同时存在于 train 和 val 中，这是正确的！
    train_bin_ids = set(train_snapshots_raw.keys())
    val_bin_ids = set(val_snapshots_raw.keys())
    test_bin_ids = set(test_snapshots_raw.keys())

    # b. 创建全局的、排序的、非空的 bin_id 列表
    # 使用 set.union 来获取所有唯一的 bin_id，然后排序
    all_active_bin_ids = sorted(list(train_bin_ids.union(val_bin_ids, test_bin_ids)))
    
    # c. 在一个循环中完成所有分配，确保索引连续且唯一
    train_snapshots, val_snapshots, test_snapshots = {}, {}, {}
    train_ts, val_ts, test_ts = {}, {}, {}
    manifest_data = {"train": {}, "val": {}, "test": {}}

    current_train_idx, current_val_idx, current_test_idx = 0, 0, 0

    for bin_id in all_active_bin_ids:
        # 计算当前快照的起始时间
        current_ts = (bin_id - 1) * interval_size
        
        # 一个 bin_id 现在可能需要被分配给多个集合
        if bin_id in train_bin_ids:
            edges = train_snapshots_raw[bin_id]
            train_snapshots[current_train_idx] = np.array(edges, dtype=int).T if edges else np.empty((2, 0), dtype=int)
            train_ts[current_train_idx] = current_ts
            manifest_data["train"][current_train_idx] = train_orig_ts_raw[bin_id]
            current_train_idx += 1

        if bin_id in val_bin_ids:
            edges = val_snapshots_raw[bin_id]
            val_snapshots[current_val_idx] = np.array(edges, dtype=int).T if edges else np.empty((2, 0), dtype=int)
            val_ts[current_val_idx] = current_ts
            manifest_data["val"][current_val_idx] = val_orig_ts_raw[bin_id]
            current_val_idx += 1
            
        if bin_id in test_bin_ids:
            edges = test_snapshots_raw[bin_id]
            test_snapshots[current_test_idx] = np.array(edges, dtype=int).T if edges else np.empty((2, 0), dtype=int)
            test_ts[current_test_idx] = current_ts
            manifest_data["test"][current_test_idx] = test_orig_ts_raw[bin_id]
            current_test_idx += 1

    # 5. 写入证据文件 (此部分不变)
    with open(debug_path, 'w') as f:
        json.dump(manifest_data, f, indent=2, cls=NumpyEncoder)
    print(f"证据文件已成功写入: {debug_path}")
    
    print(f"数据处理完成。快照数量: Train={len(train_snapshots)}, Val={len(val_snapshots)}, Test={len(test_snapshots)}")
    
    # 6. 最终处理并返回 (此部分不变)
    train_data = process_edges(train_snapshots, num_nodes, train_ts)
    val_data = process_edges(val_snapshots, num_nodes, val_ts)
    test_data = process_edges(test_snapshots, num_nodes, test_ts)

    return train_data, val_data, test_data


def loader(dataset='uci', time_scale=None):
    """
    loader function to check with dataset to be loaded
    """
    print(f"INFO: DATA: {dataset}, TIME_SCALE: {time_scale}")
    # if not cached, to process and cached
    print('INFO: data does not exits, processing ...')

    if dataset in ['uci', 'canparl', 'unvote', 'uslegis', 'untrade', 'contacts', 'enron',
                   'socialevo', 'contacts', 'mooc', 'social_evo']:  # lastfm
        print("INFO: Loading TGX built-in DGB dataset: {}".format(dataset))
        train_data, val_data, test_data = load_TGX_dataset(dataset, time_scale)
        data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
        }

    elif 'tgb' in dataset:
        train_data, val_data, test_data = load_TGB_dataset(
            dataset, time_scale=time_scale)
        print("INFO: Loading TGB dataset: {}".format(dataset))
        data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
        }

    else:
        raise ValueError("ERROR: Undefined dataset!")
    return data
