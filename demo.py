import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
from collections import defaultdict
from models.NAGNN import RefinedNAGNN

# =============================================================================
#  [粘贴你所有的模型定义代码到这里]
#  - build_dynamic_neighbors
#  - LearnableReservoirUpdateSoft
#  - NeuralBucketing
#  - RolandGNNImproved
# =============================================================================

# 假设所有模型代码都在这个文件里...

# --- 辅助函数：MRR 计算 ---
def calculate_mrr(model, embeddings, pos_edges, neg_edges, device):
    """计算一个时间步内的 MRR"""
    if pos_edges.shape[1] == 0:
        return 0.0

    reciprocal_ranks = []
    
    # 将正负样本按源节点分组
    # key: source_node, value: list of (destination_node, is_positive)
    source_to_targets = defaultdict(list)
    
    pos_src, pos_dst = pos_edges
    for i in range(pos_src.size(0)):
        source_to_targets[pos_src[i].item()].append((pos_dst[i].item(), 1))

    neg_src, neg_dst = neg_edges
    for i in range(neg_src.size(0)):
        # 只为那些有正样本的源节点计算 MRR
        if neg_src[i].item() in source_to_targets:
            source_to_targets[neg_src[i].item()].append((neg_dst[i].item(), 0))

    # 为每个源节点计算排名
    for src, targets in source_to_targets.items():
        if not any(is_pos for _, is_pos in targets):
            continue # 跳过没有正样本的源节点 (虽然理论上不应该发生)

        destinations = torch.tensor([dst for dst, _ in targets], dtype=torch.long, device=device)
        is_positive = torch.tensor([is_pos for _, is_pos in targets], dtype=torch.bool, device=device)
        
        # 构建用于解码的边
        src_repeated = torch.full_like(destinations, fill_value=src)
        eval_edges = torch.stack([src_repeated, destinations])
        
        # 批量评分
        with torch.no_grad():
            scores = model.decode(embeddings, eval_edges)
        
        # 排序并找到正样本的排名
        sorted_indices = torch.argsort(scores, descending=True)
        ranks = (sorted_indices == torch.where(is_positive)[0]).nonzero(as_tuple=True)[1] + 1
        
        for rank in ranks:
            reciprocal_ranks.append(1.0 / rank.item())

    return torch.tensor(reciprocal_ranks).mean().item() if reciprocal_ranks else 0.0

# --- 训练和评估函数 ---
def train_epoch(model, snapshots, optimizer, loss_fn, device):
    model.train()
    current_states = {}
    total_loss = 0
    
    # 使用 tqdm 创建进度条
    pbar = tqdm(range(len(snapshots) - 1), desc="Training")
    for t in pbar:
        data_t = snapshots[t].to(device)
        data_t_plus_1 = snapshots[t+1].to(device)
        
        final_embeddings_t, next_states = model(
            x=data_t.x, edge_index=data_t.edge_index, states=current_states
        )
        
        pos_preds = model.decode(final_embeddings_t, data_t_plus_1.pos_edge_label_index)
        neg_preds = model.decode(final_embeddings_t, data_t_plus_1.neg_edge_label_index)
        
        preds = torch.cat([pos_preds, neg_preds])
        labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)])
        
        loss = loss_fn(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_states = next_states
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    return total_loss / (len(snapshots) - 1)

def eval_epoch(model, snapshots, loss_fn, device):
    model.eval()
    current_states = {}
    total_loss = 0
    total_mrr = 0
    
    with torch.no_grad():
        pbar = tqdm(range(len(snapshots) - 1), desc="Evaluating")
        for t in pbar:
            data_t = snapshots[t].to(device)
            data_t_plus_1 = snapshots[t+1].to(device)

            final_embeddings_t, next_states = model(
                x=data_t.x, edge_index=data_t.edge_index, states=current_states
            )
            
            # 计算损失
            pos_preds = model.decode(final_embeddings_t, data_t_plus_1.pos_edge_label_index)
            neg_preds = model.decode(final_embeddings_t, data_t_plus_1.neg_edge_label_index)
            preds = torch.cat([pos_preds, neg_preds])
            labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)])
            loss = loss_fn(preds, labels)
            total_loss += loss.item()

            # 计算 MRR
            mrr = calculate_mrr(
                model, final_embeddings_t,
                data_t_plus_1.pos_edge_label_index,
                data_t_plus_1.neg_edge_label_index,
                device
            )
            total_mrr += mrr
            pbar.set_postfix({"loss": loss.item(), "mrr": mrr})
            
            current_states = next_states
            
    num_steps = len(snapshots) - 1
    return total_loss / num_steps, total_mrr / num_steps

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 设置参数和设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_NODES = 100
    INPUT_DIM = 16
    HIDDEN_DIM = 32
    RESERVOIR_SIZE = 8
    NUM_BUCKETS = 16
    TOTAL_SNAPSHOTS = 50 
    EPOCHS = 20
    
    # 2. 实例化模型和优化器
    model = RefinedNAGNN(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_nodes=NUM_NODES,
        reservoir_size=RESERVOIR_SIZE, num_buckets=NUM_BUCKETS, device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. 创建并划分伪动态图数据
    print("Generating synthetic dynamic graph data...")
    all_snapshots = []
    for t in range(TOTAL_SNAPSHOTS):
        x = torch.randn(NUM_NODES, INPUT_DIM)
        num_edges = torch.randint(low=NUM_NODES, high=NUM_NODES * 2, size=(1,)).item()
        edge_index = torch.randint(low=0, high=NUM_NODES, size=(2, num_edges))
        
        # 确保链接预测标签不为空
        pos_edges = edge_index[:, :max(1, num_edges//2)]
        neg_edges = torch.randint(low=0, high=NUM_NODES, size=(2, max(1, num_edges//2)))
        
        all_snapshots.append(Data(x=x, edge_index=edge_index, 
                                  pos_edge_label_index=pos_edges, 
                                  neg_edge_label_index=neg_edges))

    # 划分数据集: 60% 训练, 20% 验证, 20% 测试
    train_end = int(TOTAL_SNAPSHOTS * 0.6)
    val_end = int(TOTAL_SNAPSHOTS * 0.8)
    
    train_snapshots = all_snapshots[:train_end]
    val_snapshots = all_snapshots[train_end-1:val_end] # val 从 train 的最后一个快照开始，以继承状态
    test_snapshots = all_snapshots[val_end-1:]     # test 从 val 的最后一个快照开始
    
    print(f"Data split: {len(train_snapshots)} train, {len(val_snapshots)} val, {len(test_snapshots)} test snapshots.")

    # 4. 训练、验证和测试循环
    best_val_mrr = 0
    test_mrr_at_best_val = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = train_epoch(model, train_snapshots, optimizer, loss_fn, device)
        print(f"Epoch {epoch} Train Avg Loss: {train_loss:.4f}")
        
        val_loss, val_mrr = eval_epoch(model, val_snapshots, loss_fn, device)
        print(f"Epoch {epoch} Val Avg Loss: {val_loss:.4f}, Val Avg MRR: {val_mrr:.4f}")
        
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            print(f"🚀 New best validation MRR: {best_val_mrr:.4f}. Evaluating on test set...")
            # 在验证集上取得新高时，在测试集上进行一次评估
            test_loss, test_mrr = eval_epoch(model, test_snapshots, loss_fn, device)
            test_mrr_at_best_val = test_mrr
            print(f"  Test Loss: {test_loss:.4f}, Test MRR: {test_mrr:.4f}")
            # torch.save(model.state_dict(), 'best_model.pth') # 保存最佳模型

    print("\n--- Training Finished ---")
    print(f"🏆 Best Validation MRR: {best_val_mrr:.4f}")
    print(f"🏁 Test MRR at Best Validation: {test_mrr_at_best_val:.4f}")