import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 构造动态图
def generate_dynamic_graph(num_nodes, num_edges, num_snapshots):
    graphs = []
    for t in range(num_snapshots):
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['time'] = torch.randint(0, 10, (num_edges,))  # 边的时间特征
        g.ndata['feat'] = torch.randn(num_nodes, 16)  # 节点特征
        graphs.append(g)
    return graphs

# 动态图序列
graphs = generate_dynamic_graph(num_nodes=100, num_edges=300, num_snapshots=10)

class DynamicGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(DynamicGNN, self).__init__()
        self.rnn = nn.GRU(in_feats, hidden_feats, batch_first=True)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, g_list):
        h_list = []
        for g in g_list:
            h = g.ndata['feat']  # [num_nodes, feat_dim]
            h = h.unsqueeze(1)  # 添加时间批量维度
            _, hidden = self.rnn(h)  # GRU 输出 [1, num_nodes, hidden_feats]
            h_list.append(hidden.squeeze(0))  # 移除时间批量维度

        # 时间片特征汇总
        h_dynamic = torch.stack(h_list, dim=1)  # [num_nodes, time_steps, hidden_feats]
        h_dynamic = h_dynamic.mean(dim=1)  # 平均时间特征 [num_nodes, hidden_feats]

        out = self.fc(h_dynamic)  # [num_nodes, num_classes]
        return out

# 模型初始化
model = DynamicGNN(in_feats=16, hidden_feats=32, out_feats=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 模拟标签
labels = torch.randint(0, 3, (100,))  # 3 分类任务

# 训练
for epoch in range(50):
    model.train()
    logits = model(graphs)  # 动态图输入模型
    logits = logits.squeeze(0)  # 移除批量维度
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 推理
model.eval()
with torch.no_grad():
    predictions = model(graphs)
    predictions = predictions.squeeze(0)  # 移除批量维度
    preds = torch.argmax(predictions, dim=1)

    # 计算准确率
    accuracy = (preds == labels).float().mean().item()
    print(f"Test Accuracy: {accuracy:.2%}")
