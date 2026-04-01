import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# ==========================================
# 导入你的最新版网络
# ==========================================
from SDK.training.rl_network import PolicyValueNet

class ExpertDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.states = torch.from_numpy(data['states'])
        self.stats = torch.from_numpy(data['stats']) # 🌟 新增：加载 stats
        self.masks = torch.from_numpy(data['masks'])
        self.actions = torch.from_numpy(data['actions'])
        self.values = torch.from_numpy(data['values']).unsqueeze(1) 
        # 🌟 新增：加载我们辛苦提取的动作特征！
        self.bundle_features = torch.from_numpy(data['bundle_features']) 
        print(f"✅ 数据加载成功！总样本数: {len(self.states)}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.stats[idx], self.masks[idx], self.actions[idx], self.values[idx], self.bundle_features[idx]

def train_supervised(data_path="data/expert_data.npz", save_model_path="checkpoints/sl_pretrained.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用计算设备: {device}")
    
    dataset = ExpertDataset(data_path)
    # 因为我们现在只是拿 1208 条数据做干跑测试，Batch Size 设小一点
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # 初始化你的模型
    # 💡 注意：Jules 可能已经在模型内部写死了 in_channels=28 和 feature_dim=10 的默认值。
    # 如果运行报错说形状不匹配，你可能需要在这里显式传入：
    # model = PolicyValueNet(in_channels=28, action_feature_dim=10).to(device)
    model = PolicyValueNet().to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 25
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        
        # 🌟 注意这里解包多了一个 batch_features
        for batch_states, batch_stats, batch_masks, batch_actions, batch_values, batch_features in dataloader:
            batch_states = batch_states.to(device)
            batch_stats = batch_stats.to(device)
            batch_masks = batch_masks.to(device)
            batch_actions = batch_actions.to(device)
            batch_values = batch_values.to(device)
            batch_features = batch_features.to(device)
            
            optimizer.zero_grad()
            
            # 🌟 核心修改：将 action_features 喂给网络的 Cross-Attention 机制！
            # 💡 注意：如果报错说 forward 不接受这个参数，请去 rl_network.py 确认一下 
            # Jules 到底把这个参数叫什么名字 (比如叫 action_features 或 bundle_features)
            action_logits, values = model(
                batch_states, 
                stats=batch_stats,
                mask=batch_masks, 
                action_features=batch_features
            )
            
            # --- 损失计算 ---
            # 策略损失 (Cross Entropy 天然需要 Logits)
            policy_loss = F.cross_entropy(action_logits, batch_actions)
            # 价值损失 (MSE)
            # 注意：如果这行报错说维度不匹配，可能需要改成 F.mse_loss(values.view(-1), batch_values.view(-1))
            value_loss = F.mse_loss(values, batch_values)
            
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 🌟 计算预测准确率 (直接对 logits 取 argmax，结果和对 probs 取是一模一样的！)
            preds = torch.argmax(action_logits, dim=1)
            correct_preds += (preds == batch_actions).sum().item()
            total_samples += batch_actions.size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_preds / total_samples * 100
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | 动作预测准确率: {accuracy:.2f}%")
        
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    # 假设你的网络有自定义的保存方法，如果没有就用 torch.save(model.state_dict(), ...)
    if hasattr(model, 'save_checkpoint'):
        model.save_checkpoint(save_model_path)
    else:
        torch.save(model.state_dict(), save_model_path)
        
    print(f"🎉 干跑测试完成！模型已保存至 {save_model_path}")

if __name__ == "__main__":
    train_supervised()