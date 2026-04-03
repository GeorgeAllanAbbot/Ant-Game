import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random

# ==========================================
# 导入你的最新版网络
# ==========================================
from SDK.training.rl_network import PolicyValueNet

class SingleChunkDataset(Dataset):
    """
    轻量级数据集：每次只负责加载并持有一个 part 文件的内容（绝不会爆内存）
    """
    def __init__(self, data_path):
        data = np.load(data_path)
        self.states = torch.from_numpy(data['states'])
        self.stats = torch.from_numpy(data['stats'])
        self.masks = torch.from_numpy(data['masks'])
        self.actions = torch.from_numpy(data['actions'])
        self.values = torch.from_numpy(data['values']).unsqueeze(1)
        self.bundle_features = torch.from_numpy(data['bundle_features'])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.stats[idx], self.masks[idx], self.actions[idx], self.values[idx], self.bundle_features[idx]


def train_supervised(data_dir="data/", save_model_path="checkpoints/sl_pretrained.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING OUT-OF-CORE TRAINING ---")
    print(f"Compute Device: {str(device).upper()}")
    
    # 获取所有的碎片文件
    all_files = glob.glob(os.path.join(data_dir, "*_part_*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    print(f"Found {len(all_files)} chunk files.")
    
    # 初始化你的模型
    model = PolicyValueNet().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 25
    batch_size = 256
    
    model.train()
    for epoch in range(epochs):
        # 🌟 核心：每个 Epoch 开始前，打乱文件的读取顺序！
        # 这保证了宏观层面上的全局数据打乱 (Global Shuffle)
        random.shuffle(all_files)
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        
        print(f"\n[Epoch {epoch+1:02d}/{epochs}] Started...")
        
        # 循环读取每一个文件 (按 chunk 训练)
        for file_idx, file_path in enumerate(all_files):
            # 1. 读入一个 128 局的 chunk (占用极小内存)
            dataset = SingleChunkDataset(file_path)
            # 2. 对这个 chunk 内部进行细粒度的 Shuffle
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            
            chunk_loss = 0
            
            for batch_states, batch_stats, batch_masks, batch_actions, batch_values, batch_features in dataloader:
                batch_states = batch_states.to(device)
                batch_stats = batch_stats.to(device)
                batch_masks = batch_masks.to(device)
                batch_actions = batch_actions.to(device)
                batch_values = batch_values.to(device)
                batch_features = batch_features.to(device)
                
                optimizer.zero_grad()
                
                action_logits, values = model(
                    batch_states, 
                    stats=batch_stats,
                    mask=batch_masks, 
                    action_features=batch_features
                )
                
                policy_loss = F.cross_entropy(action_logits, batch_actions)
                value_loss = F.mse_loss(values, batch_values)
                loss = policy_loss + value_loss
                
                loss.backward()
                optimizer.step()
                
                chunk_loss += loss.item()
                epoch_loss += loss.item()
                
                preds = torch.argmax(action_logits, dim=1)
                epoch_correct += (preds == batch_actions).sum().item()
                epoch_samples += batch_actions.size(0)
            
            # 清理无用的内存 (可选，Python 一般会自动回收)
            del dataset
            del dataloader
            
            # 打印每个文件的进度 (可选)
            # print(f"  -> Processed File {file_idx+1}/{len(all_files)} | Chunk Loss: {chunk_loss/len(dataloader):.4f}")
            
        # 汇总这一个 Epoch 的平均指标
        avg_loss = epoch_loss / (epoch_samples / batch_size) 
        accuracy = epoch_correct / epoch_samples * 100
        print(f"✅ Epoch [{epoch+1:02d}/{epochs}] Summary | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    # 保存模型
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    if hasattr(model, 'save_checkpoint'):
        model.save_checkpoint(save_model_path)
    else:
        torch.save(model.state_dict(), save_model_path)
        
    print(f"🎉 Training Complete! Model safely saved to {save_model_path}")

if __name__ == "__main__":
    train_supervised()