import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResidualBlockSE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class PolicyValueNet(nn.Module):
    def __init__(self, in_channels: int = 12, action_feat_dim: int = 8, num_actions: int = 96, num_channels: int = 128, num_blocks: int = 6):
        super().__init__()
        self.num_actions = num_actions

        # Initial Conv
        self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # ResNet Backbone with SE
        self.res_blocks = nn.ModuleList([
            ResidualBlockSE(num_channels) for _ in range(num_blocks)
        ])

        # Global State Feature Extractor for Policy
        # Compress spatial dimensions to a global context vector
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)

        # Query projection for State
        self.state_query = nn.Linear(32, 64)

        # Key projection for Actions
        # Actions are represented dynamically: (Batch, NumActions, action_feat_dim)
        self.action_key = nn.Sequential(
            nn.Linear(action_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, state_x, action_features, mask=None):
        """
        state_x: (Batch, 12, 20, 20)
        action_features: (Batch, 96, action_feat_dim) -> e.g. location, type one-hot, score
        mask: (Batch, 96)
        """
        out = F.relu(self.bn_in(self.conv_in(state_x)))
        for block in self.res_blocks:
            out = block(out)

        # --- Policy Head (Attention-based dynamic action selection) ---
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        # Global Average Pooling to retain spatial summarization without full flattening
        p = F.adaptive_avg_pool2d(p, 1).view(p.size(0), -1) # (Batch, 32)

        # Project state to Query: (Batch, 64) -> (Batch, 1, 64)
        q = self.state_query(p).unsqueeze(1)

        # Project dynamic actions to Keys: (Batch, NumActions, 64)
        k = self.action_key(action_features)

        # Dot product for logits: (Batch, 1, 64) @ (Batch, 64, NumActions) -> (Batch, 1, NumActions)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) # (Batch, NumActions)

        # Scale down logits to prevent softmax saturation
        logits = logits / np.sqrt(64)

        if mask is not None:
            # Mask invalid actions
            logits = logits.masked_fill(mask == 0, -1e9)

        policy = F.softmax(logits, dim=1)

        # --- Value Head ---
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = F.adaptive_avg_pool2d(v, 1).view(v.size(0), -1) # (Batch, 1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

    def save_checkpoint(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
        }, str(path))

    @classmethod
    def load_checkpoint(cls, path: str, device=None, **kwargs):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cls(**kwargs)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
