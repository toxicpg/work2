"""
消融实验 Dispatcher - 根据消融类型创建不同的网络架构

支持的消融类型:
1. full_model: 完整的 MGCN + Dueling DQN + 注意力融合
2. no_mgcn: 使用简化 MLP 替代 MGCN
3. no_dueling: 使用标准 DQN 替代 Dueling DQN
4. no_attention_fusion: 使用拼接替代注意力融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from MGCN_Separate import MGCN_Separate
except ImportError:
    from models.MGCN_Separate import MGCN_Separate


class SimplifiedMLP(nn.Module):
    """简化的 MLP 模型，用于替代 MGCN"""

    def __init__(self, config):
        super(SimplifiedMLP, self).__init__()
        input_dim = config.INPUT_DIM * config.NUM_GRIDS  # 展平所有网格
        hidden_dim = config.HIDDEN_DIMS[0]

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.HIDDEN_DIMS[-1])
        )

    def forward(self, node_features):
        """
        Args:
            node_features: (batch_size, num_grids, input_dim)
        Returns:
            (batch_size, hidden_dims[-1])
        """
        batch_size = node_features.shape[0]
        flattened = node_features.view(batch_size, -1)
        return self.network(flattened)


class StandardDQNHead(nn.Module):
    """标准 DQN 头部（非 Dueling）"""

    def __init__(self, input_dim, num_actions):
        super(StandardDQNHead, self).__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.value_stream(x)


class DuelingDQNHead(nn.Module):
    """Dueling DQN 头部"""

    def __init__(self, input_dim, num_actions):
        super(DuelingDQNHead, self).__init__()

        # 值流
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # 使用平均优势进行归一化
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values


class AblationDispatcher(nn.Module):
    """消融实验 Dispatcher"""

    def __init__(self, config, neighbor_adj, poi_adj, ablation_type='full_model'):
        super(AblationDispatcher, self).__init__()

        self.config = config
        self.num_grids = config.NUM_GRIDS
        self.num_actions = config.NUM_ACTIONS
        self.ablation_type = ablation_type

        # ===== 特征提取器 =====
        if ablation_type == 'no_mgcn' or ablation_type == 'minimal':
            # 使用简化的 MLP
            self.feature_extractor = SimplifiedMLP(config)
            feature_dim = config.HIDDEN_DIMS[-1]
        else:
            # 使用 MGCN
            self.feature_extractor = MGCN_Separate(
                input_dim=config.INPUT_DIM,
                hidden_dims=config.HIDDEN_DIMS,
                neighbor_adj=neighbor_adj,
                poi_adj=poi_adj
            )
            feature_dim = config.HIDDEN_DIMS[-1]

        # ===== 池化层 =====
        if ablation_type == 'no_attention_fusion':
            # 简单平均池化
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            # 注意力池化
            self.pooling = AttentionPooling(feature_dim)

        # ===== Embedding 层 =====
        self.position_embedding = nn.Embedding(config.NUM_GRIDS, config.POSITION_EMB_DIM)
        self.day_embedding = nn.Embedding(7, config.DAY_EMB_DIM)

        # ===== 融合网络 =====
        if ablation_type == 'no_attention_fusion':
            # 使用简单拼接
            fusion_input_dim = feature_dim + config.POSITION_EMB_DIM + config.DAY_EMB_DIM
            self.fusion_network = nn.Sequential(
                nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(config.FUSION_HIDDEN_DIM, config.FINAL_HIDDEN_DIM),
                nn.ReLU()
            )
        else:
            # 使用注意力融合（默认）
            fusion_input_dim = feature_dim + config.POSITION_EMB_DIM + config.DAY_EMB_DIM
            self.fusion_network = nn.Sequential(
                nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(config.FUSION_HIDDEN_DIM, config.FINAL_HIDDEN_DIM),
                nn.ReLU()
            )

        # ===== Q-Value 头部 =====
        if ablation_type == 'no_dueling' or ablation_type == 'minimal':
            # 标准 DQN
            self.q_head = StandardDQNHead(config.FINAL_HIDDEN_DIM, config.NUM_ACTIONS)
        else:
            # Dueling DQN
            self.q_head = DuelingDQNHead(config.FINAL_HIDDEN_DIM, config.NUM_ACTIONS)

    def forward(self, node_features, vehicle_locations, day_of_week):
        """
        前向传播
        Args:
            node_features: (batch_size, num_grids, input_dim)
            vehicle_locations: (batch_size,)
            day_of_week: (batch_size,)
        Returns:
            (batch_size, num_actions) - Q值
        """
        # 范围检查
        if (vehicle_locations.max() >= self.num_grids or vehicle_locations.min() < 0):
            vehicle_locations = torch.clamp(vehicle_locations, 0, self.num_grids - 1)
        if (day_of_week.max() >= 7 or day_of_week.min() < 0):
            day_of_week = torch.clamp(day_of_week, 0, 6)

        # 特征提取
        features = self.feature_extractor(node_features)  # (B, feature_dim)

        # 池化（如果需要）
        if isinstance(self.pooling, nn.AdaptiveAvgPool1d):
            features = features.unsqueeze(1)  # (B, 1, feature_dim)
            features = self.pooling(features).squeeze(1)  # (B, feature_dim)
        else:
            features = self.pooling(features)  # (B, feature_dim)

        # 获取 Embedding
        vehicle_pos_emb = self.position_embedding(vehicle_locations)  # (B, pos_emb_dim)
        day_emb = self.day_embedding(day_of_week)  # (B, day_emb_dim)

        # 融合特征
        fused_features = torch.cat([features, vehicle_pos_emb, day_emb], dim=1)  # (B, fusion_input_dim)
        hidden = self.fusion_network(fused_features)  # (B, final_hidden_dim)

        # 计算 Q 值
        q_values = self.q_head(hidden)  # (B, num_actions)

        return q_values


class AttentionPooling(nn.Module):
    """注意力池化层"""

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_grids, input_dim) 或 (batch_size, input_dim)
        """
        if len(x.shape) == 2:
            # 如果已经是 (B, dim)，直接返回
            return x

        weights = self.attention(x)  # (B, num_grids, 1)
        weights = F.softmax(weights, dim=1)  # (B, num_grids, 1)
        pooled = torch.sum(x * weights, dim=1)  # (B, input_dim)
        return pooled


def create_ablation_dispatcher(config, neighbor_adj, poi_adj, ablation_type='full_model'):
    """工厂函数：创建消融实验 Dispatcher"""
    return AblationDispatcher(config, neighbor_adj, poi_adj, ablation_type)

