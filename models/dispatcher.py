# models/dispatcher.py (V-Final-Embedding: INPUT_DIM=5)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# (导入 MGCN_Separate, 假设它在同一目录)
try:
    from MGCN_Separate import MGCN_Separate
except ImportError:
    # 尝试绝对导入 (如果 models 是一个包)
    from models.MGCN_Separate import MGCN_Separate


class AttentionPooling(nn.Module):
    """注意力池化层 (保持不变)"""

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        weights = self.attention(x);
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1);
        return pooled


class MGCNVehicleDispatcher(nn.Module):
    """MGCN车辆调度器 (V-Final-Embedding, INPUT_DIM=5)"""

    def __init__(self, config, neighbor_adj, poi_adj):
        super(MGCNVehicleDispatcher, self).__init__()

        self.config = config
        self.num_grids = config.NUM_GRIDS
        self.num_actions = config.NUM_ACTIONS

        # ===== 关键修改：检查 INPUT_DIM 是否为 5 =====
        if config.INPUT_DIM != 5:
            # 打印一个警告，但继续（以防万一）
            print(f"警告: MGCNVehicleDispatcher 期望 config.INPUT_DIM = 5, 但收到了 {config.INPUT_DIM}")
        # ==========================================

        self.mgcn = MGCN_Separate(
            input_dim=config.INPUT_DIM,  # 应该是 5
            hidden_dims=config.HIDDEN_DIMS,
            neighbor_adj=neighbor_adj,
            poi_adj=poi_adj
        )
        mgcn_output_dim = config.HIDDEN_DIMS[-1]

        self.pooling_type = getattr(config, 'POOLING_TYPE', 'attention')
        if self.pooling_type == 'attention':
            self.global_pooling = AttentionPooling(mgcn_output_dim)

        # Embedding 层 (保持不变)
        self.position_embedding = nn.Embedding(config.NUM_GRIDS, config.POSITION_EMB_DIM)
        self.day_embedding = nn.Embedding(7, config.DAY_EMB_DIM)

        # 融合网络输入维度 (保持不变, 因为 INPUT_DIM=5 只影响 MGCN)
        fusion_input_dim = mgcn_output_dim + config.POSITION_EMB_DIM + config.DAY_EMB_DIM

        # 融合网络 (保持不变)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.FINAL_HIDDEN_DIM),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(config.FINAL_HIDDEN_DIM, config.NUM_ACTIONS)

    def forward(self, node_features, vehicle_locations, day_of_week):
        """
        前向传播
        Args:
            node_features: (batch_size, num_grids, input_dim=5)
            vehicle_locations: (batch_size,)
            day_of_week: (batch_size,)
        """
        # (范围检查 - 保持不变)
        if (vehicle_locations.max() >= self.num_grids or vehicle_locations.min() < 0):
            vehicle_locations = torch.clamp(vehicle_locations, 0, self.num_grids - 1)
        if (day_of_week.max() >= 7 or day_of_week.min() < 0):
            day_of_week = torch.clamp(day_of_week, 0, 6)

        # MGCN 处理 (输入是 5 维)
        mgcn_features = self.mgcn(node_features)  # (B, 400, hidden)

        # 全局池化
        if self.pooling_type == 'attention':
            pooled_features = self.global_pooling(mgcn_features)  # (B, hidden)
        else:
            pooled_features = mgcn_features.mean(dim=1)

        # Embedding (不变)
        position_emb = self.position_embedding(vehicle_locations)  # (B, pos_dim)
        day_emb = self.day_embedding(day_of_week)  # (B, day_dim)

        # 拼接 (不变)
        combined_features = torch.cat([pooled_features, position_emb, day_emb], dim=1)

        # 融合与输出 (不变)
        fused_features = self.fusion_network(combined_features)
        q_values = self.output_layer(fused_features)
        return q_values

    def select_action(self, node_features, vehicle_location, day_of_week, epsilon=0.1):
        """选择动作 (不变)"""
        day_of_week = max(0, min(6, int(day_of_week)))

        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
            return action, None
        else:
            with torch.no_grad():
                device = node_features.device  # 获取设备
                node_features_unsqueezed = node_features.unsqueeze(0)
                vehicle_location_tensor = torch.tensor([vehicle_location], dtype=torch.long, device=device)
                day_of_week_tensor = torch.tensor([day_of_week], dtype=torch.long, device=device)

                q_values = self.forward(node_features_unsqueezed, vehicle_location_tensor, day_of_week_tensor)

                q_values = q_values.squeeze(0)
                action = q_values.argmax().item()
                return action, q_values


class DuelingMGCNDispatcher(MGCNVehicleDispatcher):
    """Dueling网络 (V-Final-Embedding, INPUT_DIM=5)"""

    def __init__(self, config, neighbor_adj, poi_adj):
        # (父类 __init__ 现在期望 INPUT_DIM=5)
        super().__init__(config, neighbor_adj, poi_adj)

        self.output_layer = None  # 移除父类的 output_layer

        # Dueling 网络层 (不变)
        self.value_network = nn.Linear(config.FINAL_HIDDEN_DIM, 1)
        self.advantage_network = nn.Linear(config.FINAL_HIDDEN_DIM, config.NUM_ACTIONS)

    def forward(self, node_features, vehicle_locations, day_of_week):
        """Dueling网络的前向传播 (不变)"""

        # 获取融合特征 (父类逻辑)
        mgcn_features = self.mgcn(node_features)

        if self.pooling_type == 'attention':
            pooled_features = self.global_pooling(mgcn_features)
        else:
            pooled_features = mgcn_features.mean(dim=1)

        position_emb = self.position_embedding(vehicle_locations)
        day_emb = self.day_embedding(day_of_week)

        combined_features = torch.cat([pooled_features, position_emb, day_emb], dim=-1)
        fused_features = self.fusion_network(combined_features)

        # Dueling 架构 (不变)
        state_value = self.value_network(fused_features)
        advantages = self.advantage_network(fused_features)
        q_values = state_value + advantages - advantages.mean(dim=1, keepdim=True)

        return q_values

    # (select_action 继承父类)


# 工厂函数 (不变)
def create_dispatcher(config, neighbor_adj, poi_adj, model_type='standard'):
    if model_type == 'standard':
        return MGCNVehicleDispatcher(config, neighbor_adj, poi_adj)
    elif model_type == 'dueling':
        return DuelingMGCNDispatcher(config, neighbor_adj, poi_adj)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


