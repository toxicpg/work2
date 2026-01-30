"""
CNN-DDQN Model for Ride-Hailing Dispatching
使用卷积神经网络（CNN）提取空间特征 + Dueling DQN架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDDQN(nn.Module):
    """
    CNN-based Dueling DQN for ride-hailing dispatching

    Architecture:
        Input: (B, 400, 5) -> Reshape to (B, 5, 20, 20)
        CNN: Conv2D layers to extract spatial features
        Flatten + MLP: Feature fusion with position and time embeddings
        Dueling Head: Separate value and advantage streams
        Output: (B, 179) Q-values for 179 hotspot grids
    """

    def __init__(self, config):
        super(CNNDDQN, self).__init__()
        self.config = config
        self.num_grids = config.NUM_GRIDS  # 400
        self.grid_rows = config.GRID_SIZE[0]  # 20
        self.grid_cols = config.GRID_SIZE[1]  # 20
        self.input_channels = config.INPUT_DIM  # 5
        self.num_actions = config.NUM_ACTIONS  # 179

        # ===== CNN Layers =====
        # Input: (B, 5, 20, 20)
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        # After conv layers: (B, 64, 20, 20)
        # Flatten: (B, 64*20*20) = (B, 25600)
        self.cnn_output_dim = 64 * self.grid_rows * self.grid_cols

        # ===== Position and Time Embeddings =====
        self.position_embedding = nn.Embedding(self.num_grids, 16)
        self.day_embedding = nn.Embedding(7, 8)

        # ===== Feature Fusion Network =====
        # Input: CNN output (25600) + position (16) + day (8) = 25624
        fusion_input_dim = self.cnn_output_dim + 16 + 8
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # ===== Dueling DQN Head =====
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, node_features, vehicle_locations, day_of_week):
        """
        Forward pass

        Args:
            node_features: (B, 400, 5) - grid features
            vehicle_locations: (B,) - current vehicle grid index
            day_of_week: (B,) - day of week (0-6)

        Returns:
            q_values: (B, 179) - Q-values for each action
        """
        batch_size = node_features.size(0)

        # Range check
        vehicle_locations = torch.clamp(vehicle_locations, 0, self.num_grids - 1)
        day_of_week = torch.clamp(day_of_week, 0, 6)

        # ===== Step 1: Reshape to 2D grid =====
        # (B, 400, 5) -> (B, 5, 20, 20)
        grid_2d = node_features.view(batch_size, self.grid_rows, self.grid_cols, self.input_channels)
        grid_2d = grid_2d.permute(0, 3, 1, 2)  # (B, 5, 20, 20)

        # ===== Step 2: CNN Feature Extraction =====
        x = F.relu(self.bn1(self.conv1(grid_2d)))  # (B, 32, 20, 20)
        x = F.relu(self.bn2(self.conv2(x)))        # (B, 64, 20, 20)
        x = F.relu(self.bn3(self.conv3(x)))        # (B, 64, 20, 20)

        # ===== Step 3: Flatten =====
        cnn_features = x.view(batch_size, -1)  # (B, 25600)

        # ===== Step 4: Get Embeddings =====
        pos_emb = self.position_embedding(vehicle_locations)  # (B, 16)
        day_emb = self.day_embedding(day_of_week)            # (B, 8)

        # ===== Step 5: Feature Fusion =====
        fused = torch.cat([cnn_features, pos_emb, day_emb], dim=1)  # (B, 25624)
        hidden = self.fusion_network(fused)  # (B, 128)

        # ===== Step 6: Dueling DQN =====
        value = self.value_stream(hidden)           # (B, 1)
        advantage = self.advantage_stream(hidden)   # (B, 179)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test the model"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config

    config = Config()
    model = CNNDDQN(config)

    print("CNN-DDQN Model Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("=" * 60)

    # Test forward pass
    batch_size = 4
    node_features = torch.randn(batch_size, 400, 5)
    vehicle_locations = torch.randint(0, 400, (batch_size,))
    day_of_week = torch.randint(0, 7, (batch_size,))

    q_values = model(node_features, vehicle_locations, day_of_week)
    print(f"\nInput shapes:")
    print(f"  node_features: {node_features.shape}")
    print(f"  vehicle_locations: {vehicle_locations.shape}")
    print(f"  day_of_week: {day_of_week.shape}")
    print(f"\nOutput shape:")
    print(f"  q_values: {q_values.shape}")
    print(f"\nTest passed! ✓")

