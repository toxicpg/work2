import torch.nn as nn
import torch
import torch.nn.functional as F

# ===== 使用绝对导入 (假设 gcn.py 在 models 文件夹下) =====
from models.GCN import GCN, Adj_Preprocessor
# =======================================================

# ===== Multi-Graph Convolutional Network =====
class MGCN_Separate(nn.Module):
    """
    分离式多图卷积网络：为不同图构建独立的GCN分支，最后融合特征
    (V-FIX: Moves Adjacency matrices to correct device in forward pass)
    """

    def __init__(self, neighbor_adj, poi_adj, input_dim=3, hidden_dims=[64, 32],
                 kernel_type='localpool', K=1, activation=nn.ReLU, fusion='concat',
                 use_neighbor_only=False, use_poi_only=False, no_graph=False):
        super().__init__()
        self.fusion = fusion.lower()
        self.use_neighbor_only = use_neighbor_only
        self.use_poi_only = use_poi_only
        self.no_graph = bool(no_graph)

        # Create preprocessor instance (Adj_Preprocessor is now imported)
        self.preprocessor = Adj_Preprocessor(kernel_type, K)

        # Process adjacency matrices ONCE during init (on CPU)
        neighbor_kernels = self.preprocessor.process(neighbor_adj)
        poi_kernels = self.preprocessor.process(poi_adj)

        # Store processed kernels as buffers (remain on CPU initially)
        self.register_buffer('A_neighbor', neighbor_kernels)
        self.register_buffer('A_poi', poi_kernels)

        # Create separate GCN branches or MLP
        self.neighbor_layers = nn.ModuleList()
        self.poi_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims
        
        if self.no_graph:
            # MLP mode: just linear layers
            for i in range(len(dims) - 1):
                self.mlp_layers.append(
                    nn.Sequential(
                        nn.Linear(dims[i], dims[i+1]),
                        activation()
                    )
                )
        else:
            k_neighbor = self.A_neighbor.shape[0] # Actual number of neighbor supports
            k_poi = self.A_poi.shape[0]       # Actual number of POI supports

            # Build GCN layers for each branch using actual K
            for i in range(len(dims) - 1):
                self.neighbor_layers.append(
                    GCN(K=k_neighbor, input_dim=dims[i], hidden_dim=dims[i + 1], activation=activation)
                )
                if not self.use_neighbor_only:
                    self.poi_layers.append(
                        GCN(K=k_poi, input_dim=dims[i], hidden_dim=dims[i + 1], activation=activation)
                    )

        # Define fusion layer
        final_dim = hidden_dims[-1]
        
        if self.no_graph or self.use_neighbor_only or self.use_poi_only:
            self.fusion_layer = None # No fusion needed
        elif self.fusion == 'concat':
            self.fusion_layer = nn.Linear(final_dim * 2, final_dim)
        elif self.fusion == 'add':
            self.fusion_layer = None
        elif self.fusion == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(final_dim * 2, final_dim // 2),
                nn.ReLU(),
                nn.Linear(final_dim // 2, 2),
                nn.Softmax(dim=-1) # Output weights for neighbor and poi branches
            )
            self.fusion_layer = None # No extra linear layer needed
        else:
            raise ValueError(f"Unsupported fusion type: {fusion}. Choose from ['concat', 'add', 'attention']")

    def forward(self, x):
        """
        前向传播 (Forward pass) - Corrected for device mismatch.

        Args:
            x (torch.Tensor): Input node features (Batch, N, input_dim).
                               Should be on the target device (e.g., GPU).
        Returns:
            torch.Tensor: Fused output features (Batch, N, hidden_dims[-1]).

        """
        device = x.device
        
        # --- No Graph Mode (MLP) ---
        if self.no_graph:
            x_out = x
            for layer in self.mlp_layers:
                x_out = layer(x_out)
            return x_out

        # --- Neighbor Graph Branch ---
        x_neighbor = x
        # ===== 将 A_neighbor 移动到 x 所在的设备 =====
        A_neighbor_on_device = self.A_neighbor.to(device)
        # ==========================================
        for layer in self.neighbor_layers:
            # 使用移动到正确设备上的邻接矩阵
            x_neighbor = layer(A_neighbor_on_device, x_neighbor)
            
        if self.use_neighbor_only:
            return x_neighbor

        # --- POI Graph Branch ---
        x_poi = x
        # ===== 将 A_poi 移动到 x 所在的设备 =====
        A_poi_on_device = self.A_poi.to(device)
        # =====================================
        for layer in self.poi_layers:
            # 使用移动到正确设备上的邻接矩阵
            x_poi = layer(A_poi_on_device, x_poi)
            
        if self.use_poi_only:
            return x_poi

        # --- Fusion ---
        if self.fusion == 'concat':
            x_fused = torch.cat([x_neighbor, x_poi], dim=-1)
            x_fused = self.fusion_layer(x_fused)
        elif self.fusion == 'add':
            x_fused = x_neighbor + x_poi
        elif self.fusion == 'attention':
            x_cat = torch.cat([x_neighbor, x_poi], dim=-1)
            weights = self.attention(x_cat) # (B, N, 2)
            # 加权求和: weight_neighbor * feature_neighbor + weight_poi * feature_poi
            x_fused = weights[:, :, 0:1] * x_neighbor + weights[:, :, 1:2] * x_poi
            
        return x_fused

    def get_branch_outputs(self, x):
        """
        获取两个分支的独立输出 (也修复了设备问题).
        """
        # ===== 获取设备并移动 A =====
        device = x.device
        A_neighbor_on_device = self.A_neighbor.to(device)
        A_poi_on_device = self.A_poi.to(device)
        # ==========================

        # Neighbor branch
        x_neighbor = x
        for layer in self.neighbor_layers:
            x_neighbor = layer(A_neighbor_on_device, x_neighbor)

        # POI branch
        x_poi = x
        for layer in self.poi_layers:
            x_poi = layer(A_poi_on_device, x_poi)

        return x_neighbor, x_poi

    def get_support_info(self):
        """返回支持矩阵的信息 (保持不变)"""
        # Ensure preprocessor exists before accessing attributes
        kernel_type = self.preprocessor.kernel_type if hasattr(self, 'preprocessor') else 'N/A'
        k_val = self.preprocessor.K if hasattr(self, 'preprocessor') else 'N/A'

        return {
            'neighbor_supports': self.A_neighbor.shape[0] if hasattr(self, 'A_neighbor') else 0,
            'poi_supports': self.A_poi.shape[0] if hasattr(self, 'A_poi') else 0,
            'neighbor_shape': self.A_neighbor.shape if hasattr(self, 'A_neighbor') else None,
            'poi_shape': self.A_poi.shape if hasattr(self, 'A_poi') else None,
            'fusion_type': self.fusion,
            'kernel_type': kernel_type,
            'K': k_val
        }

