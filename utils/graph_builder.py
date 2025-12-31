# utils/graph_builder.py
import pandas as pd
import numpy as np
import torch
import pickle
import os
from config import Config
import sys

class GraphBuilder:
    """图加载器 - 从预处理的CSV文件加载邻接矩阵"""

    def __init__(self, config):
        self.config = config

    def load_adjacency_from_csv(self, csv_file, expected_shape=(400, 400)):
        """从CSV文件加载邻接矩阵"""
        print(f"从CSV加载邻接矩阵: {csv_file}")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"邻接矩阵文件不存在: {csv_file}")

        # 读取CSV文件
        adj_df = pd.read_csv(csv_file, header=None)  # 假设没有header
        if adj_df.shape[0] == adj_df.shape[1] and adj_df.shape[0] == 401:
            print("检测到多余的索引行列，自动裁剪为 400x400")
            adj_df = adj_df.iloc[1:, 1:]  # 丢掉首行首列

        adj_matrix = adj_df.values.astype(np.float32)
        # 转换为numpy数组
        adj_matrix = adj_df.values.astype(np.float32)

        # 验证形状
        if adj_matrix.shape != expected_shape:
            raise ValueError(f"邻接矩阵形状错误: {adj_matrix.shape}, 期望: {expected_shape}")

        # 转换为PyTorch张量
        adj_tensor = torch.FloatTensor(adj_matrix)

        print(f"邻接矩阵加载成功: {adj_tensor.shape}")
        print(f"连接数量: {(adj_tensor > 0).sum().item()}")
        print(f"矩阵数值范围: {adj_tensor.min():.3f} - {adj_tensor.max():.3f}")

        return adj_tensor

    def load_graphs_from_csv(self, neighbor_csv=None, poi_csv=None):
        """从CSV文件加载两个图结构"""

        # 设置默认路径
        if neighbor_csv is None:
            neighbor_csv = os.path.join(self.config.RAW_DATA_PATH, 'neighbor_adj.csv')
        if poi_csv is None:
            poi_csv = os.path.join(self.config.RAW_DATA_PATH, 'poi_adj.csv')

        # 加载邻接图
        neighbor_adj = self.load_adjacency_from_csv(neighbor_csv)

        # 加载POI相似性图
        poi_adj = self.load_adjacency_from_csv(poi_csv)

        return neighbor_adj, poi_adj

    def validate_adjacency_matrix(self, adj_matrix, matrix_name="邻接矩阵"):
        """验证邻接矩阵的基本性质"""
        print(f"\n验证{matrix_name}:")
        print("-" * 30)

        # 基本形状检查
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            print(f"❌ 矩阵不是方阵: {adj_matrix.shape}")
            return False

        # 数值范围检查
        min_val, max_val = adj_matrix.min().item(), adj_matrix.max().item()
        print(f"数值范围: {min_val:.3f} - {max_val:.3f}")

        # 对称性检查（可选，看你的图是否为无向图）
        is_symmetric = torch.allclose(adj_matrix, adj_matrix.T, atol=1e-6)
        print(f"对称性: {'✓' if is_symmetric else '✗'}")

        # 对角线检查
        diag_nonzero = (torch.diag(adj_matrix) > 0).sum().item()
        print(f"对角线非零元素: {diag_nonzero}/{adj_matrix.shape[0]}")

        # 稀疏性统计
        nonzero_count = (adj_matrix > 0).sum().item()
        total_count = adj_matrix.numel()
        density = nonzero_count / total_count
        print(f"稀疏性: {nonzero_count}/{total_count} ({density:.4f})")

        # 度分布
        degrees = adj_matrix.sum(1)
        print(f"度数统计: 平均={degrees.mean():.2f}, 最大={degrees.max():.0f}, 最小={degrees.min():.0f}")

        print(f"✓ {matrix_name}验证完成")
        return True

    def save_graphs_as_pt(self, neighbor_adj, poi_adj, save_dir=None):
        """将图保存为PyTorch格式（便于后续快速加载）"""
        if save_dir is None:
            save_dir = self.config.PROCESSED_DATA_PATH

        os.makedirs(save_dir, exist_ok=True)

        # 保存路径
        neighbor_path = os.path.join(save_dir, self.config.NEIGHBOR_ADJ_FILE)
        poi_path = os.path.join(save_dir, self.config.POI_ADJ_FILE)

        # 保存为.pt格式
        torch.save(neighbor_adj, neighbor_path)
        torch.save(poi_adj, poi_path)

        print(f"邻接矩阵已保存到: {neighbor_path}")
        print(f"POI相似性矩阵已保存到: {poi_path}")

        # 保存图的统计信息
        graph_info = {
            'neighbor_adj_shape': neighbor_adj.shape,
            'poi_adj_shape': poi_adj.shape,
            'neighbor_connections': (neighbor_adj > 0).sum().item(),
            'poi_connections': (poi_adj > 0).sum().item(),
            'neighbor_avg_degree': neighbor_adj.sum(1).mean().item(),
            'poi_avg_degree': poi_adj.sum(1).mean().item(),
            'neighbor_density': (neighbor_adj > 0).sum().item() / neighbor_adj.numel(),
            'poi_density': (poi_adj > 0).sum().item() / poi_adj.numel()
        }

        info_path = os.path.join(save_dir, 'graph_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(graph_info, f)

        print(f"图统计信息已保存到: {info_path}")

        return graph_info

    def load_graphs_pt(self, load_dir=None):
        """从.pt文件加载图结构（快速加载）"""
        if load_dir is None:
            load_dir = self.config.PROCESSED_DATA_PATH

        neighbor_path = os.path.join(load_dir, self.config.NEIGHBOR_ADJ_FILE)
        poi_path = os.path.join(load_dir, self.config.POI_ADJ_FILE)

        if not os.path.exists(neighbor_path) or not os.path.exists(poi_path):
            raise FileNotFoundError("图的.pt文件不存在，请先运行CSV转换")

        neighbor_adj = torch.load(neighbor_path, weights_only=True)
        poi_adj = torch.load(poi_path, weights_only=True)

        print(f"已从.pt文件加载邻接矩阵: {neighbor_adj.shape}")
        print(f"已从.pt文件加载POI相似性矩阵: {poi_adj.shape}")

        return neighbor_adj, poi_adj


# 测试和使用示例
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    sys.path.append(PROJECT_ROOT)
    # 初始化配置和图构建器
    config = Config()
    builder = GraphBuilder(config)

    try:
        # 从CSV文件加载图结构
        print("=== 从CSV文件加载图结构 ===")
        neighbor_adj, poi_adj = builder.load_graphs_from_csv()

        # 验证图结构
        builder.validate_adjacency_matrix(neighbor_adj, "地理邻接图")
        builder.validate_adjacency_matrix(poi_adj, "POI相似性图")

        # 转换并保存为.pt格式（便于训练时快速加载）
        print("\n=== 保存为PyTorch格式 ===")
        graph_info = builder.save_graphs_as_pt(neighbor_adj, poi_adj)

        print("\n=== 图统计信息 ===")
        for key, value in graph_info.items():
            print(f"{key}: {value}")

        # 测试.pt文件加载
        print("\n=== 测试.pt文件加载 ===")
        loaded_neighbor, loaded_poi = builder.load_graphs_pt()

        # 验证加载的数据一致性
        neighbor_match = torch.allclose(neighbor_adj, loaded_neighbor)
        poi_match = torch.allclose(poi_adj, loaded_poi)

        print(f"邻接矩阵一致性: {'✓' if neighbor_match else '✗'}")
        print(f"POI矩阵一致性: {'✓' if poi_match else '✗'}")

        print("\n图结构处理完成!")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件存在:")
        print(f"- {os.path.join(config.RAW_DATA_PATH, 'neighbor_adj.csv')}")
        print(f"- {os.path.join(config.RAW_DATA_PATH, 'poi_adj.csv')}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")