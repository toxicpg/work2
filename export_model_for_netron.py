#!/usr/bin/env python3
"""
导出模型为 .pth 文件，用于 Netron 可视化
包含：GCN + MGCN + Dueling DQN 完整架构
"""

import torch
import numpy as np
from pathlib import Path

# 假设你的项目结构
# 根据你的项目修改这些导入
try:
    from models.dispatcher import create_dispatcher
    from config import Config
except ImportError:
    print("请确保在项目根目录运行此脚本")
    print("或者修改导入路径")
    exit(1)


def create_dummy_adjacency_matrices(num_grids=400):
    """
    创建虚拟的邻接矩阵
    用于模型初始化
    """
    # 邻接矩阵1：基于地理位置的邻接关系
    A_neighbor = np.eye(num_grids)

    # 邻接矩阵2：基于 POI 相似性
    A_poi = np.eye(num_grids)

    return A_neighbor, A_poi


def export_complete_model(output_path='model.pth'):
    """
    导出完整模型（GCN + MGCN + Dueling DQN）

    输入：
    - node_features: (B, 400, 5) - 400个网格的5维特征
    - vehicle_locations: (B,) - 车辆位置
    - day_of_week: (B,) - 星期几

    输出：
    - q_values: (B, 179) - 179个热点的Q值
    """
    print("=" * 60)
    print("导出完整模型（GCN + MGCN + Dueling DQN）")
    print("=" * 60)

    # 初始化配置
    config = Config()
    print(f"✓ 配置已加载")

    # 创建虚拟邻接矩阵
    A_neighbor, A_poi = create_dummy_adjacency_matrices(config.NUM_GRIDS)
    print(f"✓ 邻接矩阵已创建 ({config.NUM_GRIDS} x {config.NUM_GRIDS})")

    # 创建模型
    model = create_dispatcher(config, A_neighbor, A_poi, 'dueling')
    model.eval()
    print(f"✓ 模型已创建（使用 Dueling DQN 架构）")

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型参数数量：")
    print(f"  - 总参数：{total_params:,}")
    print(f"  - 可训练参数：{trainable_params:,}")

    # 保存模型
    torch.save(model.state_dict(), output_path)
    print(f"\n✓ 模型已保存到：{output_path}")
    print(f"  文件大小：{Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

    return model, output_path


def export_model_with_architecture_info(output_path='model_with_info.pth'):
    """
    导出模型并附加架构信息
    """
    print("\n" + "=" * 60)
    print("导出模型（包含架构信息）")
    print("=" * 60)

    config = Config()
    A_neighbor, A_poi = create_dummy_adjacency_matrices(config.NUM_GRIDS)
    model = create_dispatcher(config, A_neighbor, A_poi, 'dueling')
    model.eval()

    # 保存模型和额外信息
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'config': {
            'num_grids': config.NUM_GRIDS,
            'num_hotspots': config.NUM_HOTSPOTS,
            'input_dim': 5,
            'hidden_dim': 64,
            'output_dim': 179,
        },
        'model_info': {
            'type': 'Dueling DQN with MGCN',
            'components': [
                'GCN Branch 1 (Adjacency Graph)',
                'GCN Branch 2 (POI Graph)',
                'Attention Fusion',
                'Global Pooling + Embeddings',
                'Dueling DQN Head (Value + Advantage)',
            ]
        }
    }

    torch.save(checkpoint, output_path)
    print(f"✓ 模型已保存到：{output_path}")
    print(f"  文件包含：模型权重 + 架构信息 + 配置")

    return output_path


def test_model_forward_pass(model):
    """
    测试模型的前向传播
    验证输入输出维度是否正确
    """
    print("\n" + "=" * 60)
    print("测试模型前向传播")
    print("=" * 60)

    with torch.no_grad():
        # 创建虚拟输入
        batch_size = 2
        node_features = torch.randn(batch_size, 400, 5)
        vehicle_locations = torch.randint(0, 400, (batch_size,))
        day_of_week = torch.randint(0, 7, (batch_size,))

        print(f"输入维度：")
        print(f"  - node_features: {node_features.shape}")
        print(f"  - vehicle_locations: {vehicle_locations.shape}")
        print(f"  - day_of_week: {day_of_week.shape}")

        # 前向传播
        q_values = model(node_features, vehicle_locations, day_of_week)

        print(f"\n✓ 前向传播成功！")
        print(f"输出维度：")
        print(f"  - q_values: {q_values.shape}")
        print(f"  - 值范围：[{q_values.min():.4f}, {q_values.max():.4f}]")

        return q_values


def print_model_structure(model):
    """
    打印模型结构
    """
    print("\n" + "=" * 60)
    print("模型结构详情")
    print("=" * 60)
    print(model)


def main():
    """
    主函数：导出模型
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "模型导出脚本 - 用于 Netron 可视化" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")

    # 第一步：导出完整模型
    model, model_path = export_complete_model('model.pth')

    # 第二步：测试前向传播
    test_model_forward_pass(model)

    # 第三步：打印模型结构
    print_model_structure(model)

    # 第四步：导出带信息的模型
    export_model_with_architecture_info('model_with_info.pth')

    # 第五步：使用说明
    print("\n" + "=" * 60)
    print("✓ 导出完成！")
    print("=" * 60)
    print("\n接下来的步骤：")
    print("1. 打开 https://netron.app/")
    print("2. 上传文件 'model.pth' 或 'model_with_info.pth'")
    print("3. 查看你的模型架构图")
    print("\n注意：")
    print("- model.pth：只包含模型权重（较小）")
    print("- model_with_info.pth：包含权重 + 架构信息（较大）")
    print("- Netron 会自动识别模型结构并生成漂亮的架构图")
    print("\n模型包含的组件：")
    print("✓ GCN Branch 1 (邻接图卷积)")
    print("✓ GCN Branch 2 (POI图卷积)")
    print("✓ Attention Fusion (注意力融合)")
    print("✓ Global Pooling + Embeddings (全局池化和嵌入)")
    print("✓ Dueling DQN Head (值流 + 优势流)")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

