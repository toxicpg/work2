#!/usr/bin/env python3
"""
生成 3D 立体架构图
用于论文中展示深度学习模型
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path


class Layer3D:
    """3D 层的表示"""
    def __init__(self, x, y, z, width, height, depth, color, label, label_info=""):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth
        self.color = color
        self.label = label
        self.label_info = label_info

    def get_vertices(self):
        """获取立方体的顶点"""
        x, y, z = self.x, self.y, self.z
        w, h, d = self.width, self.height, self.depth

        vertices = [
            [x, y, z],
            [x + w, y, z],
            [x + w, y + h, z],
            [x, y + h, z],
            [x, y, z + d],
            [x + w, y, z + d],
            [x + w, y + h, z + d],
            [x, y + h, z + d],
        ]
        return vertices

    def get_faces(self):
        """获取立方体的面"""
        faces = [
            [0, 1, 5, 4],  # 前面
            [2, 3, 7, 6],  # 后面
            [0, 3, 7, 4],  # 左面
            [1, 2, 6, 5],  # 右面
            [0, 1, 2, 3],  # 底面
            [4, 5, 6, 7],  # 顶面
        ]
        return faces


def draw_3d_architecture():
    """绘制 3D 架构图"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 设置视角
    ax.view_init(elev=20, azim=45)

    # 创建层
    layers = []

    # 第1层：输入层
    layers.append(Layer3D(
        x=0, y=0, z=0,
        width=4, height=2, depth=0.5,
        color='#E8F4F8',
        label='Input State\n(B, 400, 5)',
        label_info='Orders|Idle|Busy|sin(t)|cos(t)'
    ))

    # 第2层：GCN 分支 1（邻接图）
    layers.append(Layer3D(
        x=-3, y=0, z=2,
        width=3, height=2, depth=1,
        color='#B3E5FC',
        label='GCN Branch 1\n(Adjacency)',
        label_info='(B, 400, 64)'
    ))

    # 第3层：GCN 分支 2（POI图）
    layers.append(Layer3D(
        x=5, y=0, z=2,
        width=3, height=2, depth=1,
        color='#C8E6C9',
        label='GCN Branch 2\n(POI)',
        label_info='(B, 400, 64)'
    ))

    # 第4层：Attention Fusion
    layers.append(Layer3D(
        x=0.5, y=0, z=4,
        width=3, height=2, depth=1.2,
        color='#F3E5F5',
        label='Attention Fusion\nα_n⊙H_n + α_p⊙H_p',
        label_info='(B, 400, 32)'
    ))

    # 第5层：Global Pooling + Embeddings
    layers.append(Layer3D(
        x=1, y=0, z=6,
        width=2.5, height=1.8, depth=1,
        color='#DCEDC8',
        label='Global Pooling\n+ Embeddings',
        label_info='(B, 64)'
    ))

    # 第6层：Dueling DQN - Value Stream
    layers.append(Layer3D(
        x=-1.5, y=0, z=8,
        width=2, height=1.5, depth=0.8,
        color='#FFEBEE',
        label='Value Stream\n64→128→1',
        label_info='V(s) (B, 1)'
    ))

    # 第7层：Dueling DQN - Advantage Stream
    layers.append(Layer3D(
        x=2.5, y=0, z=8,
        width=2, height=1.5, depth=0.8,
        color='#E3F2FD',
        label='Advantage Stream\n64→128→179',
        label_info='A(s,a) (B, 179)'
    ))

    # 第8层：Q-value Aggregation
    layers.append(Layer3D(
        x=0.5, y=0, z=10,
        width=3, height=1.8, depth=1,
        color='#FFF3E0',
        label='Q-value Aggregation\nQ = V + [A - mean(A)]',
        label_info='(B, 179)'
    ))

    # 绘制所有层
    for i, layer in enumerate(layers):
        vertices = layer.get_vertices()
        faces = layer.get_faces()

        # 创建多边形集合
        poly = [[vertices[j] for j in face] for face in faces]

        # 添加到图中
        ax.add_collection3d(Poly3DCollection(
            poly,
            alpha=0.7,
            facecolor=layer.color,
            edgecolor='#333333',
            linewidth=2
        ))

        # 添加标签
        center_x = layer.x + layer.width / 2
        center_y = layer.y + layer.height / 2
        center_z = layer.z + layer.depth / 2

        ax.text(
            center_x, center_y, center_z,
            layer.label,
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            color='#000000'
        )

        # 添加信息标签
        ax.text(
            center_x, center_y - 1.2, center_z,
            layer.label_info,
            fontsize=7,
            ha='center',
            va='top',
            color='#666666',
            style='italic'
        )

    # 绘制连接线
    connections = [
        ((2, 1, 0.25), (-1.5, 1, 2)),      # Input -> GCN1
        ((2, 1, 0.25), (6.5, 1, 2)),       # Input -> GCN2
        ((-1.5, 1, 3), (2.25, 1, 4)),      # GCN1 -> Fusion
        ((6.5, 1, 3), (2.25, 1, 4)),       # GCN2 -> Fusion
        ((2.25, 1, 5.2), (2.25, 1, 6)),    # Fusion -> Pooling
        ((2.25, 1, 7), (-0.5, 1, 8)),      # Pooling -> Value
        ((2.25, 1, 7), (3.5, 1, 8)),       # Pooling -> Advantage
        ((-0.5, 1, 8.4), (2.25, 1, 10)),   # Value -> Output
        ((3.5, 1, 8.4), (2.25, 1, 10)),    # Advantage -> Output
    ]

    for start, end in connections:
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        ax.plot(xs, ys, zs, 'k-', linewidth=1.5, alpha=0.3)

    # 设置坐标轴
    ax.set_xlabel('X', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (Layer Depth)', fontsize=10, fontweight='bold')

    # 设置标题
    ax.set_title(
        'Complete Model Architecture\nGCN + MGCN + Dueling DQN',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # 设置坐标轴范围
    ax.set_xlim(-5, 10)
    ax.set_ylim(-2, 4)
    ax.set_zlim(-1, 12)

    # 去掉网格
    ax.grid(False)

    # 设置背景色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 设置边框
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.tight_layout()

    # 保存图片
    output_path = 'model_architecture_3d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 3D 架构图已保存到：{output_path}")
    print(f"  分辨率：300 DPI（适合论文）")
    print(f"  文件大小：{Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

    plt.show()


def draw_2d_architecture_simplified():
    """绘制简化的 2D 架构图（备选方案）"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # 定义颜色
    colors = {
        'input': '#E8F4F8',
        'gcn': '#B3E5FC',
        'fusion': '#F3E5F5',
        'pooling': '#DCEDC8',
        'value': '#FFEBEE',
        'advantage': '#E3F2FD',
        'output': '#FFF3E0',
    }

    # 绘制层
    layers_info = [
        (1, 9, 'Input State\n(B, 400, 5)', colors['input'], 'Orders|Idle|Busy|sin(t)|cos(t)'),
        (0.2, 7, 'GCN Branch 1\n(Adjacency)\n(B, 400, 64)', colors['gcn'], '邻接图'),
        (1.8, 7, 'GCN Branch 2\n(POI)\n(B, 400, 64)', colors['gcn'], 'POI图'),
        (1, 5, 'Attention Fusion\n(B, 400, 32)', colors['fusion'], 'α_n⊙H_n + α_p⊙H_p'),
        (1, 3, 'Global Pooling\n+ Embeddings\n(B, 64)', colors['pooling'], '池化+嵌入'),
        (0.2, 1, 'Value Stream\n(B, 1)', colors['value'], 'V(s)'),
        (1.8, 1, 'Advantage Stream\n(B, 179)', colors['advantage'], 'A(s,a)'),
    ]

    for x, y, label, color, info in layers_info:
        # 绘制方框
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (x - 0.35, y - 0.4), 0.7, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(box)

        # 添加标签
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, y - 0.6, info, ha='center', va='top', fontsize=7, style='italic', color='#666666')

    # 绘制连接线
    connections = [
        ((1, 8.6), (0.2, 7.4)),
        ((1, 8.6), (1.8, 7.4)),
        ((0.2, 6.6), (1, 5.4)),
        ((1.8, 6.6), (1, 5.4)),
        ((1, 4.6), (1, 3.4)),
        ((1, 2.6), (0.2, 1.4)),
        ((1, 2.6), (1.8, 1.4)),
    ]

    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='#333333', alpha=0.5))

    # 最后添加聚合层
    ax.text(1, -0.5, 'Q-value Aggregation: Q(s,a) = V(s) + [A(s,a) - mean(A)]',
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#333333', linewidth=2))

    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-1.5, 10)
    ax.axis('off')

    plt.title('Complete Model Architecture\nGCN + MGCN + Dueling DQN',
             fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # 保存图片
    output_path = 'model_architecture_2d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 2D 架构图已保存到：{output_path}")
    print(f"  分辨率：300 DPI（适合论文）")
    print(f"  文件大小：{Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

    plt.show()


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "生成 3D 立体架构图" + " " * 28 + "║")
    print("╚" + "=" * 58 + "╝")

    print("\n生成 3D 立体架构图...")
    draw_3d_architecture()

    print("\n生成 2D 简化架构图...")
    draw_2d_architecture_simplified()

    print("\n" + "=" * 60)
    print("✓ 所有图表已生成完成！")
    print("=" * 60)
    print("\n生成的文件：")
    print("  1. model_architecture_3d.png - 3D 立体图（推荐用于论文）")
    print("  2. model_architecture_2d.png - 2D 简化图（备选方案）")
    print("\n这些图表包含的内容：")
    print("  ✓ 输入层：(B, 400, 5)")
    print("  ✓ GCN 分支 1（邻接图）：(B, 400, 64)")
    print("  ✓ GCN 分支 2（POI图）：(B, 400, 64)")
    print("  ✓ Attention Fusion：(B, 400, 32)")
    print("  ✓ Global Pooling + Embeddings：(B, 64)")
    print("  ✓ Value Stream：(B, 1)")
    print("  ✓ Advantage Stream：(B, 179)")
    print("  ✓ Q-value Aggregation：(B, 179)")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

