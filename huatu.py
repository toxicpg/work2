#!/usr/bin/env python3
"""
在图片上绘制透明热力图 - 二维正态分布订单热力图
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def create_grid_image(image_path, output_path=None):
    """
    创建带网格线的原图
    """
    # 加载图片
    img = plt.imread(image_path)
    height, width = img.shape[0], img.shape[1]

    # 设置网格大小
    grid_size = (20, 20)
    cell_width = width / grid_size[1]
    cell_height = height / grid_size[0]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 显示原图
    ax.imshow(img)

    # 绘制网格线
    for i in range(grid_size[1] + 1):
        ax.axvline(i * cell_width, color='white', linewidth=1.0, alpha=0.8)
    for j in range(grid_size[0] + 1):
        ax.axhline(j * cell_height, color='white', linewidth=1.0, alpha=0.8)

    # 不显示坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # 移除所有边距
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_grid.png"

    # 保存结果（完全移除白边）
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    print(f"带网格的图片已保存到: {output_path}")
    return output_path

def create_transparent_heatmap(image_path, output_path=None, opacity=0.7):
    """
    创建透明热力图
    """
    # 加载图片
    img = plt.imread(image_path)
    height, width = img.shape[0], img.shape[1]

    # 设置网格大小
    grid_size = (20, 20)

    # 创建二维正态分布数据
    center_x = grid_size[1] // 2  # 横向居中
    center_y = grid_size[0] // 2  # 纵向居中

    # 创建坐标网格
    x = np.arange(grid_size[1])
    y = np.arange(grid_size[0])
    X, Y = np.meshgrid(x, y)

    # 二维正态分布参数
    sigma_x = 4.0
    sigma_y = 4.0

    # 创建二维正态分布
    heat_data = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) +
                        (Y - center_y)**2 / (2 * sigma_y**2)))

    # 添加随机扰动（只在中心区域，避免边缘）
    noise = np.zeros(heat_data.shape)
    # 只在中心 3/4 区域内添加噪声
    center_region_x = int(grid_size[1] * 0.25)
    center_region_y = int(grid_size[0] * 0.25)
    noise_center = np.random.normal(0, 0.15,
                                   (grid_size[0] - 2*center_region_y,
                                    grid_size[1] - 2*center_region_x))
    noise[center_region_y:center_region_y+noise_center.shape[0],
          center_region_x:center_region_x+noise_center.shape[1]] = noise_center

    heat_data = heat_data + noise
    heat_data = np.clip(heat_data, 0, None)
    heat_data = heat_data / heat_data.max()

    # 调整数据形状以匹配图片
    cell_width = width / grid_size[1]
    cell_height = height / grid_size[0]
    heat_resized = np.kron(heat_data, np.ones((int(cell_height), int(cell_width))))

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 显示原图
    ax.imshow(img)

    # 叠加热力图（透明）- 使用蓝绿色调
    heatmap = ax.imshow(heat_resized, cmap='viridis', alpha=opacity, extent=[0, width, height, 0])

    # 添加网格线
    for i in range(grid_size[1] + 1):
        ax.axvline(i * cell_width, color='white', linewidth=0.5, alpha=0.5)
    for j in range(grid_size[0] + 1):
        ax.axhline(j * cell_height, color='white', linewidth=0.5, alpha=0.5)

    # 不显示坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_transparent_heatmap.png"

    # 保存结果
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    print(f"透明热力图已保存到: {output_path}")
    return output_path, heat_data

def main():
    """主函数"""

    # 输入图片路径
    image_path = "/Users/qiukuipeng/Desktop/ditu.png"

    if not os.path.exists(image_path):
        print(f"错误: 找不到图片文件 {image_path}")
        return

    print("开始创建热力图...")
    print(f"输入图片: {image_path}")

    try:
        # 创建带网格的图片
        grid_path = create_grid_image(image_path)

        # 创建透明热力图
        heatmap_path, heat_data = create_transparent_heatmap(image_path, opacity=0.6)

        print("\n创建完成!")
        print(f"1. 带网格的图片: {grid_path}")
        print(f"2. 透明热力图: {heatmap_path}")
        print(f"热力数据形状: {heat_data.shape}")
        print(f"热力值范围: [{heat_data.min():.4f}, {heat_data.max():.4f}]")

    except Exception as e:
        print(f"创建热力图时出错: {e}")
        print("请确保已安装所需的库:")
        print("pip install matplotlib numpy")

if __name__ == "__main__":
    main()

