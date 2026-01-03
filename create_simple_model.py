#!/usr/bin/env python3
"""
创建最简单的深度学习模型
用于测试 Netron 可视化效果
"""

import torch
import torch.nn as nn
from pathlib import Path


class SimpleNeuralNetwork(nn.Module):
    """
    最简单的神经网络
    输入 → 隐层1 → 隐层2 → 输出
    """
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()

        # 第一层：输入 100 → 隐层 64
        self.fc1 = nn.Linear(100, 64)
        self.relu1 = nn.ReLU()

        # 第二层：隐层 64 → 隐层 32
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        # 第三层：隐层 32 → 输出 10
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """
    最简单的卷积神经网络
    用于图像分类
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积层 1：输入 3通道 → 16通道，3x3卷积核
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层 2：16通道 → 32通道，3x3卷积核
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: (B, 3, 32, 32)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (B, 16, 16, 16)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (B, 32, 8, 8)

        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def create_and_save_simple_model():
    """创建并保存简单的全连接网络"""
    print("=" * 60)
    print("创建简单的全连接网络 (FCNN)")
    print("=" * 60)

    model = SimpleNeuralNetwork()
    model.eval()

    print("\n模型结构：")
    print(model)

    # 保存模型
    output_path = 'simple_fcnn_model.pth'
    torch.save(model.state_dict(), output_path)

    print(f"\n✓ 模型已保存到：{output_path}")
    print(f"  文件大小：{Path(output_path).stat().st_size / 1024:.2f} KB")

    # 测试
    print("\n测试前向传播：")
    with torch.no_grad():
        dummy_input = torch.randn(1, 100)
        output = model(dummy_input)
        print(f"  输入形状：{dummy_input.shape}")
        print(f"  输出形状：{output.shape}")

    return output_path


def create_and_save_simple_cnn():
    """创建并保存简单的卷积神经网络"""
    print("\n" + "=" * 60)
    print("创建简单的卷积神经网络 (CNN)")
    print("=" * 60)

    model = SimpleCNN()
    model.eval()

    print("\n模型结构：")
    print(model)

    # 保存模型
    output_path = 'simple_cnn_model.pth'
    torch.save(model.state_dict(), output_path)

    print(f"\n✓ 模型已保存到：{output_path}")
    print(f"  文件大小：{Path(output_path).stat().st_size / 1024:.2f} KB")

    # 测试
    print("\n测试前向传播：")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        print(f"  输入形状：{dummy_input.shape}")
        print(f"  输出形状：{output.shape}")

    return output_path


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "创建简单模型用于 Netron 测试" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")

    # 创建全连接网络
    fcnn_path = create_and_save_simple_model()

    # 创建卷积神经网络
    cnn_path = create_and_save_simple_cnn()

    # 使用说明
    print("\n" + "=" * 60)
    print("✓ 模型创建完成！")
    print("=" * 60)
    print("\n接下来的步骤：")
    print("1. 打开 https://netron.app/")
    print("2. 上传以下文件之一：")
    print(f"   - {fcnn_path} (全连接网络，更简单)")
    print(f"   - {cnn_path} (卷积神经网络，更复杂)")
    print("3. 查看 Netron 生成的架构图")
    print("\n建议：先用 FCNN 看看效果，再用 CNN 看复杂的图")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

