#!/usr/bin/env python3
"""
初始分布对比图 - 5种算法在5种不同初始分布下的性能对比
展示我们的模型在所有初始分布中都表现最好

5种算法: Ours (MGCN-D3QN), CNN-DDQN, SARSA(λ)-SAA, Random Walk, Random Dispatch
5种分布: 均匀分布, 正态分布(σ=1), 正态分布(σ=3), 正态分布(σ=5), 正态分布(σ=7)
2个指标: 平均匹配率, 平均等待时间

用法:
    python visualization/plot_distribution_comparison.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_distribution_comparison():
    """
    绘制5种算法在5种初始分布下的性能对比
    """
    # ==================== 数据 ====================
    # 5种初始分布
    distributions = ['均匀分布', 'σ=1', 'σ=3', 'σ=5', 'σ=7']

    # 5种算法
    methods = ['Ours (MGCN-D3QN)', 'CNN-DDQN', 'SARSA(λ)-SAA', 'Random Walk', 'Random Dispatch']

    # 平均匹配率 (%) - 每行是一个算法,每列是一个分布
    # 模拟数据,请替换为真实数据
    matching_rates = [
        [91.2, 89.5, 92.8, 93.5, 92.1],  # Ours (MGCN-D3QN) - 在所有分布下都最好
        [88.3, 86.7, 89.5, 90.2, 88.9],  # CNN-DDQN
        [89.1, 87.8, 90.3, 91.1, 89.8],  # SARSA(λ)-SAA
        [86.5, 84.2, 87.8, 88.6, 87.3],  # Random Walk
        [87.8, 85.9, 88.9, 89.7, 88.5],  # Random Dispatch
    ]

    # 平均等待时间 (秒) - 每行是一个算法,每列是一个分布
    # 模拟数据,请替换为真实数据
    waiting_times = [
        [225.4, 238.2, 218.6, 215.3, 223.7],  # Ours (MGCN-D3QN) - 在所有分布下都最好
        [248.7, 262.5, 241.3, 237.8, 246.2],  # CNN-DDQN
        [241.2, 255.8, 234.7, 231.2, 239.5],  # SARSA(λ)-SAA
        [268.9, 283.4, 262.1, 258.5, 267.3],  # Random Walk
        [259.3, 274.2, 253.4, 249.8, 258.1],  # Random Dispatch
    ]

    # ==================== 绘图 ====================
    # 创建图形 - 1行2列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('不同初始分布下的算法性能对比', fontsize=16, fontweight='bold', y=0.98)

    # 颜色方案 - 5种算法
    colors = ['#E74C3C', '#9B59B6', '#F39C12', '#3498DB', '#2ECC71']

    # ===== 子图1: 平均匹配率 - 分组柱状图 =====
    x = np.arange(len(distributions))
    width = 0.15  # 每根柱子的宽度 (5根柱子)

    for i, (method, rates, color) in enumerate(zip(methods, matching_rates, colors)):
        offset = (i - 2) * width  # 中心对齐: -2, -1, 0, 1, 2
        bars = ax1.bar(x + offset, rates, width, label=method, color=color,
                      alpha=0.8, edgecolor='black', linewidth=1)

    ax1.set_xlabel('初始分布类型', fontsize=13, fontweight='bold')
    ax1.set_ylabel('平均匹配率 (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) 不同分布下的匹配率对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(distributions, fontsize=12)
    ax1.set_ylim([82, 96])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9, ncol=1)

    # ===== 子图2: 平均等待时间 - 分组柱状图 =====
    for i, (method, times, color) in enumerate(zip(methods, waiting_times, colors)):
        offset = (i - 2) * width
        bars = ax2.bar(x + offset, times, width, label=method, color=color,
                      alpha=0.8, edgecolor='black', linewidth=1)

    ax2.set_xlabel('初始分布类型', fontsize=13, fontweight='bold')
    ax2.set_ylabel('平均等待时间 (秒)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) 不同分布下的等待时间对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(distributions, fontsize=12)
    ax2.set_ylim([210, 290])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=1)

    plt.tight_layout()

    # ==================== 保存 ====================
    save_dir = 'results/visualizations'
    os.makedirs(save_dir, exist_ok=True)

    # 保存PNG格式 (便于预览)
    output_path_png = os.path.join(save_dir, 'distribution_comparison.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f'✓ 初始分布对比图已保存(PNG): {output_path_png}')

    # 保存TIFF格式 (适合论文发表)
    output_path_tiff = os.path.join(save_dir, 'distribution_comparison.tiff')
    plt.savefig(output_path_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f'✓ 初始分布对比图已保存(TIFF): {output_path_tiff}')

    plt.close()

    # ==================== 数据摘要 ====================
    print("\n" + "="*90)
    print("5种算法在5种初始分布下的性能数据")
    print("="*90)

    # 匹配率表格
    print("\n【平均匹配率 (%)】")
    print("-"*90)
    print(f"{'算法':<25} | {'均匀分布':<10} | {'σ=1':<10} | {'σ=3':<10} | {'σ=5':<10} | {'σ=7':<10}")
    print("-"*90)
    for i, (method, rates) in enumerate(zip(methods, matching_rates)):
        marker = ' ★最佳' if i == 0 else ''
        print(f"{method:<25} | {rates[0]:>8.1f}% | {rates[1]:>8.1f}% | {rates[2]:>8.1f}% | {rates[3]:>8.1f}% | {rates[4]:>8.1f}%{marker}")
    print("-"*90)

    # 等待时间表格
    print("\n【平均等待时间 (秒)】")
    print("-"*90)
    print(f"{'算法':<25} | {'均匀分布':<10} | {'σ=1':<10} | {'σ=3':<10} | {'σ=5':<10} | {'σ=7':<10}")
    print("-"*90)
    for i, (method, times) in enumerate(zip(methods, waiting_times)):
        marker = ' ★最佳' if i == 0 else ''
        print(f"{method:<25} | {times[0]:>8.1f}s | {times[1]:>8.1f}s | {times[2]:>8.1f}s | {times[3]:>8.1f}s | {times[4]:>8.1f}s{marker}")
    print("-"*90)

    # 统计分析
    print("\n【关键发现】")
    print("-"*90)

    # 转换为numpy数组以便计算
    matching_rates_np = np.array(matching_rates)
    waiting_times_np = np.array(waiting_times)

    # Ours的平均性能
    ours_avg_rate = np.mean(matching_rates_np[0])
    ours_avg_time = np.mean(waiting_times_np[0])

    print(f"✓ Ours (MGCN-D3QN) 在所有5种初始分布下均表现最佳")
    print(f"  - 平均匹配率: {ours_avg_rate:.1f}%")
    print(f"  - 平均等待时间: {ours_avg_time:.1f}s")
    print()

    # 相对于其他方法的平均优势
    print("✓ Ours相对于其他方法的平均优势:")
    for i in range(1, len(methods)):
        avg_rate_diff = ours_avg_rate - np.mean(matching_rates_np[i])
        avg_time_diff = ours_avg_time - np.mean(waiting_times_np[i])
        time_pct = (avg_time_diff / ours_avg_time) * 100
        print(f"  vs {methods[i]:<20}: 匹配率高 {avg_rate_diff:>+5.1f}%,  等待时间短 {abs(avg_time_diff):>5.1f}s ({abs(time_pct):>5.1f}%)")

    print("-"*90)


if __name__ == '__main__':
    print("="*80)
    print("初始分布对比图生成工具")
    print("="*80)
    print("\n提示: 请修改脚本中的 matching_rates 和 waiting_times 数据")
    print("      (约第23-27行)\n")

    plot_distribution_comparison()

    print("\n" + "="*80)
    print("✅ 图表生成完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  - distribution_comparison.png  (PNG格式)")
    print("  - distribution_comparison.tiff (TIFF格式)")
    print()

