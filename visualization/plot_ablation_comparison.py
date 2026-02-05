#!/usr/bin/env python3
"""
消融实验对比图 - 特征提取方法对比
4种方法: Ours (MGCN-D3QN), CNN-D3QN, GCN(POI)-D3QN, GCN(Adj)-D3QN
2个指标: 平均匹配率, 平均等待时间

用法:
    python visualization/plot_ablation_comparison.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_ablation_comparison():
    """
    绘制消融实验对比图
    """
    # ==================== 数据 ====================
    # 请根据你的实际实验结果修改这些数值
    methods = ['Ours\n(MGCN-D3QN)', 'CNN-D3QN', 'GCN(POI)-D3QN', 'GCN(Adj)-D3QN']

    # 平均匹配率 (%) - 模拟数据,请替换为真实数据
    matching_rates = [92.5, 89.3, 87.8, 86.2]

    # 平均等待时间 (秒) - 模拟数据,请替换为真实数据
    waiting_times = [215.4, 238.7, 251.3, 264.8]

    # ==================== 绘图 ====================
    # 创建图形 - 1行2列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('消融实验对比 - 特征提取方法', fontsize=16, fontweight='bold', y=0.98)

    # 颜色方案
    colors = ['#E74C3C', '#9B59B6', '#3498DB', '#2ECC71']

    # ===== 子图1: 平均匹配率 =====
    x = np.arange(len(methods))
    bars1 = ax1.bar(x, matching_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('平均匹配率 (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) 平均匹配率对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=12)
    ax1.set_ylim([80, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, val in zip(bars1, matching_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加基准线 (Ours的值)
    ax1.axhline(y=matching_rates[0], color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Ours基准: {matching_rates[0]:.1f}%')
    ax1.legend(loc='lower right', fontsize=11)

    # ===== 子图2: 平均等待时间 =====
    bars2 = ax2.bar(x, waiting_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('平均等待时间 (秒)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) 平均等待时间对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.set_ylim([180, 280])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, val in zip(bars2, waiting_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加基准线 (Ours的值)
    ax2.axhline(y=waiting_times[0], color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Ours基准: {waiting_times[0]:.1f}s')
    ax2.legend(loc='upper right', fontsize=11)

    plt.tight_layout()

    # ==================== 保存 ====================
    save_dir = 'results/visualizations'
    os.makedirs(save_dir, exist_ok=True)

    # 保存PNG格式 (便于预览)
    output_path_png = os.path.join(save_dir, 'ablation_feature_comparison.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f'✓ 消融实验对比图已保存(PNG): {output_path_png}')

    # 保存TIFF格式 (适合论文发表)
    output_path_tiff = os.path.join(save_dir, 'ablation_feature_comparison.tiff')
    plt.savefig(output_path_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f'✓ 消融实验对比图已保存(TIFF): {output_path_tiff}')

    plt.close()

    # ==================== 数据摘要 ====================
    print("\n" + "="*70)
    print("消融实验数据摘要")
    print("="*70)
    print(f"{'方法':<25} | {'匹配率':<12} | {'等待时间'}")
    print("-"*70)
    for i, method in enumerate(methods):
        method_name = method.replace('\n', ' ')
        print(f"{method_name:<25} | {matching_rates[i]:>6.1f}%     | {waiting_times[i]:>6.1f}s")
    print("="*70)

    # 计算相对差异
    print("\n相对于Ours (MGCN-D3QN)的性能差异:")
    print("-"*70)
    print(f"{'方法':<25} | {'匹配率差异':<15} | {'等待时间差异'}")
    print("-"*70)
    for i in range(1, len(methods)):
        method_name = methods[i].replace('\n', ' ')
        rate_diff = matching_rates[i] - matching_rates[0]
        time_diff = waiting_times[i] - waiting_times[0]
        time_pct = (time_diff / waiting_times[0]) * 100
        print(f"{method_name:<25} | {rate_diff:>+6.1f}%        | {time_diff:>+6.1f}s ({time_pct:>+5.1f}%)")
    print("-"*70)


if __name__ == '__main__':
    print("="*80)
    print("消融实验对比图生成工具")
    print("="*80)
    print("\n提示: 请修改脚本中的 matching_rates 和 waiting_times 数据")
    print("      (约第26-30行)\n")

    plot_ablation_comparison()

    print("\n" + "="*80)
    print("✅ 图表生成完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  - ablation_feature_comparison.png  (PNG格式)")
    print("  - ablation_feature_comparison.tiff (TIFF格式)")
    print()

