"""
消融实验结果可视化脚本

用法:
    python plot_ablation_results.py <result_file.json>
    python plot_ablation_results.py results/ablation_studies/ablation_results_20260101_120000.json
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(result_file):
    """加载消融实验结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_comparison(results, output_dir='results/ablation_plots/'):
    """绘制对比图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    ablation_types = []
    avg_rewards = []
    std_rewards = []
    avg_losses = []

    for ablation_type, result in results.items():
        ablation_types.append(ablation_type)
        avg_rewards.append(result['avg_reward'])
        std_rewards.append(result.get('std_reward', 0))
        avg_losses.append(result['avg_loss'])

    # 创建对比图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('消融实验对比分析', fontsize=16, fontweight='bold')

    # 1. 平均奖励对比
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ablation_types)))
    bars1 = ax1.bar(range(len(ablation_types)), avg_rewards, color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(range(len(ablation_types)), avg_rewards, yerr=std_rewards, fmt='none',
                 ecolor='black', capsize=5, capthick=2)
    ax1.set_ylabel('平均奖励', fontsize=11, fontweight='bold')
    ax1.set_title('平均奖励对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(ablation_types)))
    ax1.set_xticklabels(ablation_types, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, avg_rewards)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + std_rewards[i] + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # 2. 平均损失对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(ablation_types)), avg_losses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('平均损失', fontsize=11, fontweight='bold')
    ax2.set_title('平均损失对比', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(ablation_types)))
    ax2.set_xticklabels(ablation_types, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars2, avg_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. 性能差异分析
    ax3 = axes[1, 0]
    full_model_reward = results['full_model']['avg_reward']
    performance_diff = [(r - full_model_reward) / full_model_reward * 100
                       for r in avg_rewards]

    colors_diff = ['green' if x >= 0 else 'red' for x in performance_diff]
    bars3 = ax3.bar(range(len(ablation_types)), performance_diff, color=colors_diff, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_ylabel('性能差异 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('相对于完整模型的性能差异', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(ablation_types)))
    ax3.set_xticklabels(ablation_types, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars3, performance_diff):
        ax3.text(bar.get_x() + bar.get_width()/2, val + (1 if val >= 0 else -2),
                f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    # 4. 组件重要性评分
    ax4 = axes[1, 1]
    importance_scores = []
    importance_labels = []

    for ablation_type, diff in zip(ablation_types, performance_diff):
        if ablation_type != 'full_model':
            importance_scores.append(-diff)  # 负数表示重要性
            importance_labels.append(ablation_type)

    if importance_scores:
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        sorted_labels = [importance_labels[i] for i in sorted_indices]

        # 确定颜色
        colors_importance = []
        for score in sorted_scores:
            if score > 20:
                colors_importance.append('#d62728')  # 红色 - 非常重要
            elif score > 10:
                colors_importance.append('#ff7f0e')  # 橙色 - 很重要
            elif score > 5:
                colors_importance.append('#2ca02c')  # 绿色 - 较为重要
            else:
                colors_importance.append('#1f77b4')  # 蓝色 - 影响较小

        bars4 = ax4.barh(range(len(sorted_scores)), sorted_scores, color=colors_importance, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('组件重要性 (%)', fontsize=11, fontweight='bold')
        ax4.set_title('组件重要性排序', fontsize=12, fontweight='bold')
        ax4.set_yticks(range(len(sorted_labels)))
        ax4.set_yticklabels(sorted_labels)
        ax4.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for bar, val in zip(bars4, sorted_scores):
            ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    # 保存图表
    output_file = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存到: {output_file}")

    return fig


def plot_episodes_progression(results, output_dir='results/ablation_plots/'):
    """绘制 episode 进度图"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (ablation_type, result), color in zip(results.items(), colors):
        episodes = result.get('episodes', [])
        if episodes:
            episode_nums = [e['episode'] for e in episodes]
            rewards = [e['train_reward'] for e in episodes]
            ax.plot(episode_nums, rewards, marker='o', label=ablation_type, color=color, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('训练奖励', fontsize=12, fontweight='bold')
    ax.set_title('消融实验 - Episode 进度对比', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'ablation_episodes_progression.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Episode 进度图已保存到: {output_file}")

    return fig


def generate_summary_table(results, output_dir='results/ablation_plots/'):
    """生成总结表格"""
    os.makedirs(output_dir, exist_ok=True)

    data = []
    full_model_reward = results.get('full_model', {}).get('avg_reward', 0)

    for ablation_type, result in results.items():
        avg_reward = result['avg_reward']
        avg_loss = result['avg_loss']
        std_reward = result.get('std_reward', 0)

        if full_model_reward > 0:
            reward_diff = avg_reward - full_model_reward
            reward_pct = (reward_diff / full_model_reward) * 100
        else:
            reward_diff = 0
            reward_pct = 0

        # 确定重要性等级
        if ablation_type == 'full_model':
            importance = '基准'
        elif abs(reward_pct) > 20:
            importance = '★★★★★ 非常重要'
        elif abs(reward_pct) > 10:
            importance = '★★★★☆ 很重要'
        elif abs(reward_pct) > 5:
            importance = '★★★☆☆ 较为重要'
        elif abs(reward_pct) > 2:
            importance = '★★☆☆☆ 有作用'
        else:
            importance = '★☆☆☆☆ 影响较小'

        data.append({
            'Ablation Type': ablation_type,
            'Avg Reward': f'{avg_reward:.2f}',
            'Std Reward': f'{std_reward:.2f}',
            'Avg Loss': f'{avg_loss:.4f}',
            'Reward Diff': f'{reward_diff:+.2f}',
            'Reward Diff %': f'{reward_pct:+.1f}%',
            'Importance': importance
        })

    df = pd.DataFrame(data)

    # 保存为 CSV
    output_file = os.path.join(output_dir, 'ablation_summary_table.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ 总结表格已保存到: {output_file}")

    # 打印表格
    print(f"\n{'='*120}")
    print("消融实验总结表格")
    print(f"{'='*120}\n")
    print(df.to_string(index=False))
    print()

    return df


def main():
    """主函数"""

    if len(sys.argv) < 2:
        # 查找最新的结果文件
        ablation_dir = Path('results/ablation_studies/')
        if ablation_dir.exists():
            result_files = list(ablation_dir.glob('ablation_results_*.json'))
            if result_files:
                result_file = str(sorted(result_files)[-1])
                print(f"使用最新的结果文件: {result_file}")
            else:
                print("❌ 未找到消融实验结果文件")
                print("用法: python plot_ablation_results.py <result_file.json>")
                return
        else:
            print("❌ 未找到消融实验结果目录")
            print("用法: python plot_ablation_results.py <result_file.json>")
            return
    else:
        result_file = sys.argv[1]

    if not os.path.exists(result_file):
        print(f"❌ 文件不存在: {result_file}")
        return

    print(f"加载结果文件: {result_file}")
    results = load_results(result_file)

    print("\n生成可视化图表...")

    # 生成图表
    plot_comparison(results)
    plot_episodes_progression(results)
    df = generate_summary_table(results)

    print("\n✓ 所有图表已生成！")
    print("  查看 results/ablation_plots/ 目录获取详细图表")


if __name__ == '__main__':
    main()

