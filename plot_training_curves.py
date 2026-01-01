"""
训练曲线可视化脚本 - 方案 B
用于绘制强化学习训练过程中的关键指标曲线
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import json


def plot_training_curves(stats_dict, save_dir='results/plots/'):
    """
    绘制训练曲线

    参数:
        stats_dict: 包含训练统计数据的字典
        save_dir: 保存图表的目录
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 创建一个大的 Figure 包含多个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('强化学习训练曲线 (方案 B - 多阶段奖励)', fontsize=16, fontweight='bold')

    episodes = np.arange(len(stats_dict['total_rewards']))

    # ===== 子图 1: 累积奖励 (Total Reward) =====
    ax = axes[0, 0]
    ax.plot(episodes, stats_dict['total_rewards'], 'b-', linewidth=2, label='Total Reward')
    # 添加平滑曲线（移动平均）
    if len(stats_dict['total_rewards']) > 10:
        window = max(1, len(stats_dict['total_rewards']) // 10)
        smoothed = np.convolve(stats_dict['total_rewards'], np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(stats_dict['total_rewards'])), smoothed, 'r--',
                linewidth=2, label=f'Smoothed (window={window})', alpha=0.7)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (Revenue)', fontsize=12)
    ax.set_title('累积奖励曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # ===== 子图 2: 训练损失 (Loss) =====
    ax = axes[0, 1]
    ax.plot(episodes, stats_dict['losses'], 'g-', linewidth=2, label='Loss')
    if len(stats_dict['losses']) > 10:
        window = max(1, len(stats_dict['losses']) // 10)
        smoothed = np.convolve(stats_dict['losses'], np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(stats_dict['losses'])), smoothed, 'orange',
                linewidth=2, label=f'Smoothed (window={window})', alpha=0.7)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('训练损失曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_yscale('log')  # 使用对数刻度以便更好地查看小的损失值

    # ===== 子图 3: 探索率 (Epsilon) =====
    ax = axes[1, 0]
    ax.plot(episodes, stats_dict['epsilon_history'], 'purple', linewidth=2, label='Epsilon')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax.set_title('探索率衰减曲线', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, max(stats_dict['epsilon_history']) * 1.1])

    # ===== 子图 4: 业务指标 (Completion Rate, Match Rate, etc.) =====
    ax = axes[1, 1]

    # 确保所有指标长度一致
    min_len = min(
        len(stats_dict.get('completion_rates', [])),
        len(stats_dict.get('match_rates', [])),
        len(stats_dict.get('cancel_rates', []))
    )

    if min_len > 0:
        episodes_metrics = np.arange(min_len)

        ax.plot(episodes_metrics, stats_dict['completion_rates'][:min_len],
                'b-', linewidth=2, label='Completion Rate', marker='o', markersize=4)
        ax.plot(episodes_metrics, stats_dict['match_rates'][:min_len],
                'g-', linewidth=2, label='Match Rate', marker='s', markersize=4)
        ax.plot(episodes_metrics, stats_dict['cancel_rates'][:min_len],
                'r-', linewidth=2, label='Cancel Rate', marker='^', markersize=4)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title('业务指标曲线', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, '暂无业务指标数据', ha='center', va='center', fontsize=12)
        ax.set_title('业务指标曲线 (数据不足)', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    plot_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存到: {plot_path}")

    # 返回 Figure 对象以便进一步处理
    return fig


def plot_waiting_time_curve(stats_dict, save_dir='results/plots/'):
    """
    绘制平均等待时间曲线

    参数:
        stats_dict: 包含训练统计数据的字典
        save_dir: 保存图表的目录
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    if len(stats_dict.get('avg_waiting_times', [])) > 0:
        episodes = np.arange(len(stats_dict['avg_waiting_times']))
        ax.plot(episodes, stats_dict['avg_waiting_times'], 'b-', linewidth=2, label='Avg Waiting Time')

        # 添加平滑曲线
        if len(stats_dict['avg_waiting_times']) > 10:
            window = max(1, len(stats_dict['avg_waiting_times']) // 10)
            smoothed = np.convolve(stats_dict['avg_waiting_times'], np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(stats_dict['avg_waiting_times'])), smoothed, 'r--',
                    linewidth=2, label=f'Smoothed (window={window})', alpha=0.7)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
        ax.set_title('平均等待时间曲线 (优化目标)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, '暂无等待时间数据', ha='center', va='center', fontsize=12)

    plt.tight_layout()

    # 保存图表
    plot_path = Path(save_dir) / 'waiting_time_curve.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 等待时间曲线已保存到: {plot_path}")

    return fig


def load_training_stats(checkpoint_path):
    """
    从 checkpoint 中加载训练统计数据

    参数:
        checkpoint_path: checkpoint 文件路径

    返回:
        包含训练统计数据的字典
    """
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        stats = {
            'total_rewards': checkpoint.get('total_rewards', []),
            'losses': checkpoint.get('losses', []),
            'epsilon_history': checkpoint.get('epsilon_history', []),
            'completion_rates': checkpoint.get('completion_rates', []),
            'avg_waiting_times': checkpoint.get('avg_waiting_times', []),
            'match_rates': checkpoint.get('match_rates', []),
            'cancel_rates': checkpoint.get('cancel_rates', [])
        }
        print(f"✓ 成功加载 checkpoint: {checkpoint_path}")
        return stats
    except Exception as e:
        print(f"✗ 加载 checkpoint 失败: {e}")
        return None


def main():
    """
    主函数：从最新的 checkpoint 加载数据并绘制曲线
    """
    import glob

    # 查找最新的 checkpoint
    checkpoint_dir = Path('results/models/')
    checkpoint_files = list(checkpoint_dir.glob('mgcn_dispatcher_episode_*.pt'))

    if not checkpoint_files:
        print("✗ 未找到任何 checkpoint 文件")
        print(f"  请确保 checkpoint 保存在: {checkpoint_dir}")
        return

    # 选择最新的 checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"✓ 使用最新的 checkpoint: {latest_checkpoint}")

    # 加载统计数据
    stats = load_training_stats(str(latest_checkpoint))
    if stats is None:
        return

    # 绘制曲线
    print("\n绘制训练曲线...")
    plot_training_curves(stats)
    plot_waiting_time_curve(stats)

    # 打印统计摘要
    print("\n" + "="*60)
    print("训练统计摘要")
    print("="*60)
    print(f"总 Episode 数: {len(stats['total_rewards'])}")
    if stats['total_rewards']:
        print(f"总奖励 (最后): {stats['total_rewards'][-1]:.2f}")
        print(f"总奖励 (平均): {np.mean(stats['total_rewards']):.2f}")
        print(f"总奖励 (最大): {np.max(stats['total_rewards']):.2f}")
    if stats['losses']:
        print(f"损失 (最后): {stats['losses'][-1]:.6f}")
        print(f"损失 (平均): {np.mean(stats['losses']):.6f}")
    if stats['epsilon_history']:
        print(f"Epsilon (最后): {stats['epsilon_history'][-1]:.4f}")
    if stats['completion_rates']:
        print(f"完成率 (最后): {stats['completion_rates'][-1]:.1%}")
        print(f"完成率 (平均): {np.mean(stats['completion_rates']):.1%}")
    if stats['avg_waiting_times']:
        print(f"平均等待时间 (最后): {stats['avg_waiting_times'][-1]:.1f}s")
        print(f"平均等待时间 (平均): {np.mean(stats['avg_waiting_times']):.1f}s")
    print("="*60)

    print("\n✓ 所有曲线已绘制完成！")


if __name__ == '__main__':
    main()

