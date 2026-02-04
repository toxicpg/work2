"""
主实验结果可视化
绘制堆叠柱状图 + 折线图（匹配率）和等待时间折线图
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据 ====================
# 你的3组数据：3种不同方法（都是1800辆车）
# 每组第1行=匹配率(%)，第2行=等待时间(秒)

# 方法1：你的主实验方法（比如 MGCN-DDQN）
data_main = {
    'completion_rate': [0.8321, 0.8476, 0.8635, 0.8774, 0.8890, 0.8981, 0.9051],  # 第1行
    'waiting_time': [247.35, 240.12, 232.89, 226.54, 220.78, 216.32, 212.45]      # 第2行
}

# 方法2：Random Walk
data_random_walk = {
    'completion_rate': [0.8701, 0.8833, 0.8956, 0.9065, 0.9158, 0.9236, 0.9298],
    'waiting_time': [218.67, 212.34, 206.78, 201.92, 197.56, 193.87, 190.65]
}

# 方法3：Random Dispatch
data_random_dispatch = {
    'completion_rate': [0.8912, 0.9021, 0.9118, 0.9203, 0.9277, 0.9340, 0.9393],
    'waiting_time': [198.45, 193.21, 188.56, 184.32, 180.67, 177.43, 174.56]
}

days = list(range(1, 8))  # Day 1-7

# 每天的订单总数
daily_orders = [167407, 168267, 180804, 178820, 168970, 168981, 176344]

# ==================== 图1: 匹配率（堆叠柱状图 + 折线图）====================
def plot_completion_rate_combined():
    """绘制3×1子图：每种方法一个子图，堆叠柱+折线"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = [
        (data_main, '主实验方法', axes[0]),
        (data_random_walk, 'Random Walk', axes[1]),
        (data_random_dispatch, 'Random Dispatch', axes[2])
    ]

    for data, label, ax in datasets:
        # 计算已匹配和未匹配的订单数
        completion_rate = np.array(data['completion_rate'])
        matched_orders = np.array(daily_orders) * completion_rate
        unmatched_orders = np.array(daily_orders) * (1 - completion_rate)

        # 堆叠柱状图（绝对数量）
        bars1 = ax.bar(days, matched_orders, label='已匹配订单',
                      color='#2ECC71', alpha=0.8, width=0.7)
        bars2 = ax.bar(days, unmatched_orders, bottom=matched_orders, label='未匹配订单',
                      color='#E74C3C', alpha=0.6, width=0.7)

        # 创建第二个Y轴用于匹配率折线
        ax2 = ax.twinx()

        # 折线图（匹配率百分比）
        completion_pct = completion_rate * 100
        line = ax2.plot(days, completion_pct, marker='o', color='#3498DB',
                       linewidth=2.5, markersize=8, label='匹配率', zorder=10)

        # 在折线点上标注数值
        for i, (x, y) in enumerate(zip(days, completion_pct)):
            if i == 0 or i == len(days) - 1:  # 只标注首尾
                ax2.text(x, y + 1, f'{y:.1f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#3498DB')

        # 设置左Y轴（订单数）
        ax.set_xlabel('天数', fontsize=12, fontweight='bold')
        ax.set_ylabel('订单数', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xticks(days)
        ax.set_ylim(0, max(daily_orders) * 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 设置右Y轴（匹配率）
        ax2.set_ylabel('匹配率 (%)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)

        # 只在第一个子图显示图例
        if ax == axes[0]:
            # 合并两个轴的图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                     loc='lower right', framealpha=0.9, fontsize=10)

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fig1_completion_rate_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✓ 图1已保存: {save_path}')
    plt.close()


# ==================== 图2: 等待时间（折线图）====================
def plot_waiting_time():
    """绘制等待时间变化折线图（3条线在一张图）"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3条折线
    ax.plot(days, data_main['waiting_time'], marker='o', linewidth=2.5,
           markersize=8, label='主实验方法', color='#E74C3C')
    ax.plot(days, data_random_walk['waiting_time'], marker='s', linewidth=2.5,
           markersize=8, label='Random Walk', color='#3498DB')
    ax.plot(days, data_random_dispatch['waiting_time'], marker='^', linewidth=2.5,
           markersize=8, label='Random Dispatch', color='#2ECC71')

    # 设置
    ax.set_xlabel('天数', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均等待时间 (秒)', fontsize=12, fontweight='bold')
    ax.set_title('7天等待时间变化', fontsize=14, fontweight='bold')
    ax.set_xticks(days)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=11)

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'
    save_path = os.path.join(save_dir, 'fig2_waiting_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✓ 图2已保存: {save_path}')
    plt.close()


# ==================== 图3: 平均性能对比（柱状图）====================
def plot_average_performance():
    """绘制平均性能对比柱状图"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 计算平均值
    avg_completion = [
        np.mean(data_main['completion_rate']) * 100,
        np.mean(data_random_walk['completion_rate']) * 100,
        np.mean(data_random_dispatch['completion_rate']) * 100
    ]

    avg_waiting = [
        np.mean(data_main['waiting_time']),
        np.mean(data_random_walk['waiting_time']),
        np.mean(data_random_dispatch['waiting_time'])
    ]

    methods = ['主实验', 'Random Walk', 'Random Dispatch']
    x_pos = np.arange(len(methods))

    # 子图1: 平均匹配率
    bars1 = ax1.bar(x_pos, avg_completion, color=['#E74C3C', '#3498DB', '#2ECC71'],
                   alpha=0.8, width=0.6)
    ax1.set_xlabel('方法', fontsize=12, fontweight='bold')
    ax1.set_ylabel('平均匹配率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('平均匹配率对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars1, avg_completion)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # 子图2: 平均等待时间
    bars2 = ax2.bar(x_pos, avg_waiting, color=['#E74C3C', '#3498DB', '#2ECC71'],
                   alpha=0.8, width=0.6)
    ax2.set_xlabel('方法', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均等待时间 (秒)', fontsize=12, fontweight='bold')
    ax2.set_title('平均等待时间对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标注数值
    for i, (bar, val) in enumerate(zip(bars2, avg_waiting)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 3,
                f'{val:.1f}s', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'
    save_path = os.path.join(save_dir, 'fig3_average_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✓ 图3已保存: {save_path}')
    plt.close()


# ==================== 主函数 ====================
def main():
    print('=' * 60)
    print('开始生成主实验结果图表...')
    print('=' * 60)

    plot_completion_rate_combined()
    plot_waiting_time()
    plot_average_performance()

    print('\n' + '=' * 60)
    print('✅ 所有图表生成完成！')
    print('=' * 60)


if __name__ == '__main__':
    main()

