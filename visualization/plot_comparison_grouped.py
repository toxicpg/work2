"""
方法对比图 - 分组柱状图
每天并排3根柱子，对比不同方法的匹配率
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据 ====================
# 方法1：你的主实验方法
data_main = {
    'completion_rate': [0.8321, 0.8476, 0.8635, 0.8774, 0.8890, 0.8981, 0.9051],
    'waiting_time': [247.35, 240.12, 232.89, 226.54, 220.78, 216.32, 212.45]
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

# 每天的订单总数
daily_orders = [167407, 168267, 180804, 178820, 168970, 168981, 176344]

# 日期标签
dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']


# ==================== 图1: 匹配率分组柱状图 ====================
def plot_grouped_completion_rate():
    """绘制匹配率分组柱状图 - 每天3根柱子"""

    fig, ax = plt.subplots(figsize=(14, 6))

    # 转换为百分比
    main_rate = np.array(data_main['completion_rate']) * 100
    walk_rate = np.array(data_random_walk['completion_rate']) * 100
    dispatch_rate = np.array(data_random_dispatch['completion_rate']) * 100

    # 设置柱子位置
    x = np.arange(len(dates))  # 7天的位置
    width = 0.25  # 每根柱子的宽度

    # 绘制3组柱子
    bars1 = ax.bar(x - width, main_rate, width, label='主实验方法',
                   color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x, walk_rate, width, label='Random Walk',
                   color='#3498DB', alpha=0.8)
    bars3 = ax.bar(x + width, dispatch_rate, width, label='Random Dispatch',
                   color='#2ECC71', alpha=0.8)

    # 在柱子上标注数值（可选）
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)

    # add_value_labels(bars1)  # 如果需要数值标注，取消注释
    # add_value_labels(bars2)
    # add_value_labels(bars3)

    # 设置
    ax.set_xlabel('日期', fontsize=13, fontweight='bold')
    ax.set_ylabel('匹配率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('不同方法的匹配率对比 (11.24-11.30)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'comparison_grouped_completion_rate.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✓ 匹配率对比图已保存: {save_path}')
    plt.close()


# ==================== 图2: 等待时间分组柱状图 ====================
def plot_grouped_waiting_time():
    """绘制等待时间分组柱状图 - 每天3根柱子"""

    fig, ax = plt.subplots(figsize=(14, 6))

    # 数据
    main_wait = np.array(data_main['waiting_time'])
    walk_wait = np.array(data_random_walk['waiting_time'])
    dispatch_wait = np.array(data_random_dispatch['waiting_time'])

    # 设置柱子位置
    x = np.arange(len(dates))
    width = 0.25

    # 绘制3组柱子
    bars1 = ax.bar(x - width, main_wait, width, label='主实验方法',
                   color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x, walk_wait, width, label='Random Walk',
                   color='#3498DB', alpha=0.8)
    bars3 = ax.bar(x + width, dispatch_wait, width, label='Random Dispatch',
                   color='#2ECC71', alpha=0.8)

    # 设置
    ax.set_xlabel('日期', fontsize=13, fontweight='bold')
    ax.set_ylabel('平均等待时间 (秒)', fontsize=13, fontweight='bold')
    ax.set_title('不同方法的等待时间对比 (11.24-11.30)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'
    save_path = os.path.join(save_dir, 'comparison_grouped_waiting_time.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✓ 等待时间对比图已保存: {save_path}')
    plt.close()


# ==================== 图3: 堆叠柱状图（订单匹配情况）+ 匹配率点 ====================
def plot_stacked_orders():
    """绘制堆叠柱状图 - 每天3根柱子（3种方法），柱子上方显示匹配率点"""

    fig, ax = plt.subplots(figsize=(16, 7))

    # 数据准备
    main_rate = np.array(data_main['completion_rate'])
    walk_rate = np.array(data_random_walk['completion_rate'])
    dispatch_rate = np.array(data_random_dispatch['completion_rate'])

    # 计算每种方法的已匹配/未匹配订单数
    main_matched = np.array(daily_orders) * main_rate
    main_unmatched = np.array(daily_orders) * (1 - main_rate)

    walk_matched = np.array(daily_orders) * walk_rate
    walk_unmatched = np.array(daily_orders) * (1 - walk_rate)

    dispatch_matched = np.array(daily_orders) * dispatch_rate
    dispatch_unmatched = np.array(daily_orders) * (1 - dispatch_rate)

    # 设置柱子位置
    x = np.arange(len(dates))
    width = 0.25

    # 绘制3组堆叠柱状图
    # 方法1
    ax.bar(x - width, main_matched, width, label='主实验-已匹配',
           color='#E74C3C', alpha=0.8)
    ax.bar(x - width, main_unmatched, width, bottom=main_matched,
           color='#E74C3C', alpha=0.3)

    # 方法2
    ax.bar(x, walk_matched, width, label='Random Walk-已匹配',
           color='#3498DB', alpha=0.8)
    ax.bar(x, walk_unmatched, width, bottom=walk_matched,
           color='#3498DB', alpha=0.3)

    # 方法3
    ax.bar(x + width, dispatch_matched, width, label='Random Dispatch-已匹配',
           color='#2ECC71', alpha=0.8)
    ax.bar(x + width, dispatch_unmatched, width, bottom=dispatch_matched,
           color='#2ECC71', alpha=0.3)

    # 创建第二个Y轴用于匹配率
    ax2 = ax.twinx()

    # 在柱子顶部绘制匹配率点（带连线）
    main_pct = main_rate * 100
    walk_pct = walk_rate * 100
    dispatch_pct = dispatch_rate * 100

    # 主实验方法：圆形点 + 实线
    ax2.plot(x - width, main_pct, 'o-', color='#C0392B', markersize=10,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='主实验-匹配率', zorder=10)

    # Random Walk：方形点 + 虚线
    ax2.plot(x, walk_pct, 's--', color='#2874A6', markersize=10,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='Random Walk-匹配率', zorder=10)

    # Random Dispatch：三角形点 + 点线
    ax2.plot(x + width, dispatch_pct, '^:', color='#229954', markersize=10,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='Random Dispatch-匹配率', zorder=10)

    # 设置左Y轴（订单数）
    ax.set_xlabel('日期', fontsize=13, fontweight='bold')
    ax.set_ylabel('订单数', fontsize=13, fontweight='bold')
    ax.set_title('不同方法的订单匹配情况与匹配率对比 (11.24-11.30)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dates, fontsize=12)
    ax.set_ylim(0, max(daily_orders) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 设置右Y轴（匹配率）
    ax2.set_ylabel('匹配率 (%)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 简化图例（只显示3种方法）
    ax.legend(['主实验方法', 'Random Walk', 'Random Dispatch'],
             loc='upper left', fontsize=11, framealpha=0.9, title='堆叠柱（下=已匹配，上=未匹配）')
    ax2.legend(lines2, labels2, loc='upper right', fontsize=11, framealpha=0.9, title='匹配率点')

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'
    save_path = os.path.join(save_dir, 'comparison_stacked_orders.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'✓ 订单匹配情况对比图已保存: {save_path}')
    plt.close()


# ==================== 主函数 ====================
def main():
    print('=' * 60)
    print('开始生成方法对比图表...')
    print('=' * 60)

    plot_grouped_completion_rate()
    plot_grouped_waiting_time()
    plot_stacked_orders()

    print('\n' + '=' * 60)
    print('✅ 所有对比图表生成完成！')
    print('=' * 60)
    print('\n生成的图表：')
    print('  1. comparison_grouped_completion_rate.png  - 匹配率分组柱状图')
    print('  2. comparison_grouped_waiting_time.png     - 等待时间分组柱状图')
    print('  3. comparison_stacked_orders.png           - 订单匹配情况（堆叠柱+折线）')


if __name__ == '__main__':
    main()

