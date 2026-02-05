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
# 方法1：你的主实验方法 (MGCN-D3QN)
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

# 方法4：CNN-DDQN (消融实验)
data_cnn_ddqn = {
    'completion_rate': [0.8245, 0.8398, 0.8542, 0.8678, 0.8802, 0.8915, 0.9016],
    'waiting_time': [252.18, 245.67, 239.34, 233.56, 228.21, 223.45, 219.12]
}

# 方法5：SARSA(λ)-SAA
data_sarsa_lambda = {
    'completion_rate': [0.8567, 0.8712, 0.8848, 0.8972, 0.9085, 0.9186, 0.9275],
    'waiting_time': [231.45, 224.89, 218.76, 213.12, 207.98, 203.34, 199.23]
}

# 每天的订单总数
daily_orders = [167407, 168267, 180804, 178820, 168970, 168981, 176344]

# 日期标签
dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']


# ==================== 图1: 匹配率分组柱状图 ====================
def plot_grouped_completion_rate():
    """绘制匹配率分组柱状图 - 每天5根柱子"""

    fig, ax = plt.subplots(figsize=(16, 7))

    # 转换为百分比
    main_rate = np.array(data_main['completion_rate']) * 100
    walk_rate = np.array(data_random_walk['completion_rate']) * 100
    dispatch_rate = np.array(data_random_dispatch['completion_rate']) * 100
    cnn_rate = np.array(data_cnn_ddqn['completion_rate']) * 100
    sarsa_rate = np.array(data_sarsa_lambda['completion_rate']) * 100

    # 设置柱子位置
    x = np.arange(len(dates))  # 7天的位置
    width = 0.15  # 每根柱子的宽度（5根柱子所以更窄）

    # 绘制5组柱子
    bars1 = ax.bar(x - 2*width, main_rate, width, label='MGCN-D3QN',
                   color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x - width, cnn_rate, width, label='CNN-DDQN',
                   color='#9B59B6', alpha=0.8)
    bars3 = ax.bar(x, sarsa_rate, width, label='SARSA(λ)-SAA',
                   color='#F39C12', alpha=0.8)
    bars4 = ax.bar(x + width, walk_rate, width, label='Random Walk',
                   color='#3498DB', alpha=0.8)
    bars5 = ax.bar(x + 2*width, dispatch_rate, width, label='Random Dispatch',
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

    # 保存PNG格式 (便于预览)
    save_path_png = os.path.join(save_dir, 'comparison_grouped_completion_rate.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f'✓ 匹配率对比图已保存(PNG): {save_path_png}')

    # 保存TIFF格式 (适合论文发表)
    save_path_tiff = os.path.join(save_dir, 'comparison_grouped_completion_rate.tiff')
    plt.savefig(save_path_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f'✓ 匹配率对比图已保存(TIFF): {save_path_tiff}')

    plt.close()


# ==================== 图2: 等待时间分组柱状图 ====================
def plot_grouped_waiting_time():
    """绘制等待时间分组柱状图 - 每天5根柱子"""

    fig, ax = plt.subplots(figsize=(16, 7))

    # 数据
    main_wait = np.array(data_main['waiting_time'])
    walk_wait = np.array(data_random_walk['waiting_time'])
    dispatch_wait = np.array(data_random_dispatch['waiting_time'])
    cnn_wait = np.array(data_cnn_ddqn['waiting_time'])
    sarsa_wait = np.array(data_sarsa_lambda['waiting_time'])

    # 设置柱子位置
    x = np.arange(len(dates))
    width = 0.15

    # 绘制5组柱子
    bars1 = ax.bar(x - 2*width, main_wait, width, label='MGCN-D3QN',
                   color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x - width, cnn_wait, width, label='CNN-DDQN',
                   color='#9B59B6', alpha=0.8)
    bars3 = ax.bar(x, sarsa_wait, width, label='SARSA(λ)-SAA',
                   color='#F39C12', alpha=0.8)
    bars4 = ax.bar(x + width, walk_wait, width, label='Random Walk',
                   color='#3498DB', alpha=0.8)
    bars5 = ax.bar(x + 2*width, dispatch_wait, width, label='Random Dispatch',
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

    # 保存PNG格式 (便于预览)
    save_path_png = os.path.join(save_dir, 'comparison_grouped_waiting_time.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f'✓ 等待时间对比图已保存(PNG): {save_path_png}')

    # 保存TIFF格式 (适合论文发表)
    save_path_tiff = os.path.join(save_dir, 'comparison_grouped_waiting_time.tiff')
    plt.savefig(save_path_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f'✓ 等待时间对比图已保存(TIFF): {save_path_tiff}')

    plt.close()


# ==================== 图3: 堆叠柱状图（订单匹配情况）+ 匹配率点 ====================
def plot_stacked_orders():
    """绘制堆叠柱状图 - 每天5根柱子（5种方法），柱子上方显示匹配率点"""

    fig, ax = plt.subplots(figsize=(18, 8))

    # 数据准备
    main_rate = np.array(data_main['completion_rate'])
    walk_rate = np.array(data_random_walk['completion_rate'])
    dispatch_rate = np.array(data_random_dispatch['completion_rate'])
    cnn_rate = np.array(data_cnn_ddqn['completion_rate'])
    sarsa_rate = np.array(data_sarsa_lambda['completion_rate'])

    # 计算每种方法的已匹配/未匹配订单数
    main_matched = np.array(daily_orders) * main_rate
    main_unmatched = np.array(daily_orders) * (1 - main_rate)

    cnn_matched = np.array(daily_orders) * cnn_rate
    cnn_unmatched = np.array(daily_orders) * (1 - cnn_rate)

    sarsa_matched = np.array(daily_orders) * sarsa_rate
    sarsa_unmatched = np.array(daily_orders) * (1 - sarsa_rate)

    walk_matched = np.array(daily_orders) * walk_rate
    walk_unmatched = np.array(daily_orders) * (1 - walk_rate)

    dispatch_matched = np.array(daily_orders) * dispatch_rate
    dispatch_unmatched = np.array(daily_orders) * (1 - dispatch_rate)

    # 设置柱子位置
    x = np.arange(len(dates))
    width = 0.15

    # 绘制5组堆叠柱状图
    # 方法1: MGCN-D3QN
    ax.bar(x - 2*width, main_matched, width, label='MGCN-D3QN-已匹配',
           color='#E74C3C', alpha=0.8)
    ax.bar(x - 2*width, main_unmatched, width, bottom=main_matched,
           color='#E74C3C', alpha=0.3)

    # 方法2: CNN-DDQN
    ax.bar(x - width, cnn_matched, width, label='CNN-DDQN-已匹配',
           color='#9B59B6', alpha=0.8)
    ax.bar(x - width, cnn_unmatched, width, bottom=cnn_matched,
           color='#9B59B6', alpha=0.3)

    # 方法3: SARSA(λ)-SAA
    ax.bar(x, sarsa_matched, width, label='SARSA(λ)-SAA-已匹配',
           color='#F39C12', alpha=0.8)
    ax.bar(x, sarsa_unmatched, width, bottom=sarsa_matched,
           color='#F39C12', alpha=0.3)

    # 方法4: Random Walk
    ax.bar(x + width, walk_matched, width, label='Random Walk-已匹配',
           color='#3498DB', alpha=0.8)
    ax.bar(x + width, walk_unmatched, width, bottom=walk_matched,
           color='#3498DB', alpha=0.3)

    # 方法5: Random Dispatch
    ax.bar(x + 2*width, dispatch_matched, width, label='Random Dispatch-已匹配',
           color='#2ECC71', alpha=0.8)
    ax.bar(x + 2*width, dispatch_unmatched, width, bottom=dispatch_matched,
           color='#2ECC71', alpha=0.3)

    # 创建第二个Y轴用于匹配率
    ax2 = ax.twinx()

    # 在柱子顶部绘制匹配率点（带连线）
    main_pct = main_rate * 100
    cnn_pct = cnn_rate * 100
    sarsa_pct = sarsa_rate * 100
    walk_pct = walk_rate * 100
    dispatch_pct = dispatch_rate * 100

    # MGCN-D3QN：圆形点 + 实线
    ax2.plot(x - 2*width, main_pct, 'o-', color='#C0392B', markersize=10,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='MGCN-D3QN-匹配率', zorder=10)

    # CNN-DDQN：菱形点 + 实线
    ax2.plot(x - width, cnn_pct, 'D-', color='#7D3C98', markersize=9,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='CNN-DDQN-匹配率', zorder=10)

    # SARSA(λ)-SAA：星形点 + 虚线
    ax2.plot(x, sarsa_pct, '*--', color='#D68910', markersize=12,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='SARSA(λ)-SAA-匹配率', zorder=10)

    # Random Walk：方形点 + 虚线
    ax2.plot(x + width, walk_pct, 's--', color='#2874A6', markersize=10,
            linewidth=2.5, markeredgecolor='white', markeredgewidth=2,
            label='Random Walk-匹配率', zorder=10)

    # Random Dispatch：三角形点 + 点线
    ax2.plot(x + 2*width, dispatch_pct, '^:', color='#229954', markersize=10,
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

    # 简化图例（显示5种方法）
    ax.legend(['MGCN-D3QN', 'CNN-DDQN', 'SARSA(λ)-SAA', 'Random Walk', 'Random Dispatch'],
             loc='upper left', fontsize=10, framealpha=0.9, title='堆叠柱（下=已匹配，上=未匹配）')
    ax2.legend(lines2, labels2, loc='upper right', fontsize=10, framealpha=0.9, title='匹配率点')

    plt.tight_layout()

    # 保存
    save_dir = 'results/visualizations'

    # 保存PNG格式 (便于预览)
    save_path_png = os.path.join(save_dir, 'comparison_stacked_orders.png')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f'✓ 订单匹配情况对比图已保存(PNG): {save_path_png}')

    # 保存TIFF格式 (适合论文发表)
    save_path_tiff = os.path.join(save_dir, 'comparison_stacked_orders.tiff')
    plt.savefig(save_path_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f'✓ 订单匹配情况对比图已保存(TIFF): {save_path_tiff}')

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
    print('  1. comparison_grouped_completion_rate.png  - 匹配率分组柱状图 (5种方法)')
    print('  2. comparison_grouped_waiting_time.png     - 等待时间分组柱状图 (5种方法)')
    print('  3. comparison_stacked_orders.png           - 订单匹配情况（堆叠柱+折线, 5种方法）')
    print('\n包含的方法：')
    print('  - MGCN-D3QN (主实验)')
    print('  - CNN-DDQN (消融实验)')
    print('  - SARSA(λ)-SAA')
    print('  - Random Walk')
    print('  - Random Dispatch')


if __name__ == '__main__':
    main()

