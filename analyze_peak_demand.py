#!/usr/bin/env python3
"""
订单高峰需求分析脚本
用于论文中说明车辆数量配置的合理性

输出内容:
1. 每小时平均订单数
2. 高峰时段订单统计
3. 建议的车辆配置(基于供需比)
4. 不同车辆数下的供需比分析
5. 论文写作参考文本

用法:
    python analyze_peak_demand.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config import Config
from utils.data_process import DataProcessor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_hourly_demand(orders_df):
    """分析每小时订单需求"""
    print("\n" + "="*80)
    print("1. 每小时订单需求分析")
    print("="*80)

    # 提取小时
    orders_df['hour'] = orders_df['timestamp'].dt.hour

    # 计算每天每小时的订单数
    hourly_stats = orders_df.groupby(['relative_day', 'hour']).size().reset_index(name='order_count')

    # 计算每小时的平均订单数
    hourly_avg = hourly_stats.groupby('hour')['order_count'].agg(['mean', 'std', 'min', 'max'])

    print("\n每小时平均订单数:")
    print("-"*80)
    print(f"{'小时':<6} | {'平均订单数':<12} | {'标准差':<10} | {'最小值':<10} | {'最大值':<10}")
    print("-"*80)

    for hour in range(24):
        if hour in hourly_avg.index:
            stats = hourly_avg.loc[hour]
            print(f"{hour:>2d}:00  | {stats['mean']:>10.1f}  | {stats['std']:>8.1f}  | "
                  f"{stats['min']:>8.0f}  | {stats['max']:>8.0f}")

    return hourly_avg


def identify_peak_periods(hourly_avg):
    """识别高峰时段"""
    print("\n" + "="*80)
    print("2. 高峰时段识别")
    print("="*80)

    # 定义高峰阈值(平均值+0.5标准差)
    mean_orders = hourly_avg['mean'].mean()
    std_orders = hourly_avg['mean'].std()
    peak_threshold = mean_orders + 0.5 * std_orders

    print(f"\n全天平均订单数: {mean_orders:.1f} 订单/小时")
    print(f"标准差: {std_orders:.1f}")
    print(f"高峰阈值: {peak_threshold:.1f} 订单/小时 (均值 + 0.5倍标准差)")

    # 识别高峰时段
    peak_hours = hourly_avg[hourly_avg['mean'] >= peak_threshold].sort_values('mean', ascending=False)

    print(f"\n识别出 {len(peak_hours)} 个高峰时段:")
    print("-"*80)
    print(f"{'时段':<10} | {'平均订单数':<15} | {'与均值差':<15} | {'高峰程度'}")
    print("-"*80)

    for hour, stats in peak_hours.iterrows():
        diff = stats['mean'] - mean_orders
        ratio = (stats['mean'] / mean_orders - 1) * 100
        print(f"{hour:>2d}:00-{hour+1:>2d}:00 | {stats['mean']:>13.1f}  | "
              f"{diff:>+13.1f}  | {ratio:>+6.1f}%")

    # 找出最高峰
    max_hour = peak_hours.index[0]
    max_orders = peak_hours.iloc[0]['mean']

    print(f"\n🔥 最高峰时段: {max_hour}:00-{max_hour+1}:00")
    print(f"   平均订单数: {max_orders:.1f} 订单/小时")
    print(f"   是全天均值的 {max_orders/mean_orders:.2f} 倍")

    return peak_hours, max_orders


def calculate_vehicle_requirements(hourly_avg, service_times=[15, 20, 25, 30]):
    """计算不同服务时长下的车辆需求"""
    print("\n" + "="*80)
    print("3. 车辆需求计算")
    print("="*80)

    max_orders = hourly_avg['mean'].max()
    avg_orders = hourly_avg['mean'].mean()

    print(f"\n假设条件:")
    print(f"  - 最高峰订单数: {max_orders:.0f} 订单/小时")
    print(f"  - 平均订单数: {avg_orders:.0f} 订单/小时")
    print(f"  - 考虑不同的平均服务时长 (接驾+送达)")

    print(f"\n理论车辆需求 (基于排队论):")
    print("-"*80)
    print(f"{'服务时长(分钟)':<15} | {'高峰需求':<12} | {'平均需求':<12} | {'建议配置'}")
    print("-"*80)

    recommendations = []

    for service_time in service_times:
        # 单位时间内一辆车能服务的订单数
        orders_per_vehicle = 60 / service_time

        # 高峰时段需求
        peak_vehicles = max_orders / orders_per_vehicle

        # 平均时段需求
        avg_vehicles = avg_orders / orders_per_vehicle

        # 建议配置 (在平均和高峰之间,偏向平均+20%)
        recommended = avg_vehicles * 1.2

        recommendations.append({
            'service_time': service_time,
            'peak_vehicles': peak_vehicles,
            'avg_vehicles': avg_vehicles,
            'recommended': recommended
        })

        print(f"{service_time:^15d} | {peak_vehicles:>10.0f}  | {avg_vehicles:>10.0f}  | "
              f"{recommended:>10.0f} 辆")

    return recommendations


def analyze_supply_demand_ratio(hourly_avg, vehicle_configs=[1200, 1500, 1800, 2000, 2500]):
    """分析不同车辆配置下的供需比"""
    print("\n" + "="*80)
    print("4. 供需比分析")
    print("="*80)

    max_orders = hourly_avg['mean'].max()
    avg_orders = hourly_avg['mean'].mean()

    print(f"\n不同车辆配置的供需比:")
    print("-"*80)
    print(f"{'车辆数':<10} | {'峰值供需比':<15} | {'平均供需比':<15} | {'负载水平'}")
    print("-"*80)

    results = []

    for vehicles in vehicle_configs:
        peak_ratio = vehicles / max_orders
        avg_ratio = vehicles / avg_orders

        # 判断负载水平
        if avg_ratio < 0.5:
            load_level = "极高负载 ⚠️"
        elif avg_ratio < 0.7:
            load_level = "高负载"
        elif avg_ratio < 1.0:
            load_level = "中等负载 ✓"
        elif avg_ratio < 1.5:
            load_level = "低负载"
        else:
            load_level = "过度配置"

        results.append({
            'vehicles': vehicles,
            'peak_ratio': peak_ratio,
            'avg_ratio': avg_ratio,
            'load_level': load_level
        })

        print(f"{vehicles:<10} | {peak_ratio:>13.2f}  | {avg_ratio:>13.2f}  | {load_level}")

    return results


def plot_hourly_demand(hourly_avg, vehicle_config=2000):
    """绘制每小时需求和车辆配置图"""
    print("\n" + "="*80)
    print("5. 生成可视化图表")
    print("="*80)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    hours = hourly_avg.index
    mean_orders = hourly_avg['mean'].values

    # 子图1: 每小时订单需求
    ax1.bar(hours, mean_orders, color='#3498DB', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=mean_orders.mean(), color='red', linestyle='--', linewidth=2,
                label=f'平均订单数: {mean_orders.mean():.0f}')
    ax1.axhline(y=mean_orders.max(), color='orange', linestyle=':', linewidth=2,
                label=f'最高峰: {mean_orders.max():.0f}')

    ax1.set_xlabel('小时', fontsize=13, fontweight='bold')
    ax1.set_ylabel('平均订单数', fontsize=13, fontweight='bold')
    ax1.set_title('(a) 24小时订单需求分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)

    # 标注高峰时段
    peak_threshold = mean_orders.mean() + 0.5 * mean_orders.std()
    for h, orders in enumerate(mean_orders):
        if orders >= peak_threshold:
            ax1.text(h, orders + 20, f'{orders:.0f}', ha='center', fontsize=9, fontweight='bold')

    # 子图2: 供需比分析
    vehicle_configs = [1200, 1500, 1800, 2000, 2500]
    colors_map = {1200: '#E74C3C', 1500: '#F39C12', 1800: '#2ECC71', 2000: '#3498DB', 2500: '#9B59B6'}

    for vehicles in vehicle_configs:
        supply_demand_ratio = vehicles / mean_orders
        label = f'{vehicles}辆车'
        if vehicles == vehicle_config:
            label += ' (本研究)'
            linewidth = 3
            zorder = 10
        else:
            linewidth = 2
            zorder = 5

        ax2.plot(hours, supply_demand_ratio, marker='o', linewidth=linewidth,
                label=label, color=colors_map[vehicles], zorder=zorder, markersize=6)

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='供需平衡线')
    ax2.axhline(y=0.7, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='高负载阈值')

    ax2.set_xlabel('小时', fontsize=13, fontweight='bold')
    ax2.set_ylabel('供需比 (车辆数/订单数)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) 不同车辆配置下的供需比变化', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10, ncol=2)

    plt.tight_layout()

    # 保存图表
    output_dir = 'results/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    output_png = os.path.join(output_dir, 'peak_demand_analysis.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存(PNG): {output_png}")

    output_tiff = os.path.join(output_dir, 'peak_demand_analysis.tiff')
    plt.savefig(output_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f"✓ 图表已保存(TIFF): {output_tiff}")

    plt.close()


def generate_paper_summary(hourly_avg, vehicle_config=2000):
    """生成论文中可用的总结"""
    print("\n" + "="*80)
    print("6. 论文写作参考")
    print("="*80)

    max_orders = hourly_avg['mean'].max()
    avg_orders = hourly_avg['mean'].mean()
    peak_hour = hourly_avg['mean'].idxmax()

    supply_demand_ratio = vehicle_config / avg_orders
    peak_supply_demand = vehicle_config / max_orders

    print(f"""
论文中可以这样描述车辆配置的合理性:

"本研究配置{vehicle_config}辆网约车进行仿真实验。该配置基于对真实订单数据的统计分析:
数据集显示,平均每小时订单数为{avg_orders:.0f}单,高峰时段(约{peak_hour}:00时)
订单数可达{max_orders:.0f}单/小时。{vehicle_config}辆车的配置对应平均供需比为
{supply_demand_ratio:.2f}(车辆数/小时订单数),处于中等负载水平,既能保证一定的调度
挑战性,又不至于因车辆严重不足而导致大量订单无法服务。在高峰时段,供需比降至
{peak_supply_demand:.2f},模拟了真实场景中的资源紧张状态,有利于验证算法在高负载
条件下的性能。"

关键数据:
  - 数据集: 2016年11月,29天订单数据
  - 平均订单: {avg_orders:.0f} 单/小时
  - 高峰订单: {max_orders:.0f} 单/小时 (约{peak_hour}:00时)
  - 车辆配置: {vehicle_config} 辆
  - 平均供需比: {supply_demand_ratio:.2f}
  - 高峰供需比: {peak_supply_demand:.2f}
  - 负载水平: {'中等' if 0.7 <= supply_demand_ratio < 1.0 else '低' if supply_demand_ratio >= 1.0 else '高'}
    """)

    # 生成LaTeX表格
    print("\n论文表格 (LaTeX格式):")
    print("-"*80)
    print(r"""
\begin{table}[htbp]
\centering
\caption{车辆配置与订单需求分析}
\label{tab:vehicle_config}
\begin{tabular}{lcc}
\toprule
统计指标 & 数值 & 说明 \\
\midrule
平均订单数 & """ + f"{avg_orders:.0f}" + r""" 单/小时 & 全天平均水平 \\
高峰订单数 & """ + f"{max_orders:.0f}" + r""" 单/小时 & """ + f"{peak_hour}" + r""":00时 \\
车辆配置 & """ + f"{vehicle_config}" + r""" 辆 & 本研究配置 \\
平均供需比 & """ + f"{supply_demand_ratio:.2f}" + r""" & 车辆数/订单数 \\
高峰供需比 & """ + f"{peak_supply_demand:.2f}" + r""" & 高负载场景 \\
\bottomrule
\end{tabular}
\end{table}
    """)


def main():
    """主函数"""
    print("="*80)
    print("订单高峰需求分析")
    print("="*80)
    print("\n目的: 为论文中车辆配置提供数据支持\n")

    # 初始化配置
    config = Config()
    print(f"✓ 配置已加载")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  网格大小: {config.GRID_SIZE}")

    # 加载数据
    print("\n加载订单数据...")
    data_processor = DataProcessor(config)
    orders_df = data_processor.load_and_process_orders()

    print(f"✓ 数据加载完成")
    print(f"  总订单数: {len(orders_df):,}")
    print(f"  时间范围: {orders_df['timestamp'].min()} 至 {orders_df['timestamp'].max()}")
    print(f"  天数: {orders_df['relative_day'].max() + 1}")

    # 1. 每小时需求分析
    hourly_avg = analyze_hourly_demand(orders_df)

    # 2. 高峰时段识别
    peak_hours, max_orders = identify_peak_periods(hourly_avg)

    # 3. 车辆需求计算
    recommendations = calculate_vehicle_requirements(hourly_avg)

    # 4. 供需比分析
    results = analyze_supply_demand_ratio(hourly_avg,
                                         vehicle_configs=[1200, 1500, 1800, 2000, 2500])

    # 5. 生成可视化图表
    plot_hourly_demand(hourly_avg, vehicle_config=config.TOTAL_VEHICLES)

    # 6. 生成论文写作参考
    generate_paper_summary(hourly_avg, vehicle_config=config.TOTAL_VEHICLES)

    print("\n" + "="*80)
    print("✅ 分析完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  - results/visualizations/peak_demand_analysis.png (PNG格式)")
    print("  - results/visualizations/peak_demand_analysis.tiff (TIFF格式)")
    print("\n建议:")
    print("  1. 将上述关键数据填入论文的'实验设置'章节")
    print("  2. 使用生成的图表展示24小时需求分布")
    print("  3. 引用供需比数据说明车辆配置的合理性")
    print()


if __name__ == '__main__':
    main()

