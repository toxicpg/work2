#!/usr/bin/env python3
"""
统计高峰期每小时的订单数，计算车辆配置比例
用于确定 1800/2000/2200 车辆分别对应少车/正好/盈余的情况
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils.data_process import DataProcessor


def analyze_vehicle_requirements():
    """分析高峰期车辆需求，计算车辆配置比例"""

    print("=" * 80)
    print("高峰期车辆需求分析")
    print("=" * 80)

    # 1. 初始化配置
    config = Config()

    # 2. 加载数据
    print("\n加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()

    # 3. 划分数据集
    print("划分数据集...")
    _, _, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    print(f"\n测试集总订单数: {len(test_orders):,} 条")
    print(f"时间范围: {test_orders['timestamp'].min()} 到 {test_orders['timestamp'].max()}")

    # 4. 按日期和小时分组
    print("\n" + "=" * 80)
    print("每小时订单统计（按天计算，取平均）")
    print("=" * 80)

    test_orders['date'] = test_orders['timestamp'].dt.date
    test_orders['hour'] = test_orders['timestamp'].dt.hour

    # 统计每天每小时的订单数
    daily_hourly = test_orders.groupby(['date', 'hour']).size().reset_index(name='order_count')

    # 计算每小时的平均订单数
    avg_hourly = daily_hourly.groupby('hour')['order_count'].agg(['mean', 'std', 'max', 'min']).reset_index()
    avg_hourly.columns = ['hour', 'avg_orders', 'std_orders', 'max_orders', 'min_orders']

    # 5. 假设参数
    print("\n" + "=" * 80)
    print("假设参数")
    print("=" * 80)

    # 假设每辆车每小时能完成的订单数（考虑接驾、送达、空闲时间）
    # 平均一单：接驾3分钟 + 送达5分钟 + 找下一单2分钟 = 10分钟
    # 所以一小时最多 60/10 = 6单，但考虑实际情况，取 4-5单
    orders_per_vehicle_per_hour_optimistic = 5.0  # 乐观情况
    orders_per_vehicle_per_hour_realistic = 4.0   # 现实情况
    orders_per_vehicle_per_hour_conservative = 3.0  # 保守情况

    print(f"\n每辆车每小时完成订单数假设:")
    print(f"  - 乐观估计: {orders_per_vehicle_per_hour_optimistic} 单/小时")
    print(f"  - 现实估计: {orders_per_vehicle_per_hour_realistic} 单/小时")
    print(f"  - 保守估计: {orders_per_vehicle_per_hour_conservative} 单/小时")

    # 计算所需车辆数
    avg_hourly['vehicles_needed_optimistic'] = np.ceil(avg_hourly['avg_orders'] / orders_per_vehicle_per_hour_optimistic)
    avg_hourly['vehicles_needed_realistic'] = np.ceil(avg_hourly['avg_orders'] / orders_per_vehicle_per_hour_realistic)
    avg_hourly['vehicles_needed_conservative'] = np.ceil(avg_hourly['avg_orders'] / orders_per_vehicle_per_hour_conservative)

    # 6. 打印每小时统计
    print("\n" + "=" * 80)
    print("每小时订单量与所需车辆数")
    print("=" * 80)

    print(f"\n{'小时':<6} {'平均订单':>10} {'±标准差':>10} {'所需车辆(乐观)':>15} {'所需车辆(现实)':>15} {'所需车辆(保守)':>15}")
    print("-" * 85)

    for idx, row in avg_hourly.iterrows():
        hour = int(row['hour'])
        avg = row['avg_orders']
        std = row['std_orders']
        v_opt = int(row['vehicles_needed_optimistic'])
        v_real = int(row['vehicles_needed_realistic'])
        v_cons = int(row['vehicles_needed_conservative'])
        print(f"{hour:02d}:00 {avg:>10,.1f} ±{std:>8,.1f} {v_opt:>15,} {v_real:>15,} {v_cons:>15,}")

    # 7. 识别高峰期
    print("\n" + "=" * 80)
    print("高峰期分析")
    print("=" * 80)

    # 找出订单量最多的小时（高峰期）
    peak_hours = avg_hourly.nlargest(8, 'avg_orders')

    print(f"\n订单量最高的8个小时（高峰期）:")
    print(f"{'小时':<6} {'平均订单':>10} {'所需车辆(现实)':>15}")
    print("-" * 35)

    for idx, row in peak_hours.iterrows():
        hour = int(row['hour'])
        avg = row['avg_orders']
        v_real = int(row['vehicles_needed_realistic'])
        print(f"{hour:02d}:00 {avg:>10,.1f} {v_real:>15,}")

    # 计算高峰期平均所需车辆
    peak_avg_vehicles_realistic = peak_hours['vehicles_needed_realistic'].mean()
    peak_max_vehicles_realistic = peak_hours['vehicles_needed_realistic'].max()

    print(f"\n高峰期车辆需求:")
    print(f"  - 平均需求: {peak_avg_vehicles_realistic:,.0f} 辆")
    print(f"  - 最大需求: {peak_max_vehicles_realistic:,.0f} 辆")

    # 8. 车辆配置方案分析
    print("\n" + "=" * 80)
    print("车辆配置方案分析")
    print("=" * 80)

    vehicle_configs = [1800, 2000, 2200]

    print(f"\n基于现实估计（每辆车每小时完成{orders_per_vehicle_per_hour_realistic}单）:\n")

    for vehicles in vehicle_configs:
        capacity = vehicles * orders_per_vehicle_per_hour_realistic
        peak_avg_capacity = capacity / peak_avg_vehicles_realistic * 100
        peak_max_capacity = capacity / peak_max_vehicles_realistic * 100

        print(f"配置 {vehicles} 辆车:")
        print(f"  - 每小时总容量: {capacity:,.0f} 单")
        print(f"  - 相对高峰平均需求: {peak_avg_capacity:.1f}% ({'盈余' if peak_avg_capacity > 110 else '正好' if peak_avg_capacity > 90 else '少车'})")
        print(f"  - 相对高峰最大需求: {peak_max_capacity:.1f}% ({'盈余' if peak_max_capacity > 110 else '正好' if peak_max_capacity > 90 else '少车'})")
        print()

    # 9. 更详细的比例分析
    print("=" * 80)
    print("详细比例分析")
    print("=" * 80)

    print(f"\n以高峰期最大需求 ({peak_max_vehicles_realistic:,.0f} 辆) 为基准:\n")

    for vehicles in vehicle_configs:
        ratio = vehicles / peak_max_vehicles_realistic * 100

        if ratio < 90:
            status = "严重不足"
            description = "车辆数远低于高峰需求，预计大量订单无法匹配"
        elif ratio < 95:
            status = "少车"
            description = "车辆数略低于高峰需求，会有部分订单等待或取消"
        elif ratio < 105:
            status = "正好"
            description = "车辆数接近高峰需求，绝大部分订单能及时匹配"
        elif ratio < 115:
            status = "略有盈余"
            description = "车辆数略高于高峰需求，几乎所有订单能快速匹配"
        else:
            status = "充足盈余"
            description = "车辆数显著高于高峰需求，订单匹配非常快速"

        print(f"{vehicles} 辆车:")
        print(f"  - 占比: {ratio:.1f}%")
        print(f"  - 状态: {status}")
        print(f"  - 说明: {description}")
        print()

    # 10. 可视化
    print("=" * 80)
    print("生成可视化图表")
    print("=" * 80)

    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 图1: 每小时订单数
        ax1 = axes[0]
        hours = avg_hourly['hour'].values
        avg_orders = avg_hourly['avg_orders'].values
        std_orders = avg_hourly['std_orders'].values

        ax1.bar(hours, avg_orders, alpha=0.7, color='steelblue', label='Average Orders')
        ax1.errorbar(hours, avg_orders, yerr=std_orders, fmt='none', ecolor='red',
                    capsize=3, alpha=0.5, label='Std Dev')

        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Orders', fontsize=12)
        ax1.set_title('Hourly Order Distribution (Test Set)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(24))
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        # 图2: 所需车辆数对比
        ax2 = axes[1]

        ax2.plot(hours, avg_hourly['vehicles_needed_realistic'],
                marker='o', linewidth=2, markersize=6, label='Realistic (4 orders/vehicle/hour)')
        ax2.plot(hours, avg_hourly['vehicles_needed_optimistic'],
                marker='s', linewidth=2, markersize=5, alpha=0.7, label='Optimistic (5 orders/vehicle/hour)')
        ax2.plot(hours, avg_hourly['vehicles_needed_conservative'],
                marker='^', linewidth=2, markersize=5, alpha=0.7, label='Conservative (3 orders/vehicle/hour)')

        # 标注车辆配置线
        for vehicles in vehicle_configs:
            ax2.axhline(y=vehicles, linestyle='--', alpha=0.5, linewidth=1.5,
                       label=f'{vehicles} vehicles')

        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Required Number of Vehicles', fontsize=12)
        ax2.set_title('Required Vehicles vs Vehicle Configurations', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(24))
        ax2.grid(alpha=0.3)
        ax2.legend(loc='upper right')

        plt.tight_layout()

        output_file = 'results/vehicle_requirement_analysis.png'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n可视化图表已保存到: {output_file}")

    except Exception as e:
        print(f"\n生成可视化失败: {e}")

    # 11. 保存结果
    output_file = 'results/hourly_vehicle_requirements.csv'
    avg_hourly.to_csv(output_file, index=False)
    print(f"统计数据已保存到: {output_file}")

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        analyze_vehicle_requirements()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

