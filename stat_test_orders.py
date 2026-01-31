#!/usr/bin/env python3
"""
统计测试集（最后7天）每天的订单数量
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils.data_process import DataProcessor


def analyze_test_orders():
    """分析测试集每天的订单数量"""

    print("=" * 80)
    print("测试集订单统计分析")
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

    # 4. 按日期分组统计
    print("\n" + "=" * 80)
    print("每日订单统计")
    print("=" * 80)

    # 添加日期列
    test_orders['date'] = test_orders['timestamp'].dt.date

    # 按日期统计
    daily_stats = test_orders.groupby('date').agg({
        'order_id': 'count',
        'fee': ['sum', 'mean']
    }).reset_index()

    daily_stats.columns = ['date', 'order_count', 'total_fee', 'avg_fee']

    # 打印每日统计
    print(f"\n{'日期':<15} {'订单数':>10} {'总收入':>12} {'平均票价':>10}")
    print("-" * 50)

    total_orders = 0
    total_fee = 0.0

    for idx, row in daily_stats.iterrows():
        print(f"{str(row['date']):<15} {row['order_count']:>10,} {row['total_fee']:>12,.2f} {row['avg_fee']:>10,.2f}")
        total_orders += row['order_count']
        total_fee += row['total_fee']

    print("-" * 50)
    print(f"{'总计':<15} {total_orders:>10,} {total_fee:>12,.2f} {total_fee/total_orders:>10,.2f}")

    # 5. 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)

    print(f"\n总天数: {len(daily_stats)} 天")
    print(f"每天订单数:")
    print(f"  - 平均: {daily_stats['order_count'].mean():,.0f} 条")
    print(f"  - 最大: {daily_stats['order_count'].max():,.0f} 条 (日期: {daily_stats.loc[daily_stats['order_count'].idxmax(), 'date']})")
    print(f"  - 最小: {daily_stats['order_count'].min():,.0f} 条 (日期: {daily_stats.loc[daily_stats['order_count'].idxmin(), 'date']})")
    print(f"  - 标准差: {daily_stats['order_count'].std():,.0f} 条")

    # 6. 按星期几统计
    print("\n" + "=" * 80)
    print("按星期几统计")
    print("=" * 80)

    test_orders['weekday'] = test_orders['timestamp'].dt.day_name()
    weekday_stats = test_orders.groupby('weekday').size().reset_index(name='order_count')

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

    print(f"\n{'星期':<10} {'订单数':>10} {'占比':>8}")
    print("-" * 30)

    for day_en, day_cn in zip(weekday_order, weekday_cn):
        if day_en in weekday_stats['weekday'].values:
            count = weekday_stats.loc[weekday_stats['weekday'] == day_en, 'order_count'].values[0]
            percentage = count / total_orders * 100
            print(f"{day_cn:<10} {count:>10,} {percentage:>7.2f}%")

    # 7. 按小时统计
    print("\n" + "=" * 80)
    print("每小时订单分布（所有7天汇总）")
    print("=" * 80)

    test_orders['hour'] = test_orders['timestamp'].dt.hour
    hourly_stats = test_orders.groupby('hour').size().reset_index(name='order_count')

    print(f"\n{'小时':<6} {'订单数':>10} {'平均每天':>10} {'占比':>8}")
    print("-" * 40)

    for hour in range(24):
        if hour in hourly_stats['hour'].values:
            count = hourly_stats.loc[hourly_stats['hour'] == hour, 'order_count'].values[0]
            avg_per_day = count / len(daily_stats)
            percentage = count / total_orders * 100
            print(f"{hour:02d}:00 {count:>10,} {avg_per_day:>10,.0f} {percentage:>7.2f}%")

    # 8. 识别高峰期
    print("\n" + "=" * 80)
    print("高峰期分析")
    print("=" * 80)

    # 找出订单数最多的前5个小时
    top_hours = hourly_stats.nlargest(5, 'order_count')

    print("\n订单量最高的5个小时:")
    for idx, row in top_hours.iterrows():
        count = row['order_count']
        avg_per_day = count / len(daily_stats)
        percentage = count / total_orders * 100
        print(f"  {int(row['hour']):02d}:00-{int(row['hour'])+1:02d}:00: {count:,} 条 (平均每天 {avg_per_day:,.0f} 条, {percentage:.2f}%)")

    # 9. 保存结果
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)

    output_file = 'results/test_orders_stats.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily_stats.to_csv(output_file, index=False)
    print(f"\n每日统计已保存到: {output_file}")

    hourly_file = 'results/test_orders_hourly.csv'
    hourly_stats.to_csv(hourly_file, index=False)
    print(f"小时统计已保存到: {hourly_file}")

    print("\n" + "=" * 80)
    print("统计完成!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        analyze_test_orders()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

