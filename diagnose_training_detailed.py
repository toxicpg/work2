"""
详细训练数据诊断脚本 - 按小时分析
用于诊断同时在线订单数和车辆配置
"""

import os
# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append('.')

from config import Config
from utils.data_process import DataProcessor
import pandas as pd
import numpy as np

config = Config()
data_processor = DataProcessor(config)

# 加载数据
all_orders = data_processor.load_and_process_orders()
train_orders, val_orders, test_orders = data_processor.split_data_by_time(
    all_orders, config.TRAIN_RATIO, config.VAL_RATIO
)

print("\n" + "="*70)
print("详细数据诊断报告 - 按小时分析")
print("="*70)

# 分析每小时订单数
train_orders['hour'] = train_orders['timestamp'].dt.hour
hourly_counts = train_orders.groupby('hour').size()

print(f"\n每小时平均订单数分布:")
print(f"  最少的小时: {hourly_counts.min()} 订单 (小时 {hourly_counts.idxmin()})")
print(f"  最多的小时: {hourly_counts.max()} 订单 (小时 {hourly_counts.idxmax()})")
print(f"  平均每小时: {hourly_counts.mean():.0f} 订单")

# 打印每小时详细分布
print(f"\n每小时订单数详细:")
for hour in range(24):
    count = hourly_counts.get(hour, 0)
    bar = '█' * int(count / 5000)
    print(f"  {hour:02d}:00 - {count:6d} 订单 {bar}")

# 估算同时在线订单数
# 假设平均订单时长（从下单到完成）
AVG_ORDER_DURATION_MINUTES = 15  # 假设平均15分钟完成一单

print(f"\n" + "="*70)
print(f"并发订单估算 (假设平均订单时长 {AVG_ORDER_DURATION_MINUTES} 分钟)")
print("="*70)

# 计算峰值小时的并发订单数
peak_hour = hourly_counts.idxmax()
peak_hourly_orders = hourly_counts.max()
# 并发数 = 每小时订单数 * (订单时长/60分钟)
peak_concurrent_orders = peak_hourly_orders * (AVG_ORDER_DURATION_MINUTES / 60)

avg_hourly_orders = hourly_counts.mean()
avg_concurrent_orders = avg_hourly_orders * (AVG_ORDER_DURATION_MINUTES / 60)

print(f"\n峰值时段 ({peak_hour}:00):")
print(f"  每小时订单数: {peak_hourly_orders:,.0f}")
print(f"  估算并发订单数: {peak_concurrent_orders:,.0f}")
print(f"  当前车辆数: {config.TOTAL_VEHICLES:,}")
print(f"  车辆/并发订单比: {config.TOTAL_VEHICLES / peak_concurrent_orders:.2f}")

print(f"\n平均时段:")
print(f"  每小时订单数: {avg_hourly_orders:,.0f}")
print(f"  估算并发订单数: {avg_concurrent_orders:,.0f}")
print(f"  当前车辆数: {config.TOTAL_VEHICLES:,}")
print(f"  车辆/并发订单比: {config.TOTAL_VEHICLES / avg_concurrent_orders:.2f}")

# 不同订单时长的场景
print(f"\n不同订单时长场景下的车辆需求:")
for duration in [10, 15, 20, 30]:
    concurrent = peak_hourly_orders * (duration / 60)
    needed_vehicles = concurrent * 1.2  # 建议车辆数 = 并发数 * 1.2 (考虑调度效率)
    print(f"  订单时长 {duration}分钟: 并发{concurrent:,.0f}, 建议车辆数{needed_vehicles:,.0f}")

# 车辆配置建议
print(f"\n" + "="*70)
print("车辆配置建议")
print("="*70)

recommended_min = peak_concurrent_orders * 0.8  # 最少80%覆盖
recommended_ideal = peak_concurrent_orders * 1.2  # 理想120%覆盖
recommended_max = peak_concurrent_orders * 1.5  # 最多150%覆盖

print(f"\n基于峰值时段 ({peak_hour}:00):")
print(f"  最少车辆 (80%覆盖): {recommended_min:,.0f}")
print(f"  理想车辆 (120%覆盖): {recommended_ideal:,.0f}")
print(f"  最多车辆 (150%覆盖): {recommended_max:,.0f}")
print(f"\n  当前配置: {config.TOTAL_VEHICLES:,}")

if config.TOTAL_VEHICLES < recommended_min:
    print(f"  ❌ 严重不足！建议至少增加到 {recommended_min:,.0f} 辆")
elif config.TOTAL_VEHICLES < recommended_ideal:
    print(f"  ⚠️  偏少，建议增加到 {recommended_ideal:,.0f} 辆")
elif config.TOTAL_VEHICLES > recommended_max:
    print(f"  ⚠️  过多，建议减少到 {recommended_max:,.0f} 辆")
else:
    print(f"  ✅ 配置合理")

# 实际验证：分析10分钟时间窗口
print(f"\n" + "="*70)
print("10分钟时间窗口分析 (模拟器实际运行场景)")
print("="*70)

train_orders['time_window'] = (train_orders['timestamp'].dt.hour * 60 +
                                train_orders['timestamp'].dt.minute) // 10

window_counts = train_orders.groupby('time_window').size()
print(f"\n每10分钟订单数:")
print(f"  最少: {window_counts.min()} 订单")
print(f"  最多: {window_counts.max()} 订单")
print(f"  平均: {window_counts.mean():.0f} 订单")
print(f"  峰值10分钟: {window_counts.max()} 订单")

# 10分钟窗口内的车辆需求
print(f"\n基于10分钟窗口的车辆需求:")
print(f"  峰值10分钟订单数: {window_counts.max()}")
print(f"  如果这些订单都需要车 (最坏情况): {window_counts.max()} 辆")
print(f"  考虑订单时长15分钟 (重叠1.5x): {window_counts.max() * 1.5:.0f} 辆")
print(f"  当前车辆数: {config.TOTAL_VEHICLES:,}")

if config.TOTAL_VEHICLES < window_counts.max():
    print(f"  ❌ 不足以应对峰值10分钟！")
elif config.TOTAL_VEHICLES < window_counts.max() * 1.5:
    print(f"  ⚠️  勉强够用，但可能峰值时段匹配率低")
else:
    print(f"  ✅ 足够应对峰值")

print("\n" + "="*70)

