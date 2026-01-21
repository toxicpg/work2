"""
训练数据诊断脚本
用于诊断训练数据量、分布和配置是否合理
"""

import os
# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append('.')

from config import Config
from utils.data_process import DataProcessor

config = Config()
data_processor = DataProcessor(config)

# 加载数据
all_orders = data_processor.load_and_process_orders()
train_orders, val_orders, test_orders = data_processor.split_data_by_time(
    all_orders, config.TRAIN_RATIO, config.VAL_RATIO
)

print("\n" + "="*70)
print("数据诊断报告")
print("="*70)

# 训练集分析
train_dates = train_orders['timestamp'].dt.date.unique()
print(f"\n训练集:")
print(f"  总订单数: {len(train_orders):,}")
print(f"  天数: {len(train_dates)}")
print(f"  日期范围: {train_dates.min()} 到 {train_dates.max()}")
print(f"  平均每天: {len(train_orders) / len(train_dates):.0f} 订单")

# 每天订单数
daily_counts = train_orders.groupby(train_orders['timestamp'].dt.date).size()
print(f"  最少的一天: {daily_counts.min()} 订单")
print(f"  最多的一天: {daily_counts.max()} 订单")

# 验证集分析
val_dates = val_orders['timestamp'].dt.date.unique()
print(f"\n验证集:")
print(f"  总订单数: {len(val_orders):,}")
print(f"  天数: {len(val_dates)}")
print(f"  平均每天: {len(val_orders) / len(val_dates):.0f} 订单")

# 测试集分析
test_dates = test_orders['timestamp'].dt.date.unique()
print(f"\n测试集:")
print(f"  总订单数: {len(test_orders):,}")
print(f"  天数: {len(test_dates)}")
print(f"  平均每天: {len(test_orders) / len(test_dates):.0f} 订单")

# Episode 配置
print(f"\n训练配置:")
print(f"  EPISODE_DAYS: {config.EPISODE_DAYS}")
print(f"  每个Episode订单数: ~{(len(train_orders) / len(train_dates)) * config.EPISODE_DAYS:.0f}")
print(f"  总车辆数: {config.TOTAL_VEHICLES}")
print(f"  车辆/订单比: {config.TOTAL_VEHICLES / ((len(train_orders) / len(train_dates)) * config.EPISODE_DAYS):.2f}")

# 潜在问题检测
print(f"\n" + "="*70)
print("潜在问题检测")
print("="*70)

issues = []

# 检查1: 训练集是否太小
if len(train_dates) < 10:
    issues.append(f"⚠️  训练集天数太少 ({len(train_dates)}天)，建议至少10天")

# 检查2: 每天订单数是否太少
avg_daily_orders = len(train_orders) / len(train_dates)
if avg_daily_orders < 5000:
    issues.append(f"⚠️  平均每天订单数太少 ({avg_daily_orders:.0f}条)，可能导致学习困难")

# 检查3: 车辆订单比是否合理
vehicle_order_ratio = config.TOTAL_VEHICLES / (avg_daily_orders * config.EPISODE_DAYS)
if vehicle_order_ratio > 0.5:
    issues.append(f"⚠️  车辆/订单比过高 ({vehicle_order_ratio:.2f})，车辆太多，订单太少")
elif vehicle_order_ratio < 0.05:
    issues.append(f"⚠️  车辆/订单比过低 ({vehicle_order_ratio:.2f})，车辆太少，可能导致低匹配率")

# 检查4: 验证集是否太小
if len(val_dates) < 2:
    issues.append(f"⚠️  验证集天数太少 ({len(val_dates)}天)，验证结果可能不稳定")

# 检查5: 数据分布是否均匀
daily_std = daily_counts.std()
daily_mean = daily_counts.mean()
if daily_std / daily_mean > 0.5:
    issues.append(f"⚠️  每天订单数波动较大 (标准差/均值 = {daily_std/daily_mean:.2f})，可能影响训练稳定性")

if issues:
    print("\n发现以下问题:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ 未发现明显问题")

print("\n" + "="*70)
print("建议:")
print("="*70)
print("  1. 如果训练效果不好，检查上述问题")
print("  2. 查看训练日志中的 Reward 和 Loss 曲线")
print("  3. 确认验证集匹配率是否有提升")
print("  4. 考虑调整 EPISODE_DAYS 或 TOTAL_VEHICLES")
print("="*70 + "\n")

