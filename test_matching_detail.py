#!/usr/bin/env python
"""详细测试baseline环境的匹配逻辑"""
import sys
sys.path.insert(0, '/Users/qiukuipeng/PycharmProjects/work2')

from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment
import pandas as pd

config = Config()
data_processor = DataProcessor(config)
all_orders = data_processor.load_and_process_orders()
test_orders = all_orders.tail(5000)

print('=== 创建Baseline环境 ===')
env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='random_walk')

print('\n=== 重置环境 ===')
env.reset()

# 检查初始状态
total_vehicles = len(env.vehicle_manager.vehicles)
idle_count = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'idle')
serving_count = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'serving')

print(f'总车辆数: {total_vehicles}')
print(f'idle车辆: {idle_count}')
print(f'serving车辆（冷启动）: {serving_count}')

# 运行100个tick，详细观察匹配过程
print('\n=== 运行100个Tick，观察匹配详情 ===')
match_details = []

for tick in range(100):
    _, _, _, info = env.step()
    step_info = info.get('step_info', {})

    # 统计车辆状态
    idle_v = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'idle')
    serving_v = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'serving')
    dispatching_v = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'dispatching')

    new_orders = step_info.get('new_orders', 0)
    matched = step_info.get('matched_orders', 0)
    cancelled = step_info.get('cancelled_orders', 0)
    pending = len(env.pending_orders)

    match_details.append({
        'tick': tick + 1,
        'new_orders': new_orders,
        'matched': matched,
        'cancelled': cancelled,
        'pending': pending,
        'idle': idle_v,
        'serving': serving_v,
        'dispatching': dispatching_v
    })

    if tick < 20 or (tick + 1) % 10 == 0:
        print(f"Tick {tick+1:3d}: 新={new_orders:2d} 配={matched:2d} 消={cancelled:2d} "
              f"待={pending:4d} | 闲={idle_v:4d} 服={serving_v:4d} 派={dispatching_v:4d}")

# 汇总统计
df = pd.DataFrame(match_details)
total_new = df['new_orders'].sum()
total_matched = df['matched'].sum()
total_cancelled = df['cancelled'].sum()
total_pending_end = df['pending'].iloc[-1]

print('\n=== 汇总统计（前100个Tick）===')
print(f'总新增订单: {total_new}')
print(f'总匹配订单: {total_matched}')
print(f'总取消订单: {total_cancelled}')
print(f'最终pending: {total_pending_end}')
print(f'匹配率: {total_matched/total_new*100 if total_new > 0 else 0:.1f}%')
print(f'取消率: {total_cancelled/total_new*100 if total_new > 0 else 0:.1f}%')

print(f'\n最终车辆状态分布:')
print(f'  idle: {df["idle"].iloc[-1]}')
print(f'  serving: {df["serving"].iloc[-1]}')
print(f'  dispatching: {df["dispatching"].iloc[-1]}')

