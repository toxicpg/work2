#!/usr/bin/env python
"""检查baseline环境的配置"""
import sys
sys.path.insert(0, '/Users/qiukuipeng/PycharmProjects/work2')

from baselines.random_walk import _make_env
from config import Config
from utils.data_process import DataProcessor

config = Config()
print('=== 配置检查 ===')
print(f'DISPATCH_MODE: {getattr(config, "DISPATCH_MODE", "未设置")}')
print(f'MATCHER_KNN_K: {config.MATCHER_KNN_K}')
print(f'MAX_WAITING_TIME: {config.MAX_WAITING_TIME}')
print(f'TOTAL_VEHICLES: {config.TOTAL_VEHICLES}')

# 检查是否正确导入BaselineEnvironment
data_processor = DataProcessor(config)
all_orders = data_processor.load_and_process_orders()
test_orders = all_orders.tail(1000)  # 取最后1000条测试

print('\n=== 创建环境 ===')
env = _make_env(config, test_orders)
print(f'环境类型: {type(env).__name__}')
print(f'环境dispatch_policy: {getattr(env, "dispatch_policy", "未设置")}')
print(f'OrderMatcher类型: {type(env.order_matcher).__name__}')

# 重置环境，查看冷启动
print('\n=== 重置环境（检查冷启动）===')
env.reset()
idle_count = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'idle')
serving_count = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'serving')
print(f'idle车辆数: {idle_count}')
print(f'serving车辆数（冷启动）: {serving_count}')
print(f'冷启动调度表大小: {len(getattr(env, "_warmup_schedule", {}))}')

# 运行几个tick看匹配情况
print('\n=== 运行10个Tick观察匹配 ===')
for tick in range(10):
    _, _, _, info = env.step()
    step_info = info.get('step_info', {})
    pending = len(env.pending_orders)
    idle_vehicles = sum(1 for v in env.vehicle_manager.vehicles.values() if v['status'] == 'idle')

    print(f"Tick {tick+1}: "
          f"新订单={step_info.get('new_orders', 0)}, "
          f"匹配={step_info.get('matched_orders', 0)}, "
          f"取消={step_info.get('cancelled_orders', 0)}, "
          f"pending={pending}, "
          f"idle车={idle_vehicles}")

print('\n完成检查！')

