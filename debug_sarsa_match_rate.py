#!/usr/bin/env python3
"""
诊断SARSA-SAA匹配率低的问题
"""
import sys
import os
import numpy as np
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment


def debug_match_rate():
    """诊断匹配率问题"""

    print("=" * 80)
    print("SARSA-SAA 匹配率诊断")
    print("=" * 80)

    # 1. 加载配置和数据
    config = Config()
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    _, _, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    print(f"\n测试集订单数: {len(test_orders):,}")
    print(f"车辆数: {config.TOTAL_VEHICLES}")
    print(f"匹配半径: {getattr(config, 'MATCHER_SEARCH_RADIUS', 10)}")

    # 2. 创建环境（不使用任何调度策略，只测试自然匹配）
    print("\n" + "=" * 80)
    print("测试1: 无调度策略（baseline）")
    print("=" * 80)

    env_baseline = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='none')
    env_baseline.reset(start_day=0)

    baseline_stats = {
        'matched': 0,
        'generated': 0,
        'idle_vehicles': [],
        'pending_orders': []
    }

    # 运行100个tick
    for tick in range(100):
        _, _, _, info = env_baseline.step()
        step_info = info.get('step_info', {})

        baseline_stats['matched'] += step_info.get('matched_orders', 0)
        baseline_stats['generated'] += step_info.get('new_orders', 0)
        baseline_stats['idle_vehicles'].append(sum(1 for v in env_baseline.vehicle_manager.vehicles.values() if v['status'] == 'idle'))
        baseline_stats['pending_orders'].append(len(env_baseline.pending_orders))

    baseline_match_rate = baseline_stats['matched'] / baseline_stats['generated'] if baseline_stats['generated'] > 0 else 0

    print(f"\n前100个tick统计:")
    print(f"  生成订单: {baseline_stats['generated']}")
    print(f"  匹配订单: {baseline_stats['matched']}")
    print(f"  匹配率: {baseline_match_rate:.2%}")
    print(f"  平均空闲车辆: {np.mean(baseline_stats['idle_vehicles']):.0f}")
    print(f"  平均待匹配订单: {np.mean(baseline_stats['pending_orders']):.0f}")

    # 3. 测试Random Walk
    print("\n" + "=" * 80)
    print("测试2: Random Walk调度")
    print("=" * 80)

    env_rw = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='random_walk')
    env_rw.reset(start_day=0)

    rw_stats = {
        'matched': 0,
        'generated': 0,
        'dispatches': 0,
        'idle_vehicles': [],
        'pending_orders': []
    }

    for tick in range(100):
        _, _, _, info = env_rw.step()
        step_info = info.get('step_info', {})

        rw_stats['matched'] += step_info.get('matched_orders', 0)
        rw_stats['generated'] += step_info.get('new_orders', 0)
        rw_stats['dispatches'] += step_info.get('dispatch_success', 0)
        rw_stats['idle_vehicles'].append(sum(1 for v in env_rw.vehicle_manager.vehicles.values() if v['status'] == 'idle'))
        rw_stats['pending_orders'].append(len(env_rw.pending_orders))

    rw_match_rate = rw_stats['matched'] / rw_stats['generated'] if rw_stats['generated'] > 0 else 0

    print(f"\n前100个tick统计:")
    print(f"  生成订单: {rw_stats['generated']}")
    print(f"  匹配订单: {rw_stats['matched']}")
    print(f"  匹配率: {rw_match_rate:.2%}")
    print(f"  调度次数: {rw_stats['dispatches']}")
    print(f"  平均空闲车辆: {np.mean(rw_stats['idle_vehicles']):.0f}")
    print(f"  平均待匹配订单: {np.mean(rw_stats['pending_orders']):.0f}")

    # 4. 分析车辆和订单分布
    print("\n" + "=" * 80)
    print("测试3: 车辆和订单分布分析")
    print("=" * 80)

    # 统计车辆分布
    vehicle_dist = defaultdict(int)
    for v in env_baseline.vehicle_manager.vehicles.values():
        vehicle_dist[v['current_grid']] += 1

    # 统计订单分布（前100个tick）
    env_test = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='none')
    env_test.reset(start_day=0)

    order_dist = defaultdict(int)
    for tick in range(100):
        _, _, _, info = env_test.step()
        step_info = info.get('step_info', {})

        # 统计pending orders的分布
        for order in env_test.pending_orders:
            if 'grid_index' in order:
                order_dist[order['grid_index']] += 1

    non_zero_vehicle_grids = sum(1 for count in vehicle_dist.values() if count > 0)
    non_zero_order_grids = sum(1 for count in order_dist.values() if count > 0)

    print(f"\n车辆分布:")
    print(f"  总网格数: {config.NUM_GRIDS}")
    print(f"  有车网格: {non_zero_vehicle_grids}")
    print(f"  覆盖率: {non_zero_vehicle_grids/config.NUM_GRIDS:.2%}")
    print(f"  最大集中: {max(vehicle_dist.values()) if vehicle_dist else 0} 辆")
    print(f"  平均每格: {config.TOTAL_VEHICLES/config.NUM_GRIDS:.1f} 辆")

    print(f"\n订单分布:")
    print(f"  有订单网格: {non_zero_order_grids}")
    print(f"  最大集中: {max(order_dist.values()) if order_dist else 0} 单")

    # 5. 计算车辆-订单距离分布
    print("\n" + "=" * 80)
    print("测试4: 车辆-订单距离分析")
    print("=" * 80)

    env_dist = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='none')
    env_dist.reset(start_day=0)

    # 运行10个tick收集订单
    for tick in range(10):
        env_dist.step()

    # 计算距离
    distances = []
    for order in list(env_dist.pending_orders)[:100]:  # 取前100个订单
        if 'grid_index' not in order:
            continue
        order_grid = order['grid_index']
        order_row, order_col = divmod(order_grid, config.GRID_SIZE[1])

        min_dist = float('inf')
        for v in env_dist.vehicle_manager.vehicles.values():
            if v['status'] == 'idle':
                v_row, v_col = divmod(v['current_grid'], config.GRID_SIZE[1])
                dist = abs(v_row - order_row) + abs(v_col - order_col)
                min_dist = min(min_dist, dist)

        if min_dist < float('inf'):
            distances.append(min_dist)

    if distances:
        print(f"\n订单到最近空闲车辆的距离:")
        print(f"  平均: {np.mean(distances):.2f} 格")
        print(f"  中位数: {np.median(distances):.2f} 格")
        print(f"  最大: {np.max(distances):.0f} 格")
        print(f"  在半径10内: {sum(1 for d in distances if d <= 10) / len(distances):.2%}")
        print(f"  在半径5内: {sum(1 for d in distances if d <= 5) / len(distances):.2%}")

    # 6. 结论和建议
    print("\n" + "=" * 80)
    print("诊断结果和建议")
    print("=" * 80)

    if baseline_match_rate < 0.5:
        print("\n⚠️  问题严重：即使不调度，匹配率也很低")
        print("   可能原因：")
        print("   1. 车辆分布不均（初始分布随机，可能集中在某些区域）")
        print("   2. 匹配半径太小（当前半径10，在20×20网格中可能不够）")
        print("   3. 订单生成速度太快，车辆来不及服务")
        print("\n   建议：")
        print("   - 增大匹配半径到15-20")
        print("   - 优化初始车辆分布（按历史需求分布）")
        print("   - 增加车辆数量")
    elif rw_match_rate - baseline_match_rate > 0.1:
        print("\n✓  Random Walk调度有效提升匹配率")
        print(f"   提升: {(rw_match_rate - baseline_match_rate):.2%}")
        print("\n   SARSA-SAA匹配率低的可能原因：")
        print("   1. 调度频率太低（每10个tick = 10分钟才调度一次）")
        print("   2. 调度半径在探索期被缩小到6")
        print("   3. SAA优化目标不够激进")
        print("\n   建议：")
        print("   - 提高调度频率（改为每5个tick或每分钟）")
        print("   - 探索期不要缩小调度半径")
        print("   - 调整SAA的收益函数，增加匹配权重")
    else:
        print("\n✓  调度策略对匹配率影响不大")
        print("   说明车辆分布已经较为均匀")
        print("\n   SARSA-SAA需要优化：")
        print("   - 检查SAA是否真的在调度车辆")
        print("   - 检查SARSA学习是否收敛")
        print("   - 可能需要更长的训练时间")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        debug_match_rate()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

