#!/usr/bin/env python3
"""
检查SARSA-SAA是否真的在调度车辆
"""
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment
from baselines.sarsa_saa import SarsaSAABaseline


def get_current_tick_demand(env, simulation_time, tick_duration_sec):
    """获取当前tick的订单需求分布"""
    demand = np.zeros(env.config.NUM_GRIDS)
    for order in env.pending_orders:
        if 'grid_index' in order:
            try:
                grid_idx = int(order['grid_index'])
                if 0 <= grid_idx < env.config.NUM_GRIDS:
                    demand[grid_idx] += 1
            except (ValueError, TypeError):
                pass
    return demand


def test_sarsa_dispatch():
    """测试SARSA-SAA的调度逻辑"""

    print("=" * 80)
    print("SARSA-SAA 调度逻辑测试")
    print("=" * 80)

    # 1. 初始化
    config = Config()
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    _, _, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='none')
    agent = SarsaSAABaseline(config)

    print(f"\n配置:")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  最大调度半径: {agent.max_dispatch_radius}")
    print(f"  历史样本数要求: {agent.sample_size}")

    # 2. 模拟收集历史数据（第一天）
    print("\n" + "=" * 80)
    print("第1天：收集历史数据")
    print("=" * 80)

    env.reset(start_day=0)

    day1_stats = {
        'dispatches': 0,
        'saa_calls': 0,
        'saa_empty_returns': 0,
        'matched': 0,
        'generated': 0
    }

    for tick in range(100):
        current_minutes = (tick * config.TICK_DURATION_SEC) / 60
        saa_time_slot = int(current_minutes // 30)

        # 记录需求
        current_demand = get_current_tick_demand(env, env.simulation_time, config.TICK_DURATION_SEC)
        agent.record_history(saa_time_slot, current_demand)

        # 尝试调度（每10个tick）
        if tick % 10 == 0:
            idle_vehicles_list = [0] * config.NUM_GRIDS
            for v in env.vehicle_manager.vehicles.values():
                if v['status'] == 'idle':
                    idle_vehicles_list[v['current_grid']] += 1

            day1_stats['saa_calls'] += 1
            dispatch_instructions, est_value = agent.solve_saa_dispatch(idle_vehicles_list, saa_time_slot)

            if not dispatch_instructions:
                day1_stats['saa_empty_returns'] += 1
            else:
                # 执行调度
                for src, targets in dispatch_instructions.items():
                    candidates = [
                        vid for vid, v in env.vehicle_manager.vehicles.items()
                        if v['current_grid'] == src and v['status'] == 'idle'
                    ]
                    for dst, count in targets.items():
                        for i in range(min(count, len(candidates))):
                            if i < len(candidates):
                                env.vehicle_manager.start_dispatching(
                                    candidates[i], dst, env.simulation_time
                                )
                                day1_stats['dispatches'] += 1

        # 环境步进
        _, _, _, info = env.step()
        step_info = info.get('step_info', {})
        day1_stats['matched'] += step_info.get('matched_orders', 0)
        day1_stats['generated'] += step_info.get('new_orders', 0)

    match_rate_day1 = day1_stats['matched'] / day1_stats['generated'] if day1_stats['generated'] > 0 else 0

    print(f"\n前100个tick统计:")
    print(f"  SAA调用次数: {day1_stats['saa_calls']}")
    print(f"  SAA返回空调度: {day1_stats['saa_empty_returns']} (缺少历史数据)")
    print(f"  实际调度车辆数: {day1_stats['dispatches']}")
    print(f"  匹配订单: {day1_stats['matched']}")
    print(f"  生成订单: {day1_stats['generated']}")
    print(f"  匹配率: {match_rate_day1:.2%}")

    # 检查历史样本
    print(f"\n历史样本统计:")
    for time_slot in range(5):  # 检查前5个时间槽
        samples = agent.history_samples.get(time_slot, [])
        print(f"  Time Slot {time_slot}: {len(samples)} 天数据")

    # 3. 第2-4天：继续收集
    print("\n" + "=" * 80)
    print("第2-4天：继续收集历史数据")
    print("=" * 80)

    for day in range(1, 4):
        env.reset(start_day=day)
        for tick in range(100):
            current_minutes = (tick * config.TICK_DURATION_SEC) / 60
            saa_time_slot = int(current_minutes // 30)
            current_demand = get_current_tick_demand(env, env.simulation_time, config.TICK_DURATION_SEC)
            agent.record_history(saa_time_slot, current_demand)
            env.step()

    print(f"\n历史样本更新:")
    for time_slot in range(5):
        samples = agent.history_samples.get(time_slot, [])
        print(f"  Time Slot {time_slot}: {len(samples)} 天数据")

    # 4. 第5天：有足够历史数据后测试
    print("\n" + "=" * 80)
    print("第5天：有足够历史数据后测试")
    print("=" * 80)

    env.reset(start_day=4)

    day5_stats = {
        'dispatches': 0,
        'saa_calls': 0,
        'saa_empty_returns': 0,
        'matched': 0,
        'generated': 0,
        'dispatch_details': []
    }

    for tick in range(100):
        current_minutes = (tick * config.TICK_DURATION_SEC) / 60
        saa_time_slot = int(current_minutes // 30)

        current_demand = get_current_tick_demand(env, env.simulation_time, config.TICK_DURATION_SEC)
        agent.record_history(saa_time_slot, current_demand)

        if tick % 10 == 0:
            idle_vehicles_list = [0] * config.NUM_GRIDS
            for v in env.vehicle_manager.vehicles.values():
                if v['status'] == 'idle':
                    idle_vehicles_list[v['current_grid']] += 1

            total_idle = sum(idle_vehicles_list)
            total_demand = sum(current_demand)

            day5_stats['saa_calls'] += 1
            dispatch_instructions, est_value = agent.solve_saa_dispatch(idle_vehicles_list, saa_time_slot)

            if not dispatch_instructions:
                day5_stats['saa_empty_returns'] += 1
            else:
                dispatch_count = 0
                for src, targets in dispatch_instructions.items():
                    candidates = [
                        vid for vid, v in env.vehicle_manager.vehicles.items()
                        if v['current_grid'] == src and v['status'] == 'idle'
                    ]
                    for dst, count in targets.items():
                        for i in range(min(count, len(candidates))):
                            if i < len(candidates):
                                env.vehicle_manager.start_dispatching(
                                    candidates[i], dst, env.simulation_time
                                )
                                dispatch_count += 1
                day5_stats['dispatches'] += dispatch_count
                day5_stats['dispatch_details'].append({
                    'tick': tick,
                    'time_slot': saa_time_slot,
                    'idle_vehicles': total_idle,
                    'demand': total_demand,
                    'dispatched': dispatch_count,
                    'est_value': est_value
                })

        _, _, _, info = env.step()
        step_info = info.get('step_info', {})
        day5_stats['matched'] += step_info.get('matched_orders', 0)
        day5_stats['generated'] += step_info.get('new_orders', 0)

    match_rate_day5 = day5_stats['matched'] / day5_stats['generated'] if day5_stats['generated'] > 0 else 0

    print(f"\n前100个tick统计:")
    print(f"  SAA调用次数: {day5_stats['saa_calls']}")
    print(f"  SAA返回空调度: {day5_stats['saa_empty_returns']}")
    print(f"  实际调度车辆数: {day5_stats['dispatches']}")
    print(f"  匹配订单: {day5_stats['matched']}")
    print(f"  生成订单: {day5_stats['generated']}")
    print(f"  匹配率: {match_rate_day5:.2%}")

    if day5_stats['dispatch_details']:
        print(f"\n调度详情（前3次）:")
        for detail in day5_stats['dispatch_details'][:3]:
            print(f"  Tick {detail['tick']} (Slot {detail['time_slot']}):")
            print(f"    空闲车辆: {detail['idle_vehicles']}, 需求: {detail['demand']:.0f}")
            print(f"    调度数: {detail['dispatched']}, 预估价值: {detail['est_value']:.2f}")

    # 5. 诊断结论
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)

    if day1_stats['saa_empty_returns'] == day1_stats['saa_calls']:
        print("\n✓ 第1天：SAA正确地因缺少历史数据而不调度")

    if day5_stats['saa_empty_returns'] > day5_stats['saa_calls'] * 0.8:
        print("\n❌ 问题：即使有历史数据，SAA仍然很少调度")
        print("   可能原因：")
        print("   1. SAA优化目标函数有问题（收益-成本可能为负）")
        print("   2. 约束太严格（如max_dispatch_radius=10太小）")
        print("   3. 需求预测不准确")
    elif day5_stats['dispatches'] < 50:
        print("\n⚠️  问题：调度数量太少")
        print(f"   平均每次调度: {day5_stats['dispatches']/day5_stats['saa_calls']:.1f} 辆")
        print("   建议：增加调度激进程度")
    else:
        print("\n✓ SAA调度正常")
        print(f"   平均每次调度: {day5_stats['dispatches']/day5_stats['saa_calls']:.1f} 辆")

    if match_rate_day5 < 0.5:
        print("\n❌ 匹配率仍然很低")
        print("   说明调度策略无效，问题可能在：")
        print("   1. 调度方向错误（调到错误的网格）")
        print("   2. 调度频率太低（10分钟一次不够）")
        print("   3. 匹配半径太小")
    elif match_rate_day5 < 0.7:
        print("\n⚠️  匹配率偏低但有改善")
        print(f"   Day1: {match_rate_day1:.2%} → Day5: {match_rate_day5:.2%}")
        print("   建议：增加调度频率或扩大调度半径")
    else:
        print("\n✓ 匹配率正常")
        print(f"   Day1: {match_rate_day1:.2%} → Day5: {match_rate_day5:.2%}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        test_sarsa_dispatch()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

