"""
按用户指出的三点诊断匹配率问题：
1. 订单是否在对应的时间片内产生？
2. 匹配是否正确？
3. 订单完成后，车辆位置移动了吗？标记为空闲了吗？时间对不对？数据记录对不对？
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import pandas as pd
from config import Config
from environment_baseline import BaselineEnvironment
from utils.data_process import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def debug_three_points():
    config = Config()
    set_seed(config.SEED)

    print("=" * 80)
    print("三点诊断：订单生成、匹配、完成")
    print("=" * 80)

    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]

    print(f"\n测试日期: {test_day}")
    print(f"该日总订单数: {len(day_orders)}")

    env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='random_walk')
    env.reset()

    print(f"\n初始状态:")
    print(f"  开始时间: {env.current_time}")
    print(f"  当前天: {env.current_day}, 时间片: {env.current_time_slice}")
    print(f"  车辆总数: {config.TOTAL_VEHICLES}")
    stats = env.vehicle_manager.get_statistics()
    print(f"  空闲车辆: {stats.get('idle', 0)}")

    # 运行前几步，详细观察
    print("\n" + "=" * 80)
    print("详细执行前20步")
    print("=" * 80)

    for step in range(20):
        print(f"\n{'='*80}")
        print(f"Step {step}: 时间 {env.current_time}, Day={env.current_day}, Slice={env.current_time_slice}")
        print(f"{'='*80}")

        # 记录step前的状态
        before_stats = env.vehicle_manager.get_statistics()
        before_pending = len(env.pending_orders)

        # 【检查点1】查看这个时间片应该生成多少订单
        expected_orders = env.order_generator._load_orders_for_macro_step(
            env.current_day, env.current_time_slice
        )
        print(f"【点1-订单生成】该时间片应生成订单: {len(expected_orders)}")
        if len(expected_orders) > 0:
            print(f"  第一个订单: grid={expected_orders[0].get('grid_index')}, "
                  f"timestamp={expected_orders[0].get('timestamp')}")

        # 执行step
        _, _, done, info = env.step()
        step_info = info.get('step_info', {})

        # 【检查点1-验证】实际生成了多少
        actual_new = step_info.get('new_orders', 0)
        print(f"  实际生成订单: {actual_new}")
        if actual_new != len(expected_orders):
            print(f"  ⚠️ 警告：预期{len(expected_orders)}个，实际{actual_new}个")

        # 【检查点2】匹配情况
        matched = step_info.get('matched_orders', 0)
        after_pending = len(env.pending_orders)
        cancelled = step_info.get('cancelled_orders', 0)
        after_stats = env.vehicle_manager.get_statistics()

        print(f"\n【点2-匹配】")
        print(f"  Pending订单: {before_pending} -> {after_pending}")
        print(f"  本步匹配: {matched}个")
        print(f"  本步取消: {cancelled}个")
        print(f"  车辆状态变化:")
        print(f"    idle: {before_stats.get('idle', 0)} -> {after_stats.get('idle', 0)}")
        print(f"    serving: {before_stats.get('serving', 0)} -> {after_stats.get('serving', 0)}")
        print(f"    dispatching: {before_stats.get('dispatching', 0)} -> {after_stats.get('dispatching', 0)}")

        # 验证匹配逻辑
        idle_decreased = before_stats.get('idle', 0) - after_stats.get('idle', 0)
        serving_increased = after_stats.get('serving', 0) - before_stats.get('serving', 0)
        if matched > 0:
            if idle_decreased != matched:
                print(f"  ⚠️ 警告：匹配了{matched}个订单，但idle只减少了{idle_decreased}")
            if serving_increased != matched:
                print(f"  ⚠️ 警告：匹配了{matched}个订单，但serving只增加了{serving_increased}")

        # 【检查点3】检查服务中的车辆，看是否正确完成
        print(f"\n【点3-订单完成】")
        completed = step_info.get('completed_orders', 0)
        print(f"  本步完成: {completed}个")

        # 抽查几个serving车辆的状态
        serving_vehicles = [(vid, v) for vid, v in env.vehicle_manager.vehicles.items()
                           if v.get('status') == 'serving']
        if len(serving_vehicles) > 0:
            print(f"  当前serving车辆数: {len(serving_vehicles)}")
            # 检查前3个
            for i, (vid, v) in enumerate(serving_vehicles[:3]):
                order = v.get('assigned_order')
                start_time = v.get('order_start_time')
                total_time = v.get('total_completion_time', 0)
                current_grid = v.get('current_grid')

                if start_time and order:
                    elapsed = (env.current_time - start_time).total_seconds() / 60.0
                    dest_grid = order.get('dest_grid_index', 'N/A')
                    print(f"    车辆{vid}: 已服务{elapsed:.1f}min/{total_time:.1f}min, "
                          f"当前grid={current_grid}, 目标grid={dest_grid}")

        # 统计信息
        print(f"\n累计统计:")
        print(f"  总生成: {env.episode_stats['total_orders_generated']}")
        print(f"  总匹配: {env.episode_stats['total_orders_matched']}")
        print(f"  总取消: {env.episode_stats['total_orders_cancelled']}")
        rate = (env.episode_stats['total_orders_matched'] /
                env.episode_stats['total_orders_generated']
                if env.episode_stats['total_orders_generated'] > 0 else 0)
        print(f"  匹配率: {rate:.2%}")

        if done:
            break

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    debug_three_points()

