"""
测试车辆状态问题 - 检查随着时间推移idle车辆是否越来越少
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import torch
import pandas as pd
from config import Config
from environment_baseline import BaselineEnvironment
from utils.data_process import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def test_vehicle_states():
    """测试车辆状态随时间的变化"""
    print("=" * 70)
    print("测试车辆状态变化")
    print("=" * 70)

    config = Config()
    set_seed(config.SEED)

    # 加载数据
    print("\n[1] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]

    # 初始化环境
    print("\n[2] 初始化环境...")
    env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='random_walk')
    env.reset()

    print(f"\n[3] 监控车辆状态变化 (前500步)...")
    print(f"{'Step':<8} {'Idle':<8} {'Picking':<10} {'Serving':<10} {'Dispatch':<10} {'匹配率':<10} {'订单积压':<10}")
    print("-" * 70)

    for step in range(500):
        _, _, done, info = env.step()

        if step % 50 == 0:
            stats = env.vehicle_manager.get_statistics()
            idle = stats.get('idle', 0)
            picking = stats.get('picking_up', 0)
            serving = stats.get('serving', 0)
            dispatching = stats.get('dispatching', 0)

            match_rate = (env.episode_stats['total_orders_matched'] /
                         env.episode_stats['total_orders_generated']
                         if env.episode_stats['total_orders_generated'] > 0 else 0)

            pending = len(env.pending_orders)

            print(f"{step:<8} {idle:<8} {picking:<10} {serving:<10} {dispatching:<10} {match_rate:<10.2%} {pending:<10}")

            # 如果idle车辆数量持续下降，说明有问题
            if step > 0 and idle < config.TOTAL_VEHICLES * 0.3:  # 少于30%空闲
                print(f"\n⚠ 警告: 步骤{step}时，空闲车辆只有{idle}辆 ({idle/config.TOTAL_VEHICLES:.1%})")
                print(f"  这可能表明车辆状态更新有问题")

        if done:
            break

    # 最终统计
    print("\n" + "=" * 70)
    final_stats = env.vehicle_manager.get_statistics()
    print(f"\n最终车辆状态:")
    for status, count in sorted(final_stats.items()):
        percentage = count / config.TOTAL_VEHICLES * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")

    # 检查是否有异常
    idle_count = final_stats.get('idle', 0)
    if idle_count < config.TOTAL_VEHICLES * 0.2:
        print(f"\n✗ 异常: 空闲车辆过少 ({idle_count}/{config.TOTAL_VEHICLES} = {idle_count/config.TOTAL_VEHICLES:.1%})")
        print(f"  可能原因:")
        print(f"  1. 车辆状态没有正确恢复为idle")
        print(f"  2. update_dispatching_vehicles有bug")
        print(f"  3. update_serving_vehicles有bug")
    else:
        print(f"\n✓ 正常: 空闲车辆数量合理 ({idle_count}/{config.TOTAL_VEHICLES} = {idle_count/config.TOTAL_VEHICLES:.1%})")

    metrics = env.reward_calculator.get_metrics(
        total_orders_generated=env.episode_stats['total_orders_generated']
    )
    print(f"\n匹配率: {metrics['match_rate']:.2%}")

if __name__ == "__main__":
    test_vehicle_states()

