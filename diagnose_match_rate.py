"""
诊断匹配率问题
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import torch
from config import Config
from environment_baseline import BaselineEnvironment
from utils.data_process import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def diagnose():
    config = Config()
    set_seed(config.SEED)

    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]

    print("=" * 80)
    print(f"诊断: 车辆数={config.TOTAL_VEHICLES}, 订单数={len(day_orders)}")
    print("=" * 80)

    env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='random_walk')
    env.reset()

    print(f"\n{'Step':<6} {'Idle':<6} {'Serving':<8} {'Dispatch':<9} {'Matched':<8} {'Pending':<8} {'Rate':<8}")
    print("-" * 80)

    for step in range(300):
        _, _, done, _ = env.step()

        if step % 20 == 0:
            stats = env.vehicle_manager.get_statistics()
            idle = stats.get('idle', 0)
            serving = stats.get('serving', 0)
            dispatching = stats.get('dispatching', 0)

            matched = env.episode_stats['total_orders_matched']
            generated = env.episode_stats['total_orders_generated']
            pending = len(env.pending_orders)
            rate = matched / generated if generated > 0 else 0

            print(f"{step:<6} {idle:<6} {serving:<8} {dispatching:<9} {matched:<8} {pending:<8} {rate:<8.2%}")

            # 诊断：如果serving车辆过多，说明订单完成太慢
            if serving > config.TOTAL_VEHICLES * 0.5:
                print(f"  ⚠ 警告: serving车辆过多 ({serving}/{config.TOTAL_VEHICLES})")
                print(f"     可能原因: 订单完成时间过长")

        if done or step >= 299:
            break

    print("\n" + "=" * 80)
    final_stats = env.vehicle_manager.get_statistics()
    print("最终状态:")
    for status, count in sorted(final_stats.items()):
        percentage = count / config.TOTAL_VEHICLES * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")

    metrics = env.reward_calculator.get_metrics(
        total_orders_generated=env.episode_stats['total_orders_generated']
    )
    print(f"\n匹配率: {metrics['match_rate']:.2%}")
    print(f"完成率: {metrics.get('completion_rate', 0):.2%}")

if __name__ == "__main__":
    diagnose()

