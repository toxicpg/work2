"""
诊断匹配率低的问题 - 详细版
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

    print("=" * 80)
    print("诊断匹配率低的问题")
    print("=" * 80)
    print(f"车辆数: {config.TOTAL_VEHICLES}")
    print(f"空闲阈值: {config.IDLE_THRESHOLD_SEC}秒")
    print(f"超时时间: {config.MAX_WAITING_TIME}秒")
    print(f"匹配K值: {getattr(config, 'MATCHER_KNN_K', 30)}")

    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]

    print(f"\n测试日期: {test_day}, 订单数: {len(day_orders)}")

    env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='random_walk')
    env.reset()

    print(f"\n{'='*80}")
    print(f"{'Step':<6} {'Idle':<6} {'Serv':<6} {'Disp':<6} {'NewO':<6} {'Pend':<6} {'Match':<7} {'Cancel':<7} {'Rate':<8}")
    print(f"{'='*80}")

    for step in range(200):  # 只运行200步观察
        _, _, done, info = env.step()

        step_info = info.get('step_info', {})
        stats = env.vehicle_manager.get_statistics()

        if step % 10 == 0:  # 每10步打印一次
            idle = stats.get('idle', 0)
            serving = stats.get('serving', 0)
            dispatching = stats.get('dispatching', 0)

            new_orders = step_info.get('new_orders', 0)
            matched = step_info.get('matched_orders', 0)
            cancelled = step_info.get('cancelled_orders', 0)
            pending = len(env.pending_orders)

            total_matched = env.episode_stats['total_orders_matched']
            total_generated = env.episode_stats['total_orders_generated']
            rate = total_matched / total_generated if total_generated > 0 else 0

            print(f"{step:<6} {idle:<6} {serving:<6} {dispatching:<6} {new_orders:<6} {pending:<6} {matched:<7} {cancelled:<7} {rate:<8.2%}")

            # 诊断异常情况
            if pending > 100 and idle > 500:
                print(f"  ⚠ 异常: pending订单多({pending})但idle车辆也多({idle})，匹配器可能有问题！")

            if serving > config.TOTAL_VEHICLES * 0.7:
                print(f"  ⚠ 异常: serving车辆过多({serving}/{config.TOTAL_VEHICLES})，订单完成太慢！")

            if cancelled > 50:
                print(f"  ⚠ 异常: 本步取消订单过多({cancelled})！")

        if done or step >= 199:
            break

    print(f"{'='*80}")

    # 最终诊断
    print(f"\n最终诊断:")
    print(f"  总生成: {env.episode_stats['total_orders_generated']}")
    print(f"  总匹配: {env.episode_stats['total_orders_matched']}")
    print(f"  总取消: {env.episode_stats['total_orders_cancelled']}")
    print(f"  总调度: {env.episode_stats['total_dispatches']}")

    final_stats = env.vehicle_manager.get_statistics()
    print(f"\n  最终车辆状态:")
    for status, count in sorted(final_stats.items()):
        print(f"    {status}: {count} ({count/config.TOTAL_VEHICLES:.1%})")

    metrics = env.reward_calculator.get_metrics(
        total_orders_generated=env.episode_stats['total_orders_generated']
    )
    print(f"\n  最终匹配率: {metrics['match_rate']:.2%}")

    # 分析问题
    print(f"\n问题分析:")
    if env.episode_stats['total_orders_cancelled'] > env.episode_stats['total_orders_matched'] * 0.5:
        print(f"  ⚠ 取消订单数({env.episode_stats['total_orders_cancelled']})过多，可能是超时时间太短")

    if final_stats.get('serving', 0) > config.TOTAL_VEHICLES * 0.5:
        print(f"  ⚠ serving车辆过多，订单完成时间可能计算错误")

    if final_stats.get('dispatching', 0) > config.TOTAL_VEHICLES * 0.3:
        print(f"  ⚠ dispatching车辆过多，random_walk调度过于激进")

if __name__ == "__main__":
    diagnose()

