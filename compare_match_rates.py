"""
对比主实验和baseline的匹配率
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
from config import Config
from utils.data_process import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def test_main_experiment():
    """测试主实验环境"""
    from environment import RideHailingEnvironment

    config = Config()
    set_seed(config.SEED)

    print("=" * 80)
    print("主实验环境 (environment.py)")
    print("=" * 80)
    print(f"  MAX_WAITING_TIME: {config.MAX_WAITING_TIME}秒")
    print(f"  MATCHER_KNN_K: {config.MATCHER_KNN_K}")
    print(f"  TICK_DURATION_SEC: {config.TICK_DURATION_SEC}秒")
    print(f"  TOTAL_VEHICLES: {config.TOTAL_VEHICLES}")

    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]

    env = RideHailingEnvironment(config, data_processor, day_orders)
    env.reset()

    # 运行100步
    for step in range(100):
        env.step(current_epsilon=0.0)  # 不调度，只匹配

    stats = env.get_episode_summary()
    match_rate = stats['reward_metrics']['match_rate']

    print(f"\n结果（100步后）:")
    print(f"  总生成: {env.episode_stats['total_orders_generated']}")
    print(f"  总匹配: {env.episode_stats['total_orders_matched']}")
    print(f"  总取消: {env.episode_stats['total_orders_cancelled']}")
    print(f"  匹配率: {match_rate:.2%}")

    return match_rate

def test_baseline():
    """测试baseline环境"""
    from environment_baseline import BaselineEnvironment

    config = Config()
    set_seed(config.SEED)

    print("\n" + "=" * 80)
    print("Baseline环境 (environment_baseline.py)")
    print("=" * 80)
    print(f"  MAX_WAITING_TIME: {config.MAX_WAITING_TIME}秒")
    print(f"  MATCHER_KNN_K: {config.MATCHER_KNN_K}")
    print(f"  TICK_DURATION_SEC: {config.TICK_DURATION_SEC}秒")
    print(f"  TOTAL_VEHICLES: {config.TOTAL_VEHICLES}")

    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]

    env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='none')
    env.reset()

    # 运行100步
    for step in range(100):
        env.step()

    metrics = env.reward_calculator.get_metrics(
        total_orders_generated=env.episode_stats['total_orders_generated']
    )
    match_rate = metrics['match_rate']

    print(f"\n结果（100步后）:")
    print(f"  总生成: {env.episode_stats['total_orders_generated']}")
    print(f"  总匹配: {env.episode_stats['total_orders_matched']}")
    print(f"  总取消: {env.episode_stats['total_orders_cancelled']}")
    print(f"  匹配率: {match_rate:.2%}")

    return match_rate

if __name__ == "__main__":
    main_rate = test_main_experiment()
    baseline_rate = test_baseline()

    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    print(f"  主实验匹配率: {main_rate:.2%}")
    print(f"  Baseline匹配率: {baseline_rate:.2%}")
    print(f"  差异: {abs(main_rate - baseline_rate):.2%}")

    if abs(main_rate - baseline_rate) < 0.05:
        print("  ✓ 匹配率基本一致")
    else:
        print(f"  ⚠ 匹配率差异较大")

