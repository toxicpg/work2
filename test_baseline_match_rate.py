import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment

def run_test():
    print(">>> 开始 Baseline 环境匹配率测试 (1天数据) <<<")
    print("目的: 验证匹配逻辑修复后的匹配率和车辆行为")

    # 1. 配置
    config = Config()
    config.EPISODE_DAYS = 1  # 只跑一天
    config.MAX_TICKS_PER_EPISODE = config.TICKS_PER_DAY * config.EPISODE_DAYS

    # 2. 加载数据
    print("\n[1/3] 加载数据...")
    data_processor = DataProcessor(config)
    # 加载所有订单
    all_orders = data_processor.load_and_process_orders()

    # 筛选出第一天的数据用于测试
    if 'timestamp' in all_orders.columns:
        start_date = all_orders['timestamp'].min().date()
        # 筛选第一天
        test_orders = all_orders[all_orders['timestamp'].dt.date == start_date].copy()
        print(f"  测试日期: {start_date}")
        print(f"  订单数量: {len(test_orders)}")
    else:
        print("错误: 订单数据缺少 timestamp 列")
        return

    # 3. 初始化环境 (使用 Random Walk 策略作为基准)
    # Random Walk 是最基础的策略，能很好地反映匹配逻辑本身的影响
    print("\n[2/3] 初始化环境 (策略: Random Walk)...")
    env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy='random_walk')
    env.reset()

    # 4. 运行测试
    print("\n[3/3] 开始运行仿真...")
    pbar = tqdm(total=config.MAX_TICKS_PER_EPISODE)

    total_reward = 0

    while env.episode_step < config.MAX_TICKS_PER_EPISODE:
        # 执行一步
        _, _, done, info = env.step()

        step_info = info.get('step_info', {})
        total_reward += step_info.get('revenue', 0)

        pbar.update(1)

        # 实时显示一些统计
        if env.episode_step % 100 == 0:
            stats = env.reward_calculator.get_metrics()
            # 避免除以零
            match_rate = stats['match_rate']

            pbar.set_postfix({
                'MatchRate': f"{match_rate:.2%}",
                'Cancel': f"{stats['cancelled_orders']}",
                'Wait': f"{stats.get('avg_waiting_time', 0):.1f}s"
            })

        if done:
            break

    pbar.close()

    # 5. 输出最终结果
    print("\n" + "="*50)
    print("测试结果摘要")
    print("="*50)

    summary = env.get_episode_summary()
    metrics = summary['reward_metrics']
    ep_stats = summary['episode_stats']

    print(f"总生成订单: {ep_stats['total_orders_generated']}")
    print(f"总匹配订单: {ep_stats['total_orders_matched']}")
    print(f"总取消订单: {ep_stats['total_orders_cancelled']}")
    print("-" * 30)
    print(f"★ 匹配率 (Match Rate): {metrics['match_rate']:.2%}")
    print(f"★ 完成率 (Completion Rate): {metrics['completion_rate']:.2%}")
    print(f"★ 取消率 (Cancel Rate): {metrics['cancel_rate']:.2%}")
    print("-" * 30)
    print(f"平均等待时间: {metrics['avg_waiting_time']:.2f} 秒")
    print(f"最大等待时间: {metrics['max_waiting_time']:.2f} 秒")
    print(f"总营收: {metrics['total_revenue']:.2f}")
    print("="*50)

    print("\n结果分析:")
    if metrics['match_rate'] > 0.95:
        print("⚠ 警告: 匹配率仍然非常高 (>95%)。")
        print("  可能原因: 车辆数过多、需求过少，或者超时预判逻辑未生效。")
    elif metrics['match_rate'] < 0.4:
        print("ℹ 提示: 匹配率较低 (<40%)。")
        print("  这在 Random Walk 策略下可能是正常的，说明车辆没有被有效调度到热点区域。")
    else:
        print("✓ 观察: 匹配率在中等范围 (40%-95%)。")
        print("  这通常表示环境逻辑正常运作：有匹配也有因距离远/无车导致的取消。")

if __name__ == "__main__":
    run_test()

