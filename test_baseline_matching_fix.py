"""
测试 Random Walk Baseline - 1800辆车，执行1天
修复内容：
1. 匹配器在匹配时立即assign订单，而不是返回后再assign
2. 车辆在接驾完成时更新位置到订单起点
3. 车辆在接驾途中订单取消时，如果走了一半以上更新到订单起点
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import torch
from tqdm import tqdm
from config import Config
from environment_baseline import BaselineEnvironment
from utils.data_process import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_random_walk_baseline():
    """测试 Random Walk Baseline - 1800辆车，1天"""
    print("=" * 70)
    print("Random Walk Baseline 测试")
    print("=" * 70)

    # 初始化配置
    config = Config()
    set_seed(config.SEED)

    print(f"\n配置:")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  Tick时长: {config.TICK_DURATION_SEC}秒")
    print(f"  Episode天数: {config.EPISODE_DAYS}天")
    print(f"  最大Ticks: {config.MAX_TICKS_PER_EPISODE}")

    # 加载数据
    print(f"\n[1] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    # 选择一天的数据进行测试
    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]
    print(f"  测试日期: {test_day}")
    print(f"  订单数量: {len(day_orders)}")

    # 初始化环境（使用random_walk策略）
    print(f"\n[2] 初始化环境 (策略: random_walk)...")
    env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='random_walk')
    env.reset()

    # 运行仿真
    print(f"\n[3] 运行仿真 (执行 {config.MAX_TICKS_PER_EPISODE} 个ticks)...")

    pbar = tqdm(total=config.MAX_TICKS_PER_EPISODE, desc="仿真进度")

    step_count = 0

    try:
        while step_count < config.MAX_TICKS_PER_EPISODE:
            _, _, done, info = env.step()

            step_info = info.get('step_info', {})
            matched = step_info.get('matched_orders', 0)
            new_orders = step_info.get('new_orders', 0)

            step_count += 1
            pbar.update(1)

            # 更新进度条信息
            if step_count % 100 == 0:
                current_match_rate = (env.episode_stats['total_orders_matched'] /
                                     env.episode_stats['total_orders_generated']
                                     if env.episode_stats['total_orders_generated'] > 0 else 0)
                pbar.set_postfix({
                    '匹配率': f"{current_match_rate:.2%}",
                    '已匹配': env.episode_stats['total_orders_matched'],
                    '已生成': env.episode_stats['total_orders_generated']
                })

            if done:
                break
    finally:
        pbar.close()

    # 最终统计
    print("\n" + "=" * 70)
    print("仿真结果:")
    print("=" * 70)

    total_generated = env.episode_stats['total_orders_generated']
    metrics = env.reward_calculator.get_metrics(total_orders_generated=total_generated)

    print(f"\n订单统计:")
    print(f"  总生成订单: {total_generated:,}")
    print(f"  总匹配订单: {env.episode_stats['total_orders_matched']:,}")
    print(f"  总完成订单: {metrics['completed_orders']:,}")
    print(f"  总取消订单: {env.episode_stats['total_orders_cancelled']:,}")
    print(f"  匹配率: {metrics['match_rate']:.2%}")
    print(f"  完成率: {metrics.get('completion_rate', 0):.2%}")
    print(f"  取消率: {metrics.get('cancel_rate', 0):.2%}")

    print(f"\n调度统计:")
    print(f"  总调度次数: {env.episode_stats['total_dispatches']:,}")
    print(f"  总收入: ¥{env.episode_stats['total_revenue']:,.2f}")

    if metrics.get('avg_waiting_time', 0) > 0:
        print(f"\n等待时间:")
        print(f"  平均等待: {metrics['avg_waiting_time']:.1f}秒 ({metrics['avg_waiting_time']/60:.1f}分钟)")
        print(f"  最小等待: {metrics['min_waiting_time']:.1f}秒")
        print(f"  最大等待: {metrics['max_waiting_time']:.1f}秒")
        print(f"  标准差: {metrics['std_waiting_time']:.1f}秒")

    # 车辆状态统计
    vehicle_stats = env.vehicle_manager.get_statistics()
    print(f"\n车辆状态分布:")
    for status, count in sorted(vehicle_stats.items()):
        percentage = count / config.TOTAL_VEHICLES * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")

    print("=" * 70)

    # 判断结果
    if metrics['match_rate'] > 0.60:
        print("✓ 匹配率良好 (>60%)")
    elif metrics['match_rate'] > 0.40:
        print("⚠ 匹配率中等 (40%-60%)")
    else:
        print("✗ 匹配率偏低 (<40%)")

    return metrics

if __name__ == "__main__":
    test_random_walk_baseline()

