"""
测试baseline匹配修复效果
对比修复前后的匹配率差异
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_baseline_matching():
    """测试baseline环境的匹配率"""
    print("=" * 70)
    print("测试 Baseline 匹配修复效果")
    print("=" * 70)

    # 初始化配置
    config = Config()
    set_seed(config.SEED)

    # 加载数据
    print("\n[1] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date

    # 选择一天的数据进行测试
    test_day = sorted(all_orders['date'].unique())[0]
    day_orders = all_orders[all_orders['date'] == test_day]
    print(f"  测试日期: {test_day}")
    print(f"  订单数量: {len(day_orders)}")

    # 初始化环境
    print("\n[2] 初始化环境...")
    config.DISPATCH_MODE = 'random_walk'  # 设置为random_walk模式
    env = BaselineEnvironment(config, data_processor, day_orders)
    env.reset()

    # 运行仿真
    print("\n[3] 运行仿真...")
    # 一天 = 24小时 * 3600秒 / 10秒每tick = 8640 ticks
    # 但为了安全起见，我们用config中的设置
    max_steps = min(8640, config.MAX_TICKS_PER_EPISODE)  # 测试一整天（24小时）
    print(f"  测试时长: {max_steps}个ticks (约{max_steps * config.TICK_DURATION_SEC / 3600:.1f}小时)")

    step_count = 0
    total_matched = 0
    total_generated = 0
    total_cancelled = 0

    while step_count < max_steps:
        _, _, done, info = env.step()

        step_info = info.get('step_info', {})
        matched = step_info.get('matched_orders', 0)
        new_orders = step_info.get('new_orders', 0)
        cancelled = step_info.get('cancelled_orders', 0)

        total_matched += matched
        total_generated += new_orders
        total_cancelled += cancelled

        step_count += 1

        # 每500步打印一次进度（约1.4小时）
        if step_count % 500 == 0:
            current_match_rate = total_matched / total_generated if total_generated > 0 else 0
            hours_elapsed = step_count * config.TICK_DURATION_SEC / 3600
            print(f"  Step {step_count}/{max_steps} ({hours_elapsed:.1f}小时): "
                  f"匹配率={current_match_rate:.2%}, "
                  f"已匹配={total_matched}, "
                  f"已生成={total_generated}, "
                  f"已取消={total_cancelled}")

        if done:
            break

    # 最终统计
    print("\n" + "=" * 70)
    print("测试结果:")
    print("=" * 70)

    metrics = env.reward_calculator.get_metrics(total_orders_generated=total_generated)

    print(f"  总生成订单: {total_generated}")
    print(f"  总匹配订单: {total_matched}")
    print(f"  总取消订单: {total_cancelled}")
    print(f"  匹配率: {metrics['match_rate']:.2%}")
    print(f"  完成率: {metrics.get('completion_rate', 0):.2%}")
    print(f"  取消率: {metrics.get('cancel_rate', 0):.2%}")

    if metrics.get('avg_waiting_time', 0) > 0:
        print(f"  平均等待时间: {metrics['avg_waiting_time']:.1f}秒")

    # 车辆状态统计
    vehicle_stats = env.vehicle_manager.get_statistics()
    print(f"\n  车辆状态分布:")
    for status, count in vehicle_stats.items():
        print(f"    {status}: {count}")

    print("=" * 70)

    # 判断修复效果
    print("\n预期效果分析:")
    print(f"  - 距离限制: 1格以内 (约 {1 * 1.5:.1f}分钟 = {1 * 90}秒)")
    print(f"  - 最大等待: {config.MAX_WAITING_TIME}秒 (5分钟)")
    print(f"  - 理论匹配率: 应该在5%-15%之间（极严格距离限制）")

    if metrics['match_rate'] < 0.03:
        print("\n✗ 匹配率过低(<3%)，限制过于严格")
    elif metrics['match_rate'] > 0.30:
        print("\n✗ 匹配率过高(>30%)，距离限制可能没生效")
    else:
        print("\n✓ 匹配率合理，距离限制正常工作")

    return metrics

if __name__ == "__main__":
    test_baseline_matching()

