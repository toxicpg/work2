#!/usr/bin/env python3
"""
对比SARSA-SAA和Random Walk的匹配率
找出为什么SARSA-SAA匹配率这么低
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment


def test_environment(policy_name, dispatch_policy, test_orders, config):
    """测试特定策略的匹配率"""
    print(f"\n{'='*80}")
    print(f"测试: {policy_name}")
    print(f"{'='*80}")

    data_processor = DataProcessor(config)
    env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy=dispatch_policy)

    # 测试最后一天
    day_count = env.order_generator.get_day_count()
    test_day = max(0, day_count - 1)

    print(f"数据集天数: {day_count}")
    print(f"测试天: {test_day}")
    print(f"车辆数: {config.TOTAL_VEHICLES}")

    env.reset(start_day=test_day)

    # 运行100个tick
    stats = {
        'matched': 0,
        'completed': 0,
        'cancelled': 0,
        'generated': 0,
        'dispatches': 0
    }

    for tick in range(100):
        _, _, _, info = env.step()
        step_info = info.get('step_info', {})

        stats['matched'] += step_info.get('matched_orders', 0)
        stats['completed'] += step_info.get('completed_orders', 0)
        stats['cancelled'] += step_info.get('cancelled_orders', 0)
        stats['generated'] += step_info.get('new_orders', 0)
        stats['dispatches'] += step_info.get('dispatch_success', 0)

    match_rate = stats['matched'] / stats['generated'] if stats['generated'] > 0 else 0

    print(f"\n前100个tick结果:")
    print(f"  生成订单: {stats['generated']}")
    print(f"  匹配订单: {stats['matched']}")
    print(f"  完成订单: {stats['completed']}")
    print(f"  取消订单: {stats['cancelled']}")
    print(f"  调度次数: {stats['dispatches']}")
    print(f"  匹配率: {match_rate:.2%}")

    return stats


def main():
    print("="*80)
    print("匹配率对比测试")
    print("="*80)

    # 加载配置和数据
    config = Config()
    print(f"\n配置信息:")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  匹配半径: {getattr(config, 'MATCHER_SEARCH_RADIUS', 10)}")
    print(f"  最大等待时间: {config.MAX_WAITING_TIME}秒")

    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()

    print(f"\n数据集信息:")
    print(f"  总订单数: {len(all_orders):,}")
    print(f"  时间范围: {all_orders['timestamp'].min()} 到 {all_orders['timestamp'].max()}")

    # 划分测试集
    _, _, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )
    print(f"  测试集订单数: {len(test_orders):,}")

    # 测试1: 无调度
    stats_none = test_environment("无调度(baseline)", "none", test_orders, config)

    # 测试2: Random Walk
    stats_rw = test_environment("Random Walk", "random_walk", test_orders, config)

    # 测试3: 使用all_orders（模拟SARSA-SAA的方式）
    print(f"\n{'='*80}")
    print(f"测试: 使用all_orders（SARSA-SAA方式）")
    print(f"{'='*80}")

    env_all = BaselineEnvironment(config, data_processor, all_orders, dispatch_policy='none')
    day_count = env_all.order_generator.get_day_count()
    test_day = max(0, day_count - 1)

    print(f"数据集天数: {day_count}")
    print(f"测试天: {test_day}")

    env_all.reset(start_day=test_day)

    stats_all = {
        'matched': 0,
        'generated': 0
    }

    for tick in range(100):
        _, _, _, info = env_all.step()
        step_info = info.get('step_info', {})
        stats_all['matched'] += step_info.get('matched_orders', 0)
        stats_all['generated'] += step_info.get('new_orders', 0)

    match_rate_all = stats_all['matched'] / stats_all['generated'] if stats_all['generated'] > 0 else 0

    print(f"\n前100个tick结果:")
    print(f"  生成订单: {stats_all['generated']}")
    print(f"  匹配订单: {stats_all['matched']}")
    print(f"  匹配率: {match_rate_all:.2%}")

    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")

    print(f"\n匹配率对比:")
    print(f"  无调度(test_orders):    {stats_none['matched']/stats_none['generated']:.2%}")
    print(f"  Random Walk(test_orders): {stats_rw['matched']/stats_rw['generated']:.2%}")
    print(f"  无调度(all_orders):      {match_rate_all:.2%}")

    print(f"\n订单数对比:")
    print(f"  test_orders: {stats_none['generated']}")
    print(f"  all_orders:  {stats_all['generated']}")

    if abs(stats_none['generated'] - stats_all['generated']) > 10:
        print(f"\n⚠️  警告：订单数不一致！")
        print(f"  可能原因：test_orders和all_orders的最后一天数据不同")

    if stats_none['matched'] / stats_none['generated'] > 0.7:
        print(f"\n✓ 无调度baseline匹配率正常({stats_none['matched']/stats_none['generated']:.2%})")
        print(f"  如果SARSA-SAA只有30%，问题可能在：")
        print(f"  1. SARSA-SAA使用了错误的天数（训练集的某一天？）")
        print(f"  2. 环境reset有问题")
        print(f"  3. 车辆配置不对")
        print(f"  4. 统计逻辑有误")
    else:
        print(f"\n⚠️  baseline匹配率也很低({stats_none['matched']/stats_none['generated']:.2%})")
        print(f"  说明环境本身有问题，不是SARSA-SAA特有的")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

