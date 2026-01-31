#!/usr/bin/env python3
"""
SARSA-SAA Baseline 训练脚本
用法:
  python run_sarsa_saa.py                    # 使用config中的默认车辆数
  python run_sarsa_saa.py --vehicles 1800    # 指定车辆数
  python run_sarsa_saa.py --vehicles 2000 --rounds 10  # 指定车辆数和轮数
"""

import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config
from utils.data_process import DataProcessor
from baselines.sarsa_saa import run_sarsa_saa_simulation


def main():
    parser = argparse.ArgumentParser(description='运行 SARSA-SAA Baseline 训练')
    parser.add_argument('--vehicles', type=int, default=None,
                        help='车辆数量 (默认: config.TOTAL_VEHICLES)')
    parser.add_argument('--rounds', type=int, default=5,
                        help='训练轮数 (默认: 5)')
    parser.add_argument('--start-day', type=int, default=0,
                        help='起始日期索引 (默认: 0)')

    args = parser.parse_args()

    print("=" * 80)
    print("SARSA-SAA Baseline 训练")
    print("=" * 80)

    # 初始化配置
    config = Config()

    # 如果指定了车辆数，更新配置
    if args.vehicles is not None:
        original_vehicles = config.TOTAL_VEHICLES
        config.TOTAL_VEHICLES = args.vehicles

        # 更新保存路径
        config.RESULTS_BASE_PATH = f'results/vehicles_{config.TOTAL_VEHICLES}/'
        config.MODEL_SAVE_PATH = f'{config.RESULTS_BASE_PATH}models/'
        config.LOG_SAVE_PATH = f'{config.RESULTS_BASE_PATH}logs/'
        config.BENCHMARK_SAVE_PATH = f'{config.RESULTS_BASE_PATH}benchmarks/'

        print(f"车辆数: {original_vehicles} → {config.TOTAL_VEHICLES}")
    else:
        print(f"车辆数: {config.TOTAL_VEHICLES} (默认)")

    print(f"训练轮数: {args.rounds}")
    print(f"起始日期: {args.start_day}")
    print(f"结果保存路径: {config.BENCHMARK_SAVE_PATH}")

    # 验证配置
    if not Config.validate_config():
        print("❌ 配置验证失败！")
        sys.exit(1)

    # 加载数据
    print("\n" + "=" * 80)
    print("加载数据...")
    print("=" * 80)
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    print(f"总订单数: {len(all_orders):,} 条")

    # 运行训练
    print("\n" + "=" * 80)
    print("开始训练 SARSA-SAA...")
    print("=" * 80)

    try:
        run_sarsa_saa_simulation(
            config=config,
            start_day_offset=args.start_day,
            all_orders=all_orders,
            num_rounds=args.rounds
        )

        print("\n" + "=" * 80)
        print("✓ SARSA-SAA 训练完成！")
        print("=" * 80)
        print(f"结果已保存至: {config.BENCHMARK_SAVE_PATH}")

    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

