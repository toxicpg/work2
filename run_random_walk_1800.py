"""
Random Walk Baseline - 1800辆车
直接运行测试，无需训练
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from baselines.random_walk import run_last7_days_random_walk
import json
import numpy as np
from datetime import datetime

if __name__ == '__main__':
    print("="*80)
    print("Random Walk Baseline - 1800辆车")
    print("="*80)

    # 创建配置并强制设置车辆数量
    config = Config()
    config.TOTAL_VEHICLES = 1800
    config.DISPATCH_MODE = 'random_walk'
    # 更新保存路径
    config.RESULTS_BASE_PATH = f'results/vehicles_{config.TOTAL_VEHICLES}/'
    config.MODEL_SAVE_PATH = f'{config.RESULTS_BASE_PATH}models/'
    config.LOG_SAVE_PATH = f'{config.RESULTS_BASE_PATH}logs/'
    config.ABLATION_SAVE_PATH = f'{config.RESULTS_BASE_PATH}ablation/'
    config.BENCHMARK_SAVE_PATH = f'{config.RESULTS_BASE_PATH}benchmarks/'

    print(f"车辆数量: {config.TOTAL_VEHICLES}")
    print(f"保存路径: {config.RESULTS_BASE_PATH}")

    # 加载数据
    print("\n加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    _, _, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    print(f"测试集订单数: {len(test_orders):,}")

    # 运行测试（最后7天）
    print("\n运行Random Walk测试（最后7天）...")
    daily_results = run_last7_days_random_walk(config, test_orders)

    # 保存结果
    if daily_results:
        # 计算汇总指标
        overall_results = {
            'method': 'Random Walk',
            'vehicle_count': config.TOTAL_VEHICLES,
            'num_days': len(daily_results),
            'avg_completion_rate': float(np.mean([d['completion_rate'] for d in daily_results])),
            'avg_cancel_rate': float(np.mean([d['cancel_rate'] for d in daily_results])),
            'avg_waiting_time': float(np.mean([d['avg_waiting_time'] for d in daily_results])),
            'avg_revenue': float(np.mean([d['total_revenue'] for d in daily_results])),
            'daily_results': daily_results
        }

        # 打印结果
        print(f"\n结果汇总:")
        print(f"  完成率: {overall_results['avg_completion_rate']:.2%}")
        print(f"  取消率: {overall_results['avg_cancel_rate']:.2%}")
        print(f"  平均等待时间: {overall_results['avg_waiting_time']:.1f}秒")
        print(f"  平均收入: {overall_results['avg_revenue']:.2f}")

        save_dir = f'results/vehicles_{config.TOTAL_VEHICLES}/baselines/'
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(save_dir, f'random_walk_results_{timestamp}.json')

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, default=str, ensure_ascii=False)

        print(f"\n✓ 结果已保存到: {result_file}")
        print(f"\n{'='*80}")
        print("Random Walk - 1800辆车 - 完成")
        print(f"{'='*80}")
    else:
        print("\n✗ 实验失败")
        sys.exit(1)

