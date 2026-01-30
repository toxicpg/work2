"""
Random Dispatch Baseline - 2200辆车
直接运行测试，无需训练
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from baselines.random_dispatch import run_random_dispatch_simulation
import json
from datetime import datetime

if __name__ == '__main__':
    print("="*80)
    print("Random Dispatch Baseline - 2200辆车")
    print("="*80)

    # 创建配置并强制设置车辆数量
    config = Config()
    config.TOTAL_VEHICLES = 2200

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

    # 运行测试（5轮，每轮最后7天）
    results = run_random_dispatch_simulation(config, num_episodes=7, env_data=test_orders, num_rounds=5)

    # 保存结果
    if results:
        save_dir = f'results/vehicles_{config.TOTAL_VEHICLES}/baselines/'
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(save_dir, f'random_dispatch_results_{timestamp}.json')

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        print(f"\n✓ 结果已保存到: {result_file}")
        print(f"\n{'='*80}")
        print("Random Dispatch - 2200辆车 - 完成")
        print(f"{'='*80}")
    else:
        print("\n✗ 实验失败")
        sys.exit(1)

