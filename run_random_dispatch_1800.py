"""Random Dispatch Baseline - 1800辆车"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from config import Config
from utils.data_process import DataProcessor
from baselines.random_dispatch import run_random_dispatch_simulation
import json
from datetime import datetime

if __name__ == "__main__":
    print("="*80)
    print("Random Dispatch Baseline - 1800辆车")
    print("="*80)
    config = Config()
    config.TOTAL_VEHICLES = 1800
    config.RESULTS_BASE_PATH = f"results/vehicles_1800/"
    print(f"车辆数量: {config.TOTAL_VEHICLES}")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    _, _, test_orders = data_processor.split_data_by_time(all_orders, config.TRAIN_RATIO, config.VAL_RATIO)
    print(f"测试集订单数: {len(test_orders):,}")
    results = run_random_dispatch_simulation(config, num_episodes=7, env_data=test_orders, num_rounds=1)
    if results:
        save_dir = f"results/vehicles_1800/baselines/"
        os.makedirs(save_dir, exist_ok=True)
        result_file = f"{save_dir}/random_dispatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n✓ 完成: {result_file}")
    else:
        sys.exit(1)

