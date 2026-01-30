"""批量创建baseline运行脚本"""

vehicles = [2000, 2200]

for v in vehicles:
    # Random Walk
    walk_content = f'''"""Random Walk Baseline - {v}辆车"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from config import Config
from utils.data_process import DataProcessor
from baselines.random_walk import run_last7_days_random_walk
import json, numpy as np
from datetime import datetime

if __name__ == "__main__":
    print("="*80)
    print("Random Walk Baseline - {v}辆车")
    print("="*80)
    config = Config()
    config.TOTAL_VEHICLES = {v}
    config.RESULTS_BASE_PATH = f"results/vehicles_{v}/"
    print(f"车辆数量: {{config.TOTAL_VEHICLES}}")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    _, _, test_orders = data_processor.split_data_by_time(all_orders, config.TRAIN_RATIO, config.VAL_RATIO)
    print(f"测试集订单数: {{len(test_orders):,}}")
    print("\\n运行Random Walk测试（最后7天）...")
    daily_results = run_last7_days_random_walk(config, test_orders)
    if daily_results:
        overall_results = {{
            "method": "Random Walk",
            "vehicle_count": {v},
            "num_days": len(daily_results),
            "avg_completion_rate": float(np.mean([d["completion_rate"] for d in daily_results])),
            "avg_cancel_rate": float(np.mean([d["cancel_rate"] for d in daily_results])),
            "avg_waiting_time": float(np.mean([d["avg_waiting_time"] for d in daily_results])),
            "avg_revenue": float(np.mean([d["total_revenue"] for d in daily_results])),
            "daily_results": daily_results
        }}
        print(f"\\n完成率: {{overall_results['avg_completion_rate']:.2%}}")
        save_dir = f"results/vehicles_{v}/baselines/"
        os.makedirs(save_dir, exist_ok=True)
        result_file = f"{{save_dir}}/random_walk_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(overall_results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\\n✓ 完成: {{result_file}}")
    else:
        sys.exit(1)
'''

    # Random Dispatch
    dispatch_content = f'''"""Random Dispatch Baseline - {v}辆车"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from config import Config
from utils.data_process import DataProcessor
from baselines.random_dispatch import run_random_dispatch_simulation
import json
from datetime import datetime

if __name__ == "__main__":
    print("="*80)
    print("Random Dispatch Baseline - {v}辆车")
    print("="*80)
    config = Config()
    config.TOTAL_VEHICLES = {v}
    config.RESULTS_BASE_PATH = f"results/vehicles_{v}/"
    print(f"车辆数量: {{config.TOTAL_VEHICLES}}")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    _, _, test_orders = data_processor.split_data_by_time(all_orders, config.TRAIN_RATIO, config.VAL_RATIO)
    print(f"测试集订单数: {{len(test_orders):,}}")
    results = run_random_dispatch_simulation(config, num_episodes=7, env_data=test_orders, num_rounds=5)
    if results:
        save_dir = f"results/vehicles_{v}/baselines/"
        os.makedirs(save_dir, exist_ok=True)
        result_file = f"{{save_dir}}/random_dispatch_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\\n✓ 完成: {{result_file}}")
    else:
        sys.exit(1)
'''

    with open(f'run_random_walk_{v}.py', 'w', encoding='utf-8') as f:
        f.write(walk_content)
    with open(f'run_random_dispatch_{v}.py', 'w', encoding='utf-8') as f:
        f.write(dispatch_content)

    print(f'✓ 创建: run_random_walk_{v}.py')
    print(f'✓ 创建: run_random_dispatch_{v}.py')

print('\n所有脚本已创建！')

