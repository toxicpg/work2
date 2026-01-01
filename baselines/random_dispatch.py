# baselines/random_dispatch.py
"""
Benchmark 3: Random Dispatching Policy Simulation
当车辆空闲时间达到阈值时，随机选择一个 *全局热点网格* (Pruned Global Action Space)
作为调度目标。
"""

import sys
import os
import numpy as np
import torch # 导入 torch 以设置种子
import random
from tqdm import tqdm
from datetime import datetime
import pandas as pd # 用于保存结果
from collections import defaultdict

# --- 调整 Python 路径并更改工作目录 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
try:
    os.chdir(project_root)
    print(f"当前工作目录已更改为: {os.getcwd()}")
except Exception as e:
    print(f"更改工作目录失败: {e}"); sys.exit(1)
if project_root not in sys.path:
    sys.path.append(project_root)
# ---------------------------------------------

# --- 导入必要的自定义模块 ---
try:
    from config import Config
    from utils.data_process import DataProcessor
    from utils.graph_builder import GraphBuilder
    from environment_baseline import BaselineEnvironment  # 使用 environment_baseline.py 进行基准测试
    from evaluate import print_evaluation_results
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 random_dispatch.py 在 baselines 文件夹下，")
    print("并且 config.py, utils/, models/, environment_baseline.py, evaluate.py 等在上一级目录 (work2)。")
    sys.exit(1)
# ---------------------------

# ===== 添加 set_seed 函数定义 =====
def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# =====================================

# --- 模拟运行函数 ---
def run_random_dispatch_simulation(config, num_episodes, env_data, num_rounds=5):
    """
    运行 Random Dispatch 策略的模拟.
    Args:
        num_rounds (int): 运行的轮数 (seeds), 用于统计最好/最坏情况.
    """
    print(f"\n--- 开始 Random Dispatch Benchmark (Total Rounds: {num_rounds}, Last 7 Days per Round) ---")
    print(f"  车辆总数: {config.TOTAL_VEHICLES}")
    print(f"  空闲阈值: {config.IDLE_THRESHOLD_SEC} 秒")

    # 1. 准备数据处理器 (只创建一次)
    try:
        data_processor = DataProcessor(config)
    except Exception as e:
        print(f"创建 DataProcessor 出错: {e}")
        return {}

    all_daily_rows = []
    
    # --- 开始多轮循环 ---
    for round_idx in range(num_rounds):
        current_seed = config.SEED + round_idx
        set_seed(current_seed)
        print(f"\n>>> Round {round_idx + 1}/{num_rounds} (Seed: {current_seed})")

        # 2. 创建环境实例 (每轮重新创建以确保干净状态)
        try:
            env = BaselineEnvironment(config, data_processor, env_data, dispatch_policy='random_dispatching')
        except Exception as e:
            print(f"创建环境时出错: {e}")
            continue

        # 3. 确定运行日期
        try:
            day_count = env.order_generator.get_day_count()
        except Exception:
            day_count = 0
        
        start0 = max(0, day_count - 7)
        days_to_run = list(range(start0, min(start0 + 7, day_count)))
        
        pbar = tqdm(days_to_run, desc=f"Round {round_idx+1}", unit="day")
        
        for d in pbar:
            try:
                env.reset() 
                env.current_day = d
                env.episode_start_day = d
                env.simulation_time = env.order_generator.time_range[0].normalize() + pd.Timedelta(days=d)
                if env.simulation_time.tzinfo is None:
                     env.simulation_time = env.simulation_time.tz_localize('Asia/Shanghai')
                env.current_time = env.simulation_time
                env.vehicle_manager.reset()
                env.reward_calculator.reset()
                env.pending_orders.clear()
                env.event_queue.clear()
                env.buffered_orders.clear()
                env.current_macro_slice_key = None
                env.daily_stats.clear()
                
            except Exception as e:
                print(f"错误: env 重置失败在 Day {d}: {e}")
                continue

            ticks = 0
            daily_infos = defaultdict(list)
            
            while ticks < config.TICKS_PER_DAY:
                try:
                    _, _, _, info = env.step()
                    daily_infos[0].append(info.get('step_info', {}))
                except Exception as e:
                     print(f"错误: env.step() 失败在 Day {d}, Tick {ticks}: {e}")
                     break
                ticks += 1
                
            if hasattr(env, '_calculate_daily_metrics'):
                 daily_metrics = env._calculate_daily_metrics(daily_infos, d)
            else:
                 total_matched = sum(info.get('matched_orders', 0) for info in daily_infos[0])
                 total_revenue = sum(info.get('revenue', 0.0) for info in daily_infos[0])
                 daily_metrics = [{'day_index': 0, 'actual_day': d, 'completed_orders': total_matched, 'total_revenue': total_revenue}]

            if daily_metrics:
                m = daily_metrics[0]
                m['dataset_day_index'] = d
                m['benchmark_policy'] = 'random_dispatching'
                m['round_index'] = round_idx # 记录轮次
                m['seed'] = current_seed
                all_daily_rows.append(m)
                
                pbar.set_description(f"Round {round_idx+1} (Day {d}: Cmp={m.get('completion_rate', 0):.1%})")

    # --- 汇总分析 ---
    if all_daily_rows:
        df = pd.DataFrame(all_daily_rows)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存所有原始数据
        raw_csv = os.path.join(config.LOG_SAVE_PATH, 'results', f'benchmark_random_dispatch_last7_raw_{ts}.csv')
        os.makedirs(os.path.dirname(raw_csv), exist_ok=True)
        df.to_csv(raw_csv, index=False)
        print(f"\n✓ 已保存原始数据 (All Rounds) 到: {raw_csv}")

        # 2. 计算最好/最坏统计
        # 按 dataset_day_index 分组
        summary_rows = []
        grouped = df.groupby('dataset_day_index')
        
        for day, group in grouped:
            best_cmp = group['completion_rate'].max()
            worst_cmp = group['completion_rate'].min()
            
            # 等待时间: 越小越好
            best_wait = group['avg_waiting_time'].min()
            worst_wait = group['avg_waiting_time'].max()
            
            summary_rows.append({
                'Day': day,
                'Best_Wait': best_wait,
                'Worst_Wait': worst_wait,
                'Best_Completion_Rate': best_cmp,
                'Worst_Completion_Rate': worst_cmp
            })
            
        df_summary = pd.DataFrame(summary_rows)
        
        # 打印汇总表
        print("\n" + "="*60)
        print("Random Dispatch 5-Round Summary (Best/Worst)")
        print("="*60)
        try:
            from tabulate import tabulate
            print(tabulate(df_summary, headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False))
        except ImportError:
            print(df_summary.to_string(index=False))
        
        # 保存汇总表
        summary_csv = os.path.join(config.LOG_SAVE_PATH, 'results', f'benchmark_random_dispatch_summary_{ts}.csv')
        df_summary.to_csv(summary_csv, index=False)
        print(f"\n✓ 已保存汇总分析结果到: {summary_csv}")

    # 为了兼容，返回所有数据的平均值
    if all_daily_rows:
        avg_results = {
            'benchmark': 'Random Dispatch (Last 7 Days - 5 Rounds Avg)',
            'num_episodes': len(all_daily_rows),
            'avg_reward': np.mean([r.get('total_revenue', 0) for r in all_daily_rows]),
            'completion_rate': np.mean([r.get('completion_rate', 0) for r in all_daily_rows]),
            'cancel_rate': np.mean([r.get('cancel_rate', 0) for r in all_daily_rows]),
            'avg_waiting_time': np.mean([r.get('avg_waiting_time', 0) for r in all_daily_rows]),
            'avg_dispatches_per_ep': np.mean([r.get('total_dispatches', 0) for r in all_daily_rows]),
        }
    else:
        avg_results = {}

    print(f"--- Random Dispatch Benchmark 完成 ---")
    return avg_results

# --- 保存结果函数 ---
def save_benchmark_results(results, config):
    """保存 benchmark 结果到 CSV"""
    results_dir = os.path.join(config.LOG_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)
    filename = f"benchmark_results.csv" # <--- 使用固定文件名以追加
    filepath = os.path.join(results_dir, filename)

    try:
        results_to_save = {}
        for k, v in results.items():
             if isinstance(v, (int, float, str, bool)): results_to_save[k] = v
             elif isinstance(v, np.generic): results_to_save[k] = v.item()
             else: results_to_save[k] = str(v)
        df_new = pd.DataFrame([results_to_save])

        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Benchmark 结果已追加到: {filepath}")
    except Exception as e:
        print(f"保存 Benchmark 结果失败: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
    print("=" * 80)
    print("运行 Benchmark 3: Random Dispatching") # <-- 修改名称
    print("=" * 80)

    try:
        config = Config()
        if not Config.validate_config(): sys.exit(1)
        set_seed(config.SEED)
    except Exception as e:
        print(f"加载配置或设置种子时出错: {e}"); sys.exit(1)

    NUM_EVAL_EPISODES = getattr(config, 'TEST_EPISODES', 10)

    try:
        print("加载数据...")
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()
        if hasattr(data_processor, 'split_data_by_time'):
            _, _, test_orders = data_processor.split_data_by_time(
                all_orders, config.TRAIN_RATIO, config.VAL_RATIO
            )
        else:
             print("错误: DataProcessor 中缺少 split_data_by_time 方法"); sys.exit(1)
        if test_orders.empty:
             print("错误：测试订单数据为空！无法运行模拟。"); sys.exit(1)
        print(f"使用 {len(test_orders):,} 条测试订单运行模拟。")

        # 运行模拟
        results = run_random_dispatch_simulation(config, NUM_EVAL_EPISODES, test_orders)

        # 打印结果
        if results:
            if 'print_evaluation_results' in globals():
                print_evaluation_results(results, title="Benchmark 3: Random Dispatch 结果") # <-- 修改名称
            else:
                 print("打印函数 print_evaluation_results 未找到，输出原始字典：")
                 import pprint; pprint.pprint(results)
            # 保存结果
            save_benchmark_results(results, config)

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except Exception as e:
        print(f"运行时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
