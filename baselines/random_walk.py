# baselines/random_walk.py
"""
Benchmark 1: Random Walk Dispatching Policy Simulation
当车辆空闲时间达到阈值时，随机选择移动到相邻网格（N, S, E, W）或停留在原地。
"""

import sys
import os
import numpy as np
import torch # 导入 torch 以设置种子
import random
from tqdm import tqdm
from datetime import datetime
import pandas as pd # 用于保存结果

# --- 调整 Python 路径并更改工作目录 ---
# 获取当前脚本 (random_walk.py) 所在的目录 (baselines)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (work2)
project_root = os.path.dirname(current_dir)

# ===== 关键修复：更改当前工作目录 =====
# 将工作目录更改为项目根目录 (work2)
# 这样所有相对路径 (如 'data/processed/') 都能正确解析
try:
    os.chdir(project_root)
    print(f"当前工作目录已更改为: {os.getcwd()}")
except Exception as e:
    print(f"更改工作目录失败: {e}")
    sys.exit(1)
# =====================================

# 将项目根目录添加到 Python 搜索路径中
if project_root not in sys.path:
    sys.path.append(project_root)
# ---------------------------------------------

# --- 导入必要的自定义模块 ---
# (现在可以安全导入，因为 CWD 正确了)
try:
    from config import Config
    from utils.data_process import DataProcessor
    from utils.graph_builder import GraphBuilder
    # 我们需要 RideHailingEnvironment 来运行模拟
    from environment_back import RideHailingEnvironment
    # (evaluate 模块中的 print_evaluation_results 可能有用)
    from evaluate import print_evaluation_results
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 random_walk.py 在 baselines 文件夹下，")
    print("并且 config.py, utils/, models/, environment.py, evaluate.py 等在上一级目录 (work2)。")
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
        # 注意：为了完全可复现，有时需要设置 torch.backends.cudnn.deterministic = True
        # 但这可能会降低性能，并且与 benchmark = True 冲突，所以这里暂时不设置
# =====================================

# --- 模拟运行函数 ---
def run_random_walk_simulation(config, num_episodes, env_data):
    """
    运行 Random Walk 策略的模拟.

    Args:
        config: 配置对象
        num_episodes: 要运行的模拟回合数
        env_data: 用于创建环境的订单数据 (例如 test_orders DataFrame)

    Returns:
        dict: 包含平均评估指标的字典, 或空字典如果出错
    """
    print(f"\n--- 开始 Random Walk Benchmark ({num_episodes} episodes) ---")
    print(f"  车辆总数: {config.TOTAL_VEHICLES}")
    print(f"  空闲阈值: {config.IDLE_THRESHOLD_SEC} 秒")
    print(f"  模拟天数/Episode: {config.EPISODE_DAYS}")

    # 1. 创建环境实例
    try:
        data_processor = DataProcessor(config)
        env = RideHailingEnvironment(config, data_processor, env_data, dispatch_policy='random_walk')
    except Exception as e:
        print(f"创建环境时出错: {e}")
        return {}

    # 2. 收集结果的列表
    all_rewards = []
    all_completion_rates = []
    all_cancel_rates = []
    all_avg_wait_times = []
    all_dispatches = [] # 记录每轮调度次数

    # 3. 运行模拟循环
    pbar_desc = "Running Random Walk"
    pbar = tqdm(range(num_episodes), desc=pbar_desc, unit="ep")
    for episode in pbar:
        try:
            state = env.reset() # 重置环境
        except Exception as e:
            print(f"错误: env.reset() 失败在 Ep {episode}: {e}")
            continue # 跳过这个 episode

        episode_reward = 0.0
        done = False
        step_count = 0

        # --- 单个 Episode 的模拟循环 ---
        while not done and step_count < config.MAX_TICKS_PER_EPISODE:
            # Random Walk 基准测试直接调用 env.step()
            # 环境内部会自动调用 _execute_random_walk_dispatch
            try:
                next_state, reward, done, info = env.step()
            except Exception as e:
                 print(f"错误: env.step() 失败在 Random Walk Ep {episode}, Step {step_count}: {e}")
                 break # 出错则结束本轮模拟

            episode_reward += reward
            state = next_state
            step_count += 1

        # --- Episode 结束 ---
        # 4. 获取并记录该 Episode 的统计结果
        try:
            summary = env.get_episode_summary()
            metrics = summary.get('reward_metrics', {})
            waiting_stats = summary.get('waiting_time_stats', {})
            env_stats = summary.get('episode_stats', {})

            all_rewards.append(episode_reward)
            all_completion_rates.append(metrics.get('completion_rate', 0.0))
            all_cancel_rates.append(metrics.get('cancel_rate', 0.0))
            all_avg_wait_times.append(waiting_stats.get('avg_waiting_time', 0.0))
            all_dispatches.append(env_stats.get('total_dispatches', 0))

            # 更新进度条描述
            pbar.set_description(f"{pbar_desc} (Ep {episode+1}: R={episode_reward:.2f}, Cmp={metrics.get('completion_rate', 0.0):.1%})")
        except Exception as e:
             print(f"错误: 获取或记录 Ep {episode} 的 summary 失败: {e}")

    # --- 计算所有 Episodes 的平均结果 ---
    avg_results = {
        'benchmark': 'Random Walk (Adjacent + Stay)',
        'num_episodes': num_episodes,
        'avg_reward': np.mean(all_rewards) if all_rewards else 0.0,
        'std_reward': np.std(all_rewards) if all_rewards else 0.0,
        'completion_rate': np.mean(all_completion_rates) if all_completion_rates else 0.0,
        'cancel_rate': np.mean(all_cancel_rates) if all_cancel_rates else 0.0,
        'avg_waiting_time': np.mean(all_avg_wait_times) if all_avg_wait_times else 0.0,
        'avg_dispatches_per_ep': np.mean(all_dispatches) if all_dispatches else 0.0,
        # 可以添加其他指标的平均值
    }

    print(f"--- Random Walk Benchmark ({num_episodes} episodes) 完成 ---")
    return avg_results

def run_last7_days_random_walk(config, env_data):
    try:
        data_processor = DataProcessor(config)
        env = RideHailingEnvironment(config, data_processor, env_data, dispatch_policy='random_walk')
    except Exception as e:
        print(f"创建环境时出错: {e}")
        return []
    available_days = 0
    try:
        available_days = env.order_generator.get_day_count()
    except Exception as e:
        print(f"无法获取数据天数: {e}")
    start0 = max(0, available_days - 7)
    daily_results = []
    for i in range(7):
        day_index = start0 + i
        try:
            env.reset(start_day=day_index)
        except Exception as e:
            print(f"重置到第 {day_index} 天失败: {e}")
            continue
        ticks = 0
        daily_infos = []
        while ticks < config.TICKS_PER_DAY:
            try:
                _, _, _, info = env.step()
            except Exception as e:
                print(f"Step 错误: {e}")
                break
            daily_infos.append(info.get('step_info', {}))
            ticks += 1
        total_matched = sum(si.get('matched_orders', 0) for si in daily_infos)
        total_cancelled = sum(si.get('cancelled_orders', 0) for si in daily_infos)
        all_waiting = [wt for si in daily_infos for wt in si.get('waiting_times', [])]
        total_revenue = sum(si.get('revenue', 0.0) for si in daily_infos)
        total_dispatches = sum(si.get('dispatch_total', 0) for si in daily_infos)
        total_new_orders = sum(si.get('new_orders', 0) for si in daily_infos)
        total_processed = total_matched + total_cancelled
        completion_rate = (total_matched / total_processed) if total_processed > 0 else 0.0
        cancel_rate = (total_cancelled / total_processed) if total_processed > 0 else 0.0
        avg_waiting_time = (float(np.mean(all_waiting)) if all_waiting else 0.0)
        daily_results.append({
            'day_index': i,
            'actual_day': day_index,
            'total_revenue': round(total_revenue, 2),
            'completed_orders': total_matched,
            'cancelled_orders': total_cancelled,
            'completion_rate': round(completion_rate, 4),
            'cancel_rate': round(cancel_rate, 4),
            'avg_waiting_time': round(avg_waiting_time, 1),
            'total_dispatches': total_dispatches,
            'total_new_orders': total_new_orders,
        })
    return daily_results

def save_daily_results(daily_results, config):
    results_dir = os.path.join(config.LOG_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)
    filename = f"random_walk_last7days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(results_dir, filename)
    try:
        df = pd.DataFrame(daily_results)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Last7Days 结果已保存到: {filepath}")
    except Exception as e:
        print(f"保存 Last7Days 结果失败: {e}")

# --- 保存结果函数 ---
def save_benchmark_results(results, config):
    """保存 benchmark 结果到 CSV"""
    # 确保保存目录存在
    results_dir = os.path.join(config.LOG_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # (文件名保持不变)
    filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(results_dir, filename)

    try:
        results_to_save = {}
        # (确保 results 中的值是标量 - 代码不变)
        for k, v in results.items():
             if isinstance(v, (int, float, str, bool)):
                  results_to_save[k] = v
             elif isinstance(v, np.generic): # Handle numpy types
                  results_to_save[k] = v.item()
             else:
                  results_to_save[k] = str(v) # Convert others to string

        df_new = pd.DataFrame([results_to_save])

        # (追加或创建 CSV - 代码不变)
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Benchmark 结果已追加/保存到: {filepath}")
    except Exception as e:
        print(f"保存 Benchmark 结果失败: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
    print("=" * 80)
    print("运行 Benchmark 1: Random Walk")
    print("=" * 80)

    # 1. 加载配置
    try:
        config = Config()
        if not Config.validate_config(): sys.exit(1) # 验证配置
        set_seed(config.SEED) # 设置随机种子
    except Exception as e:
        print(f"加载配置或设置种子时出错: {e}"); sys.exit(1)

    # (可以选择性覆盖配置，例如回合数)
    NUM_EVAL_EPISODES = 1

    try:
        # 2. 加载数据 (只需要测试集)
        print("加载数据...")
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()
        # 确保划分函数存在且有效
        if hasattr(data_processor, 'split_data_by_time'):
            _, _, test_orders = data_processor.split_data_by_time(
                all_orders, config.TRAIN_RATIO, config.VAL_RATIO
            )
        else:
             print("错误: DataProcessor 中缺少 split_data_by_time 方法")
             test_orders = all_orders # Fallback to using all data? Risky.
             # sys.exit(1)

        if test_orders.empty:
             print("错误：测试订单数据为空！无法运行模拟。")
             sys.exit(1)
        print(f"使用 {len(test_orders):,} 条测试订单运行模拟。")

        daily_results = run_last7_days_random_walk(config, test_orders)
        if daily_results:
            overall = {
                'benchmark': 'Random Walk (Adjacent + Stay) - Last7Days',
                'num_days': len(daily_results),
                'avg_completion_rate': float(np.mean([d['completion_rate'] for d in daily_results])),
                'avg_cancel_rate': float(np.mean([d['cancel_rate'] for d in daily_results])),
                'avg_waiting_time': float(np.mean([d['avg_waiting_time'] for d in daily_results])),
                'avg_revenue': float(np.mean([d['total_revenue'] for d in daily_results]))
            }
            if 'print_evaluation_results' in globals():
                print_evaluation_results(overall, title="Random Walk 最近7天汇总")
            else:
                import pprint
                pprint.pprint(overall)
            save_daily_results(daily_results, config)

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except Exception as e:
        print(f"运行时发生严重错误: {e}")
        import traceback
        traceback.print_exc()

