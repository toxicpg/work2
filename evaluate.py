# evaluate.py (V5.2 - 10-sec Tick / Event-Driven / Daily Results)
"""
评估模块 (V5.2 修复版)
修改：
1. evaluate_model 循环 config.MAX_TICKS_PER_EPISODE 次。
2. evaluate_model 调用 env.step(current_epsilon=0.0) 进行贪婪评估。
3. 移除 episode_reward += reward。
4. 在 episode 结束后从 env.reward_calculator.total_revenue 获取奖励。
5. 修复 print_evaluation_results 中的标签。
6. (V5.2) 在 evaluate_model 中按天收集 step_info。
7. (V5.2) 新增 _calculate_daily_metrics 函数计算每日指标。
8. (V5.2) evaluate_model 返回 avg_results 和 daily_results_df。
9. (V5.2) print_evaluation_results 可以打印 DataFrame。
"""
import numpy as np
import torch
from tqdm import tqdm
import random
import pandas as pd
import os
from datetime import datetime
import json
from collections import defaultdict  # 用于按天分组

try:
    from config import Config
except ImportError:
    print("警告 (evaluate.py): 无法导入 Config。")


    class Config:
        MAX_TICKS_PER_EPISODE = 8640 * 2


def print_evaluation_results(metrics, daily_df=None, title="评估结果 (V5.2)"):
    """
    打印评估结果 (V5.2 - 可选打印每日结果)
    """
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"--- 总体平均指标 (基于 {metrics.get('num_episodes', 'N/A')} 个 Episodes) ---")
    print(f"  平均总收入 (奖励): {metrics.get('avg_reward', 0.0):.2f} ± {metrics.get('std_reward', 0.0):.2f}")
    print(f"  完成率: {metrics.get('completion_rate', 0.0):.2%}")
    print(f"  取消率: {metrics.get('cancel_rate', 0.0):.2%}")
    print(f"  平均等待: {metrics.get('avg_waiting_time', 0.0):.1f}秒")
    print(f"  车辆利用率: {metrics.get('vehicle_utilization', 0.0):.2%}")
    print(f"  处理率: {metrics.get('processing_rate', 0.0):.2%}")
    print(f"  平均调度次数/Ep: {metrics.get('avg_dispatches_per_ep', 0.0):.1f}")

    if daily_df is not None and not daily_df.empty:
        print(f"\n--- 每日详细指标 ---")
        try:
            # 尝试打印更美观的 DataFrame (如果环境支持)
            from tabulate import tabulate
            print(tabulate(daily_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
        except ImportError:
            # 备用打印方式
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(daily_df.to_string(index=False, float_format='%.2f'))

    print(f"{'=' * 70}")


def _calculate_daily_metrics(daily_step_infos, start_day):
    """
    (V5.2 新增) 计算每个模拟日的指标
    Args:
        daily_step_infos (dict): {day_index: [step_info_tick1, step_info_tick2, ...]}
        start_day (int): Episode 的起始日期索引
    Returns:
        list[dict]: 包含每天指标的字典列表
    """
    daily_metrics_list = []
    # 按天排序处理
    for day_index in sorted(daily_step_infos.keys()):
        step_infos_today = daily_step_infos[day_index]

        total_matched = sum(info.get('matched_orders', 0) for info in step_infos_today)
        total_cancelled = sum(info.get('cancelled_orders', 0) for info in step_infos_today)
        all_waiting_times = [wt for info in step_infos_today for wt in info.get('waiting_times', [])]
        total_revenue = sum(info.get('revenue', 0.0) for info in step_infos_today)
        total_dispatches = sum(info.get('dispatch_total', 0) for info in step_infos_today)
        total_new_orders = sum(info.get('new_orders', 0) for info in step_infos_today)

        total_processed = total_matched + total_cancelled
        completion_rate = total_matched / total_processed if total_processed > 0 else 0.0
        cancel_rate = total_cancelled / total_processed if total_processed > 0 else 0.0
        avg_waiting_time = np.mean(all_waiting_times) if all_waiting_times else 0.0

        daily_metrics_list.append({
            'day_index': day_index,  # 相对天数 (e.g., 0, 1, ...)
            'actual_day': start_day + day_index,  # 数据集中的实际天数
            'total_revenue': round(total_revenue, 2),
            'completed_orders': total_matched,
            'cancelled_orders': total_cancelled,
            'completion_rate': round(completion_rate, 4),
            'cancel_rate': round(cancel_rate, 4),
            'avg_waiting_time': round(avg_waiting_time, 1),
            'total_dispatches': total_dispatches,
            'total_new_orders': total_new_orders,
        })
    return daily_metrics_list


def evaluate_model(trainer, env, num_episodes, config, verbose=True):
    """
    评估模型性能 (V5.2 - 输出每日结果)

    Args:
        trainer: 训练器 (仅用于设置 model.eval() 和 model.train())
        env: 环境 (V5.1 10-sec Tick 版本)
        num_episodes: 评估的episode数量
        config: 配置
        verbose: 是否打印详细信息
    Returns:
        tuple: (avg_results (dict), daily_results_df (pd.DataFrame))
               avg_results 包含所有 episodes 的平均指标
               daily_results_df 包含所有 episodes 中每一天的详细指标
    """
    print(f"\n===== [V5.2 评估] 进入 evaluate_model =====")

    if env.model is None:
        print("错误 (evaluate): 'env.model' 未设置!")
        return {}, pd.DataFrame()  # 返回空结果

    model_to_eval = env.model
    model_to_eval.eval()

    # 存储所有 episodes 的结果
    all_episode_results = {
        'rewards': [], 'completion_rates': [], 'cancel_rates': [],
        'avg_waiting_times': [], 'vehicle_utilizations': [], 'processing_rates': [],
        'dispatches_per_ep': []
    }
    # (V5.2) 存储所有 episodes 的每日详细结果
    daily_results_all_episodes = []

    pbar_desc = "评估中 (V5.2)"
    pbar = tqdm(range(num_episodes), desc=pbar_desc, unit="ep")

    for ep in pbar:
        try:
            state = env.reset()  # S_0
            episode_start_day = env.episode_start_day  # 记录起始日期
        except Exception as e:
            print(f"错误: env.reset() 在评估时失败 (Ep {ep}): {e}")
            continue

        done = False
        tick_count = 0

        # (V5.2) 用于存储当前 episode 的每日 step_info
        daily_step_infos = defaultdict(list)
        current_day_in_episode = 0  # 当前是 episode 内的第几天 (0-indexed)

        while not done and tick_count < config.MAX_TICKS_PER_EPISODE:
            try:
                # --- 与环境交互 (epsilon=0.0) ---
                next_state, _, done, info = env.step(current_epsilon=0.0)
                step_info = info.get('step_info', {})  # 获取详细信息

                # (V5.2) 按天存储 step_info
                # (env.current_day 是数据集中的绝对天数, 我们需要相对天数)
                current_day_in_episode = env.current_day - episode_start_day
                daily_step_infos[current_day_in_episode].append(step_info)

            except Exception as e:
                print(f"错误: env.step(eps=0.0) 失败在评估 Ep {ep}, Tick {tick_count}: {e}")
                next_state, _, done, info = state, 0, True, {}
                break  # 终止当前 episode

            state = next_state
            tick_count += 1

        # --- (Episode 结束) ---
        try:
            # (V5.2) 计算当前 episode 的每日指标
            daily_metrics_this_ep = _calculate_daily_metrics(daily_step_infos, episode_start_day)
            daily_results_all_episodes.extend(daily_metrics_this_ep)  # 添加到总列表

            # (V5.1) 获取当前 episode 的总体指标 (用于计算总平均值)
            summary = env.get_episode_summary()
            metrics = summary.get('reward_metrics', {})
            waiting_stats = summary.get('waiting_time_stats', {})
            env_stats = summary.get('episode_stats', {})
            reward_this_ep = env.reward_calculator.total_revenue  # 获取总收入

            # 记录总体结果
            all_episode_results['rewards'].append(reward_this_ep)
            all_episode_results['completion_rates'].append(metrics.get('completion_rate', 0.0))
            all_episode_results['cancel_rates'].append(metrics.get('cancel_rate', 0.0))
            all_episode_results['avg_waiting_times'].append(waiting_stats.get('avg_waiting_time', 0.0))
            all_episode_results['vehicle_utilizations'].append(metrics.get('vehicle_utilization', 0.0))
            all_episode_results['processing_rates'].append(metrics.get('processing_rate', 0.0))
            all_episode_results['dispatches_per_ep'].append(env_stats.get('total_dispatches', 0))

            pbar.set_description(
                f"{pbar_desc} (Ep {ep + 1}: Rev={reward_this_ep:.2f}, Cmp={metrics.get('completion_rate', 0.0):.1%})")

        except Exception as e:
            print(f"错误: 获取或记录 Ep {ep} 的 summary 失败: {e}")

    # --- (所有 Episodes 结束) ---
    model_to_eval.train()  # 恢复训练模式

    # 计算总体平均值
    avg_results = {
        'num_episodes': num_episodes,  # 添加 episode 数量
        'avg_reward': np.mean(all_episode_results['rewards']) if all_episode_results['rewards'] else 0.0,
        'std_reward': np.std(all_episode_results['rewards']) if all_episode_results['rewards'] else 0.0,
        'completion_rate': np.mean(all_episode_results['completion_rates']) if all_episode_results[
            'completion_rates'] else 0.0,
        'cancel_rate': np.mean(all_episode_results['cancel_rates']) if all_episode_results['cancel_rates'] else 0.0,
        'avg_waiting_time': np.mean(all_episode_results['avg_waiting_times']) if all_episode_results[
            'avg_waiting_times'] else 0.0,
        'vehicle_utilization': np.mean(all_episode_results['vehicle_utilizations']) if all_episode_results[
            'vehicle_utilizations'] else 0.0,
        'processing_rate': np.mean(all_episode_results['processing_rates']) if all_episode_results[
            'processing_rates'] else 0.0,
        'std_completion_rate': np.std(all_episode_results['completion_rates']) if all_episode_results[
            'completion_rates'] else 0.0,
        'std_waiting_time': np.std(all_episode_results['avg_waiting_times']) if all_episode_results[
            'avg_waiting_times'] else 0.0,
        'avg_dispatches_per_ep': np.mean(all_episode_results['dispatches_per_ep']) if all_episode_results[
            'dispatches_per_ep'] else 0.0,
        'avg_total_revenue': np.mean(all_episode_results['rewards']) if all_episode_results['rewards'] else 0.0
    }

    # (V5.2) 将每日结果转换为 DataFrame
    daily_results_df = pd.DataFrame(daily_results_all_episodes)

    print(f"===== [V5.2 评估] 结束 =====")

    # 返回平均结果和每日结果 DataFrame
    return avg_results, daily_results_df


# (如果 train.py 也导入了这个函数，确保它在这里被定义)
def save_benchmark_results(results, daily_df, config):
    """(辅助函数) 保存 benchmark 结果 (V5.2: 保存总体和每日)"""
    results_dir = os.path.join(config.LOG_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 保存总体结果 (JSON)
    overall_filename = f"benchmark_overall_{timestamp}.json"
    overall_filepath = os.path.join(results_dir, overall_filename)
    try:
        results_to_save = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                results_to_save[k] = v
            elif isinstance(v, np.generic):
                results_to_save[k] = v.item()
            else:
                results_to_save[k] = str(v)  # fallback
        with open(overall_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print(f"总体 Benchmark 结果已保存到: {overall_filepath}")
    except Exception as e:
        print(f"保存总体 Benchmark 结果失败: {e}")

    # 2. 保存每日结果 (CSV)
    if daily_df is not None and not daily_df.empty:
        daily_filename = f"benchmark_daily_{timestamp}.csv"
        daily_filepath = os.path.join(results_dir, daily_filename)
        try:
            daily_df.to_csv(daily_filepath, index=False, encoding='utf-8-sig')
            print(f"每日 Benchmark 结果已保存到: {daily_filepath}")
        except Exception as e:
            print(f"保存每日 Benchmark 结果失败: {e}")
