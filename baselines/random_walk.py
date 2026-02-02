# baselines/random_walk.py
"""
Benchmark 1: Random Walk Dispatching Policy Simulation
当车辆空闲时间达到阈值时，随机选择移动到相邻网格（N, S, E, W）或停留在原地。
使用主环境 RideHailingEnvironment，并强制 DISPATCH_MODE='random_walk'。

本版本修复：
1) 日指标分母用 total_new_orders（避免虚高）
2) 加入“日终清算”(flush)：跑完一天后停止生成新订单，继续推进若干 tick，直到 pending 清空或达到上限
3) 输出 pending_end_of_day / pending_after_flush / unprocessed 等字段，便于排查
"""

import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- 调整 Python 路径并更改工作目录 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# ===== 关键修复：更改当前工作目录 =====
try:
    os.chdir(project_root)
    print(f"当前工作目录已更改为: {os.getcwd()}")
except Exception as e:
    print(f"更改工作目录失败: {e}")
    sys.exit(1)
# =====================================

if project_root not in sys.path:
    sys.path.append(project_root)

# --- 导入必要的自定义模块 ---
try:
    from config import Config
    from utils.data_process import DataProcessor
    from environment_baseline import BaselineEnvironment
    from evaluate import print_evaluation_results
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 random_walk.py 在 baselines 文件夹下，")
    print("并且 config.py, utils/, environment.py, evaluate.py 等在上一级目录。")
    sys.exit(1)


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(config, env_data):
    """统一创建Baseline环境，并强制使用 random_walk 模式（双保险）。"""
    config.DISPATCH_MODE = 'random_walk'
    data_processor = DataProcessor(config)
    env = BaselineEnvironment(config, data_processor, env_data, dispatch_policy='random_walk')
    return env


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _flush_day_no_new_orders(env, config, max_flush_ticks=None):
    """
    日终清算：停止生成新订单，继续推进 tick，让 pending 有机会完成/取消。
    不修改 environment.py，通过 monkey-patch env._load_orders_for_tick() 来实现“今日无新单”。

    Returns:
        flush_infos: list[step_info]  # flush 阶段每 tick 的 step_info
        flush_ticks_run: int
        pending_after_flush: int
    """
    if max_flush_ticks is None:
        # 默认最多再跑 2 小时的 tick：2*3600 / tick_sec
        tick_sec = getattr(config, 'TICK_DURATION_SEC', 10)
        max_flush_ticks = max(1, int(2 * 3600 / max(1, tick_sec)))

    # 保存原方法
    original_loader = getattr(env, "_load_orders_for_tick", None)

    def _no_new_orders_loader():
        return []

    # 打补丁
    if original_loader is not None:
        env._load_orders_for_tick = _no_new_orders_loader

    flush_infos = []
    flush_ticks = 0

    try:
        while flush_ticks < max_flush_ticks:
            pending_now = len(getattr(env, "pending_orders", []))
            if pending_now <= 0:
                break

            _, _, done, info = env.step()
            flush_infos.append(info.get("step_info", {}))
            flush_ticks += 1

            # 理论上 flush 不会触发 episode done（除非 MAX_TICKS_PER_EPISODE 太小）
            if done:
                break
    finally:
        # 还原方法
        if original_loader is not None:
            env._load_orders_for_tick = original_loader

    pending_after = len(getattr(env, "pending_orders", []))
    return flush_infos, flush_ticks, pending_after


def run_random_walk_simulation(config, num_episodes, env_data):
    """
    运行 Random Walk 策略的模拟（episode 口径，沿用 env.get_episode_summary())。
    这部分保持你原逻辑，但使用主环境。
    """
    print(f"\n--- 开始 Random Walk Benchmark ({num_episodes} episodes) ---")
    print(f"  车辆总数: {config.TOTAL_VEHICLES}")
    print(f"  空闲阈值: {config.IDLE_THRESHOLD_SEC} 秒")
    print(f"  模拟天数/Episode: {config.EPISODE_DAYS}")
    print(f"  DISPATCH_MODE: {getattr(config, 'DISPATCH_MODE', None)} -> 强制设置为 random_walk")

    try:
        env = _make_env(config, env_data)
    except Exception as e:
        print(f"创建环境时出错: {e}")
        return {}

    all_rewards = []
    all_completion_rates = []
    all_cancel_rates = []
    all_avg_wait_times = []
    all_dispatches = []

    pbar_desc = "Running Random Walk"
    pbar = tqdm(range(num_episodes), desc=pbar_desc, unit="ep")

    for episode in pbar:
        try:
            env.reset()
        except Exception as e:
            print(f"错误: env.reset() 失败在 Ep {episode}: {e}")
            continue

        episode_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < config.MAX_TICKS_PER_EPISODE:
            try:
                _, reward, done, _ = env.step()
            except Exception as e:
                print(f"错误: env.step() 失败在 Random Walk Ep {episode}, Step {step_count}: {e}")
                break
            episode_reward += reward
            step_count += 1

        try:
            summary = env.get_episode_summary()
            metrics = summary.get('reward_metrics', {})
            waiting_stats = summary.get('waiting_time_stats', {})
            env_stats = summary.get('episode_stats', {})

            all_rewards.append(episode_reward)
            all_completion_rates.append(_safe_float(metrics.get('completion_rate', 0.0)))
            all_cancel_rates.append(_safe_float(metrics.get('cancel_rate', 0.0)))
            all_avg_wait_times.append(_safe_float(waiting_stats.get('avg_waiting_time', 0.0)))
            all_dispatches.append(_safe_float(env_stats.get('total_dispatches', 0.0)))

            pbar.set_description(
                f"{pbar_desc} (Ep {episode+1}: R={episode_reward:.2f}, Cmp={_safe_float(metrics.get('completion_rate', 0.0)):.1%})"
            )
        except Exception as e:
            print(f"错误: 获取或记录 Ep {episode} 的 summary 失败: {e}")

    avg_results = {
        'benchmark': 'Random Walk (Adjacent + Stay)',
        'num_episodes': int(num_episodes),
        'avg_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
        'std_reward': float(np.std(all_rewards)) if all_rewards else 0.0,
        'completion_rate': float(np.mean(all_completion_rates)) if all_completion_rates else 0.0,
        'cancel_rate': float(np.mean(all_cancel_rates)) if all_cancel_rates else 0.0,
        'avg_waiting_time': float(np.mean(all_avg_wait_times)) if all_avg_wait_times else 0.0,
        'avg_dispatches_per_ep': float(np.mean(all_dispatches)) if all_dispatches else 0.0,
    }

    print(f"--- Random Walk Benchmark ({num_episodes} episodes) 完成 ---")
    return avg_results


def run_last7_days_random_walk(config, env_data, flush=True, max_flush_ticks=None):
    """
    跑测试集最后 7 天：
    - 先跑满 config.TICKS_PER_DAY
    - 然后可选 flush：停止新订单，继续推进若干 tick，直到 pending 清空或达到上限
    """
    try:
        env = _make_env(config, env_data)
    except Exception as e:
        print(f"创建环境时出错: {e}")
        return []

    try:
        available_days = env.order_generator.get_day_count()
    except Exception as e:
        print(f"无法获取数据天数: {e}")
        available_days = 0

    start0 = max(0, available_days - 7)
    daily_results = []

    for i in range(7):
        day_index = start0 + i
        try:
            env.reset(start_day=day_index)  # 依赖 environment.py 支持 start_day
        except Exception as e:
            print(f"重置到第 {day_index} 天失败: {e}")
            continue

        # 1) 正常跑一天
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

        pending_end_of_day = len(getattr(env, "pending_orders", []))

        # 2) flush（不再生成新订单）
        flush_infos = []
        flush_ticks_run = 0
        pending_after_flush = pending_end_of_day

        if flush and pending_end_of_day > 0:
            flush_infos, flush_ticks_run, pending_after_flush = _flush_day_no_new_orders(
                env, config, max_flush_ticks=max_flush_ticks
            )

        # 3) 汇总：把 flush 阶段也算进“当日处理结果”
        combined_infos = daily_infos + flush_infos

        total_matched = sum(_safe_int(si.get('matched_orders', 0)) for si in combined_infos)
        total_completed = sum(_safe_int(si.get('completed_orders', 0)) for si in combined_infos)
        total_cancelled = sum(_safe_int(si.get('cancelled_orders', 0)) for si in combined_infos)
        all_waiting = [wt for si in combined_infos for wt in si.get('waiting_times', [])]
        total_revenue = sum(_safe_float(si.get('revenue', 0.0)) for si in combined_infos)
        total_dispatches = sum(_safe_int(si.get('dispatch_total', 0)) for si in combined_infos)

        # 注意：新订单只来自“正常一天”阶段，flush 阶段被禁止生成
        total_new_orders = sum(_safe_int(si.get('new_orders', 0)) for si in daily_infos)

        # ✅ 关键修复：分母用 total_new_orders（避免虚高）
        den = total_new_orders
        match_rate = (total_matched / den) if den > 0 else 0.0
        completion_rate = (total_completed / den) if den > 0 else 0.0
        cancel_rate = (total_cancelled / den) if den > 0 else 0.0

        # 未处理（按生成口径）
        unprocessed_by_gen = den - total_completed - total_cancelled
        # 注意：unprocessed_by_gen 可能为负（如果你的 env 在 flush 阶段仍然会生成新订单，或者统计口径不一致）
        # 这里做个 clamp，避免出现难看的负数
        unprocessed_by_gen = int(max(0, unprocessed_by_gen))

        avg_waiting_time = float(np.mean(all_waiting)) if all_waiting else 0.0

        daily_results.append({
            'day_index': int(i),
            'actual_day': int(day_index),

            'total_new_orders': int(total_new_orders),
            'matched_orders': int(total_matched),
            'completed_orders': int(total_completed),
            'cancelled_orders': int(total_cancelled),

            'match_rate': round(match_rate, 4),
            'completion_rate': round(completion_rate, 4),
            'cancel_rate': round(cancel_rate, 4),

            'avg_waiting_time': round(avg_waiting_time, 1),
            'total_revenue': round(total_revenue, 2),
            'total_dispatches': int(total_dispatches),

            # 诊断字段
            'pending_end_of_day': int(pending_end_of_day),
            'pending_after_flush': int(pending_after_flush),
            'flush_ticks_run': int(flush_ticks_run),
            'unprocessed_by_gen': int(unprocessed_by_gen),
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


def save_benchmark_results(results, config):
    """保存 benchmark 结果到 CSV"""
    results_dir = os.path.join(config.LOG_SAVE_PATH, 'results')
    os.makedirs(results_dir, exist_ok=True)

    filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(results_dir, filename)

    try:
        results_to_save = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                results_to_save[k] = v
            elif isinstance(v, np.generic):
                results_to_save[k] = v.item()
            else:
                results_to_save[k] = str(v)

        df_new = pd.DataFrame([results_to_save])

        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Benchmark 结果已追加/保存到: {filepath}")
    except Exception as e:
        print(f"保存 Benchmark 结果失败: {e}")


if __name__ == '__main__':
    print("=" * 80)
    print("运行 Benchmark 1: Random Walk")
    print("=" * 80)

    try:
        config = Config()
        if not Config.validate_config():
            sys.exit(1)
        set_seed(config.SEED)
    except Exception as e:
        print(f"加载配置或设置种子时出错: {e}")
        sys.exit(1)

    try:
        print("加载数据...")
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()

        if hasattr(data_processor, 'split_data_by_time'):
            _, _, test_orders = data_processor.split_data_by_time(
                all_orders, config.TRAIN_RATIO, config.VAL_RATIO
            )
        else:
            print("错误: DataProcessor 中缺少 split_data_by_time 方法")
            test_orders = all_orders

        if test_orders.empty:
            print("错误：测试订单数据为空！无法运行模拟。")
            sys.exit(1)

        print(f"使用 {len(test_orders):,} 条测试订单运行模拟。")

        # 你可以在这里调 flush 上限：
        # max_flush_ticks=None 使用默认（2小时）
        daily_results = run_last7_days_random_walk(config, test_orders, flush=True, max_flush_ticks=None)

        if daily_results:
            overall = {
                'benchmark': 'Random Walk (Adjacent + Stay) - Last7Days (FixedDen + Flush)',
                'num_days': len(daily_results),
                'avg_completion_rate': float(np.mean([d['completion_rate'] for d in daily_results])),
                'avg_cancel_rate': float(np.mean([d['cancel_rate'] for d in daily_results])),
                'avg_waiting_time': float(np.mean([d['avg_waiting_time'] for d in daily_results])),
                'avg_revenue': float(np.mean([d['total_revenue'] for d in daily_results])),
                'avg_unprocessed_by_gen': float(np.mean([d.get('unprocessed_by_gen', 0) for d in daily_results])),
                'avg_pending_end_of_day': float(np.mean([d.get('pending_end_of_day', 0) for d in daily_results])),
                'avg_pending_after_flush': float(np.mean([d.get('pending_after_flush', 0) for d in daily_results])),
                'avg_flush_ticks_run': float(np.mean([d.get('flush_ticks_run', 0) for d in daily_results])),
            }

            if 'print_evaluation_results' in globals():
                print_evaluation_results(overall, title="Random Walk 最近7天汇总（修正口径 + 日终清算）")
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