"""
Baseline 实验批量执行脚本

实验1: 均匀分布，不同车辆数 (1800, 2000, 2200)，跑最后7天
实验2: 不同初始分布 (正态σ=1,3,5,7)，2000辆车，只跑1天

记录指标: 等待时间、匹配率、完成率
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.getcwd())
from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def modify_vehicle_distribution(env, distribution_type='uniform', std=4.0, seed=None):
    """
    修改车辆初始分布

    Args:
        env: 环境实例
        distribution_type: 'uniform' 或 'normal'
        std: 正态分布的标准差
        seed: 随机种子
    """
    vm = env.vehicle_manager
    total_vehicles = vm.config.TOTAL_VEHICLES
    rows, cols = vm.config.GRID_SIZE
    rng = np.random.default_rng(seed)

    if distribution_type == 'normal':
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0

        row_pos = rng.normal(loc=center_row, scale=std, size=total_vehicles)
        col_pos = rng.normal(loc=center_col, scale=std, size=total_vehicles)

        row_idx = np.clip(np.rint(row_pos), 0, rows-1).astype(int)
        col_idx = np.clip(np.rint(col_pos), 0, cols-1).astype(int)
        positions = row_idx * cols + col_idx
    else:
        # 均匀分布
        positions = rng.integers(0, vm.config.NUM_GRIDS, total_vehicles)

    # 更新车辆位置
    for i, v in vm.vehicles.items():
        v['current_grid'] = int(positions[i])

    return positions


def run_single_day(config, test_orders, baseline_policy, distribution_type='uniform', std=4.0, day_idx=None):
    """
    运行单天测试

    Returns:
        dict: 包含等待时间、匹配率等指标
    """
    data_processor = DataProcessor(config)
    env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy=baseline_policy)

    # 确定测试天数
    try:
        day_count = env.order_generator.get_day_count()
    except Exception:
        day_count = 0

    if day_idx is None:
        day_idx = max(0, day_count - 1)

    # 重置环境
    try:
        env.reset(start_day=day_idx)
    except Exception as e:
        print(f"      ❌ 环境重置失败: {e}")
        return None

    # 修改车辆分布
    modify_vehicle_distribution(env, distribution_type, std, config.SEED)

    # 运行一天
    ticks = 0
    daily_infos = []

    while ticks < config.TICKS_PER_DAY:
        try:
            _, _, _, info = env.step()
            daily_infos.append(info.get('step_info', {}))
        except Exception as e:
            print(f"      ❌ Step失败 (tick {ticks}): {e}")
            break
        ticks += 1

    # 汇总结果
    if not daily_infos:
        return None

    total_new_orders = sum(si.get('new_orders', 0) for si in daily_infos)
    total_matched = sum(si.get('matched_orders', 0) for si in daily_infos)
    total_completed = sum(si.get('completed_orders', 0) for si in daily_infos)
    total_cancelled = sum(si.get('cancelled_orders', 0) for si in daily_infos)
    all_waiting = [wt for si in daily_infos for wt in si.get('waiting_times', [])]
    total_revenue = sum(si.get('revenue', 0.0) for si in daily_infos)

    den = total_new_orders if total_new_orders > 0 else 1
    match_rate = total_matched / den
    completion_rate = total_completed / den
    cancel_rate = total_cancelled / den
    avg_waiting_time = float(np.mean(all_waiting)) if all_waiting else 0.0

    return {
        'day_index': day_idx,
        'total_new_orders': total_new_orders,
        'matched_orders': total_matched,
        'completed_orders': total_completed,
        'cancelled_orders': total_cancelled,
        'match_rate': match_rate,
        'completion_rate': completion_rate,
        'cancel_rate': cancel_rate,
        'avg_waiting_time': avg_waiting_time,
        'total_revenue': total_revenue,
    }


def run_last_7_days(config, test_orders, baseline_policy, distribution_type='uniform', std=4.0):
    """
    运行最后7天

    Returns:
        list[dict]: 7天的结果
    """
    data_processor = DataProcessor(config)
    env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy=baseline_policy)

    try:
        day_count = env.order_generator.get_day_count()
    except Exception:
        day_count = 0

    start_day = max(0, day_count - 7)
    days_to_run = list(range(start_day, min(start_day + 7, day_count)))

    daily_results = []

    for day_idx in days_to_run:
        # 重置环境
        try:
            env.reset(start_day=day_idx)
        except Exception as e:
            print(f"      ❌ Day {day_idx} 重置失败: {e}")
            continue

        # 修改车辆分布（每天都重新分布）
        modify_vehicle_distribution(env, distribution_type, std, config.SEED + day_idx)

        # 运行一天
        ticks = 0
        daily_infos = []

        while ticks < config.TICKS_PER_DAY:
            try:
                _, _, _, info = env.step()
                daily_infos.append(info.get('step_info', {}))
            except Exception as e:
                print(f"      ❌ Day {day_idx} Step失败: {e}")
                break
            ticks += 1

        # 汇总当日结果
        if not daily_infos:
            continue

        total_new_orders = sum(si.get('new_orders', 0) for si in daily_infos)
        total_matched = sum(si.get('matched_orders', 0) for si in daily_infos)
        total_completed = sum(si.get('completed_orders', 0) for si in daily_infos)
        total_cancelled = sum(si.get('cancelled_orders', 0) for si in daily_infos)
        all_waiting = [wt for si in daily_infos for wt in si.get('waiting_times', [])]
        total_revenue = sum(si.get('revenue', 0.0) for si in daily_infos)

        den = total_new_orders if total_new_orders > 0 else 1
        match_rate = total_matched / den
        completion_rate = total_completed / den
        cancel_rate = total_cancelled / den
        avg_waiting_time = float(np.mean(all_waiting)) if all_waiting else 0.0

        daily_results.append({
            'day_index': day_idx,
            'total_new_orders': total_new_orders,
            'matched_orders': total_matched,
            'completed_orders': total_completed,
            'cancelled_orders': total_cancelled,
            'match_rate': match_rate,
            'completion_rate': completion_rate,
            'cancel_rate': cancel_rate,
            'avg_waiting_time': avg_waiting_time,
            'total_revenue': total_revenue,
        })

    return daily_results


def experiment_1_uniform_multiple_vehicles():
    """
    实验1: 均匀分布，不同车辆数 (1800, 2000, 2200)，跑最后7天
    """
    print("\n" + "="*80)
    print("实验1: 均匀分布 × 不同车辆数 (最后7天)")
    print("="*80)

    baselines = [
        ('random_walk', 'Random Walk'),
        ('random_dispatching', 'Random Dispatch'),
    ]

    vehicle_counts = [1800, 2000, 2200]

    all_results = []

    for num_vehicles in vehicle_counts:
        print(f"\n{'─'*80}")
        print(f"车辆数: {num_vehicles}")
        print(f"{'─'*80}")

        # 加载配置
        config = Config()
        config.TOTAL_VEHICLES = num_vehicles
        set_seed(config.SEED)

        # 加载数据
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()

        if hasattr(data_processor, 'split_data_by_time'):
            _, _, test_orders = data_processor.split_data_by_time(
                all_orders, config.TRAIN_RATIO, config.VAL_RATIO
            )
        else:
            test_orders = all_orders

        for baseline_policy, baseline_name in baselines:
            print(f"  → {baseline_name}...")

            daily_results = run_last_7_days(
                config, test_orders, baseline_policy,
                distribution_type='uniform', std=0.0
            )

            if daily_results:
                # 添加实验信息
                for result in daily_results:
                    result['experiment'] = 'Exp1_Uniform'
                    result['baseline'] = baseline_name
                    result['num_vehicles'] = num_vehicles
                    result['distribution'] = '均匀'
                    result['std'] = 0.0
                    all_results.append(result)

                # 打印汇总
                avg_match = np.mean([r['match_rate'] for r in daily_results])
                avg_wait = np.mean([r['avg_waiting_time'] for r in daily_results])
                avg_comp = np.mean([r['completion_rate'] for r in daily_results])

                print(f"      完成率={avg_comp:.2%}, 匹配率={avg_match:.2%}, 等待={avg_wait:.1f}s")

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_dir = 'results/baseline_experiments/'
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, f'exp1_uniform_vehicles_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 实验1结果已保存: {csv_path}")

        return df

    return None


def experiment_2_normal_distributions():
    """
    实验2: 不同初始分布 (正态σ=1,3,5,7)，2000辆车，只跑1天
    """
    print("\n" + "="*80)
    print("实验2: 不同初始分布 × 2000辆车 (只跑1天)")
    print("="*80)

    baselines = [
        ('random_walk', 'Random Walk'),
        ('random_dispatching', 'Random Dispatch'),
    ]

    std_values = [1.0, 3.0, 5.0, 7.0]

    all_results = []

    # 加载配置
    config = Config()
    config.TOTAL_VEHICLES = 2000
    set_seed(config.SEED)

    # 加载数据
    print("\n加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()

    if hasattr(data_processor, 'split_data_by_time'):
        _, _, test_orders = data_processor.split_data_by_time(
            all_orders, config.TRAIN_RATIO, config.VAL_RATIO
        )
    else:
        test_orders = all_orders

    print(f"✓ 测试集订单数: {len(test_orders):,}")

    for std_val in std_values:
        print(f"\n{'─'*80}")
        print(f"正态分布: σ = {std_val}")
        print(f"{'─'*80}")

        for baseline_policy, baseline_name in baselines:
            print(f"  → {baseline_name}...")

            result = run_single_day(
                config, test_orders, baseline_policy,
                distribution_type='normal', std=std_val, day_idx=None
            )

            if result:
                # 添加实验信息
                result['experiment'] = 'Exp2_Distribution'
                result['baseline'] = baseline_name
                result['num_vehicles'] = 2000
                result['distribution'] = f'正态(σ={std_val})'
                result['std'] = std_val
                all_results.append(result)

                print(f"      完成率={result['completion_rate']:.2%}, "
                      f"匹配率={result['match_rate']:.2%}, "
                      f"等待={result['avg_waiting_time']:.1f}s")

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_dir = 'results/baseline_experiments/'
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, f'exp2_distributions_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 实验2结果已保存: {csv_path}")

        return df

    return None


def print_summary_table(df1, df2):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("📊 实验结果汇总")
    print("="*80)

    # 实验1汇总
    if df1 is not None:
        print("\n【实验1: 均匀分布 × 不同车辆数】")
        print("-"*80)

        for baseline in df1['baseline'].unique():
            print(f"\n{baseline}:")
            sub_df = df1[df1['baseline'] == baseline]

            summary = sub_df.groupby('num_vehicles').agg({
                'match_rate': 'mean',
                'completion_rate': 'mean',
                'avg_waiting_time': 'mean',
                'cancel_rate': 'mean'
            }).reset_index()

            print(f"{'车辆数':<10} {'匹配率':>10} {'完成率':>10} {'等待时间':>12} {'取消率':>10}")
            print("-"*60)
            for _, row in summary.iterrows():
                print(f"{row['num_vehicles']:<10} "
                      f"{row['match_rate']:>9.2%} "
                      f"{row['completion_rate']:>9.2%} "
                      f"{row['avg_waiting_time']:>11.1f}s "
                      f"{row['cancel_rate']:>9.2%}")

    # 实验2汇总
    if df2 is not None:
        print("\n" + "-"*80)
        print("【实验2: 不同初始分布 × 2000辆车】")
        print("-"*80)

        for baseline in df2['baseline'].unique():
            print(f"\n{baseline}:")
            sub_df = df2[df2['baseline'] == baseline]

            print(f"{'分布':<15} {'匹配率':>10} {'完成率':>10} {'等待时间':>12} {'取消率':>10}")
            print("-"*60)
            for _, row in sub_df.iterrows():
                print(f"{row['distribution']:<15} "
                      f"{row['match_rate']:>9.2%} "
                      f"{row['completion_rate']:>9.2%} "
                      f"{row['avg_waiting_time']:>11.1f}s "
                      f"{row['cancel_rate']:>9.2%}")

    print("\n" + "="*80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Baseline实验批量执行')
    parser.add_argument('--exp', type=str, choices=['1', '2', 'all'], default='all',
                        help='选择实验: 1=均匀分布×车辆数, 2=不同分布×2000车, all=全部')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚗 Baseline 实验批量执行")
    print("="*80)
    print("\n实验配置:")
    print("  实验1: 均匀分布 × 车辆数(1800,2000,2200) × 最后7天")
    print("  实验2: 正态分布(σ=1,3,5,7) × 2000辆车 × 1天")
    print("  Baseline: Random Walk + Random Dispatch")
    print("  记录指标: 等待时间、匹配率、完成率")
    print("="*80)

    df1, df2 = None, None

    # 执行实验
    if args.exp in ['1', 'all']:
        df1 = experiment_1_uniform_multiple_vehicles()

    if args.exp in ['2', 'all']:
        df2 = experiment_2_normal_distributions()

    # 打印汇总
    print_summary_table(df1, df2)

    print("\n" + "="*80)
    print("✅ 所有实验完成!")
    print("="*80)
    print("\n结果文件:")
    print("  results/baseline_experiments/exp1_uniform_vehicles_*.csv")
    print("  results/baseline_experiments/exp2_distributions_*.csv")
    print("")


if __name__ == '__main__':
    main()

