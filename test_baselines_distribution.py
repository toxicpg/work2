"""
测试所有 Baseline 算法在不同车辆初始分布下的性能
只运行1天，快速对比结果
支持分布：均匀 + 正态(std=1,3,5,7)
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
        std: 正态分布的标准差（二维各向同性）
        seed: 随机种子
    """
    vm = env.vehicle_manager
    total_vehicles = vm.config.TOTAL_VEHICLES
    rows, cols = vm.config.GRID_SIZE
    rng = np.random.default_rng(seed)

    if distribution_type == 'normal':
        # 中心点：网格中心
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0

        # 生成二维正态分布
        row_pos = rng.normal(loc=center_row, scale=std, size=total_vehicles)
        col_pos = rng.normal(loc=center_col, scale=std, size=total_vehicles)

        # 裁剪到网格范围
        row_idx = np.clip(np.rint(row_pos), 0, rows-1).astype(int)
        col_idx = np.clip(np.rint(col_pos), 0, cols-1).astype(int)
        positions = row_idx * cols + col_idx

        unique_grids = len(np.unique(positions))
        print(f"    ✓ 正态分布(std={std:.1f}): 中心({center_row:.1f},{center_col:.1f}), 覆盖{unique_grids}/{vm.config.NUM_GRIDS}格")
    else:
        # 均匀分布
        positions = rng.integers(0, vm.config.NUM_GRIDS, total_vehicles)
        print(f"    ✓ 均匀分布: 覆盖{len(np.unique(positions))}/{vm.config.NUM_GRIDS}格")

    # 更新车辆位置
    for i, v in vm.vehicles.items():
        v['current_grid'] = int(positions[i])

    return positions


def run_single_day_baseline(config, test_orders, baseline_policy, distribution_type='uniform', std=4.0, test_day_idx=None):
    """
    运行单天的 baseline 测试

    Args:
        config: 配置对象
        test_orders: 测试集订单数据
        baseline_policy: 'random_walk', 'random_dispatching', 'none'
        distribution_type: 'uniform' 或 'normal'
        std: 正态分布标准差
        test_day_idx: 指定测试哪一天（None=最后一天）

    Returns:
        dict: 单日结果
    """
    # 创建环境
    data_processor = DataProcessor(config)
    env = BaselineEnvironment(config, data_processor, test_orders, dispatch_policy=baseline_policy)

    # 确定测试天数
    try:
        day_count = env.order_generator.get_day_count()
    except Exception:
        day_count = 0

    if test_day_idx is None:
        test_day_idx = max(0, day_count - 1)  # 默认最后一天

    # 重置环境到指定天
    try:
        env.reset(start_day=test_day_idx)
    except Exception as e:
        print(f"    ❌ 环境重置失败: {e}")
        return None

    # 修改车辆分布
    seed = config.SEED
    modify_vehicle_distribution(env, distribution_type, std, seed)

    # 运行一天
    ticks = 0
    daily_infos = []

    while ticks < config.TICKS_PER_DAY:
        try:
            _, _, _, info = env.step()
            daily_infos.append(info.get('step_info', {}))
        except Exception as e:
            print(f"    ❌ Step失败 (tick {ticks}): {e}")
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
    total_dispatches = sum(si.get('dispatch_total', 0) for si in daily_infos)

    # 计算指标
    den = total_new_orders if total_new_orders > 0 else 1
    completion_rate = total_completed / den
    cancel_rate = total_cancelled / den
    avg_waiting_time = float(np.mean(all_waiting)) if all_waiting else 0.0

    return {
        'day_index': test_day_idx,
        'total_new_orders': total_new_orders,
        'completed_orders': total_completed,
        'cancelled_orders': total_cancelled,
        'completion_rate': completion_rate,
        'cancel_rate': cancel_rate,
        'avg_waiting_time': avg_waiting_time,
        'total_revenue': total_revenue,
        'total_dispatches': total_dispatches,
    }


def test_all_baselines_all_distributions(config, test_orders, test_day_idx=None):
    """
    测试所有 baseline 在所有分布下的性能

    Returns:
        DataFrame: 完整结果表
    """
    # 定义测试配置
    baselines = [
        ('random_walk', 'Random Walk'),
        ('random_dispatching', 'Random Dispatch'),
        # 'none' 可用于测试无调度的基准
    ]

    distributions = [
        ('uniform', 0.0, '均匀'),
        ('normal', 1.0, '正态(σ=1)'),
        ('normal', 3.0, '正态(σ=3)'),
        ('normal', 5.0, '正态(σ=5)'),
        ('normal', 7.0, '正态(σ=7)'),
    ]

    all_results = []

    print("\n" + "="*80)
    print("🚗 Baseline 车辆分布对比测试（只跑1天）")
    print("="*80)
    print(f"车辆数: {config.TOTAL_VEHICLES}")
    print(f"测试日: Day {test_day_idx if test_day_idx is not None else '(最后一天)'}")
    print("="*80 + "\n")

    # 遍历所有组合
    total_tests = len(baselines) * len(distributions)
    pbar = tqdm(total=total_tests, desc="运行测试", unit="test")

    for baseline_policy, baseline_name in baselines:
        for dist_type, std_val, dist_label in distributions:
            pbar.set_description(f"{baseline_name} - {dist_label}")

            result = run_single_day_baseline(
                config, test_orders, baseline_policy, dist_type, std_val, test_day_idx
            )

            if result:
                result['baseline'] = baseline_name
                result['distribution'] = dist_label
                result['distribution_type'] = dist_type
                result['std'] = std_val
                all_results.append(result)

            pbar.update(1)

    pbar.close()

    # 转为 DataFrame
    if not all_results:
        print("\n❌ 没有测试结果！")
        return None

    df = pd.DataFrame(all_results)
    return df


def print_comparison_table(df):
    """打印对比表格"""
    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80)

    # 按 baseline 分组打印
    for baseline in df['baseline'].unique():
        sub_df = df[df['baseline'] == baseline].copy()

        print(f"\n【{baseline}】")
        print("-"*80)
        print(f"{'分布':<15} {'完成率':>10} {'取消率':>10} {'等待时间':>12} {'收入':>12} {'调度次数':>10}")
        print("-"*80)

        for _, row in sub_df.iterrows():
            print(f"{row['distribution']:<15} "
                  f"{row['completion_rate']:>9.2%} "
                  f"{row['cancel_rate']:>9.2%} "
                  f"{row['avg_waiting_time']:>11.1f}s "
                  f"{row['total_revenue']:>11.1f} "
                  f"{row['total_dispatches']:>10}")

        print("-"*80)

    # 分布间对比（以均匀为基准）
    print(f"\n【分布影响分析】")
    print("-"*80)
    print(f"{'Baseline':<20} {'指标':<15} {'均匀':<10} {'σ=1':<10} {'σ=3':<10} {'σ=5':<10} {'σ=7':<10}")
    print("-"*80)

    metrics = [
        ('completion_rate', '完成率', '.2%'),
        ('avg_waiting_time', '等待时间', '.1f'),
    ]

    for baseline in df['baseline'].unique():
        sub_df = df[df['baseline'] == baseline]

        for metric, label, fmt in metrics:
            row_str = f"{baseline:<20} {label:<15}"

            for dist in ['均匀', '正态(σ=1)', '正态(σ=3)', '正态(σ=5)', '正态(σ=7)']:
                val = sub_df[sub_df['distribution'] == dist][metric].values
                if len(val) > 0:
                    row_str += f" {val[0]:>{fmt}}"
                else:
                    row_str += f" {'N/A':>10}"

            print(row_str)

        print("-"*80)


def save_results(df, config):
    """保存结果到文件"""
    results_dir = f'results/vehicles_{config.TOTAL_VEHICLES}/distribution_tests/'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存 CSV
    csv_path = os.path.join(results_dir, f'baseline_distribution_comparison_{timestamp}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: {csv_path}")

    # 保存格式化的报告
    report_path = os.path.join(results_dir, f'baseline_distribution_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Baseline 车辆分布对比测试报告\n")
        f.write("="*80 + "\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"车辆数: {config.TOTAL_VEHICLES}\n")
        f.write(f"测试天数: 1 天\n")
        f.write("="*80 + "\n\n")

        # 写入完整表格
        f.write(df.to_string(index=False))

    print(f"✓ 报告已保存到: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试所有Baseline在不同车辆分布下的性能')
    parser.add_argument('--day', type=int, default=None, help='指定测试天数索引（默认：最后一天）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（默认：使用config中的种子）')
    args = parser.parse_args()

    # 加载配置
    config = Config()
    if not config.validate_config():
        print("❌ 配置验证失败!")
        return

    # 设置种子
    if args.seed is not None:
        config.SEED = args.seed
    set_seed(config.SEED)

    # 加载数据
    print("加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()

    if hasattr(data_processor, 'split_data_by_time'):
        _, _, test_orders = data_processor.split_data_by_time(
            all_orders, config.TRAIN_RATIO, config.VAL_RATIO
        )
    else:
        test_orders = all_orders

    if test_orders.empty:
        print("❌ 测试订单数据为空！")
        return

    print(f"✓ 测试集订单数: {len(test_orders):,}")

    # 运行测试
    df = test_all_baselines_all_distributions(config, test_orders, test_day_idx=args.day)

    if df is not None:
        # 打印对比表
        print_comparison_table(df)

        # 保存结果
        save_results(df, config)

        print("\n" + "="*80)
        print("✅ 所有测试完成!")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()

