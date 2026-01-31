"""测试不同车辆初始分布对模型性能的影响"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.trainer import MGCNTrainer
from evaluate import evaluate_model, print_evaluation_results

def modify_vehicle_distribution(env, distribution_type='uniform', center=(0.5,0.5), std=(4.0,4.0), seed=None):
    """修改车辆初始分布"""
    vm = env.vehicle_manager
    total_vehicles, rows, cols = vm.config.TOTAL_VEHICLES, *vm.config.GRID_SIZE
    rng = np.random.default_rng(seed)

    if distribution_type == 'normal':
        center_row, center_col = center[0]*(rows-1), center[1]*(cols-1)
        row_pos = rng.normal(loc=center_row, scale=std[0], size=total_vehicles)
        col_pos = rng.normal(loc=center_col, scale=std[1], size=total_vehicles)
        row_idx = np.clip(np.rint(row_pos), 0, rows-1).astype(int)
        col_idx = np.clip(np.rint(col_pos), 0, cols-1).astype(int)
        positions = row_idx * cols + col_idx
        print(f"  ✓ 正态分布: 中心({center_row:.1f},{center_col:.1f}), 标准差({std[0]:.1f},{std[1]:.1f})")
        print(f"    覆盖 {len(np.unique(positions))}/{vm.config.NUM_GRIDS} 格")
    else:
        positions = rng.integers(0, vm.config.NUM_GRIDS, total_vehicles)
        print(f"  ✓ 均匀分布")

    for i, v in vm.vehicles.items():
        v['current_grid'] = int(positions[i])
    return positions

def test_single(trainer, config, test_orders, data_processor, distribution_type, center, std, num_episodes, seed):
    print(f"\n{'='*80}\n📊 测试: {distribution_type}", end='')
    if distribution_type == 'normal':
        print(f", 中心({center[0]:.2f},{center[1]:.2f}), std({std[0]:.1f},{std[1]:.1f})")
    else:
        print()
    print(f"{'='*80}\n")

    test_env = RideHailingEnvironment(config, data_processor, test_orders)
    modify_vehicle_distribution(test_env, distribution_type, center, std, seed)

    if hasattr(test_env, 'set_model_and_buffer'):
        test_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        test_env.model = trainer.main_net
        test_env.replay_buffer = None
        test_env.device = config.DEVICE

    avg_results, daily_results_df = evaluate_model(trainer, test_env, num_episodes, config, verbose=True)
    return avg_results, daily_results_df

def test_multiple(trainer, config, test_orders, data_processor, std_list, num_episodes):
    results_list = []

    # 基线
    print("\n" + "="*80 + "\n🔵 基线: 均匀分布\n" + "="*80)
    avg_r, _ = test_single(trainer, config, test_orders, data_processor, 'uniform', (0.5,0.5), (4.0,4.0), num_episodes, None)
    results_list.append({'distribution':'Uniform', 'std':'-', **{k:avg_r[k] for k in ['completion_rate','cancel_rate','avg_waiting_time','vehicle_utilization','avg_total_revenue']}})

    # 正态分布
    for std_val in std_list:
        print("\n" + "="*80 + f"\n🔴 正态分布: std={std_val}\n" + "="*80)
        avg_r, _ = test_single(trainer, config, test_orders, data_processor, 'normal', (0.5,0.5), (std_val,std_val), num_episodes, None)
        results_list.append({'distribution':'Normal', 'std':f'{std_val:.1f}', **{k:avg_r[k] for k in ['completion_rate','cancel_rate','avg_waiting_time','vehicle_utilization','avg_total_revenue']}})

    # 对比报告
    print("\n" + "="*80 + "\n📊 性能对比\n" + "="*80 + "\n")
    df = pd.DataFrame(results_list)

    for metric, name in [('completion_rate','完成率'),('avg_waiting_time','等待时间'),('vehicle_utilization','车辆利用率'),('avg_total_revenue','总收入')]:
        print(f"【{name}】\n" + "-"*70)
        print(f"{'分布':<12} {'std':<8} {name:>15} {'相对变化':>15}")
        print("-"*70)
        baseline = df.iloc[0][metric]
        for _, row in df.iterrows():
            val = row[metric]
            rel = ((val-baseline)/baseline*100) if baseline>0 else 0
            fmt = '.2%' if 'rate' in metric or 'utilization' in metric else '.1f'
            print(f"{row['distribution']:<12} {row['std']:<8} {val:>15{fmt}} {rel:>+14.2f}%")
        print("-"*70 + "\n")

    results_dir = f'results/vehicles_{config.TOTAL_VEHICLES}/distribution_tests/'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'distribution_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ 保存到: {csv_path}\n")
    return df

def main():
    parser = argparse.ArgumentParser(description='测试不同车辆初始分布')
    parser.add_argument('--model', type=str, default=None, help='模型路径 (默认: 自动选择最新模型)')
    parser.add_argument('--episodes', type=int, default=7, help='测试episodes')
    parser.add_argument('--distribution', type=str, default='uniform', choices=['uniform','normal'])
    parser.add_argument('--center', type=float, nargs=2, default=[0.5,0.5], help='正态分布中心 (0-1)')
    parser.add_argument('--std', type=float, default=4.0, help='标准差')
    parser.add_argument('--test-multiple', action='store_true', help='批量测试')
    parser.add_argument('--std-list', type=float, nargs='+', default=[2.0,4.0,6.0,8.0])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    print("\n"+"="*80+"\n🚗 车辆分布测试\n"+"="*80+"\n")

    config = Config()
    if not config.validate_config():
        print("❌ 配置验证失败!")
        return
    print(f"✓ 配置: 车辆{config.TOTAL_VEHICLES}, Episodes{args.episodes}")

    print(f"\n[1/4] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(all_orders, config.TRAIN_RATIO, config.VAL_RATIO)
    print(f"  ✓ 测试集: {len(test_orders)} 订单")

    print(f"\n[2/4] 加载图...")
    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()
    print(f"  ✓ 完成")

    print(f"\n[3/4] 加载模型...")
    trainer = MGCNTrainer(config, neighbor_adj, poi_adj)

    # 自动查找最新模型
    if args.model is None:
        import glob
        model_dir = f'results/vehicles_{config.TOTAL_VEHICLES}/models/'
        if not os.path.exists(model_dir):
            model_dir = 'results/models/'  # 兼容旧路径

        model_files = glob.glob(os.path.join(model_dir, '*.pt'))
        if not model_files:
            print(f"❌ 在 {model_dir} 中未找到模型文件!")
            print(f"提示: 请先训练模型或使用 --model 指定模型路径")
            return

        args.model = max(model_files, key=os.path.getmtime)
        print(f"  → 自动选择最新模型: {args.model}")

    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        return
    try:
        ep = trainer.load_checkpoint(args.model)
        print(f"  ✓ {os.path.basename(args.model)} (Episode {ep})")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"\n[4/4] 测试...")
    if args.test_multiple:
        test_multiple(trainer, config, test_orders, data_processor, args.std_list, args.episodes)
    else:
        avg_r, daily_df = test_single(trainer, config, test_orders, data_processor, args.distribution, tuple(args.center), (args.std,args.std), args.episodes, args.seed)
        print_evaluation_results(avg_r, daily_df, title=f"📊 结果 ({args.distribution}, 车辆{config.TOTAL_VEHICLES})")

    print("\n"+"="*80+"\n✅ 完成!\n"+"="*80+"\n")

if __name__ == '__main__':
    main()

