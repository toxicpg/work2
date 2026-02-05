"""
测试训练好的 CNN-D3QN 消融模型 - 在不同车辆数和初始分布下测试

使用方法:

1. 测试单个配置:
   python test_cnn_ablation.py --model results/vehicles_1800/ablation/cnn_model.pt --vehicles 2000

2. 测试不同初始分布:
   python test_cnn_ablation.py --model results/vehicles_1800/ablation/cnn_model.pt \
                                --vehicles 2000 \
                                --distribution normal \
                                --std 5.0

3. 批量测试多个车辆数:
   python test_cnn_ablation.py --model results/vehicles_1800/ablation/cnn_model.pt \
                                --test-all-vehicles \
                                --vehicle-list 1800 2000 2200

4. 批量测试多个分布:
   python test_cnn_ablation.py --model results/vehicles_1800/ablation/cnn_model.pt \
                                --vehicles 2000 \
                                --test-all-distributions \
                                --std-list 3 5 7

5. 全面测试 (所有车辆数 × 所有分布):
   python test_cnn_ablation.py --model results/vehicles_1800/ablation/cnn_model.pt \
                                --comprehensive-test
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
from datetime import datetime
import json
import argparse

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.ablation_trainer import AblationMGCNTrainer
from evaluate import evaluate_model


def test_single_config(model_path, ablation_type, num_vehicles, distribution_type='uniform',
                      distribution_params=None, num_test_episodes=7, seed=None):
    """
    在单个配置下测试模型

    Args:
        model_path: 模型检查点路径
        ablation_type: 消融类型 ('cnn', 'no_mgcn', 'full_model', 等)
        num_vehicles: 车辆数
        distribution_type: 初始分布类型 ('uniform' 或 'normal')
        distribution_params: 分布参数 (对于normal分布: {'center': (0.5, 0.5), 'std': (5.0, 5.0)})
        num_test_episodes: 测试episode数 (默认7天)
        seed: 随机种子
    """

    print(f"\n{'='*80}")
    print(f"测试配置: CNN-D3QN")
    print(f"{'='*80}")
    print(f"  模型路径: {model_path}")
    print(f"  消融类型: {ablation_type}")
    print(f"  车辆数: {num_vehicles}")
    print(f"  初始分布: {distribution_type}")
    if distribution_type == 'normal' and distribution_params:
        print(f"  分布参数: center={distribution_params.get('center')}, std={distribution_params.get('std')}")
    print(f"  测试Episodes: {num_test_episodes}")
    print(f"{'='*80}\n")

    # 设置随机种子
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # 1. 创建配置 (动态修改车辆数)
    config = Config()
    config.TOTAL_VEHICLES = num_vehicles

    # 动态调整保存路径
    config.RESULTS_BASE_PATH = f'results/vehicles_{num_vehicles}/'
    config.MODEL_SAVE_PATH = f'{config.RESULTS_BASE_PATH}models/'
    config.LOG_SAVE_PATH = f'{config.RESULTS_BASE_PATH}logs/'
    config.ABLATION_SAVE_PATH = f'{config.RESULTS_BASE_PATH}ablation/'
    config.BENCHMARK_SAVE_PATH = f'{config.RESULTS_BASE_PATH}benchmarks/'

    if not config.validate_config():
        print("❌ 配置验证失败!")
        return None

    # 2. 加载数据
    print(f"[1/5] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )
    print(f"  ✓ 测试集: {len(test_orders)} 条订单")

    # 3. 加载图
    print(f"[2/5] 加载图结构...")
    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()
    print(f"  ✓ 图加载完成")

    # 4. 初始化训练器
    print(f"[3/5] 初始化模型...")
    trainer = AblationMGCNTrainer(config, neighbor_adj, poi_adj, ablation_type)

    # 5. 加载模型
    print(f"[4/5] 加载模型检查点...")
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    try:
        episode = trainer.load_checkpoint(model_path)
        print(f"  ✓ 模型加载成功 (训练到Episode {episode})")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 6. 创建测试环境 (设置初始分布)
    print(f"[5/5] 开始测试 (初始分布: {distribution_type})...")
    test_env = RideHailingEnvironment(config, data_processor, test_orders)

    # 设置车辆初始分布
    if distribution_type == 'normal' and distribution_params:
        center = distribution_params.get('center', (0.5, 0.5))
        std = distribution_params.get('std', (5.0, 5.0))
        test_env.set_vehicle_distribution(
            distribution_type='normal',
            center=center,
            std=std
        )
        print(f"  → 设置正态分布: center={center}, std={std}")
    else:
        test_env.set_vehicle_distribution(distribution_type='uniform')
        print(f"  → 设置均匀分布")

    # 设置模型
    if hasattr(test_env, 'set_model_and_buffer'):
        test_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        test_env.model = trainer.main_net
        test_env.replay_buffer = None
        test_env.device = config.DEVICE

    # 7. 在测试集上评估
    print(f"\n评估模型...")
    avg_results, daily_results_df = evaluate_model(
        trainer, test_env, num_test_episodes, config, verbose=False
    )

    # 8. 打印结果
    print(f"\n{'='*80}")
    print(f"📊 测试结果")
    print(f"{'='*80}")
    print(f"  完成率 (匹配率): {avg_results['completion_rate']:.2%}")
    print(f"  取消率: {avg_results['cancel_rate']:.2%}")
    print(f"  平均等待时间: {avg_results['avg_waiting_time']:.1f} 秒")
    print(f"  车辆利用率: {avg_results['vehicle_utilization']:.2%}")
    print(f"  总收入: {avg_results['avg_total_revenue']:.2f}")
    print(f"{'='*80}\n")

    # 9. 保存结果
    result_summary = {
        'model_path': model_path,
        'ablation_type': ablation_type,
        'num_vehicles': num_vehicles,
        'distribution_type': distribution_type,
        'distribution_params': distribution_params,
        'num_test_episodes': num_test_episodes,
        'avg_results': {
            'completion_rate': float(avg_results['completion_rate']),
            'cancel_rate': float(avg_results['cancel_rate']),
            'avg_waiting_time': float(avg_results['avg_waiting_time']),
            'vehicle_utilization': float(avg_results['vehicle_utilization']),
            'total_revenue': float(avg_results['avg_total_revenue'])
        },
        'daily_results': daily_results_df.to_dict('records') if daily_results_df is not None else []
    }

    return result_summary


def test_multiple_vehicles(model_path, ablation_type, vehicle_list, distribution_type='uniform',
                          distribution_params=None, num_test_episodes=7):
    """测试多个车辆数配置"""

    print(f"\n{'#'*80}")
    print(f"# 批量测试: 多个车辆数")
    print(f"# 车辆数列表: {vehicle_list}")
    print(f"{'#'*80}\n")

    all_results = {}

    for num_vehicles in vehicle_list:
        try:
            result = test_single_config(
                model_path, ablation_type, num_vehicles,
                distribution_type, distribution_params, num_test_episodes
            )
            all_results[f'vehicles_{num_vehicles}'] = result
        except Exception as e:
            print(f"\n❌ 车辆数 {num_vehicles} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def test_multiple_distributions(model_path, ablation_type, num_vehicles, std_list,
                                num_test_episodes=7):
    """测试多个分布配置 (正态分布，不同标准差)"""

    print(f"\n{'#'*80}")
    print(f"# 批量测试: 多个初始分布")
    print(f"# 标准差列表: {std_list}")
    print(f"{'#'*80}\n")

    all_results = {}

    # 测试均匀分布
    try:
        result = test_single_config(
            model_path, ablation_type, num_vehicles,
            'uniform', None, num_test_episodes
        )
        all_results['uniform'] = result
    except Exception as e:
        print(f"\n❌ 均匀分布测试失败: {e}")

    # 测试不同标准差的正态分布
    for std in std_list:
        try:
            distribution_params = {
                'center': (0.5, 0.5),
                'std': (std, std)
            }
            result = test_single_config(
                model_path, ablation_type, num_vehicles,
                'normal', distribution_params, num_test_episodes
            )
            all_results[f'normal_std{std}'] = result
        except Exception as e:
            print(f"\n❌ 正态分布 (std={std}) 测试失败: {e}")

    return all_results


def comprehensive_test(model_path, ablation_type, vehicle_list=[1800, 2000, 2200],
                      std_list=[3.0, 5.0, 7.0], num_test_episodes=7):
    """全面测试: 所有车辆数 × 所有分布"""

    print(f"\n{'#'*80}")
    print(f"# 全面测试: 车辆数 × 初始分布")
    print(f"# 车辆数: {vehicle_list}")
    print(f"# 分布: uniform + normal(std={std_list})")
    print(f"{'#'*80}\n")

    all_results = {}

    for num_vehicles in vehicle_list:
        # 测试均匀分布
        config_key = f'vehicles_{num_vehicles}_uniform'
        try:
            result = test_single_config(
                model_path, ablation_type, num_vehicles,
                'uniform', None, num_test_episodes
            )
            all_results[config_key] = result
        except Exception as e:
            print(f"\n❌ {config_key} 测试失败: {e}")

        # 测试不同标准差的正态分布
        for std in std_list:
            config_key = f'vehicles_{num_vehicles}_normal_std{std}'
            try:
                distribution_params = {
                    'center': (0.5, 0.5),
                    'std': (std, std)
                }
                result = test_single_config(
                    model_path, ablation_type, num_vehicles,
                    'normal', distribution_params, num_test_episodes
                )
                all_results[config_key] = result
            except Exception as e:
                print(f"\n❌ {config_key} 测试失败: {e}")

    return all_results


def save_test_results(all_results, ablation_type, output_dir='results/cnn_ablation_tests'):
    """保存测试结果"""

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'{ablation_type}_test_results_{timestamp}.json')

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 测试结果已保存到: {result_file}")

    # 生成汇总表格
    summary_data = []
    for config_key, result in all_results.items():
        if result is None:
            continue
        summary_data.append({
            'Config': config_key,
            'Vehicles': result['num_vehicles'],
            'Distribution': result['distribution_type'],
            'Completion Rate': result['avg_results']['completion_rate'],
            'Cancel Rate': result['avg_results']['cancel_rate'],
            'Avg Wait Time': result['avg_results']['avg_waiting_time'],
            'Vehicle Util': result['avg_results']['vehicle_utilization'],
            'Total Revenue': result['avg_results']['total_revenue']
        })

    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f'{ablation_type}_summary_{timestamp}.csv')
        df.to_csv(summary_file, index=False)
        print(f"✓ 汇总表格已保存到: {summary_file}")

        # 打印汇总表格
        print(f"\n{'='*120}")
        print(f"测试结果汇总")
        print(f"{'='*120}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(description='测试训练好的 CNN-D3QN 消融模型')

    # 基本参数
    parser.add_argument('--model', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--ablation', type=str, default='cnn',
                       choices=['cnn', 'no_mgcn', 'full_model', 'neighbor_only', 'poi_only', 'no_dueling'],
                       help='消融类型 (默认: cnn)')
    parser.add_argument('--vehicles', type=int, default=1800,
                       help='车辆数 (默认: 1800)')
    parser.add_argument('--episodes', type=int, default=7,
                       help='测试episode数 (默认: 7)')

    # 分布参数
    parser.add_argument('--distribution', type=str, default='uniform',
                       choices=['uniform', 'normal'],
                       help='初始分布类型 (默认: uniform)')
    parser.add_argument('--center', type=float, nargs=2, default=[0.5, 0.5],
                       help='正态分布中心坐标 (默认: 0.5 0.5)')
    parser.add_argument('--std', type=float, default=5.0,
                       help='正态分布标准差 (默认: 5.0)')

    # 批量测试选项
    parser.add_argument('--test-all-vehicles', action='store_true',
                       help='测试多个车辆数')
    parser.add_argument('--vehicle-list', type=int, nargs='+', default=[1800, 2000, 2200],
                       help='车辆数列表 (默认: 1800 2000 2200)')

    parser.add_argument('--test-all-distributions', action='store_true',
                       help='测试多个初始分布')
    parser.add_argument('--std-list', type=float, nargs='+', default=[3.0, 5.0, 7.0],
                       help='标准差列表 (默认: 3.0 5.0 7.0)')

    parser.add_argument('--comprehensive-test', action='store_true',
                       help='全面测试 (所有车辆数 × 所有分布)')

    # 其他选项
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--output-dir', type=str, default='results/cnn_ablation_tests',
                       help='结果保存目录')

    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print(f"\n提示: 请检查模型路径，通常在以下位置:")
        print(f"  - results/vehicles_1800/ablation/")
        print(f"  - results/vehicles_1800/models/")
        return

    # 执行测试
    if args.comprehensive_test:
        # 全面测试
        all_results = comprehensive_test(
            args.model, args.ablation, args.vehicle_list, args.std_list, args.episodes
        )
    elif args.test_all_vehicles:
        # 测试多个车辆数
        distribution_params = None
        if args.distribution == 'normal':
            distribution_params = {
                'center': tuple(args.center),
                'std': (args.std, args.std)
            }
        all_results = test_multiple_vehicles(
            args.model, args.ablation, args.vehicle_list,
            args.distribution, distribution_params, args.episodes
        )
    elif args.test_all_distributions:
        # 测试多个分布
        all_results = test_multiple_distributions(
            args.model, args.ablation, args.vehicles, args.std_list, args.episodes
        )
    else:
        # 单个配置测试
        distribution_params = None
        if args.distribution == 'normal':
            distribution_params = {
                'center': tuple(args.center),
                'std': (args.std, args.std)
            }
        result = test_single_config(
            args.model, args.ablation, args.vehicles,
            args.distribution, distribution_params, args.episodes, args.seed
        )
        all_results = {'single_test': result}

    # 保存结果
    save_test_results(all_results, args.ablation, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✅ 所有测试完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

