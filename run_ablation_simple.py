"""
简化的消融实验脚本 - 快速运行消融实验

使用方法:
1. 运行所有消融实验:
   python run_ablation_simple.py

2. 运行特定的消融类型:
   python run_ablation_simple.py --ablation full_model
   python run_ablation_simple.py --ablation no_mgcn
   python run_ablation_simple.py --ablation no_dueling
   python run_ablation_simple.py --ablation no_per
   python run_ablation_simple.py --ablation no_multi_stage_reward
   python run_ablation_simple.py --ablation no_attention_fusion
   python run_ablation_simple.py --ablation minimal

3. 指定 episode 数量:
   python run_ablation_simple.py --train-episodes 10 --test-episodes 7
"""

import os
import sys

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import argparse

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.ablation_trainer import AblationMGCNTrainer
from evaluate import evaluate_model


ABLATION_TYPES = {
    'full_model': '完整模型 (双图MGCN + Dueling)',
    'no_mgcn': '无MGCN - 使用简化MLP',
    'neighbor_only': '单图MGCN - 仅邻接图',
    'poi_only': '单图MGCN - 仅POI图',
    'no_dueling': '无Dueling - 使用标准DQN',
}


def run_single_ablation(ablation_type, config, data_processor, neighbor_adj, poi_adj,
                       train_orders, val_orders, test_orders,
                       num_train_episodes=10, num_test_episodes=7):
    """运行单个消融实验 (训练 + 测试)"""

    print(f"\n{'='*80}")
    print(f"消融实验: {ablation_type}")
    print(f"描述: {ABLATION_TYPES[ablation_type]}")
    print(f"训练 Episodes: {num_train_episodes}, 测试 Episodes: {num_test_episodes}")
    print(f"{'='*80}\n")

    # ===== 阶段1: 训练 =====
    print(f"\n{'='*80}")
    print(f"阶段 1/2: 训练阶段")
    print(f"{'='*80}\n")

    # 创建训练器
    trainer = AblationMGCNTrainer(config, neighbor_adj, poi_adj, ablation_type)

    # 创建训练环境
    train_env = RideHailingEnvironment(config, data_processor, train_orders)
    if hasattr(train_env, 'set_model_and_buffer'):
        train_env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, config.DEVICE)
    else:
        train_env.model = trainer.main_net
        train_env.replay_buffer = trainer.replay_buffer
        train_env.device = config.DEVICE

    # 训练循环
    train_episode_results = []

    # 早停参数
    best_reward = float('-inf')
    early_stopping_counter = 0
    early_stopping_patience = config.EARLY_STOPPING_PATIENCE

    for episode in range(1, num_train_episodes + 1):
        print(f"\n--- 训练 Episode {episode}/{num_train_episodes} ---")

        reward, loss = trainer.train_episode(train_env, episode)

        episode_result = {
            'episode': episode,
            'train_reward': reward,
            'train_loss': loss,
            'epsilon': trainer.epsilon,
        }
        train_episode_results.append(episode_result)

        print(f"  Reward: {reward:.2f}, Loss: {loss:.4f}, Epsilon: {trainer.epsilon:.4f}")

        # 早停检查
        if reward > best_reward:
            best_reward = reward
            early_stopping_counter = 0
            print(f"  ✓ 新的最佳奖励: {best_reward:.2f}")
        else:
            early_stopping_counter += 1
            print(f"  早停计数: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            print(f"\n早停触发！连续 {early_stopping_patience} 个episode奖励未提升。")
            break

    # ===== 阶段2: 测试 =====
    print(f"\n{'='*80}")
    print(f"阶段 2/2: 测试阶段")
    print(f"{'='*80}\n")

    # 创建测试环境
    test_env = RideHailingEnvironment(config, data_processor, test_orders)
    if hasattr(test_env, 'set_model_and_buffer'):
        test_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        test_env.model = trainer.main_net
        test_env.replay_buffer = None
        test_env.device = config.DEVICE

    # 在测试集上评估
    print(f"在测试集上评估 (共 {num_test_episodes} 个Episodes)...")
    avg_test_results, daily_test_results = evaluate_model(
        trainer, test_env, num_test_episodes, config, verbose=False
    )

    print(f"\n测试结果:")
    print(f"  完成率: {avg_test_results['completion_rate']:.2%}")
    print(f"  取消率: {avg_test_results['cancel_rate']:.2%}")
    print(f"  平均等待时间: {avg_test_results['avg_waiting_time']:.1f}秒")
    print(f"  车辆利用率: {avg_test_results['vehicle_utilization']:.2%}")
    print(f"  总收入: {avg_test_results['avg_total_revenue']:.2f}")

    # 获取总结
    summary = trainer.get_ablation_summary()
    summary['train_episodes'] = train_episode_results
    summary['test_results'] = avg_test_results
    summary['test_completion_rate'] = avg_test_results['completion_rate']
    summary['test_cancel_rate'] = avg_test_results['cancel_rate']
    summary['test_avg_waiting_time'] = avg_test_results['avg_waiting_time']
    summary['test_vehicle_utilization'] = avg_test_results['vehicle_utilization']
    summary['test_total_revenue'] = avg_test_results['avg_total_revenue']

    return summary


def run_all_ablations(config, data_processor, neighbor_adj, poi_adj,
                     train_orders, val_orders, test_orders,
                     num_train_episodes=10, num_test_episodes=7):
    """运行所有消融实验"""

    all_results = {}

    for ablation_type in ABLATION_TYPES.keys():
        try:
            result = run_single_ablation(
                ablation_type, config, data_processor, neighbor_adj, poi_adj,
                train_orders, val_orders, test_orders,
                num_train_episodes, num_test_episodes
            )
            all_results[ablation_type] = result
        except Exception as e:
            print(f"\n❌ 消融实验 {ablation_type} 失败: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def print_comparison_report(all_results):
    """打印对比报告"""

    print(f"\n\n{'='*120}")
    print(f"消融实验对比报告 - 测试集性能")
    print(f"{'='*120}\n")

    # 创建对比表格 (测试集指标)
    comparison_data = []
    for ablation_type, result in all_results.items():
        comparison_data.append({
            'Ablation Type': ablation_type,
            'Description': ABLATION_TYPES[ablation_type][:20] + '...',  # 截断描述
            'Completion Rate': result.get('test_completion_rate', 0.0),
            'Cancel Rate': result.get('test_cancel_rate', 0.0),
            'Avg Wait Time': result.get('test_avg_waiting_time', 0.0),
            'Vehicle Util': result.get('test_vehicle_utilization', 0.0),
            'Total Revenue': result.get('test_total_revenue', 0.0),
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()

    # 计算性能差异 (基于测试集完成率)
    full_model = all_results.get('full_model', {})
    if full_model:
        print(f"\n{'='*120}")
        print(f"性能差异分析 (相对于完整模型, 基于测试集完成率)")
        print(f"{'='*120}\n")

        full_completion = full_model.get('test_completion_rate', 0.0)

        for ablation_type, result in all_results.items():
            if ablation_type == 'full_model':
                continue

            reward_diff = result['avg_reward'] - full_reward
            reward_pct = (reward_diff / full_reward * 100) if full_reward != 0 else 0

            print(f"\n{ablation_type}")
            print(f"  平均奖励: {result['avg_reward']:.2f} "
                  f"(vs {full_reward:.2f}, 差异: {reward_diff:+.2f}, {reward_pct:+.1f}%)")
            print(f"  平均损失: {result['avg_loss']:.4f}")


def save_results(all_results, config):
    """保存结果"""
    save_dir = getattr(config, 'ABLATION_SAVE_PATH', 'results/ablation_studies/')
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(save_dir, f'ablation_results_{timestamp}.json')

    # 转换为可序列化的格式
    serializable_results = {}
    for ablation_type, result in all_results.items():
        serializable_results[ablation_type] = {
            'ablation_type': result['ablation_type'],
            'total_episodes': result['total_episodes'],
            'train_avg_reward': float(result['avg_reward']),
            'train_std_reward': float(result['std_reward']),
            'train_avg_loss': float(result['avg_loss']),
            'final_epsilon': float(result['final_epsilon']),
            'test_completion_rate': float(result.get('test_completion_rate', 0.0)),
            'test_cancel_rate': float(result.get('test_cancel_rate', 0.0)),
            'test_avg_waiting_time': float(result.get('test_avg_waiting_time', 0.0)),
            'test_vehicle_utilization': float(result.get('test_vehicle_utilization', 0.0)),
            'test_total_revenue': float(result.get('test_total_revenue', 0.0)),
            'train_episodes': result['train_episodes']
        }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 消融实验结果已保存到: {result_file}")


def main():
    """主函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--ablation', type=str, default=None,
                       help=f'指定消融类型: {", ".join(ABLATION_TYPES.keys())}')
    parser.add_argument('--train-episodes', type=int, default=10,
                       help='训练的 episode 数量 (默认: 10)')
    parser.add_argument('--test-episodes', type=int, default=7,
                       help='测试的 episode 数量，对应测试集7天 (默认: 7)')
    args = parser.parse_args()

    # 初始化配置和数据
    print("初始化配置和数据...")
    config = Config()
    if not config.validate_config():
        return

    print("加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    print("加载图...")
    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()

    # 运行消融实验
    if args.ablation:
        # 运行指定的消融类型
        if args.ablation not in ABLATION_TYPES:
            print(f"❌ 未知的消融类型: {args.ablation}")
            print(f"可用的消融类型: {', '.join(ABLATION_TYPES.keys())}")
            return

        result = run_single_ablation(
            args.ablation, config, data_processor, neighbor_adj, poi_adj,
            train_orders, val_orders, test_orders,
            args.train_episodes, args.test_episodes
        )
        all_results = {args.ablation: result}
    else:
        # 运行所有消融类型
        all_results = run_all_ablations(
            config, data_processor, neighbor_adj, poi_adj,
            train_orders, val_orders, test_orders,
            args.train_episodes, args.test_episodes
        )

    # 打印报告和保存结果
    print_comparison_report(all_results)
    save_results(all_results, config)


if __name__ == '__main__':
    main()

