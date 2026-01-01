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
   python run_ablation_simple.py --episodes 10
"""

import sys
import os
import torch
import numpy as np
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


ABLATION_TYPES = {
    'full_model': '完整模型 (基准)',
    'no_mgcn': '无 MGCN - 使用简化 MLP',
    'no_dueling': '无 Dueling DQN - 使用标准 DQN',
    'no_per': '无 PER - 使用统一采样',
    'no_multi_stage_reward': '无多阶段奖励 - 使用简化奖励',
    'no_attention_fusion': '无注意力融合 - 使用拼接',
    'minimal': '最小化模型 - 仅保留基础组件',
}


def run_single_ablation(ablation_type, config, data_processor, neighbor_adj, poi_adj,
                       train_orders, val_orders, num_episodes=5):
    """运行单个消融实验"""

    print(f"\n{'='*80}")
    print(f"消融实验: {ablation_type}")
    print(f"描述: {ABLATION_TYPES[ablation_type]}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*80}\n")

    # 创建训练器
    trainer = AblationMGCNTrainer(config, neighbor_adj, poi_adj, ablation_type)

    # 创建环境
    env = RideHailingEnvironment(config, data_processor, train_orders)
    if hasattr(env, 'set_model_and_buffer'):
        env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, config.DEVICE)
    else:
        env.model = trainer.main_net
        env.replay_buffer = trainer.replay_buffer
        env.device = config.DEVICE

    # 训练循环
    episode_results = []
    for episode in range(1, num_episodes + 1):
        print(f"\n--- Episode {episode}/{num_episodes} ---")

        reward, loss = trainer.train_episode(env, episode)

        episode_result = {
            'episode': episode,
            'train_reward': reward,
            'train_loss': loss,
            'epsilon': trainer.epsilon,
        }
        episode_results.append(episode_result)

        print(f"  Reward: {reward:.2f}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Epsilon: {trainer.epsilon:.4f}")
        print(f"  Replay Buffer Size: {len(trainer.replay_buffer)}")

    # 获取总结
    summary = trainer.get_ablation_summary()
    summary['episodes'] = episode_results

    return summary


def run_all_ablations(config, data_processor, neighbor_adj, poi_adj,
                     train_orders, val_orders, num_episodes=5):
    """运行所有消融实验"""

    all_results = {}

    for ablation_type in ABLATION_TYPES.keys():
        try:
            result = run_single_ablation(
                ablation_type, config, data_processor, neighbor_adj, poi_adj,
                train_orders, val_orders, num_episodes
            )
            all_results[ablation_type] = result
        except Exception as e:
            print(f"\n❌ 消融实验 {ablation_type} 失败: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def print_comparison_report(all_results):
    """打印对比报告"""

    print(f"\n\n{'='*100}")
    print(f"消融实验对比报告")
    print(f"{'='*100}\n")

    # 创建对比表格
    comparison_data = []
    for ablation_type, result in all_results.items():
        comparison_data.append({
            'Ablation Type': ablation_type,
            'Description': ABLATION_TYPES[ablation_type],
            'Avg Reward': result['avg_reward'],
            'Std Reward': result['std_reward'],
            'Avg Loss': result['avg_loss'],
            'Final Epsilon': result['final_epsilon'],
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()

    # 计算性能差异
    full_model = all_results.get('full_model', {})
    if full_model:
        print(f"\n{'='*100}")
        print(f"性能差异分析 (相对于完整模型)")
        print(f"{'='*100}\n")

        full_reward = full_model['avg_reward']

        for ablation_type, result in all_results.items():
            if ablation_type == 'full_model':
                continue

            reward_diff = result['avg_reward'] - full_reward
            reward_pct = (reward_diff / full_reward * 100) if full_reward != 0 else 0

            print(f"\n{ablation_type}")
            print(f"  平均奖励: {result['avg_reward']:.2f} "
                  f"(vs {full_reward:.2f}, 差异: {reward_diff:+.2f}, {reward_pct:+.1f}%)")
            print(f"  平均损失: {result['avg_loss']:.4f}")


def save_results(all_results):
    """保存结果"""
    os.makedirs('results/ablation_studies/', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/ablation_studies/ablation_results_{timestamp}.json'

    # 转换为可序列化的格式
    serializable_results = {}
    for ablation_type, result in all_results.items():
        serializable_results[ablation_type] = {
            'ablation_type': result['ablation_type'],
            'total_episodes': result['total_episodes'],
            'avg_reward': float(result['avg_reward']),
            'std_reward': float(result['std_reward']),
            'avg_loss': float(result['avg_loss']),
            'final_epsilon': float(result['final_epsilon']),
            'episodes': result['episodes']
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
    parser.add_argument('--episodes', type=int, default=5,
                       help='每个消融实验的 episode 数量')
    args = parser.parse_args()

    # 初始化配置和数据
    print("初始化配置和数据...")
    config = Config()
    if not config.validate_config():
        return

    print("加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, _ = data_processor.split_data_by_time(
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
            train_orders, val_orders, args.episodes
        )
        all_results = {args.ablation: result}
    else:
        # 运行所有消融类型
        all_results = run_all_ablations(
            config, data_processor, neighbor_adj, poi_adj,
            train_orders, val_orders, args.episodes
        )

    # 打印报告和保存结果
    print_comparison_report(all_results)
    save_results(all_results)


if __name__ == '__main__':
    main()

