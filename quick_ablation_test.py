"""
快速消融实验测试脚本
目标: 用极小配置快速验证消融实验流程是否能跑通

策略:
1. 只测试 2-3 个消融配置
2. 每个配置只训练 2 个 Episode
3. 只用 1 天数据
4. 快速验证流程后再跑完整实验
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.trainer import MGCNTrainer
from evaluate import _calculate_daily_metrics


class QuickAblationConfig(Config):
    """快速消融实验配置"""

    def __init__(self):
        super().__init__()

        # ========== 极小化数据配置 ==========
        self.EPISODE_DAYS = 1
        self.MAX_TICKS_PER_EPISODE = self.TICKS_PER_DAY * self.EPISODE_DAYS

        # ========== 极小化训练配置 ==========
        self.NUM_EPISODES = 2  # 每个消融配置只训练2个episode
        self.BATCH_SIZE = 16
        self.REPLAY_BUFFER_SIZE = 1000
        self.MIN_REPLAY_SIZE = 100

        # ========== 训练参数 ==========
        self.TRAIN_EVERY_N_TICKS = 50
        self.TARGET_UPDATE_FREQ = 50
        self.VALIDATION_INTERVAL = 1
        self.SAVE_FREQ = 999  # 不保存模型，加快速度

        # ========== 加快训练 ==========
        self.EPSILON_START = 0.3
        self.EPSILON_END = 0.1
        self.EPSILON_DECAY = 0.9

        # ========== 日志配置 ==========
        self.VERBOSE = True
        self.LOG_SAVE_PATH = 'logs/quick_ablation_test/'

        print("\n" + "="*70)
        print("⚡ 快速消融实验配置已加载")
        print("="*70)
        print(f"  每个消融配置: {self.NUM_EPISODES} episodes")
        print(f"  每Episode天数: {self.EPISODE_DAYS}天")
        print(f"  预计每个配置运行时间: 2-3分钟")
        print("="*70 + "\n")


# 只测试这3个消融配置（快速验证）
QUICK_ABLATION_TYPES = {
    'full_model': {
        'description': '完整模型 (Ours)',
        'use_mgcn': True,
        'use_dueling': True,
        'use_per': True,
    },
    'no_mgcn': {
        'description': 'w/o MGCN - CNN-D3QN',
        'use_mgcn': False,
        'use_dueling': True,
        'use_per': True,
    },
    'no_dueling': {
        'description': 'w/o Dueling - 标准DQN',
        'use_mgcn': True,
        'use_dueling': False,
        'use_per': True,
    },
}


def run_quick_ablation_test():
    """运行快速消融实验测试"""

    print("\n" + "🔬"*35)
    print("开始快速消融实验测试")
    print("🔬"*35 + "\n")

    # 1. 加载配置
    config = QuickAblationConfig()

    # 2. 加载数据
    print("[1/4] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, _ = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )
    print(f"  ✓ 训练集: {len(train_orders)} 条")
    print(f"  ✓ 验证集: {len(val_orders)} 条")

    # 3. 加载图
    print("\n[2/4] 加载图结构...")
    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()
    print("  ✓ 图已加载")

    # 4. 存储所有结果
    all_results = {}

    # 5. 遍历消融配置
    print(f"\n[3/4] 开始测试 {len(QUICK_ABLATION_TYPES)} 个消融配置...")

    for idx, (ablation_type, ablation_params) in enumerate(QUICK_ABLATION_TYPES.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(QUICK_ABLATION_TYPES)}] 测试消融配置: {ablation_type}")
        print(f"描述: {ablation_params['description']}")
        print(f"{'='*70}")

        try:
            # 根据消融类型修改配置
            test_config = QuickAblationConfig()
            if not ablation_params.get('use_dueling', True):
                test_config.USE_DUELING = False
            if not ablation_params.get('use_per', True):
                test_config.PER_ALPHA = 0.0
            if not ablation_params.get('use_mgcn', True):
                test_config.USE_SIMPLIFIED_MODEL = True

            # 创建训练器
            trainer = MGCNTrainer(test_config, neighbor_adj, poi_adj)
            env = RideHailingEnvironment(test_config, data_processor, train_orders)

            if hasattr(env, 'set_model_and_buffer'):
                env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, test_config.DEVICE)
            else:
                env.model = trainer.main_net
                env.replay_buffer = trainer.replay_buffer
                env.device = test_config.DEVICE

            # 训练
            episode_results = []
            for episode in range(1, test_config.NUM_EPISODES + 1):
                print(f"\n  Episode {episode}/{test_config.NUM_EPISODES}:")

                reward, loss = trainer.train_episode(env, episode)

                # 简单验证（只跑500 ticks）
                val_env = RideHailingEnvironment(test_config, data_processor, val_orders)
                if hasattr(val_env, 'set_model_and_buffer'):
                    val_env.set_model_and_buffer(trainer.main_net, None, test_config.DEVICE)
                else:
                    val_env.model = trainer.main_net
                    val_env.replay_buffer = None
                    val_env.device = test_config.DEVICE

                val_env.reset()
                val_infos = []
                for tick in range(min(500, test_config.TICKS_PER_DAY)):
                    _, _, _, info = val_env.step(current_epsilon=0.0)
                    val_infos.append(info.get('step_info', {}))

                # 计算指标
                total_orders = sum(info.get('orders_generated', 0) for info in val_infos)
                matched_orders = sum(info.get('orders_matched', 0) for info in val_infos)
                match_rate = matched_orders / total_orders if total_orders > 0 else 0.0

                episode_result = {
                    'episode': episode,
                    'train_reward': reward,
                    'train_loss': loss,
                    'match_rate': match_rate,
                    'epsilon': trainer.epsilon
                }
                episode_results.append(episode_result)

                print(f"    训练: Reward={reward:.2f}, Loss={loss:.4f}")
                print(f"    验证: 匹配率={match_rate:.4f}, Epsilon={trainer.epsilon:.4f}")

            # 保存结果
            df = pd.DataFrame(episode_results)
            all_results[ablation_type] = {
                'description': ablation_params['description'],
                'avg_reward': float(df['train_reward'].mean()),
                'avg_loss': float(df['train_loss'].mean()),
                'avg_match_rate': float(df['match_rate'].mean()),
                'episodes': episode_results
            }

            print(f"\n  ✓ {ablation_type} 完成")
            print(f"    平均匹配率: {all_results[ablation_type]['avg_match_rate']:.4f}")

        except Exception as e:
            print(f"\n  ❌ {ablation_type} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[ablation_type] = {
                'description': ablation_params['description'],
                'error': str(e)
            }

    # 6. 打印总结报告
    print(f"\n{'='*70}")
    print("[4/4] 快速消融实验测试完成！")
    print(f"{'='*70}")

    print(f"\n📊 结果总结:\n")
    for ablation_type, results in all_results.items():
        if 'error' in results:
            print(f"  {ablation_type}: ❌ 失败 - {results['error']}")
        else:
            print(f"  {ablation_type} ({results['description']})")
            print(f"    平均匹配率: {results['avg_match_rate']:.4f}")
            print(f"    平均奖励: {results['avg_reward']:.2f}")
            print(f"    平均Loss: {results['avg_loss']:.4f}")
            print()

    # 保存结果
    os.makedirs('results/quick_ablation/', exist_ok=True)
    result_file = f"results/quick_ablation/quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ 结果已保存到: {result_file}")

    print(f"\n{'='*70}")
    print("💡 下一步:")
    print("  如果快速测试通过，可以运行完整消融实验:")
    print("  python ablation_study.py")
    print(f"{'='*70}\n")

    return all_results


if __name__ == '__main__':
    run_quick_ablation_test()

