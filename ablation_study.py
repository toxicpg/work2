"""
消融实验框架 - 用于评估主算法各组件的贡献度

主要消融配置:
1. 完整模型 (Full Model): 所有组件启用
2. 无 MGCN: 移除 MGCN 图卷积网络，使用简化的 MLP
3. 无 Dueling DQN: 使用标准 DQN 替代 Dueling DQN
4. 无 PER: 使用统一采样替代优先级经验回放
5. 无多阶段奖励: 使用简化的单阶段奖励
6. 无注意力融合: 使用简单拼接替代注意力融合
7. 组合消融: 多个组件的组合
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


class AblationConfig:
    """消融实验配置"""

    # 消融类型
    ABLATION_TYPES = {
        'full_model': {
            'description': '完整模型 (基准)',
            'use_mgcn': True,
            'use_dueling': True,
            'use_per': True,
            'use_multi_stage_reward': True,
            'use_attention_fusion': True,
        },
        'no_mgcn': {
            'description': '无 MGCN - 使用简化 MLP',
            'use_mgcn': False,
            'use_dueling': True,
            'use_per': True,
            'use_multi_stage_reward': True,
            'use_attention_fusion': True,
        },
        'no_dueling': {
            'description': '无 Dueling DQN - 使用标准 DQN',
            'use_mgcn': True,
            'use_dueling': False,
            'use_per': True,
            'use_multi_stage_reward': True,
            'use_attention_fusion': True,
        },
        'no_per': {
            'description': '无 PER - 使用统一采样',
            'use_mgcn': True,
            'use_dueling': True,
            'use_per': False,
            'use_multi_stage_reward': True,
            'use_attention_fusion': True,
        },
        'no_multi_stage_reward': {
            'description': '无多阶段奖励 - 使用简化奖励',
            'use_mgcn': True,
            'use_dueling': True,
            'use_per': True,
            'use_multi_stage_reward': False,
            'use_attention_fusion': True,
        },
        'no_attention_fusion': {
            'description': '无注意力融合 - 使用拼接',
            'use_mgcn': True,
            'use_dueling': True,
            'use_per': True,
            'use_multi_stage_reward': True,
            'use_attention_fusion': False,
        },
        'minimal': {
            'description': '最小化模型 - 仅保留 MGCN 和 DQN',
            'use_mgcn': True,
            'use_dueling': False,
            'use_per': False,
            'use_multi_stage_reward': False,
            'use_attention_fusion': False,
        },
    }


class AblationTrainer:
    """消融实验训练器"""

    def __init__(self, config, ablation_type='full_model'):
        self.config = config
        self.ablation_type = ablation_type
        self.ablation_params = AblationConfig.ABLATION_TYPES.get(ablation_type,
                                                                   AblationConfig.ABLATION_TYPES['full_model'])
        self.results = {
            'ablation_type': ablation_type,
            'ablation_params': self.ablation_params,
            'episodes': [],
            'metrics': {}
        }

    def create_modified_config(self):
        """根据消融参数创建修改后的配置"""
        modified_config = Config()

        # 根据消融类型修改配置
        if not self.ablation_params['use_dueling']:
            # 禁用 Dueling DQN
            modified_config.USE_DUELING = False

        if not self.ablation_params['use_per']:
            # 禁用 PER，使用统一采样
            modified_config.PER_ALPHA = 0.0  # 禁用优先级
            modified_config.REPLAY_BUFFER_SIZE = 50000

        if not self.ablation_params['use_multi_stage_reward']:
            # 简化奖励函数
            modified_config.REWARD_WEIGHTS = {
                'W_MATCH': 1.0,
                'W_WAIT': 0.0,
                'W_CANCEL': 1.0,
                'W_WAIT_SCORE': 0.0,
                'W_COMPLETION': 1.0,
                'W_MATCH_SPEED': 0.0
            }

        if not self.ablation_params['use_attention_fusion']:
            # 禁用注意力融合
            modified_config.MGCN_FUSION_TYPE = 'concat'

        if not self.ablation_params['use_mgcn']:
            # 禁用 MGCN - 使用简化模型
            modified_config.USE_SIMPLIFIED_MODEL = True

        return modified_config

    def run_ablation_study(self, trainer, env, data_processor, val_orders, num_episodes=10):
        """运行消融实验"""
        print(f"\n{'='*70}")
        print(f"消融实验: {self.ablation_type}")
        print(f"描述: {self.ablation_params['description']}")
        print(f"配置: {json.dumps(self.ablation_params, indent=2)}")
        print(f"{'='*70}\n")

        episode_results = []

        for episode in range(1, num_episodes + 1):
            print(f"\n--- Ablation Episode {episode}/{num_episodes} ({self.ablation_type}) ---")

            # 训练一个 episode
            reward, loss = trainer.train_episode(env, episode)

            # 验证
            val_metrics = self._run_validation(trainer, data_processor, val_orders)

            episode_result = {
                'episode': episode,
                'train_reward': reward,
                'train_loss': loss,
                'epsilon': trainer.epsilon,
                **val_metrics
            }
            episode_results.append(episode_result)

            print(f"  Train: Reward={reward:.2f}, Loss={loss:.4f}, Epsilon={trainer.epsilon:.4f}")
            print(f"  Valid: Completion={val_metrics['completion_rate']:.4f}, "
                  f"Match={val_metrics['match_rate']:.4f}, "
                  f"AvgWait={val_metrics['avg_waiting_time']:.1f}s")

        # 计算聚合指标
        df = pd.DataFrame(episode_results)
        self.results['episodes'] = episode_results
        self.results['metrics'] = {
            'avg_train_reward': float(df['train_reward'].mean()),
            'std_train_reward': float(df['train_reward'].std()),
            'avg_train_loss': float(df['train_loss'].mean()),
            'avg_completion_rate': float(df['completion_rate'].mean()),
            'std_completion_rate': float(df['completion_rate'].std()),
            'avg_match_rate': float(df['match_rate'].mean()),
            'avg_waiting_time': float(df['avg_waiting_time'].mean()),
            'final_epsilon': float(df['epsilon'].iloc[-1])
        }

        return self.results

    def _run_validation(self, trainer, data_processor, val_orders):
        """运行验证"""
        val_env = RideHailingEnvironment(trainer.config, data_processor, val_orders)
        if hasattr(val_env, 'set_model_and_buffer'):
            val_env.set_model_and_buffer(trainer.main_net, None, trainer.config.DEVICE)
        else:
            val_env.model = trainer.main_net
            val_env.replay_buffer = None
            val_env.device = trainer.config.DEVICE

        all_daily_infos = {}
        num_val_days = val_env.order_generator.get_day_count()
        if num_val_days == 0:
            return self._get_empty_metrics()

        for day_index in range(min(num_val_days, 1)):  # 仅验证第一天以加快速度
            val_env.reset()
            daily_infos = []
            for tick in range(val_env.config.TICKS_PER_DAY):
                _, _, _, info = val_env.step(current_epsilon=0.0)
                daily_infos.append(info.get('step_info', {}))
            all_daily_infos[day_index] = daily_infos

        daily_metrics = _calculate_daily_metrics(all_daily_infos, 0)
        if not daily_metrics:
            return self._get_empty_metrics()

        df = pd.DataFrame(daily_metrics)
        return {
            'completion_rate': float(df['completion_rate'].mean()),
            'match_rate': float(df['match_rate'].mean()),
            'avg_waiting_time': float(df['avg_waiting_time'].mean()),
            'cancel_rate': float(df['cancel_rate'].mean())
        }

    def _get_empty_metrics(self):
        """返回空指标"""
        return {
            'completion_rate': 0.0,
            'match_rate': 0.0,
            'avg_waiting_time': 0.0,
            'cancel_rate': 0.0
        }


def run_ablation_experiments(num_episodes_per_ablation=5):
    """运行所有消融实验"""

    # 初始化配置和数据
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

    # 存储所有消融实验的结果
    all_ablation_results = {}

    # 遍历所有消融类型
    for ablation_type in AblationConfig.ABLATION_TYPES.keys():
        print(f"\n{'#'*70}")
        print(f"# 开始消融实验: {ablation_type}")
        print(f"{'#'*70}")

        # 为每个消融类型创建新的训练器
        trainer = MGCNTrainer(config, neighbor_adj, poi_adj)
        env = RideHailingEnvironment(config, data_processor, train_orders)

        if hasattr(env, 'set_model_and_buffer'):
            env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, config.DEVICE)
        else:
            env.model = trainer.main_net
            env.replay_buffer = trainer.replay_buffer
            env.device = config.DEVICE

        # 运行消融实验
        ablation_trainer = AblationTrainer(config, ablation_type)
        results = ablation_trainer.run_ablation_study(
            trainer, env, data_processor, val_orders,
            num_episodes=num_episodes_per_ablation
        )

        all_ablation_results[ablation_type] = results

    # 生成对比报告
    print_ablation_comparison_report(all_ablation_results)
    save_ablation_results(all_ablation_results)

    return all_ablation_results


def print_ablation_comparison_report(all_results):
    """打印消融实验对比报告"""
    print(f"\n{'='*100}")
    print(f"消融实验对比报告")
    print(f"{'='*100}\n")

    # 创建对比表格
    comparison_data = []
    for ablation_type, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Ablation Type': ablation_type,
            'Description': results['ablation_params']['description'],
            'Avg Reward': metrics.get('avg_train_reward', 0),
            'Avg Loss': metrics.get('avg_train_loss', 0),
            'Completion Rate': metrics.get('avg_completion_rate', 0),
            'Match Rate': metrics.get('avg_match_rate', 0),
            'Avg Wait Time (s)': metrics.get('avg_waiting_time', 0),
        })

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()

    # 计算性能差异
    full_model_metrics = all_results.get('full_model', {}).get('metrics', {})
    if full_model_metrics:
        print(f"\n{'='*100}")
        print(f"性能差异分析 (相对于完整模型)")
        print(f"{'='*100}\n")

        for ablation_type, results in all_results.items():
            if ablation_type == 'full_model':
                continue

            metrics = results['metrics']
            print(f"\n{ablation_type} ({results['ablation_params']['description']})")
            print(f"  Completion Rate: {metrics.get('avg_completion_rate', 0):.4f} "
                  f"(vs {full_model_metrics.get('avg_completion_rate', 0):.4f}, "
                  f"差异: {(metrics.get('avg_completion_rate', 0) - full_model_metrics.get('avg_completion_rate', 0)):.4f})")
            print(f"  Match Rate: {metrics.get('avg_match_rate', 0):.4f} "
                  f"(vs {full_model_metrics.get('avg_match_rate', 0):.4f}, "
                  f"差异: {(metrics.get('avg_match_rate', 0) - full_model_metrics.get('avg_match_rate', 0)):.4f})")
            print(f"  Avg Wait Time: {metrics.get('avg_waiting_time', 0):.1f}s "
                  f"(vs {full_model_metrics.get('avg_waiting_time', 0):.1f}s, "
                  f"差异: {(metrics.get('avg_waiting_time', 0) - full_model_metrics.get('avg_waiting_time', 0)):.1f}s)")


def save_ablation_results(all_results):
    """保存消融实验结果"""
    os.makedirs('results/ablation_studies/', exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/ablation_studies/ablation_results_{timestamp}.json'

    # 转换为可序列化的格式
    serializable_results = {}
    for ablation_type, results in all_results.items():
        serializable_results[ablation_type] = {
            'ablation_params': results['ablation_params'],
            'metrics': results['metrics'],
            'episodes': results['episodes']
        }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 消融实验结果已保存到: {result_file}")


if __name__ == '__main__':
    # 运行消融实验 (每个消融类型 5 个 episode)
    # 你可以根据需要调整 num_episodes_per_ablation
    run_ablation_experiments(num_episodes_per_ablation=5)

