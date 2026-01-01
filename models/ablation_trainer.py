"""
消融实验 Trainer - 基于标准 Trainer 的修改版本

主要改进:
1. 支持不同的消融类型
2. 可以动态启用/禁用 PER、Dueling DQN 等组件
3. 记录更详细的消融指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import pickle
import os
from datetime import datetime

try:
    from models.ablation_dispatcher import create_ablation_dispatcher
    from models.replay_buffer import PrioritizedReplayBuffer
except ImportError as e:
    print(f"导入错误: {e}")
    raise


class SimpleReplayBuffer:
    """简单的统一采样 Replay Buffer（用于 no_per 消融）"""

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, priority=None):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """均匀采样"""
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_data = [self.buffer[i] for i in indices]

        # 解包数据
        states, actions, rewards, next_states, dones = zip(*batch_data)

        batch = {
            'current_state': list(states),
            'action': torch.tensor(actions, dtype=torch.long),
            'reward': torch.tensor(rewards, dtype=torch.float32),
            'next_state': list(next_states),
            'done': torch.tensor(dones, dtype=torch.float32)
        }

        weights = torch.ones(batch_size, dtype=torch.float32)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """PER 接口（此处无操作）"""
        pass

    def __len__(self):
        return len(self.buffer)


class AblationMGCNTrainer:
    """消融实验训练器"""

    def __init__(self, config, neighbor_adj, poi_adj, ablation_type='full_model'):
        self.config = config
        self.ablation_type = ablation_type
        self.device = config.DEVICE

        print(f"\n{'='*70}")
        print(f"初始化消融训练器: {ablation_type}")
        print(f"{'='*70}")

        # 创建网络
        try:
            self.main_net = create_ablation_dispatcher(
                config, neighbor_adj, poi_adj, ablation_type
            ).to(self.device)
            self.target_net = create_ablation_dispatcher(
                config, neighbor_adj, poi_adj, ablation_type
            ).to(self.device)
            self.target_net.load_state_dict(self.main_net.state_dict())
            self.target_net.eval()
            print(f"✓ 网络创建成功 (Ablation Type: {ablation_type})")
        except Exception as e:
            print(f"❌ 创建网络失败: {e}")
            raise

        # 优化器
        self.optimizer = optim.Adam(
            self.main_net.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )

        # Replay Buffer
        if ablation_type == 'no_per':
            print("✓ 使用简单统一采样 Replay Buffer")
            self.replay_buffer = SimpleReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
            self.use_per = False
        else:
            print("✓ 使用优先级经验回放 (PER)")
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.REPLAY_BUFFER_SIZE,
                alpha=config.PER_ALPHA,
                beta_start=config.PER_BETA_START,
                beta_frames=config.PER_BETA_FRAMES
            )
            self.use_per = True

        # 训练参数
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.target_update_freq = config.TARGET_UPDATE_FREQ

        # 统计变量
        self.train_step_count = 0
        self.episode_count = 0
        self.total_rewards = []
        self.losses = []
        self.epsilon_history = []
        self.best_validation_metric = 0.0

        # 消融实验特定的统计
        self.ablation_metrics = {
            'train_steps': [],
            'train_rewards': [],
            'train_losses': [],
            'epsilon_values': []
        }

        self.setup_logging()

    def setup_logging(self):
        """设置日志"""
        try:
            os.makedirs(self.config.LOG_SAVE_PATH, exist_ok=True)
            self.log_file = os.path.join(
                self.config.LOG_SAVE_PATH,
                f"ablation_{self.ablation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"Ablation Study Training Log ({self.ablation_type})\n")
                f.write("=" * 50 + "\n")
                f.write(f"Start Time: {datetime.now()}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Ablation Type: {self.ablation_type}\n")
                f.write(f"LR: {self.config.LEARNING_RATE}, "
                       f"Target Update: {self.config.TARGET_UPDATE_FREQ}, "
                       f"Batch: {self.config.BATCH_SIZE}\n")
                f.write(f"Use PER: {self.use_per}\n\n")
        except Exception as e:
            print(f"设置日志失败: {e}")

    def log_message(self, message):
        """记录消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        if self.config.VERBOSE:
            print(message)
        try:
            if hasattr(self, 'log_file') and self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
        except Exception as e:
            print(f"写入日志失败: {e}")

    def train_step(self):
        """执行一次训练步骤"""
        if len(self.replay_buffer) < self.config.MIN_REPLAY_SIZE:
            return None

        sample_result = self.replay_buffer.sample(self.config.BATCH_SIZE)
        if sample_result is None:
            return None

        batch, indices, weights = sample_result
        weights = weights.to(self.device)

        try:
            current_node_features = torch.stack(
                [s['node_features'] for s in batch['current_state']]
            ).to(self.device)
            current_vehicle_locations = torch.tensor(
                [s['vehicle_location'] for s in batch['current_state']],
                dtype=torch.long
            ).to(self.device)
            current_day_of_week = torch.tensor(
                [s['day_of_week'] for s in batch['current_state']],
                dtype=torch.long
            ).to(self.device)

            next_node_features = torch.stack(
                [s['node_features'] for s in batch['next_state']]
            ).to(self.device)
            next_vehicle_locations = torch.tensor(
                [s['vehicle_location'] for s in batch['next_state']],
                dtype=torch.long
            ).to(self.device)
            next_day_of_week = torch.tensor(
                [s['day_of_week'] for s in batch['next_state']],
                dtype=torch.long
            ).to(self.device)

            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            dones = batch['done'].to(self.device)

        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return None

        # 计算当前 Q 值
        with torch.no_grad():
            q_next_main = self.main_net(
                next_node_features, next_vehicle_locations, next_day_of_week
            )
            best_actions = torch.argmax(q_next_main, dim=1)

            q_next_target = self.target_net(
                next_node_features, next_vehicle_locations, next_day_of_week
            )
            q_next_target_best = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + self.config.GAMMA * q_next_target_best * (1 - dones)

        # 计算当前 Q 值
        q_current = self.main_net(
            current_node_features, current_vehicle_locations, current_day_of_week
        )
        q_current_action = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算 Loss
        td_error = q_current_action - target_q
        loss = (weights * td_error ** 2).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        # 更新 PER 优先级
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        # 更新目标网络
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()

    def train_episode(self, env, episode_num):
        """训练一个 episode"""
        state = env.reset()
        episode_reward = 0.0
        episode_losses = []

        for tick in range(self.config.MAX_TICKS_PER_EPISODE):
            # 执行一个 step
            _, _, _, info = env.step(current_epsilon=self.epsilon)

            # 每 TRAIN_EVERY_N_TICKS 进行一次训练
            if tick % self.config.TRAIN_EVERY_N_TICKS == 0:
                for _ in range(self.config.TRAIN_LOOPS_PER_BATCH):
                    loss = self.train_step()
                    if loss is not None:
                        episode_losses.append(loss)

        # 更新 Epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 获取 Episode 奖励
        episode_reward = env.reward_calculator.total_revenue

        # 记录指标
        self.total_rewards.append(episode_reward)
        self.epsilon_history.append(self.epsilon)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        self.losses.append(avg_loss)

        # 消融实验指标
        self.ablation_metrics['train_steps'].append(self.train_step_count)
        self.ablation_metrics['train_rewards'].append(episode_reward)
        self.ablation_metrics['train_losses'].append(avg_loss)
        self.ablation_metrics['epsilon_values'].append(self.epsilon)

        self.episode_count += 1

        return episode_reward, avg_loss

    def get_ablation_summary(self):
        """获取消融实验总结"""
        return {
            'ablation_type': self.ablation_type,
            'total_episodes': self.episode_count,
            'avg_reward': np.mean(self.total_rewards) if self.total_rewards else 0.0,
            'std_reward': np.std(self.total_rewards) if self.total_rewards else 0.0,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'final_epsilon': self.epsilon,
            'metrics': self.ablation_metrics
        }

