"""
CNN-DDQN Baseline Training Script
使用卷积神经网络（CNN）+ Dueling DQN 进行网约车调度训练和评估
"""

import os
import random
import sys
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 调整路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
try:
    os.chdir(project_root)
    print(f"当前工作目录已更改为: {os.getcwd()}")
except Exception as e:
    print(f"更改工作目录失败: {e}")
    sys.exit(1)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模块
try:
    from config import Config
    from utils.data_process import DataProcessor
    from utils.graph_builder import GraphBuilder
    from environment import RideHailingEnvironment
    from evaluate import evaluate_model, print_evaluation_results
    from baselines.cnn_ddqn_model import CNNDDQN
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNNTrainer:
    """简化的CNN-DDQN训练器"""

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # 创建模型
        self.main_net = CNNDDQN(config).to(self.device)
        self.target_net = CNNDDQN(config).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.main_net.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Replay Buffer (简单版本)
        self.replay_buffer = deque(maxlen=config.REPLAY_BUFFER_SIZE)

        # 训练参数
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        self.train_step_count = 0

    def select_action(self, state, epsilon):
        """选择动作（epsilon-greedy）"""
        if random.random() < epsilon:
            return random.randint(0, self.config.NUM_ACTIONS - 1)
        else:
            with torch.no_grad():
                node_features = state['node_features'].unsqueeze(0).to(self.device)
                vehicle_location = torch.tensor([state['vehicle_location']], dtype=torch.long).to(self.device)
                day_of_week = torch.tensor([state['day_of_week']], dtype=torch.long).to(self.device)

                q_values = self.main_net(node_features, vehicle_location, day_of_week)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        """执行一次训练步骤"""
        if len(self.replay_buffer) < self.config.MIN_REPLAY_SIZE:
            return None

        # 采样batch
        batch_size = min(self.config.BATCH_SIZE, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        # 解包数据
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为tensor
        current_node_features = torch.stack([s['node_features'] for s in states]).to(self.device)
        current_vehicle_locations = torch.tensor([s['vehicle_location'] for s in states], dtype=torch.long).to(self.device)
        current_day_of_week = torch.tensor([s['day_of_week'] for s in states], dtype=torch.long).to(self.device)

        next_node_features = torch.stack([s['node_features'] for s in next_states]).to(self.device)
        next_vehicle_locations = torch.tensor([s['vehicle_location'] for s in next_states], dtype=torch.long).to(self.device)
        next_day_of_week = torch.tensor([s['day_of_week'] for s in next_states], dtype=torch.long).to(self.device)

        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # 计算当前Q值
        q_current = self.main_net(current_node_features, current_vehicle_locations, current_day_of_week)
        q_current_action = q_current.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            q_next_main = self.main_net(next_node_features, next_vehicle_locations, next_day_of_week)
            best_actions = q_next_main.argmax(dim=1)

            q_next_target = self.target_net(next_node_features, next_vehicle_locations, next_day_of_week)
            q_next_best = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards_tensor + self.config.GAMMA * q_next_best * (~dones_tensor)

        # 计算损失
        loss = torch.nn.functional.mse_loss(q_current_action, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新target network
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()

    def train_episode(self, env, episode):
        """训练一个episode"""
        env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0

        pbar = tqdm(total=self.config.MAX_TICKS_PER_EPISODE,
                   desc=f"Episode {episode}", leave=False)

        while env.episode_step < self.config.MAX_TICKS_PER_EPISODE:
            step_info = env.step(epsilon=self.epsilon)
            total_reward += step_info.get('revenue', 0)

            # 训练
            loss = self.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            pbar.update(1)

        pbar.close()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        return total_reward, avg_loss


def train_cnn_ddqn():
    """主训练函数"""
    print("=" * 80)
    print("CNN-DDQN Baseline Training")
    print("=" * 80)

    # 配置
    config = Config()
    set_seed(config.SEED)

    # 加载数据
    print("\n加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    print(f"训练集订单数: {len(train_orders)}")
    print(f"验证集订单数: {len(val_orders)}")
    print(f"测试集订单数: {len(test_orders)}")

    # 创建训练器
    print("\n初始化CNN-DDQN模型...")
    trainer = SimpleCNNTrainer(config)

    # 创建环境
    train_env = RideHailingEnvironment(config, data_processor, train_orders)
    train_env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, config.DEVICE)

    test_env = RideHailingEnvironment(config, data_processor, test_orders)
    test_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)

    # 训练
    num_episodes = config.NUM_EPISODES  # 使用config中的配置
    print(f"\n开始训练 (共 {num_episodes} episodes)...")

    # 早停参数
    best_reward = float('-inf')
    early_stopping_counter = 0
    early_stopping_patience = config.EARLY_STOPPING_PATIENCE

    results = []
    for episode in range(1, num_episodes + 1):
        print(f"\n{'='*80}")
        print(f"Episode {episode}/{num_episodes}")
        print(f"{'='*80}")

        reward, loss = trainer.train_episode(train_env, episode)

        print(f"\n训练结果:")
        print(f"  总奖励: {reward:.2f}")
        print(f"  平均Loss: {loss:.4f}")
        print(f"  Epsilon: {trainer.epsilon:.4f}")

        results.append({
            'episode': episode,
            'train_reward': reward,
            'train_loss': loss,
            'epsilon': trainer.epsilon
        })

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

    # 测试
    print(f"\n{'='*80}")
    print("在测试集上评估...")
    print(f"{'='*80}\n")

    avg_test_results, daily_test_results = evaluate_model(
        trainer, test_env, num_test_episodes=7, config=config, verbose=True
    )

    # 打印测试结果
    print(f"\n{'='*80}")
    print("CNN-DDQN 测试结果")
    print(f"{'='*80}")
    print_evaluation_results(avg_test_results, daily_test_results)

    # 保存结果
    save_dir = f'results/vehicles_{config.TOTAL_VEHICLES}/baselines/'
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存训练记录
    train_df = pd.DataFrame(results)
    train_df.to_csv(os.path.join(save_dir, f'cnn_ddqn_training_{timestamp}.csv'), index=False)

    # 保存测试结果
    test_results = {
        'method': 'CNN-DDQN',
        'avg_results': avg_test_results,
        'daily_results': daily_test_results,
        'training_episodes': num_episodes
    }

    import json
    with open(os.path.join(save_dir, f'cnn_ddqn_results_{timestamp}.json'), 'w') as f:
        json.dump(test_results, f, indent=2, default=str)

    # 保存模型
    torch.save(trainer.main_net.state_dict(),
              os.path.join(save_dir, f'cnn_ddqn_model_{timestamp}.pt'))

    print(f"\n✓ 结果已保存到: {save_dir}")
    print(f"✓ 训练完成！")

    return avg_test_results


if __name__ == '__main__':
    train_cnn_ddqn()

