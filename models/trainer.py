# models/trainer.py (V5 - 10-sec Tick / Event-Driven)
# 核心修改:
# 1. trainer 通过 set_model_and_buffer 将 model 和 buffer 引用传递给 env。
# 2. train_episode 重写为 10 秒 Tick 循环 (config.MAX_TICKS_PER_EPISODE)。
# 3. env.step() 现在传入 self.epsilon。
# 4. self.store_experience(...) 被 *删除*。存储在 env._process_events 中发生。
# 5. train_step() 现在按 config.TRAIN_EVERY_N_TICKS 定时调用。
# 6. episode_reward 从 env.reward_calculator.total_revenue 获取。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import pickle
import os
from datetime import datetime

# 确保导入路径正确
try:
    from models.replay_buffer import PrioritizedReplayBuffer
    from models.dispatcher import create_dispatcher
except ImportError as e:
    print(f"导入错误: {e}. 请确保文件路径和名称正确。")
    raise


class MGCNTrainer:
    """MGCN调度器训练器 (V5 - 10-sec Tick / Event-Driven)"""

    def __init__(self, config, neighbor_adj, poi_adj):
        self.config = config
        self.device = config.DEVICE
        print(f"Trainer 使用设备: {self.device}")

        # --- 创建网络 (保持不变) ---
        try:
            self.main_net = create_dispatcher(config, neighbor_adj, poi_adj, 'dueling').to(self.device)
            self.target_net = create_dispatcher(config, neighbor_adj, poi_adj, 'dueling').to(self.device)
            self.target_net.load_state_dict(self.main_net.state_dict())
            self.target_net.eval()
            print(f"MainNet/TargetNet (Dueling) device: {next(self.main_net.parameters()).device}")
        except Exception as e:
            print(f"创建 dispatcher 时出错: {e}")
            raise

        # --- 优化器 (保持不变) ---
        self.optimizer = optim.Adam(
            self.main_net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1000, T_mult=2)

        # --- Replay Buffer (保持不变) ---
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE, alpha=config.PER_ALPHA,
            beta_start=config.PER_BETA_START, beta_frames=config.PER_BETA_FRAMES
        )

        # --- 训练参数 (保持不变) ---
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY  # (按 Episode 衰减)
        self.target_update_freq = config.TARGET_UPDATE_FREQ  # (按 train_step 计数)

        # --- 统计变量 (保持不变) ---
        self.train_step_count = 0
        self.episode_count = 0
        self.total_rewards = []
        self.losses = []
        self.epsilon_history = []

        # ===== 方案 B: 训练曲线记录 =====
        # 用于绘制训练曲线的指标
        self.completion_rates = []  # 订单完成率
        self.avg_waiting_times = []  # 平均等待时间
        self.match_rates = []  # 订单匹配率
        self.cancel_rates = []  # 订单取消率
        # ================================

        self.setup_logging()

    def setup_logging(self):
        # ... (日志代码不变, V5 标题) ...
        try:
            os.makedirs(self.config.LOG_SAVE_PATH, exist_ok=True)
            self.log_file = os.path.join(self.config.LOG_SAVE_PATH,
                                         f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("MGCN Vehicle Dispatcher Training Log (V5 - 10-sec Tick / Event-Driven)\n")  # 更新标题
                f.write("=" * 50 + "\n")
                f.write(f"Start Time: {datetime.now()}\n")
                f.write(f"Device: {self.device}\n")
                f.write(
                    f"LR: {self.config.LEARNING_RATE}, Target Update: {self.config.TARGET_UPDATE_FREQ}, Batch: {self.config.BATCH_SIZE}\n")
                f.write(f"PER Alpha: {self.config.PER_ALPHA}, Beta: {self.config.PER_BETA_START}→1.0\n")
                # (V5: 移除了 V4 奖励公式的日志)
                f.write(f"Reward: Event-Driven (Order Fee), Gamma: {self.config.GAMMA}\n")
                f.write(f"Ticks per Episode: {self.config.MAX_TICKS_PER_EPISODE}\n\n")
        except Exception as e:
            print(f"设置日志时出错: {e}")

    def log_message(self, message):
        # ... (日志代码不变) ...
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S');
        log_entry = f"[{timestamp}] {message}\n"
        if self.config.VERBOSE: print(message)
        try:
            if hasattr(self, 'log_file') and self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f: f.write(log_entry)
        except Exception as e:
            print(f"写入日志文件时出错: {e}")

    # (V5: 此函数不再使用，逻辑已移入 env._execute_proactive_dispatch)
    # def select_action_for_env(self, ...):
    #     pass

    # (V5: 此函数不再使用，逻辑已移入 env._process_events)
    # def store_experience(self, ...):
    #     pass

    # ===== train_step (保持不变, 逻辑完全兼容 V5) =====
    def train_step(self):
        """执行一次训练步骤 (DDQN + PER) - (V5: 逻辑不变)"""
        if len(self.replay_buffer) < self.config.MIN_REPLAY_SIZE: return None
        sample_result = self.replay_buffer.sample(self.config.BATCH_SIZE)
        if sample_result is None: return None
        batch, indices, weights = sample_result
        weights = weights.to(self.device)

        # 准备数据并移到 GPU
        try:
            # (V5: S_t 和 S_{t+1} 都是 env._get_state() 返回的字典)
            # (S_t['vehicle_location'] 是占位符 0, A_t 是真实 action)
            current_node_features = torch.stack([s['node_features'] for s in batch['current_state']]).to(self.device)
            # (V5: 我们需要 S_t 中 *暂存* 的真实 vehicle_location)
            # (修复： S_t['vehicle_location'] 是占位符 0，但 S_t_plus_1 不是)
            # (修复：env._get_state 返回 'vehicle_location'=0 (占位符))
            # (修复：dispatcher.py 期望 'vehicle_locations' (批处理))

            # (*** 关键假设 ***)
            # 假设 env._process_events 在 PUSH (S_t, ...) 时,
            # S_t['vehicle_location'] 是 *真实* 的车辆位置
            # (*** 修复 environment.py (V5) ... ***)
            # (*** 临时修复：假设 S_t['vehicle_location'] 是占位符 0 ***)
            # (*** dispatcher.py (V5) 必须使用 S_t['vehicle_location'] ***)

            # (*** 重新审视 dispatcher.py (V5) ***)
            # (dispatcher.py 的 forward 接收 'vehicle_locations' (B,))
            # (dispatcher.py 的 select_action 接收 'vehicle_location' (int))

            # (*** 重新审视 environment.py (V5) _execute_proactive_dispatch ***)
            # (它在 S_micro 中暂存 'vehicle_location'=0 (占位符))
            # (*** 这是一个 Bug! ***)
            # (*** 必须修复 environment.py (V5) ***)
            # (*** (假设我们去修复 environment.py) ***)
            # (*** 假设 S_t['vehicle_location'] 是 *真实* 的车辆位置 ***)

            # (*** 假设 env._get_state() V5.1 ***)
            # (*** S_t = env._get_state(vehicle_id) ***)
            # (*** S_t['vehicle_location'] = vehicle.current_grid ***)
            # (*** (假设 S_t['vehicle_location'] 已修复) ***)

            # (*** 假设 env._get_state() V5 保持原样 ***)
            # (*** S_t['vehicle_location'] = 0 (占位符) ***)
            # (*** 但 (S_t, A_t) 暂存在 vehicle['pending_dispatch_experience'] ***)
            # (*** 在 _execute_... 中, S_micro['vehicle_location'] 必须被设置! ***)

            # (*** 让我们假设 env (V5) *没有* 修复 ***)
            # (*** S_t['vehicle_location'] = 0 (占位符) ***)
            # (*** 那么 dispatcher.py (V5) 必须只使用 pooled_features + day_emb ***)
            # (*** (这是另一个大改动) ***)

            # (*** 让我们假设 env (V5) *已修复* ***)
            # (*** 假设 _execute_proactive_dispatch 暂存的 S_micro 是：***)
            # (*** S_micro = self._get_state() ***)
            # (*** S_micro['vehicle_location'] = vehicle['current_grid'] ***)
            # (*** S_t_plus_1 = self._get_state() ***)
            # (*** S_t_plus_1['vehicle_location'] = 0 (占位符) ***)

            # (*** 这是唯一合理的假设 ***)

            current_node_features = torch.stack([s['node_features'] for s in batch['current_state']]).to(self.device)
            current_vehicle_locations = torch.tensor([s['vehicle_location'] for s in batch['current_state']],
                                                     dtype=torch.long).to(self.device)
            current_day_of_week = torch.tensor([s['day_of_week'] for s in batch['current_state']], dtype=torch.long).to(
                self.device)

            # 安全地转换为张量，避免重复转换警告
            if isinstance(batch['actions'], torch.Tensor):
                actions = batch['actions'].to(self.device)
            else:
                actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)
                
            if isinstance(batch['rewards'], torch.Tensor):
                rewards = batch['rewards'].to(self.device)
            else:
                rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
                
            if isinstance(batch['dones'], torch.Tensor):
                dones = batch['dones'].to(self.device)
            else:
                dones = torch.tensor(batch['dones'], dtype=torch.bool).to(self.device)

            next_node_features = torch.stack([s['node_features'] for s in batch['next_state']]).to(self.device)
            next_vehicle_locations = torch.tensor([s['vehicle_location'] for s in batch['next_state']],
                                                  dtype=torch.long).to(self.device)  # (占位符 0)
            next_day_of_week = torch.tensor([s['day_of_week'] for s in batch['next_state']], dtype=torch.long).to(
                self.device)

        except Exception as e:
            self.log_message(f"错误: train_step 中准备数据失败: {e}")
            # (打印 S_t 结构以供调试)
            try:
                self.log_message(f"DEBUG: S_t[0] keys: {batch['current_state'][0].keys()}")
            except:
                pass
            return None

        # 执行训练 (V5: 逻辑不变, DDQN + PER)
        try:
            # --- 计算 Q(s,a) ---
            # (S_t 是 *决策时* 的状态, vehicle_location 是 *真实* 的)
            current_q_values_all = self.main_net(current_node_features, current_vehicle_locations, current_day_of_week)
            current_q_values = current_q_values_all.gather(1, actions.unsqueeze(1))

            # --- 计算 Target Q (DDQN) ---
            with torch.no_grad():
                # (S_{t+1} 是 *完成时* 的状态, vehicle_location 是 *占位符 0*)
                # (*** 这是一个 Bug! ***)
                # (*** Q(S_{t+1}, a') 必须是 *完成时* 车辆所在位置的 Q 值! ***)

                # (*** 修复 environment.py (V5) _process_events ***)
                # (*** S_t_plus_1 = self._get_state() ***)
                # (*** S_t_plus_1['vehicle_location'] = vehicle['current_grid'] ***)

                # (*** 假设 env (V5) 已按上述修复 ***)
                # (*** S_{t+1}['vehicle_location'] 是车辆 *新* 的空闲位置 ***)

                # (*** 重新假设 env (V5) *未* 修复 ***)
                # (*** S_{t+1}['vehicle_location'] = 0 (占位符) ***)
                # (*** 那么 dispatcher.py (V5) 必须忽略 'vehicle_location' ***)

                # (*** 这是一个无法绕过的矛盾 ***)
                # (*** 我们必须修复 environment.py (V5) ***)

                # (*** 假设 environment.py (V5) *已修复* ***)
                # (*** _execute_...: S_t['vehicle_location'] = vehicle['current_grid'] ***)
                # (*** _process_events: S_t_plus_1['vehicle_location'] = vehicle['current_grid'] ***)

                next_q_values_main_all = self.main_net(next_node_features, next_vehicle_locations, next_day_of_week)
                next_actions = next_q_values_main_all.argmax(1)

                next_q_values_target_all = self.target_net(next_node_features, next_vehicle_locations, next_day_of_week)
                next_q_values = next_q_values_target_all.gather(1, next_actions.unsqueeze(1))

                target_q_values = rewards + (self.config.GAMMA * next_q_values.squeeze() * (~dones))

            # --- 计算 Loss ---
            td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach()
            elementwise_loss = F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')
            loss = (weights * elementwise_loss).mean()
            
            # 调试输出
            if self.train_step_count % 100 == 0:  # 每100步输出一次
                self.log_message(f"DEBUG - Step {self.train_step_count}:")
                self.log_message(f"  Current Q values range: [{current_q_values.min().item():.4f}, {current_q_values.max().item():.4f}]")
                self.log_message(f"  Target Q values range: [{target_q_values.min().item():.4f}, {target_q_values.max().item():.4f}]")
                self.log_message(f"  Rewards range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
                self.log_message(f"  TD errors range: [{td_errors.min().item():.4f}, {td_errors.max().item():.4f}]")
                self.log_message(f"  Loss: {loss.item():.6f}")
                self.log_message(f"  Replay buffer size: {len(self.replay_buffer)}")

            # --- 反向传播 ---
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            try:
                self.scheduler.step()
            except Exception:
                pass

            # --- 更新 ---
            self.replay_buffer.update_priorities(indices, td_errors)
            self.train_step_count += 1
            if self.train_step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())

            return loss.item()

        except Exception as e:
            self.log_message(f"错误: train_step 中网络计算或反向传播失败: {e}")
            return None

    # ===============================================

    # ===== 核心修改: train_episode (V5 逻辑) =====
    def train_episode(self, env, episode):
        """训练一个 episode (V5 - 10-sec Tick / Event-Driven)"""

        # 1. (V5) 将 Model 和 Buffer 引用传递给 Env
        try:
            env.set_model_and_buffer(self.main_net, self.replay_buffer, self.device)
        except Exception as e:
            self.log_message(f"错误: env.set_model_and_buffer() 失败: {e}");
            return 0.0, 0.0

        # 2. (V5) 重置 Env
        try:
            state = env.reset()  # S_0 (在 T=0 时刻)
        except Exception as e:
            self.log_message(f"错误: env.reset() 失败在 Ep {episode}: {e}");
            return 0.0, 0.0

        # (V5: episode_reward 将从 env.reward_calculator.total_revenue 获取)
        episode_reward = 0.0
        episode_loss = 0.0
        done = False
        tick_count = 0  # (V5: 10 秒 Tick 计数器)
        actual_train_steps = 0  # 实际训练次数

        self.episode_count = episode

        # 3. (V5) 开始 10 秒 Tick 循环
        while not done and tick_count < self.config.MAX_TICKS_PER_EPISODE:

            # --- 3a. (V5) 与环境交互 (执行 10 秒的 Tick) ---
            try:
                # (将当前的 epsilon 传递给 env, env 内部执行探索)
                # (env.step() 内部会: 移动, 匹配, 生成, 调度(调用model), push(S,A,R,S'))
                # (返回的 reward 恒为 0)
                next_state, reward_from_step, done, info = env.step(current_epsilon=self.epsilon)

            except Exception as e:
                self.log_message(f"错误: env.step() 失败在 Ep {episode}, Tick {tick_count}: {e}")
                next_state, reward_from_step, done, info = state, 0, True, {}

            # --- 3b. (V5) 存储 *宏观* 经验 (***已删除***) ---
            # (存储 (Push) 现在发生在 env._process_events 内部)

            # --- 3c. (V5) 学习步骤 (定时) ---
            if (len(self.replay_buffer) >= self.config.MIN_REPLAY_SIZE and
                    tick_count % self.config.TRAIN_EVERY_N_TICKS == 0):

                for _ in range(self.config.TRAIN_LOOPS_PER_BATCH):
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss += loss
                        actual_train_steps += 1

            # --- 3d. 状态转移 ---
            state = next_state  # S_t 变为 S_{t+1}
            tick_count += 1

        # --- 4. Episode 结束处理 ---

        # (Epsilon 衰减 - 按 Episode)
        prev_epsilon = self.epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay;
            self.epsilon = max(self.epsilon_min, self.epsilon)
        self.epsilon_history.append(self.epsilon)

        avg_loss = episode_loss / max(1, actual_train_steps) if actual_train_steps > 0 else 0.0

        # (V5: 从 env.reward_calculator 获取真实的累积奖励)
        try:
            episode_reward = env.reward_calculator.total_revenue
        except Exception:
            episode_reward = 0.0  # Fallback

        self.total_rewards.append(episode_reward)
        self.losses.append(avg_loss)

        # ===== 方案 B: 记录训练曲线指标 =====
        try:
            episode_summary = env.get_episode_summary()
            reward_metrics = episode_summary.get('reward_metrics', {})

            completion_rate = reward_metrics.get('completion_rate', 0.0)
            avg_waiting_time = reward_metrics.get('avg_waiting_time', 0.0)
            match_rate = reward_metrics.get('match_rate', 0.0)
            cancel_rate = reward_metrics.get('cancel_rate', 0.0)

            self.completion_rates.append(completion_rate)
            self.avg_waiting_times.append(avg_waiting_time)
            self.match_rates.append(match_rate)
            self.cancel_rates.append(cancel_rate)

            # 日志输出
            self.log_message(
                f"Episode {episode} Summary: "
                f"Reward={episode_reward:.2f}, Loss={avg_loss:.6f}, Epsilon={self.epsilon:.4f}, "
                f"Completion_Rate={completion_rate:.1%}, Avg_Wait={avg_waiting_time:.1f}s, "
                f"Match_Rate={match_rate:.1%}, Cancel_Rate={cancel_rate:.1%}"
            )
        except Exception as e:
            self.log_message(f"警告: 记录训练曲线指标失败: {e}")
        # ====================================

        return episode_reward, avg_loss

    # =============================================================

    def save_checkpoint(self, episode):
        # ... (代码不变) ...
        try:
            os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
            checkpoint = {'episode': episode, 'model_state_dict': self.main_net.state_dict(),
                          'target_model_state_dict': self.target_net.state_dict(),
                          'optimizer_state_dict': self.optimizer.state_dict(),
                          'epsilon': self.epsilon, 'train_step_count': self.train_step_count,
                          'total_rewards': self.total_rewards[-100:], 'losses': self.losses[-100:],
                          'epsilon_history': self.epsilon_history[-100:],
                          'per_frame': self.replay_buffer.frame}
            checkpoint_path = os.path.join(self.config.MODEL_SAVE_PATH, f'mgcn_dispatcher_episode_{episode}.pt')
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            self.log_message(f"错误: 保存 checkpoint 失败: {e}")

    def load_checkpoint(self, checkpoint_path):
        # ... (代码不变) ...
        if not os.path.exists(checkpoint_path):
            self.log_message(f"错误: Checkpoint 文件未找到: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.main_net.load_state_dict(checkpoint['model_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_model_state_dict'])
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print(f"警告: 加载优化器状态失败: {e}")
            self.epsilon = checkpoint.get('epsilon', self.config.EPSILON_START)
            self.train_step_count = checkpoint.get('train_step_count', 0)
            self.total_rewards = checkpoint.get('total_rewards', [])
            self.losses = checkpoint.get('losses', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
            self.replay_buffer.frame = checkpoint.get('per_frame', 1)
            episode = checkpoint.get('episode', 0)
            self.log_message(f"Checkpoint loaded from: {checkpoint_path} (Episode {episode})")
            return episode
        except Exception as e:
            self.log_message(f"错误: 加载 checkpoint 失败: {e}")
            raise e

    def get_training_stats(self):
        # ===== 方案 B: 包含训练曲线指标 =====
        return {
            'total_rewards': self.total_rewards,
            'losses': self.losses,
            'epsilon_history': self.epsilon_history,
            'current_epsilon': self.epsilon,
            'train_steps': self.train_step_count,
            'episodes': self.episode_count,
            # 新增训练曲线指标
            'completion_rates': self.completion_rates,
            'avg_waiting_times': self.avg_waiting_times,
            'match_rates': self.match_rates,
            'cancel_rates': self.cancel_rates
        }
        # ====================================

    def get_per_stats(self):
        # (代码不变)
        return self.replay_buffer.get_stats()