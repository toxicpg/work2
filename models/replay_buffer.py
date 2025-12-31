# models/replay_buffer.py
"""
Prioritized Experience Replay Buffer (V-Final-Embedding: Uses day_of_week)
基于优先级的经验回放缓冲区
"""

import numpy as np
import torch
import random


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区

    参考论文：
    Schaul et al. (2016). Prioritized Experience Replay. ICLR 2016.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity (int): 缓冲区容量
            alpha (float): 优先级指数 [0,1]
            beta_start (float): 重要性采样权重的初始值
            beta_frames (int): beta线性增长到1.0的帧数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # 当前帧数

        # 存储
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

        print(f"初始化PrioritizedReplayBuffer:")
        print(f"  容量: {capacity}")
        print(f"  alpha: {alpha}")
        print(f"  beta: {beta_start} → 1.0 (over {beta_frames} frames)")

    def beta_by_frame(self, frame_idx):
        """计算当前的beta值（线性退火）"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        """
        添加新经验到缓冲区 (V-Final-Embedding: 存储 day_of_week)
        """
        # 新经验的优先级
        max_priority = self.priorities.max() if self.buffer else 1.0

        # ===== 核心修改：移除 time_slice, 添加 day_of_week =====
        # 确保 state 和 next_state 字典中包含 'day_of_week'
        if 'day_of_week' not in state or 'day_of_week' not in next_state:
            print("错误: state 或 next_state 缺少 'day_of_week' 字段!")
            # (可以添加更健壮的错误处理)
            # return

        # 确保 node_features 在 CPU 上存储 (节省显存)
        state_node_features = state['node_features']
        next_state_node_features = next_state['node_features']

        if isinstance(state_node_features, torch.Tensor):
            state_node_features = state_node_features.clone().cpu()
        else:
            state_node_features = torch.FloatTensor(state_node_features)  # 确保是 Tensor

        if isinstance(next_state_node_features, torch.Tensor):
            next_state_node_features = next_state_node_features.clone().cpu()
        else:
            next_state_node_features = torch.FloatTensor(next_state_node_features)  # 确保是 Tensor

        experience = {
            'state': {
                'node_features': state_node_features,
                'vehicle_location': state['vehicle_location'],
                'day_of_week': state['day_of_week']  # <-- 新增
                # 'time_slice': state['time_slice'] # <-- 已移除
            },
            'action': action,
            'reward': reward,
            'next_state': {
                'node_features': next_state_node_features,
                'vehicle_location': next_state['vehicle_location'],
                'day_of_week': next_state['day_of_week']  # <-- 新增
                # 'time_slice': next_state['time_slice'] # <-- 已移除
            },
            'done': done
        }
        # =======================================================

        # 添加到缓冲区
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        按优先级采样一个批次
        """
        if len(self.buffer) < batch_size:
            return None  # 经验不足

        # 当前有效的优先级
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # 计算采样概率 (加一个小 epsilon 防止 0 概率)
        priorities_pow = priorities ** self.alpha + 1e-8
        probabilities = priorities_pow / priorities_pow.sum()

        # 采样（不放回）
        # (确保 p 的和约为 1.0)
        probabilities = probabilities / probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)

        # 计算重要性采样权重
        beta = self.beta_by_frame(self.frame)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 归一化到[0,1]
        weights = torch.FloatTensor(weights)

        # 提取batch
        batch = [self.buffer[idx] for idx in indices]
        batch_data = self._format_batch(batch)

        self.frame += 1

        return batch_data, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        根据TD-error更新优先级
        """
        # 确保 td_errors 是 numpy array
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        for idx, td_error in zip(indices, td_errors):
            # 确保 idx 在合法范围内
            if 0 <= idx < self.capacity:
                # 优先级 = |TD-error| + epsilon (避免0优先级)
                priority = float(abs(td_error)) + 1e-6
                self.priorities[idx] = priority
            else:
                print(f"警告: update_priorities 收到无效索引 {idx}")

    def _format_batch(self, batch):
        """格式化批次数据为tensor (V-Final-Embedding: 打包 day_of_week)"""

        # ===== 核心修改：移除 time_slices, 添加 day_of_week =====
        # 当前状态
        current_node_features = torch.stack([exp['state']['node_features'] for exp in batch])
        current_vehicle_locations = torch.LongTensor([exp['state']['vehicle_location'] for exp in batch])
        current_day_of_week = torch.LongTensor([exp['state']['day_of_week'] for exp in batch])  # <-- 新增
        # current_time_slices = torch.LongTensor(...) # <-- 已移除

        # 动作和奖励
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])  # 使用 BoolTensor

        # 下一状态
        next_node_features = torch.stack([exp['next_state']['node_features'] for exp in batch])
        next_vehicle_locations = torch.LongTensor([exp['next_state']['vehicle_location'] for exp in batch])
        next_day_of_week = torch.LongTensor([exp['next_state']['day_of_week'] for exp in batch])  # <-- 新增
        # next_time_slices = torch.LongTensor(...) # <-- 已移除
        # =======================================================

        # 修复：返回正确的结构，trainer.py期望的是状态字典列表
        current_states = []
        next_states = []
        
        for i in range(len(batch)):
            current_states.append({
                'node_features': batch[i]['state']['node_features'],
                'vehicle_location': batch[i]['state']['vehicle_location'],
                'day_of_week': batch[i]['state']['day_of_week']
            })
            next_states.append({
                'node_features': batch[i]['next_state']['node_features'],
                'vehicle_location': batch[i]['next_state']['vehicle_location'],
                'day_of_week': batch[i]['next_state']['day_of_week']
            })

        return {
            'current_state': current_states,
            'actions': actions,
            'rewards': rewards,
            'next_state': next_states,
            'dones': dones
        }

    def get_stats(self):
        """获取缓冲区统计信息"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'current_beta': self.beta_by_frame(self.frame),
            }

        valid_priorities = self.priorities[:len(self.buffer)]

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'current_beta': self.beta_by_frame(self.frame),
            'priority_mean': valid_priorities.mean(),
            'priority_max': valid_priorities.max(),
            'priority_min': valid_priorities.min()
        }

    def __len__(self):
        return len(self.buffer)

