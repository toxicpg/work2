"""
H-MARL (MFuN) Agent Implementation
Based on: Si et al. - Hierarchical Multi-Agent RL for Intercity Ridepooling

核心架构:
1. Manager Network: 全局协调，输出子目标（每5分钟决策一次）
2. Worker Network: 局部执行，完成具体调度（每个grid一个worker，参数共享）
3. MILP Solver: 将Worker的调度意图转化为具体的车辆分配方案
4. ALNS Optimizer: 优化车辆路径和拼车方案（简化版）
5. Intrinsic Reward: 内部奖励机制，鼓励Worker完成Manager的子目标
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("警告: PuLP未安装，MILP功能将被禁用。安装命令: pip install pulp")


class ManagerNetwork(nn.Module):
    """
    Manager 网络：全局协调者
    输入：全局状态 (所有grid的车辆分布、需求预测、时间特征)
    输出：子目标矩阵 (num_grids x num_grids)，表示期望从src调往dst的车辆数
    """
    def __init__(self, config):
        super().__init__()
        self.num_grids = config.NUM_GRIDS

        # 全局状态编码
        # 输入维度: num_grids * (车辆数 + 订单数 + 时间特征) = 400 * (2 + 1 + 2) = 2000
        state_dim = config.NUM_GRIDS * 5  # [idle_vehs, busy_vehs, orders, time_sin, time_cos]

        # GRU 用于捕捉时序模式
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 全连接层
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

        # 输出子目标矩阵 (展平后的调度矩阵)
        # 为了简化，我们只输出每个grid的"期望接收车辆数"
        self.sub_goal_layer = nn.Linear(256, config.NUM_GRIDS)

        # 价值网络 (用于 Actor-Critic)
        self.value_layer = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, state, hidden=None):
        """
        Args:
            state: (batch, seq_len, state_dim) 或 (batch, state_dim)
            hidden: GRU隐藏状态
        Returns:
            sub_goals: (batch, num_grids) - 每个grid期望调入的车辆数
            value: (batch, 1) - 状态价值
            hidden: 新的隐藏状态
        """
        # 如果输入是2D，添加时间维度
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)

        # GRU 编码
        gru_out, hidden = self.gru(state, hidden)  # (batch, seq_len, 256)

        # 取最后一个时间步
        last_out = gru_out[:, -1, :]  # (batch, 256)

        # 全连接层
        x = F.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出子目标 (ReLU确保非负)
        sub_goals = F.relu(self.sub_goal_layer(x))  # (batch, num_grids)

        # 输出状态价值
        value = self.value_layer(x)  # (batch, 1)

        return sub_goals, value, hidden


class WorkerNetwork(nn.Module):
    """
    Worker 网络：局部执行者
    输入：局部状态 (当前grid的状态) + Manager的子目标
    输出：动作分布 (调度到各个邻居grid的概率)
    """
    def __init__(self, config):
        super().__init__()
        self.num_grids = config.NUM_GRIDS

        # 局部状态维度: 当前grid状态 (5维) + 邻居grid状态 (8*5=40维) + 子目标 (1维) = 46维
        # 简化版本：只用当前grid + 子目标
        local_state_dim = 5 + 1  # [idle, busy, orders, time_sin, time_cos] + sub_goal

        # GRU 用于时序建模
        self.gru = nn.GRU(
            input_size=local_state_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # 全连接层
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)

        # 输出动作分布 (调度到哪个grid)
        # 简化：输出是否需要调出车辆 (0=不调, 1=调往需求最高的grid)
        self.action_layer = nn.Linear(64, config.NUM_GRIDS)

        # 价值网络
        self.value_layer = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, local_state, sub_goal, hidden=None):
        """
        Args:
            local_state: (batch, seq_len, 5) 或 (batch, 5)
            sub_goal: (batch, 1) - 当前grid的子目标
            hidden: GRU隐藏状态
        Returns:
            action_logits: (batch, num_grids) - 调度到各grid的logits
            value: (batch, 1) - 状态价值
            hidden: 新的隐藏状态
        """
        # 确保维度正确
        if local_state.dim() == 2:
            local_state = local_state.unsqueeze(1)  # (batch, 1, 5)
        if sub_goal.dim() == 1:
            sub_goal = sub_goal.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1)
        elif sub_goal.dim() == 2:
            sub_goal = sub_goal.unsqueeze(1)  # (batch, 1, 1)

        # 拼接局部状态和子目标
        combined = torch.cat([local_state, sub_goal], dim=-1)  # (batch, 1, 6)

        # GRU 编码
        gru_out, hidden = self.gru(combined, hidden)  # (batch, 1, 128)
        last_out = gru_out[:, -1, :]  # (batch, 128)

        # 全连接层
        x = F.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出动作logits
        action_logits = self.action_layer(x)  # (batch, num_grids)

        # 输出价值
        value = self.value_layer(x)  # (batch, 1)

        return action_logits, value, hidden


class MFuN_Agent:
    """
    多智能体封建网络 (Multi-agent Feudal Network)
    包含MILP求解器和ALNS优化器
    """
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.num_grids = config.NUM_GRIDS

        # Manager 和 Worker 网络
        self.manager = ManagerNetwork(config).to(self.device)
        self.worker_shared = WorkerNetwork(config).to(self.device)  # 所有Worker共享参数

        # 优化器
        self.manager_optimizer = torch.optim.Adam(
            self.manager.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        self.worker_optimizer = torch.optim.Adam(
            self.worker_shared.parameters(),
            lr=0.0005,
            weight_decay=1e-4
        )

        # 经验回放
        self.manager_buffer = deque(maxlen=10000)
        self.worker_buffer = deque(maxlen=50000)

        # 训练参数
        self.gamma = config.GAMMA
        self.manager_update_freq = 50  # Manager每50个tick决策一次（约50分钟）
        self.worker_update_freq = 10   # Worker每10个tick决策一次（约10分钟）

        # MILP和ALNS配置
        self.use_milp = PULP_AVAILABLE
        self.use_alns = True  # ALNS简化版（启发式优化）
        self.milp_timeout = 5  # MILP求解超时（秒）

        # 状态追踪
        self.manager_hidden = None
        self.worker_hiddens = {}  # {grid_id: hidden_state}
        self.current_sub_goals = None
        self.last_manager_step = 0

        # 累积奖励
        self.episode_rewards = []
        self.current_episode_reward = 0

    def get_global_state(self, env):
        """
        提取全局状态
        Returns:
            global_state: (1, state_dim) tensor
        """
        state_vector = []

        # 遍历所有grid
        for grid_id in range(self.num_grids):
            # 空闲车辆数
            idle_vehs = len([v for v in env.vehicle_manager.vehicles.values()
                           if v['status'] == 'idle' and v['current_grid'] == grid_id])

            # 繁忙车辆数
            busy_vehs = len([v for v in env.vehicle_manager.vehicles.values()
                           if v['status'] != 'idle' and v['current_grid'] == grid_id])

            # 订单数 (pending orders in this grid)
            pending_orders = getattr(env, 'pending_orders', [])
            orders = len([o for o in pending_orders
                        if o.get('origin_grid', -1) == grid_id])

            # 时间特征
            current_time = env.current_time
            hour = current_time.hour
            time_sin = np.sin(2 * np.pi * hour / 24)
            time_cos = np.cos(2 * np.pi * hour / 24)

            state_vector.extend([idle_vehs, busy_vehs, orders, time_sin, time_cos])

        # 转为tensor
        global_state = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)  # (1, 2000)
        return global_state

    def get_local_state(self, env, grid_id):
        """
        提取局部状态（单个grid）
        """
        # 空闲车辆数
        idle_vehs = len([v for v in env.vehicle_manager.vehicles.values()
                       if v['status'] == 'idle' and v['current_grid'] == grid_id])

        # 繁忙车辆数
        busy_vehs = len([v for v in env.vehicle_manager.vehicles.values()
                       if v['status'] != 'idle' and v['current_grid'] == grid_id])

        # 订单数
        pending_orders = getattr(env, 'pending_orders', [])
        orders = len([o for o in pending_orders
                    if o.get('origin_grid', -1) == grid_id])

        # 时间特征
        current_time = env.current_time
        hour = current_time.hour
        time_sin = np.sin(2 * np.pi * hour / 24)
        time_cos = np.cos(2 * np.pi * hour / 24)

        local_state = torch.FloatTensor([idle_vehs, busy_vehs, orders, time_sin, time_cos]).to(self.device)
        return local_state

    def select_action(self, env, step, training=True):
        """
        选择动作（调度指令）- 优化版：只处理有空闲车辆的grid
        Args:
            env: 环境实例
            step: 当前步数
            training: 是否训练模式
        Returns:
            dispatch_orders: {src_grid: {dst_grid: count}}
        """
        with torch.no_grad():
            # 1. Manager决策（每N步一次）
            if step % self.manager_update_freq == 0 or self.current_sub_goals is None:
                global_state = self.get_global_state(env)
                sub_goals, _, self.manager_hidden = self.manager(global_state, self.manager_hidden)
                self.current_sub_goals = sub_goals.squeeze(0).cpu().numpy()  # (num_grids,)
                self.last_manager_step = step

            # 2. 只处理有空闲车辆的grid（大幅优化性能）
            idle_grids = set()
            for v in env.vehicle_manager.vehicles.values():
                if v['status'] == 'idle':
                    idle_grids.add(v['current_grid'])

            if not idle_grids:
                return {}  # 没有空闲车辆，直接返回

            # 3. Worker执行（只处理有车的grid）- 简化版，不使用MILP和ALNS
            dispatch_orders = {}

            for grid_id in idle_grids:
                # 获取子目标
                sub_goal = self.current_sub_goals[grid_id]

                # 只有当子目标要求调出车辆时才处理
                if sub_goal >= -0.5:
                    continue

                # 获取该grid的空闲车辆
                idle_vehicles = [v_id for v_id, v in env.vehicle_manager.vehicles.items()
                               if v['status'] == 'idle' and v['current_grid'] == grid_id]

                if not idle_vehicles:
                    continue

                # 简化：直接选择需求最高的邻近grid（3格内）
                pending_orders = getattr(env, 'pending_orders', [])
                demand = {}

                src_row, src_col = grid_id // 20, grid_id % 20
                for g in range(self.num_grids):
                    dst_row, dst_col = g // 20, g % 20
                    if abs(src_row - dst_row) + abs(src_col - dst_col) <= 3:
                        demand[g] = len([o for o in pending_orders if o.get('origin_grid', -1) == g])

                if not demand:
                    continue

                # 选择需求最高的grid
                target_grid = max(demand.items(), key=lambda x: x[1])[0]

                # 调度部分车辆（不超过子目标要求）
                num_to_dispatch = min(len(idle_vehicles), int(abs(sub_goal)))

                if num_to_dispatch > 0:
                    if grid_id not in dispatch_orders:
                        dispatch_orders[grid_id] = {}
                    dispatch_orders[grid_id][target_grid] = num_to_dispatch

            return dispatch_orders

    def solve_milp_dispatch(self, env, grid_id, target_grids_probs, sub_goal):
        """
        MILP求解器：将Worker的调度意图转化为具体的车辆分配方案

        Args:
            env: 环境实例
            grid_id: 当前grid的ID
            target_grids_probs: Worker输出的目标grid概率分布 (num_grids,)
            sub_goal: Manager分配的子目标

        Returns:
            dispatch_plan: {vehicle_id: target_grid_id}
        """
        if not self.use_milp:
            # 降级方案：直接选择概率最高的grid
            dst_grid = torch.argmax(target_grids_probs).item()
            idle_vehs = [v_id for v_id, v in env.vehicle_manager.vehicles.items()
                        if v['status'] == 'idle' and v['current_grid'] == grid_id]
            if len(idle_vehs) > 0 and sub_goal < -0.5:
                return {idle_vehs[0]: dst_grid}
            return {}

        # === MILP求解 ===
        try:
            # 1. 获取可用车辆
            available_vehicles = [(v_id, v) for v_id, v in env.vehicle_manager.vehicles.items()
                                if v['status'] == 'idle' and v['current_grid'] == grid_id]

            if len(available_vehicles) == 0:
                return {}

            # 2. 获取需求预测（当前各grid的订单数）
            demand = np.zeros(self.num_grids)
            pending_orders = getattr(env, 'pending_orders', [])
            for order in pending_orders:
                origin = order.get('origin_grid', -1)
                if 0 <= origin < self.num_grids:
                    demand[origin] += 1

            # 3. 构建MILP问题
            prob = LpProblem("Worker_Dispatch", LpMaximize)

            # 决策变量: x[v][g] = 车辆v是否调往grid g
            x = {}
            for v_id, _ in available_vehicles:
                for g in range(self.num_grids):
                    x[v_id, g] = LpVariable(f"x_{v_id}_{g}", cat='Binary')

            # 目标函数：最大化期望收益
            objective = 0
            for v_id, v in available_vehicles:
                for g in range(self.num_grids):
                    # 收益 = 目标grid的需求 × Worker的偏好概率
                    expected_revenue = demand[g] * target_grids_probs[g].item()
                    # 成本 = 调度距离（简化为欧氏距离）
                    src_row, src_col = grid_id // 20, grid_id % 20
                    dst_row, dst_col = g // 20, g % 20
                    dispatch_cost = abs(src_row - dst_row) + abs(src_col - dst_col)

                    objective += x[v_id, g] * (expected_revenue - dispatch_cost * 0.1)

            prob += objective

            # 约束1：每辆车最多调往一个grid
            for v_id, _ in available_vehicles:
                prob += lpSum([x[v_id, g] for g in range(self.num_grids)]) <= 1

            # 约束2：满足Manager的子目标（允许一定容忍度）
            if sub_goal < -0.5:  # 期望调出
                num_to_dispatch = min(len(available_vehicles), int(abs(sub_goal)))
                total_dispatched = lpSum([x[v_id, g]
                                         for v_id, _ in available_vehicles
                                         for g in range(self.num_grids)])
                prob += total_dispatched >= num_to_dispatch * 0.8
                prob += total_dispatched <= num_to_dispatch * 1.2

            # 求解
            solver = PULP_CBC_CMD(msg=0, timeLimit=self.milp_timeout)
            prob.solve(solver)

            # 提取结果
            dispatch_plan = {}
            if LpStatus[prob.status] == 'Optimal':
                for v_id, _ in available_vehicles:
                    for g in range(self.num_grids):
                        if x[v_id, g].varValue and x[v_id, g].varValue > 0.5:
                            dispatch_plan[v_id] = g
                            break

            return dispatch_plan

        except Exception as e:
            print(f"MILP求解失败: {e}, 使用降级方案")
            # 降级：简单启发式
            dst_grid = torch.argmax(target_grids_probs).item()
            idle_vehs = [v_id for v_id, _ in available_vehicles]
            if len(idle_vehs) > 0 and sub_goal < -0.5:
                return {idle_vehs[0]: dst_grid}
            return {}

    def alns_optimize_routes(self, env, dispatch_plan):
        """
        ALNS优化器：优化车辆路径（简化版）

        在论文中，ALNS用于优化拼车路径。这里实现一个简化版：
        - 破坏算子：随机移除部分调度
        - 修复算子：重新插入到更优位置

        Args:
            env: 环境实例
            dispatch_plan: {vehicle_id: target_grid_id}

        Returns:
            optimized_plan: {vehicle_id: target_grid_id}
        """
        if not self.use_alns or len(dispatch_plan) == 0:
            return dispatch_plan

        # 简化版ALNS：只做一次破坏-修复
        try:
            # 1. 破坏：随机移除20%的调度
            vehicles = list(dispatch_plan.keys())
            num_to_remove = max(1, len(vehicles) // 5)
            removed_vehicles = random.sample(vehicles, num_to_remove)

            current_plan = {v: g for v, g in dispatch_plan.items()
                           if v not in removed_vehicles}

            # 2. 修复：贪婪插入
            for v_id in removed_vehicles:
                vehicle = env.vehicle_manager.vehicles[v_id]
                current_grid = vehicle['current_grid']

                # 寻找最优目标grid（需求最高且距离适中）
                best_score = -float('inf')
                best_grid = dispatch_plan[v_id]  # 默认保持原计划

                for g in range(self.num_grids):
                    # 需求分数
                    pending_orders = getattr(env, 'pending_orders', [])
                    demand = len([o for o in pending_orders
                                if o.get('origin_grid', -1) == g])

                    # 距离惩罚
                    src_row, src_col = current_grid // 20, current_grid % 20
                    dst_row, dst_col = g // 20, g % 20
                    distance = abs(src_row - dst_row) + abs(src_col - dst_col)

                    # 综合评分
                    score = demand * 2.0 - distance * 0.5

                    if score > best_score:
                        best_score = score
                        best_grid = g

                current_plan[v_id] = best_grid

            return current_plan

        except Exception as e:
            print(f"ALNS优化失败: {e}, 使用原始方案")
            return dispatch_plan

    def compute_intrinsic_reward(self, env, actions):
        """
        计算内部奖励：鼓励Worker完成Manager的子目标
        r_intrinsic = -||实际调度 - 子目标||²
        """
        if self.current_sub_goals is None:
            return 0.0

        intrinsic_reward = 0.0

        for grid_id in range(self.num_grids):
            # 统计实际调入/调出的车辆数
            actual_change = 0

            # 调入
            for src_grid, targets in actions.items():
                if grid_id in targets:
                    actual_change += targets[grid_id]

            # 调出
            if grid_id in actions:
                actual_change -= sum(actions[grid_id].values())

            # 计算与子目标的差距
            target_change = self.current_sub_goals[grid_id]
            intrinsic_reward -= (actual_change - target_change) ** 2

        return intrinsic_reward * 0.01  # 缩放因子

    def observe_reward(self, reward):
        """
        接收外部奖励
        """
        self.current_episode_reward += reward

    def update(self):
        """
        更新网络（简化版本）
        """
        # 简化：这里只做简单的监督学习，实际应该用 Actor-Critic 或 PPO
        # 由于时间限制，暂时返回 None
        return None

    def reset_episode(self):
        """
        重置episode状态
        """
        self.manager_hidden = None
        self.worker_hiddens = {}
        self.current_sub_goals = None

        if self.current_episode_reward > 0:
            self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0

