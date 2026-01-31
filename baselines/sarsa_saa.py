import os
import random
import sys
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- 调整 Python 路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 导入项目模块 ---
from config import Config
from utils.data_process import DataProcessor
from environment_baseline import BaselineEnvironment


# ==========================================
# SAA-SARSA Agent 类（修复版）
# ==========================================
class SarsaSAABaseline:
    """
    修复版 SARSA(Δ)-SAA 算法
    基于 Yan et al. (2023) EJOR
    """

    def __init__(self, config, delta=12, alpha=0.1, gamma=0.99, sample_size=7):
        """
        Args:
            config: 全局配置对象
            delta: 前瞻步数 (Look-ahead periods)
            alpha: 学习率 (Learning rate)
            gamma: 折扣因子
            sample_size: SAA 历史样本窗口大小 (|J_t|)
        """
        self.config = config
        self.num_grids = config.NUM_GRIDS
        self.grid_rows = config.GRID_SIZE[0]
        self.grid_cols = config.GRID_SIZE[1]

        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.sample_size = sample_size

        # 调度参数（可动态调整）
        self.max_dispatch_radius = 10  # 默认最大调度半径

        # 1. Q表 (Q-Table)
        self.q_table = defaultdict(float)

        # 2. 历史样本库 (History Memory for SAA)
        # 结构: {time_slot_id: [order_demand_vector_day1, ...]}
        self.history_samples = defaultdict(list)

        # 3. 轨迹缓冲区 (Trajectory Buffer)
        self.trajectory_buffer = deque()

        # 4. 学习率衰减参数
        self.initial_alpha = alpha
        self.alpha_decay = 0.995
        self.min_alpha = 0.01

        # 5. 探索率
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1

    def get_state_hash(self, time_step, vehicle_distribution):
        """
        状态哈希化：时间 + 车辆分布的离散化
        Args:
            time_step: int, 时间槽索引
            vehicle_distribution: list/array, 每个格子的空闲车辆数
        """
        # 将车辆分布离散化为 3 档：低(0-2)、中(3-5)、高(6+)
        discretized = []
        for count in vehicle_distribution:
            if count <= 2:
                discretized.append('L')
            elif count <= 5:
                discretized.append('M')
            else:
                discretized.append('H')

        # 只保留关键格子（需求 Top 10）
        top_indices = np.argsort(vehicle_distribution)[-10:]
        key_state = ''.join([discretized[i] for i in sorted(top_indices)])

        return f"t{time_step}_{key_state}"

    def record_history(self, time_step, current_order_demand):
        """
        记录历史需求
        Args:
            time_step: int, 当天的时间片索引
            current_order_demand: np.array (NUM_GRIDS,), 每个格子的订单数
        """
        # 保持样本窗口滚动
        if len(self.history_samples[time_step]) >= self.sample_size:
            self.history_samples[time_step].pop(0)
        self.history_samples[time_step].append(current_order_demand.copy())

    def solve_saa_dispatch(self, available_vehicles, time_step):
        """
        核心 SAA 优化模块（使用优化模型）
        Args:
            available_vehicles: (NUM_GRIDS,) 数组，当前每个格子的空车数
            time_step: 当前时间步

        Returns:
            dispatch_actions: 字典 {src_grid: {dst_grid: count}}
            expected_value: 预估的价值 (Q值)
        """
        # 1. 获取样本 (Sample Set J_t)
        samples = self.history_samples.get(time_step, [])
        if not samples or len(samples) < 3:  # 至少需要3天数据
            return {}, 0.0

        # 2. 计算平均需求 (Average Demand)
        avg_demand = np.mean(samples, axis=0)

        # 3. 使用优化模型求解
        try:
            from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD

            prob = LpProblem("SAA_Dispatch", LpMaximize)

            # 决策变量: x[src][dst] = 从 src 调度到 dst 的车辆数
            x = {}
            for src in range(self.num_grids):
                for dst in range(self.num_grids):
                    x[src, dst] = LpVariable(f"x_{src}_{dst}", lowBound=0, cat='Integer')

            # 目标函数参数
            revenue_per_order = 10.0
            cost_per_km = 0.5

            # 引入辅助变量：实际匹配数 = min(到达车辆, 需求)
            matched = {}
            for dst in range(self.num_grids):
                matched[dst] = LpVariable(f"matched_{dst}", lowBound=0, cat='Integer')
                arriving_vehicles = lpSum([x[src, dst] for src in range(self.num_grids)])

                # 约束：匹配数不超过到达车辆
                prob += matched[dst] <= arriving_vehicles
                # 约束：匹配数不超过需求
                prob += matched[dst] <= avg_demand[dst]

            # 目标函数: 最大化匹配收益 - 调度成本
            objective = 0

            # 匹配收益
            for dst in range(self.num_grids):
                objective += revenue_per_order * matched[dst]

            # 调度成本
            for src in range(self.num_grids):
                for dst in range(self.num_grids):
                    if src != dst:
                        distance = self._manhattan_distance(src, dst)
                        objective -= cost_per_km * distance * x[src, dst]

            prob += objective

            # 约束1: 每个格子调出的车不超过可用车辆
            for src in range(self.num_grids):
                prob += lpSum([x[src, dst] for dst in range(self.num_grids)]) <= available_vehicles[src]

            # 约束2: 限制调度半径（使用可动态调整的参数）
            for src in range(self.num_grids):
                for dst in range(self.num_grids):
                    if self._manhattan_distance(src, dst) > self.max_dispatch_radius:
                        prob += x[src, dst] == 0

            # 约束3: 不调度到供大于求的格子
            for dst in range(self.num_grids):
                if available_vehicles[dst] > avg_demand[dst] * 1.5:
                    for src in range(self.num_grids):
                        if src != dst:
                            prob += x[src, dst] == 0

            # 求解（静默模式）
            prob.solve(PULP_CBC_CMD(msg=0))

            # 提取结果
            dispatch_instructions = defaultdict(dict)
            expected_value = 0.0

            if LpStatus[prob.status] == 'Optimal':
                expected_value = prob.objective.value() if prob.objective.value() else 0.0

                for src in range(self.num_grids):
                    for dst in range(self.num_grids):
                        count = int(x[src, dst].varValue or 0)
                        if count > 0:
                            dispatch_instructions[src][dst] = count

            return dispatch_instructions, expected_value

        except ImportError:
            # 如果没有 PuLP，回退到贪婪算法
            print("警告: PuLP 未安装，使用贪婪算法")
            return self._greedy_dispatch(available_vehicles, avg_demand)
        except Exception as e:
            print(f"SAA 优化失败: {e}，使用贪婪算法")
            return self._greedy_dispatch(available_vehicles, avg_demand)

    def _greedy_dispatch(self, available_vehicles, avg_demand):
        """
        贪婪调度算法（备用）
        """
        current_supply = np.array(available_vehicles)
        net_flow = current_supply - avg_demand

        surplus_grids = []
        deficit_grids = []

        for g in range(self.num_grids):
            if net_flow[g] > 0.5:
                surplus_grids.append([g, int(net_flow[g])])
            elif net_flow[g] < -0.5:
                deficit_grids.append([g, abs(int(net_flow[g]))])

        surplus_grids.sort(key=lambda x: x[1], reverse=True)
        deficit_grids.sort(key=lambda x: x[1], reverse=True)

        dispatch_instructions = defaultdict(dict)
        total_expected_revenue = 0.0

        s_idx, d_idx = 0, 0
        while s_idx < len(surplus_grids) and d_idx < len(deficit_grids):
            src, s_count = surplus_grids[s_idx]
            dst, d_count = deficit_grids[d_idx]

            distance = self._manhattan_distance(src, dst)

            if distance > 10:
                s_idx += 1
                continue

            move_amount = min(s_count, d_count)

            if move_amount > 0:
                dispatch_instructions[src][dst] = move_amount
                surplus_grids[s_idx][1] -= move_amount
                deficit_grids[d_idx][1] -= move_amount
                total_expected_revenue += move_amount * (10.0 - 0.5 * distance)

            if surplus_grids[s_idx][1] == 0:
                s_idx += 1
            if deficit_grids[d_idx][1] == 0:
                d_idx += 1

        return dispatch_instructions, total_expected_revenue

    def _manhattan_distance(self, src, dst):
        """计算曼哈顿距离"""
        r1, c1 = src // self.grid_cols, src % self.grid_cols
        r2, c2 = dst // self.grid_cols, dst % self.grid_cols
        return abs(r1 - r2) + abs(c1 - c2)

    def update_sarsa(self, state_h, action_val, reward):
        """
        SARSA(Δ) 更新逻辑（修复版）
        基于论文 Algorithm 1 Line 15-19
        """
        self.trajectory_buffer.append({
            'state_h': state_h,
            'q_val': action_val,
            'reward': reward
        })

        # 当缓冲区长度 > Δ 时，更新 Δ 步前的状态
        if len(self.trajectory_buffer) > self.delta:
            past_exp = self.trajectory_buffer.popleft()
            past_state_h = past_exp['state_h']

            # 计算 G_t（带折扣的累积奖励）
            G_t = 0.0
            for i, exp in enumerate(self.trajectory_buffer):
                G_t += (self.gamma ** i) * exp['reward']

            # TD 误差
            old_q = self.q_table[past_state_h]
            td_error = G_t - old_q

            # Q 值更新
            self.q_table[past_state_h] += self.alpha * td_error

    def decay_parameters(self):
        """衰减学习率和探索率"""
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_agent(self, path):
        """保存 Agent 状态"""
        import pickle
        data = {
            'q_table': dict(self.q_table),
            'history_samples': {k: v for k, v in self.history_samples.items()},
            'alpha': self.alpha,
            'epsilon': self.epsilon
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {path}")

    def load_agent(self, path):
        """加载 Agent 状态"""
        import pickle
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(float, data['q_table'])
                self.history_samples = defaultdict(list, data['history_samples'])
                self.alpha = data.get('alpha', self.alpha)
                self.epsilon = data.get('epsilon', self.epsilon)
            print(f"Agent loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")

    def prefill_history(self, env, train_days_indices):
        """
        快速预填充 SAA 历史样本（向量化版本）
        """
        print("正在预填充 SAA 历史需求数据...")

        try:
            # 1. 获取所有订单
            all_orders = env.order_generator.orders_df.copy()

            # 2. 筛选训练集
            train_orders = all_orders[all_orders['relative_day'].isin(train_days_indices)]

            if len(train_orders) == 0:
                print("警告: 训练集订单为空")
                return

            # 3. 计算 SAA 时间槽（30分钟）
            train_orders['saa_slot'] = (train_orders['timestamp'].dt.hour * 60 +
                                        train_orders['timestamp'].dt.minute) // 30

            # 4. 按 (relative_day, saa_slot, grid_index) 聚合
            grouped = train_orders.groupby(['relative_day', 'saa_slot', 'grid_index']).size()

            # 5. 填充历史
            for (day, slot, grid), count in tqdm(grouped.items(), desc="Filling History"):
                # 初始化该 slot 的历史
                while len(self.history_samples[slot]) <= day:
                    self.history_samples[slot].append(np.zeros(self.num_grids, dtype=int))

                self.history_samples[slot][day][grid] = count

            print(f"历史数据预填充完成。共 {len(self.history_samples)} 个时间槽。")

        except Exception as e:
            print(f"预填充失败: {e}")
            import traceback
            traceback.print_exc()


# ==========================================
# 辅助函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_current_tick_demand(env, current_time, duration_sec):
    """
    获取当前 tick 内每个网格生成的新订单数
    """
    try:
        start_timestamp = env.order_generator.time_range[0].normalize()
        relative_day = (current_time - start_timestamp).days

        minutes_from_midnight = current_time.hour * 60 + current_time.minute
        time_slice = minutes_from_midnight // env.config.MACRO_STATISTICS_STEP_MINUTES
        time_slice = max(0, min(time_slice, 143))

        key = (relative_day, time_slice)
        slice_orders = env.order_generator.orders_by_day_and_slice.get(key, [])

        end_time = current_time + pd.Timedelta(seconds=duration_sec)
        demand_vector = np.zeros(env.config.NUM_GRIDS, dtype=int)

        for order in slice_orders:
            t = order['timestamp']
            if current_time <= t < end_time:
                grid_id = order['grid_index']
                if 0 <= grid_id < env.config.NUM_GRIDS:
                    demand_vector[grid_id] += 1

        return demand_vector

    except Exception:
        return np.zeros(env.config.NUM_GRIDS, dtype=int)


# ==========================================
# 主模拟循环
# ==========================================
def run_simulation_phase(phase_name, days_to_run, env, agent, config, round_idx,
                         current_seed, is_training=False):
    """
    运行模拟阶段 (训练或测试)
    """
    rows = []
    daily_metrics_list = []

    pbar = tqdm(days_to_run, desc=f"{phase_name} (Round {round_idx + 1})", unit="day")

    for d in pbar:
        try:
            # 重置环境
            env.reset()
            env.current_day = d
            env.episode_start_day = d
            env.simulation_time = env.order_generator.time_range[0].normalize() + pd.Timedelta(days=d)
            if env.simulation_time.tzinfo is None:
                env.simulation_time = env.simulation_time.tz_localize('Asia/Shanghai')
            env.current_time = env.simulation_time
            env.vehicle_manager.reset()
            env.reward_calculator.reset()
            env.pending_orders.clear()
            env.event_queue.clear()
            env.buffered_orders.clear()
            env.current_macro_slice_key = None
            env.daily_stats.clear()

            # 清空 buffer
            agent.trajectory_buffer.clear()

        except Exception as e:
            print(f"错误: env 重置失败在 Day {d}: {e}")
            continue

        ticks = 0
        daily_infos = defaultdict(list)

        # --- 单日 Ticks 循环 ---
        while ticks < config.TICKS_PER_DAY:
            try:
                current_minutes = (ticks * config.TICK_DURATION_SEC) / 60
                saa_time_slot = int(current_minutes // 30)

                # 获取真实需求并记录
                current_demand = get_current_tick_demand(env, env.simulation_time, config.TICK_DURATION_SEC)
                agent.record_history(saa_time_slot, current_demand)

                # 调度决策（每 5 分钟一次）
                dispatch_instructions = {}
                est_value = 0.0

                if ticks % 10 == 0:
                    # 获取车辆分布
                    idle_vehicles_list = [0] * config.NUM_GRIDS
                    for v in env.vehicle_manager.vehicles.values():
                        if v['status'] == 'idle':
                            idle_vehicles_list[v['current_grid']] += 1

                    # 始终执行 SAA 调度（ε 用于控制调度激进程度）
                    # 探索期：更保守的调度半径和阈值
                    # 利用期：使用学到的最优参数
                    if is_training and random.random() < agent.epsilon:
                        # 探索：使用较小的调度半径（更保守）
                        original_radius = agent.max_dispatch_radius
                        agent.max_dispatch_radius = max(5, int(original_radius * 0.6))
                        dispatch_instructions, est_value = agent.solve_saa_dispatch(
                            idle_vehicles_list, saa_time_slot
                        )
                        agent.max_dispatch_radius = original_radius
                    else:
                        # 利用：使用标准参数
                        dispatch_instructions, est_value = agent.solve_saa_dispatch(
                            idle_vehicles_list, saa_time_slot
                        )

                    # 执行调度
                    for src, targets in dispatch_instructions.items():
                        candidates = [
                            vid for vid, v in env.vehicle_manager.vehicles.items()
                            if v['current_grid'] == src and v['status'] == 'idle'
                        ]
                        candidate_idx = 0
                        for dst, count in targets.items():
                            for _ in range(min(count, len(candidates) - candidate_idx)):
                                if candidate_idx < len(candidates):
                                    env.vehicle_manager.start_dispatching(
                                        candidates[candidate_idx], dst, env.simulation_time
                                    )
                                    candidate_idx += 1

                # 环境步进
                _, reward, _, info = env.step()
                daily_infos[0].append(info.get('step_info', {}))

                # SARSA 更新
                state_key = agent.get_state_hash(saa_time_slot, idle_vehicles_list)
                agent.update_sarsa(state_key, est_value, reward)

            except Exception as e:
                print(f"错误: Day {d}, Tick {ticks}: {e}")
                break

            ticks += 1

        # 每日结算
        total_matched = sum(info.get('matched_orders', 0) for info in daily_infos[0])
        total_completed = sum(info.get('completed_orders', 0) for info in daily_infos[0])
        total_cancelled = sum(info.get('cancelled_orders', 0) for info in daily_infos[0])
        total_orders = sum(info.get('new_orders', 0) for info in daily_infos[0])
        total_revenue = sum(info.get('revenue', 0.0) for info in daily_infos[0])
        all_waiting = [wt for info in daily_infos[0] for wt in info.get('waiting_times', [])]

        # 计算指标
        match_rate = total_matched / total_orders if total_orders > 0 else 0
        total_processed = total_completed + total_cancelled
        completion_rate = total_completed / total_processed if total_processed > 0 else 0
        avg_waiting_time = float(np.mean(all_waiting)) if all_waiting else 0.0

        if not is_training:
            # 测试阶段记录详细日志
            metrics = {
                'day_index': 0,
                'actual_day': d,
                'dataset_day_index': d,
                'matched_orders': total_matched,
                'completed_orders': total_completed,
                'cancelled_orders': total_cancelled,
                'total_orders': total_orders,
                'match_rate': match_rate,
                'completion_rate': completion_rate,
                'avg_waiting_time': avg_waiting_time,
                'total_revenue': total_revenue,
                'benchmark_policy': 'sarsa_saa',
                'round_index': round_idx,
                'seed': current_seed
            }
            rows.append(metrics)
            daily_metrics_list.append(match_rate)

            # 增量保存
            temp_df = pd.DataFrame(rows)
            ts_part = datetime.now().strftime('%Y%m%d')
            temp_csv = os.path.join(config.LOG_SAVE_PATH, 'results',
                                    f'benchmark_sarsa_saa_progress_{ts_part}.csv')
            os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
            temp_df.to_csv(temp_csv, index=False)

            pbar.set_description(f"{phase_name} R{round_idx + 1} (Day {d}: Match={match_rate:.1%}, Cmp={completion_rate:.1%})")
        else:
            daily_metrics_list.append(match_rate)
            pbar.set_description(f"{phase_name} R{round_idx + 1} (Day {d}: Match={match_rate:.1%})")

        # 训练阶段衰减参数
        if is_training:
            agent.decay_parameters()

    # 返回平均完成率
    avg_performance = np.mean(daily_metrics_list) if daily_metrics_list else 0.0
    return rows, avg_performance


# ==========================================
# 主函数
# ==========================================
def run_sarsa_saa_simulation(config, num_episodes, env_data, num_rounds=5):
    """
    运行 SAA-SARSA Benchmark (5轮训练+测试)
    """
    print(f"\n--- 开始 SAA-SARSA Benchmark (Total Rounds: {num_rounds}) ---")
    print(f"  车辆总数: {config.TOTAL_VEHICLES}")

    # 准备数据
    try:
        data_processor = DataProcessor(config)
    except Exception as e:
        print(f"创建 DataProcessor 出错: {e}")
        return {}

    all_daily_rows = []

    # 多轮循环
    for round_idx in range(num_rounds):
        current_seed = config.SEED + round_idx
        set_seed(current_seed)
        print(f"\n>>> Round {round_idx + 1}/{num_rounds} (Seed: {current_seed})")

        # 初始化 Agent
        agent = SarsaSAABaseline(config, delta=12, sample_size=7)

        # 创建环境
        try:
            env = BaselineEnvironment(config, data_processor, env_data, dispatch_policy='none')
        except Exception as e:
            print(f"创建环境时出错: {e}")
            continue

        # 确定日期划分
        try:
            day_count = env.order_generator.get_day_count()
        except Exception:
            day_count = 0

        test_days_count = 7
        if day_count <= test_days_count:
            print("数据量不足，无法拆分训练/测试集")
            train_days = []
            test_days = list(range(day_count))
        else:
            test_start = day_count - test_days_count
            train_days = list(range(test_start))
            test_days = list(range(test_start, day_count))

        print(f"  训练集: {len(train_days)} 天")
        print(f"  测试集: {len(test_days)} 天")

        # Phase 0: 预填充历史
        agent.prefill_history(env, train_days)

        # Phase 1: 训练
        train_simulation_days = train_days[-15:] if len(train_days) > 15 else train_days
        print(f"  正在运行 SARSA 强化训练 ({len(train_simulation_days)} 天)...")

        best_train_performance = 0.0
        early_stopping_counter = 0
        early_stopping_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 3)

        # 动态调整检查窗口：确保至少检查10次
        check_window = max(3, len(train_simulation_days) // 10)
        print(f"    训练检查窗口: {check_window} 天/批次")

        for i in range(0, len(train_simulation_days), check_window):
            batch_days = train_simulation_days[i:i + check_window]
            _, current_performance = run_simulation_phase(
                "Training", batch_days, env, agent, config, round_idx, current_seed, is_training=True
            )

            if current_performance > best_train_performance:
                best_train_performance = current_performance
                early_stopping_counter = 0
                print(f"    ✓ 性能提升: {current_performance:.2%}")
            else:
                early_stopping_counter += 1
                print(f"    早停计数: {early_stopping_counter}/{early_stopping_patience}")

            if early_stopping_counter >= early_stopping_patience:
                print(f"  早停触发！最佳性能: {best_train_performance:.2%}")
                break

        # Phase 2: 测试
        print(f"  正在运行测试 (最后 7 天)...")
        test_rows, _ = run_simulation_phase(
            "Testing", test_days, env, agent, config, round_idx, current_seed, is_training=False
        )
        all_daily_rows.extend(test_rows)

    # 汇总分析
    if all_daily_rows:
        df = pd.DataFrame(all_daily_rows)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存原始数据
        raw_csv = os.path.join(config.LOG_SAVE_PATH, 'results',
                               f'benchmark_sarsa_saa_last7_raw_{ts}.csv')
        os.makedirs(os.path.dirname(raw_csv), exist_ok=True)
        df.to_csv(raw_csv, index=False)
        print(f"\n✓ 已保存原始数据到: {raw_csv}")

        # 计算汇总
        summary_rows = []
        grouped = df.groupby('dataset_day_index')
        for day, group in grouped:
            summary_rows.append({
                'Day': day,
                'Avg_Match_Rate': group['match_rate'].mean(),
                'Avg_Completion_Rate': group['completion_rate'].mean(),
                'Avg_Waiting_Time': group['avg_waiting_time'].mean(),
                'Best_Completion_Rate': group['completion_rate'].max(),
                'Worst_Completion_Rate': group['completion_rate'].min(),
                'Avg_Revenue': group['total_revenue'].mean()
            })

        df_summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(config.LOG_SAVE_PATH, 'results',
                                   f'benchmark_sarsa_saa_summary_{ts}.csv')
        df_summary.to_csv(summary_csv, index=False)
        print(f"✓ 已保存汇总结果到: {summary_csv}")

        try:
            from tabulate import tabulate
            print("\n" + "=" * 60)
            print("SAA-SARSA 5-Round Summary")
            print("=" * 60)
            print(tabulate(df_summary, headers='keys', tablefmt='psql',
                           floatfmt='.4f', showindex=False))
        except:
            print(df_summary.to_string(index=False))

    return all_daily_rows


if __name__ == '__main__':
    print("=" * 80)
    print("运行 Benchmark: SARSA-SAA (Yan et al. 2023)")
    print("=" * 80)

    try:
        config = Config()
        if not Config.validate_config():
            sys.exit(1)

        print("加载数据...")
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()
        print(f"总订单数: {len(all_orders):,} 条")

        # 运行
        run_sarsa_saa_simulation(config, 0, all_orders, num_rounds=5)

    except Exception as e:
        print(f"运行时发生严重错误: {e}")
        import traceback

        traceback.print_exc()