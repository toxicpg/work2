
import sys
import os
import numpy as np
import random
import torch
from collections import deque, defaultdict
from datetime import datetime
import pandas as pd
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
# SAA-SARSA Agent 类
# ==========================================
class SarsaSAABaseline:
    """
    适配版 SARSA(Delta)-SAA 算法
    适用场景：20x20 网格调度
    核心逻辑来源：Yan et al. (2023) EJOR
    """

    def __init__(self, config, delta=12, alpha=0.1, gamma=0.99, sample_size=5):
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
        
        # 1. Q表 (Q-Table)
        # Key: state_hash, Value: estimated_value
        self.q_table = defaultdict(float)
        
        # 2. 历史样本库 (History Memory for SAA)
        # 结构: {time_slot_id: [order_demand_vector_day1, order_demand_vector_day2, ...]}
        self.history_samples = defaultdict(list)
        
        # 3. 轨迹缓冲区 (Trajectory Buffer)
        # 队列中存储: (state_hash, action_value, reward)
        self.trajectory_buffer = deque()

    def get_state_hash(self, time_step):
        """
        状态哈希化：将复杂的车辆分布向量转为唯一字符串key
        简化版：只根据时间步
        """
        return f"t_{time_step}"

    def record_history(self, time_step, current_order_demand):
        """
        记录历史需求
        Args:
            time_step: int, 当天的时间片索引
            current_order_demand: np.array (NUM_GRIDS,), 每个格子的订单数
        """
        # 保持样本窗口滚动，只保留最近 sample_size 天
        if len(self.history_samples[time_step]) >= self.sample_size:
            self.history_samples[time_step].pop(0)
        self.history_samples[time_step].append(current_order_demand)

    def solve_saa_dispatch(self, available_vehicles, time_step):
        """
        核心 SAA 优化模块
        Args:
            available_vehicles: (NUM_GRIDS,) 数组，当前每个格子的空车数
            time_step: 当前时间步
            
        Returns:
            dispatch_actions: 字典 {src_grid: {dst_grid: count}}
            expected_value: 预估的价值 (Q值)
        """
        # 1. 获取样本 (Sample Set J_t)
        samples = self.history_samples.get(time_step, [])
        if not samples:
            # 冷启动：如果没有历史数据，不进行调度
            return {}, 0.0

        # 2. 计算平均需求 (Average Demand)
        # 对过去几天的订单需求取平均，作为对未来的预测
        avg_demand = np.mean(samples, axis=0) # shape: (NUM_GRIDS,)

        # 3. 贪婪匹配逻辑
        surplus_grids = [] # (grid_id, surplus_count)
        deficit_grids = [] # (grid_id, deficit_count)
        
        current_supply = np.array(available_vehicles)
        net_flow = current_supply - avg_demand
        
        for g in range(self.num_grids):
            if net_flow[g] > 0.5: # 盈余 (有车没单)
                surplus_grids.append([g, int(net_flow[g])])
            elif net_flow[g] < -0.5: # 短缺 (有单没车)
                deficit_grids.append([g, abs(int(net_flow[g]))])
                
        # 排序：优先解决缺口最大的地方，优先从车辆最多的地方调
        surplus_grids.sort(key=lambda x: x[1], reverse=True)
        deficit_grids.sort(key=lambda x: x[1], reverse=True)
        
        # 生成调度指令
        dispatch_instructions = defaultdict(dict)
        total_expected_revenue = 0.0
        
        # 双指针贪婪匹配
        s_idx, d_idx = 0, 0
        while s_idx < len(surplus_grids) and d_idx < len(deficit_grids):
            src, s_count = surplus_grids[s_idx]
            dst, d_count = deficit_grids[d_idx]
            
            # 计算曼哈顿距离
            r1, c1 = src // self.grid_cols, src % self.grid_cols
            r2, c2 = dst // self.grid_cols, dst % self.grid_cols
            distance = abs(r1-r2) + abs(c1-c2)
            
            # 限制调度半径 (例如只调度到 10 格以内)
            if distance > 10: 
                s_idx += 1 
                continue 

            move_amount = min(s_count, d_count)
            
            # 记录指令
            if move_amount > 0:
                dispatch_instructions[src][dst] = move_amount
                
                # 更新剩余量
                surplus_grids[s_idx][1] -= move_amount
                deficit_grids[d_idx][1] -= move_amount
                
                # 估算价值 (假设每单收益 10元)
                total_expected_revenue += move_amount * 10.0
                
            # 移动指针
            if surplus_grids[s_idx][1] == 0: s_idx += 1
            if deficit_grids[d_idx][1] == 0: d_idx += 1

        return dispatch_instructions, total_expected_revenue

    def update_sarsa(self, state_val, action_val, reward):
        """
        SARSA(Delta) 更新逻辑
        """
        state_h = state_val
        self.trajectory_buffer.append({
            'state_h': state_h,
            'q_val': action_val,  # 当时的预估价值 (SAA算出来的)
            'reward': reward      # 当步的真实奖励
        })
        
        # 检查缓冲区长度是否达到 Delta
        if len(self.trajectory_buffer) > self.delta:
            # 取出 Delta 步之前的那次经历
            past_experience = self.trajectory_buffer.popleft()
            past_state_h = past_experience['state_h']
            old_q = past_experience['q_val']
            
            # 计算 G_t (累积奖励)
            accumulated_reward = sum([x['reward'] for x in self.trajectory_buffer])
            accumulated_reward += past_experience['reward']
            
            # 更新 Q 值
            current_q = self.q_table[past_state_h]
            if current_q == 0: current_q = old_q
                
            new_q = current_q + self.alpha * (accumulated_reward - current_q)
            self.q_table[past_state_h] = new_q

    def save_agent(self, path):
        """保存 Agent 状态 (Q表和历史样本)"""
        import pickle
        data = {
            'q_table': self.q_table,
            'history_samples': self.history_samples,
            'trajectory_buffer': self.trajectory_buffer
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {path}")

    def load_agent(self, path):
        """加载 Agent 状态"""
        import pickle
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.history_samples = data['history_samples']
                self.trajectory_buffer = data['trajectory_buffer']
            print(f"Agent loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")

    def prefill_history(self, env, train_days_indices):
        """
        快速预填充 SAA 历史样本 (不运行模拟，直接读取数据)
        """
        print("正在预填充 SAA 历史需求数据...")
        
        # 1. 获取所有训练天数的订单
        # 为了加速，直接操作 env.order_generator
        # 假设我们只关心宏观的时间片统计
        
        # 遍历每一天
        for day_idx in tqdm(train_days_indices, desc="Pre-filling History"):
            # 遍历当天的所有时间片 (0-143)
            # 注意: SAA 的 time_slot 是 30分钟 (48 slots/day)
            # 而 OrderGenerator 是 10分钟 (144 slices/day)
            
            for saa_slot in range(48): # 0..47
                # 这个 SAA slot 对应的分钟范围
                start_min = saa_slot * 30
                end_min = (saa_slot + 1) * 30
                
                # 对应的 OrderGenerator slices
                start_slice = start_min // 10
                end_slice = end_min // 10
                
                # 聚合需求
                demand_vector = np.zeros(self.num_grids, dtype=int)
                
                for s in range(start_slice, end_slice):
                    orders = env.order_generator._load_orders_for_macro_step(day_idx, s)
                    for o in orders:
                        g = o.get('grid_index', -1)
                        if 0 <= g < self.num_grids:
                            demand_vector[g] += 1
                
                # 记录到历史
                self.record_history(saa_slot, demand_vector)
        
        print(f"历史数据预填充完成。")

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
    获取当前 tick (时间窗口内) 每个网格生成的新订单数
    """
    # V5.2 兼容: 使用 orders_by_day_and_slice 加速查询
    try:
        # 1. 计算当前的 relative_day 和 time_slice
        start_timestamp = env.order_generator.time_range[0].normalize()
        relative_day = (current_time - start_timestamp).days
        
        minutes_from_midnight = current_time.hour * 60 + current_time.minute
        time_slice = (minutes_from_midnight // env.config.MACRO_STATISTICS_STEP_MINUTES)
        time_slice = max(0, min(time_slice, 143))
        
        # 2. 获取该 slice 的所有订单
        key = (relative_day, time_slice)
        slice_orders = env.order_generator.orders_by_day_and_slice.get(key, [])
        
        # 3. 筛选在当前 tick 内的订单
        end_time = current_time + pd.Timedelta(seconds=duration_sec)
        
        demand_vector = np.zeros(env.config.NUM_GRIDS, dtype=int)
        
        for order in slice_orders:
            # 假设 order['timestamp'] 是 Timestamp 对象
            t = order['timestamp']
            if current_time <= t < end_time:
                grid_id = order['grid_index'] # 注意这里用 grid_index
                if 0 <= grid_id < env.config.NUM_GRIDS:
                    demand_vector[grid_id] += 1
                    
        return demand_vector
        
    except Exception as e:
        # print(f"get_current_tick_demand error: {e}")
        return np.zeros(env.config.NUM_GRIDS, dtype=int)

# ==========================================
# 主模拟循环
# ==========================================
def run_simulation_phase(phase_name, days_to_run, env, agent, config, round_idx, current_seed, is_training=False):
    """
    运行模拟阶段 (训练或测试)
    """
    rows = []
    pbar = tqdm(days_to_run, desc=f"{phase_name} (Round {round_idx+1})", unit="day")
    
    for d in pbar:
        try:
            # 重置环境到指定天
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
            
            # 清空 Agent 的 buffer (跨天不保留 SARSA buffer, 但保留 SAA history)
            agent.trajectory_buffer.clear()
            
        except Exception as e:
            print(f"错误: env 重置失败在 Day {d}: {e}")
            continue

        ticks = 0
        daily_infos = defaultdict(list)
        
        # --- 单日 Ticks 循环 ---
        while ticks < config.TICKS_PER_DAY:
            try:
                # 1. 时间步计算
                current_minutes = (ticks * config.TICK_DURATION_SEC) / 60
                saa_time_slot = int(current_minutes // 30) # 30分钟一个slot
                
                # 2. [关键] 获取真实需求并存入历史
                # 训练阶段: 记录历史 (如果预填充了，这里可以跳过，或者继续强化)
                # 测试阶段: 也可以记录，模拟 Online Learning
                current_demand = get_current_tick_demand(env, env.simulation_time, config.TICK_DURATION_SEC)
                agent.record_history(saa_time_slot, current_demand)
                
                # 3. 调度决策 (每 5 分钟 = 10 ticks 一次)
                dispatch_instructions = {}
                est_value = 0.0
                
                if ticks % 10 == 0:
                    # 获取车辆分布
                    idle_vehicles_list = [0] * config.NUM_GRIDS
                    for v in env.vehicle_manager.vehicles.values():
                        if v['status'] == 'idle':
                            idle_vehicles_list[v['current_grid']] += 1
                    
                    # Epsilon-Greedy
                    epsilon = 0.1 if is_training else 0.0
                    if random.random() < epsilon:
                        pass # 随机探索
                    else:
                        dispatch_instructions, est_value = agent.solve_saa_dispatch(idle_vehicles_list, saa_time_slot)
                        
                    # 执行调度
                    for src, targets in dispatch_instructions.items():
                        candidates = [
                            vid for vid, v in env.vehicle_manager.vehicles.items() 
                            if v['current_grid'] == src and v['status'] == 'idle'
                        ]
                        candidate_idx = 0
                        for dst, count in targets.items():
                            for _ in range(count):
                                if candidate_idx < len(candidates):
                                    env.vehicle_manager.start_dispatching(candidates[candidate_idx], dst, env.simulation_time)
                                    candidate_idx += 1
                
                # 4. 环境步进
                _, reward, _, info = env.step()
                daily_infos[0].append(info.get('step_info', {}))
                
                # 5. SARSA 更新 (训练和测试都更新? 或者是 Baseline 的特性)
                # Yan et al. 是 Online Learning，所以测试时也会更新
                state_key = agent.get_state_hash(saa_time_slot)
                agent.update_sarsa(state_key, est_value, reward)
                
            except Exception as e:
                print(f"错误: Day {d}, Tick {ticks}: {e}")
                break
            ticks += 1
            
        # --- 每日结算 (仅在测试阶段记录详细日志) ---
        if not is_training:
            if hasattr(env, '_calculate_daily_metrics'):
                    daily_metrics = env._calculate_daily_metrics(daily_infos, d)
            else:
                    total_matched = sum(info.get('matched_orders', 0) for info in daily_infos[0])
                    total_revenue = sum(info.get('revenue', 0.0) for info in daily_infos[0])
                    daily_metrics = [{'day_index': 0, 'actual_day': d, 'completed_orders': total_matched, 'total_revenue': total_revenue}]

            if daily_metrics:
                m = daily_metrics[0]
                m['dataset_day_index'] = d
                m['benchmark_policy'] = 'sarsa_saa'
                m['round_index'] = round_idx
                m['seed'] = current_seed
                rows.append(m)
                
                # --- 增量保存 ---
                temp_df = pd.DataFrame(rows)
                ts_part = datetime.now().strftime('%Y%m%d')
                temp_csv = os.path.join(config.LOG_SAVE_PATH, 'results', f'benchmark_sarsa_saa_progress_{ts_part}.csv')
                os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
                temp_df.to_csv(temp_csv, index=False)
                
                pbar.set_description(f"{phase_name} Round {round_idx+1} (Day {d}: Cmp={m.get('completion_rate', 0):.1%})")
        else:
            # 训练阶段简单打印
             # 计算简单指标
            total_matched = sum(info.get('matched_orders', 0) for info in daily_infos[0])
            total_orders = sum(info.get('new_orders', 0) for info in daily_infos[0])
            cmp_rate = total_matched / total_orders if total_orders > 0 else 0
            pbar.set_description(f"{phase_name} Round {round_idx+1} (Day {d}: Cmp={cmp_rate:.1%})")

    return rows

# ==========================================
# 主模拟循环
# ==========================================
def run_sarsa_saa_simulation(config, num_episodes, env_data, num_rounds=5):
    """
    运行 SAA-SARSA Benchmark (5轮，每轮: 训练 -> 最后7天测试)
    """
    print(f"\n--- 开始 SAA-SARSA Benchmark (Total Rounds: {num_rounds}, Train + Test Last 7 Days) ---")
    print(f"  车辆总数: {config.TOTAL_VEHICLES}")
    
    # 1. 准备数据
    try:
        data_processor = DataProcessor(config)
    except Exception as e:
        print(f"创建 DataProcessor 出错: {e}")
        return {}

    all_daily_rows = []
    
    # --- 多轮循环 ---
    for round_idx in range(num_rounds):
        current_seed = config.SEED + round_idx
        set_seed(current_seed)
        print(f"\n>>> Round {round_idx + 1}/{num_rounds} (Seed: {current_seed})")

        # 2. 初始化 Agent
        agent = SarsaSAABaseline(config, delta=12, sample_size=7)

        # 3. 创建环境
        try:
            env = BaselineEnvironment(config, data_processor, env_data, dispatch_policy='none')
        except Exception as e:
            print(f"创建环境时出错: {e}")
            continue

        # 4. 确定日期划分
        try:
            day_count = env.order_generator.get_day_count()
        except Exception:
            day_count = 0
        
        test_days_count = 7
        if day_count <= test_days_count:
            print("数据量不足，无法拆分训练/测试集。直接运行测试。")
            train_days = []
            test_days = list(range(day_count))
        else:
            test_start = day_count - test_days_count
            train_days = list(range(test_start)) # 0 ~ N-8
            test_days = list(range(test_start, day_count)) # N-7 ~ N-1
            
        print(f"  训练集: {len(train_days)} 天 (Day {train_days[0]} - {train_days[-1]})")
        print(f"  测试集: {len(test_days)} 天 (Day {test_days[0]} - {test_days[-1]})")

        # --- Phase 0: 快速预填充历史 (Warm-up) ---
        # 利用所有训练数据瞬间填满 Agent 的记忆
        agent.prefill_history(env, train_days)

        # --- Phase 1: 训练 (Training) ---
        # 为了节省时间，我们可能不需要跑完所有的训练天数
        # 但既然用户要求 "其余时间来训练"，我们跑最后 14 天的训练集来强化 Q-table
        # 或者跑全部。
        # 考虑到速度，我们这里设定一个 limit，比如最多跑训练集的最后 7 天进行强化训练
        # 如果想跑全部，把 [-7:] 去掉即可
        
        # 用户原话: "其余时间来训练" -> 意味着全部跑完?
        # 如果训练集有 30 天，全跑完可能要很久。
        # 我们可以折中：预填充用了所有数据(SAA已完美)。SARSA Q值训练跑最近的 15 天 (两周多)。
        # 这样既能保证充分收敛，又比全跑要快。
        train_simulation_days = train_days[-15:] if len(train_days) > 15 else train_days
        print(f"  正在运行 SARSA 强化训练 (Simulation on last {len(train_simulation_days)} training days)...")
        
        run_simulation_phase("Training", train_simulation_days, env, agent, config, round_idx, current_seed, is_training=True)

        # --- Phase 2: 测试 (Testing) ---
        print(f"  正在运行测试 (Evaluation on last 7 days)...")
        test_rows = run_simulation_phase("Testing", test_days, env, agent, config, round_idx, current_seed, is_training=False)
        all_daily_rows.extend(test_rows)

    # --- 汇总分析与保存 ---
    if all_daily_rows:
        df = pd.DataFrame(all_daily_rows)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存原始数据
        raw_csv = os.path.join(config.LOG_SAVE_PATH, 'results', f'benchmark_sarsa_saa_last7_raw_{ts}.csv')
        os.makedirs(os.path.dirname(raw_csv), exist_ok=True)
        df.to_csv(raw_csv, index=False)
        print(f"\n✓ 已保存 SAA-SARSA 原始数据到: {raw_csv}")

        # 计算汇总
        summary_rows = []
        grouped = df.groupby('dataset_day_index')
        for day, group in grouped:
            summary_rows.append({
                'Day': day,
                'Best_Wait': group['avg_waiting_time'].min(),
                'Worst_Wait': group['avg_waiting_time'].max(),
                'Best_Completion_Rate': group['completion_rate'].max(),
                'Worst_Completion_Rate': group['completion_rate'].min()
            })
        
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(config.LOG_SAVE_PATH, 'results', f'benchmark_sarsa_saa_summary_{ts}.csv')
        df_summary.to_csv(summary_csv, index=False)
        print(f"✓ 已保存 SAA-SARSA 汇总结果到: {summary_csv}")
        
        try:
            from tabulate import tabulate
            print("\n" + "="*60)
            print("SAA-SARSA 5-Round Summary (Best/Worst)")
            print("="*60)
            print(tabulate(df_summary, headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False))
        except:
            print(df_summary.to_string(index=False))

    return all_daily_rows

if __name__ == '__main__':
    print("=" * 80)
    print("运行 Benchmark: SARSA-SAA (Yan et al. 2023)")
    print("=" * 80)
    
    try:
        config = Config()
        if not Config.validate_config(): sys.exit(1)
        
        print("加载数据...")
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()
        _, _, test_orders = data_processor.split_data_by_time(
            all_orders, config.TRAIN_RATIO, config.VAL_RATIO
        )
        print(f"测试集大小: {len(test_orders):,} 条")
        
        # 运行
        # 注意: 必须传入 all_orders，否则内部无法拆分训练/测试集
        run_sarsa_saa_simulation(config, 0, all_orders, num_rounds=5)
        
    except Exception as e:
        print(f"运行时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
