# environment.py
"""
网约车调度环境 - 完整版 (V5.5 - Tick / 事件驱动 / 等待时间奖励 / 支持策略切换 [drl, random_walk, random_dispatching])
核心修改:
1. ... (V5.4 修改不变) ...
15.(V5.5) _execute_proactive_dispatch 添加 'random_dispatching' 策略。
"""

import numpy as np
import pandas as pd
import torch
import time
import torch.nn.functional as F
from collections import deque, defaultdict # 导入 defaultdict
import json
import os
import random
import bisect # 用于排序插入 event_queue (可选优化)
import traceback # 用于打印详细错误

# (将默认值移到 try 之前)
EPISODE_BONUS_PARAMS_DEFAULT = {0.8: 100, 0.7: 50, 0.6: 20}
try:
    # 尝试导入用户定义的参数
    from reward_params_calibrated import EPISODE_BONUS_PARAMS
except ImportError:
    # 如果导入失败，使用默认值
    EPISODE_BONUS_PARAMS = EPISODE_BONUS_PARAMS_DEFAULT
except Exception as e:
    print(f"警告: 导入 reward_params_calibrated 时发生错误: {e}. 使用默认 EPISODE_BONUS_PARAMS。")
    EPISODE_BONUS_PARAMS = EPISODE_BONUS_PARAMS_DEFAULT


# ========== OrderGenerator Class (V5.2) ==========
# (与 V5.1 相同，无需修改)
class OrderGenerator:
    """订单生成器 - (V5.2 - 10秒 Tick 版本)"""
    def __init__(self, config, orders_df):
        self.config = config
        print(f"  初始化订单生成器（{len(orders_df):,}条订单）...", end='', flush=True)
        start_time = time.time()
        if orders_df.empty:
            print("\n错误：传入 OrderGenerator 的 orders_df 为空！")
            self.time_range = (pd.Timestamp.min, pd.Timestamp.max); self.orders_by_day_and_slice = {}
            self.total_days = 0; self.orders_df = pd.DataFrame(); config.MAX_START_DAY = 0
            return
        self.time_range = (orders_df['timestamp'].min(), orders_df['timestamp'].max())
        orders_df = orders_df.sort_values('timestamp').reset_index(drop=True)
        if 'timestamp' not in orders_df.columns:
            # 尝试从 departure_time 转换 (假设原始数据是 UTC 秒或毫秒)
            time_col = 'departure_time'
            if time_col not in orders_df.columns:
                print(f"\n错误: 订单数据缺少 '{time_col}' 或 'timestamp' 列!")
                # 创建一个空的 timestamp 列以避免后续错误
                orders_df['timestamp'] = pd.NaT
                orders_df = orders_df.dropna(subset=['timestamp']) # 移除无效行 (如果需要)
            else:
                try:
                    # 自动检测单位 (秒 s 或毫秒 ms)
                    example_time = orders_df[time_col].iloc[0]
                    unit = 'ms' if example_time > 1e10 else 's'
                    print(f"\n  检测到时间单位: {unit}")
                    orders_df['timestamp_utc'] = pd.to_datetime(orders_df[time_col], unit=unit, utc=True)
                    orders_df['timestamp'] = orders_df['timestamp_utc'].dt.tz_convert('Asia/Shanghai')
                    print(f"  已将 '{time_col}' 转换为 'Asia/Shanghai' 时区。")
                except Exception as e:
                    print(f"\n  错误: 自动转换时间戳失败: {e}. (请检查 '{time_col}' 列的数据格式和 'pip install pytz')")
                    # 使用 UTC 时间作为备用
                    if 'timestamp_utc' in orders_df:
                         orders_df['timestamp'] = orders_df['timestamp_utc']
                    else:
                         # 如果连 UTC 都没创建成功，则无法继续
                         print("  无法创建有效的时间戳列。")
                         orders_df['timestamp'] = pd.NaT
                         orders_df = orders_df.dropna(subset=['timestamp'])

        # 检查 timestamp 列是否存在且有效
        if 'timestamp' not in orders_df.columns or orders_df['timestamp'].isnull().all():
             print("\n错误: 无法生成有效的 'timestamp' 列!")
             # 进行必要的清理或错误处理
             self.time_range = (pd.Timestamp.min, pd.Timestamp.max); self.orders_by_day_and_slice = {}
             self.total_days = 0; self.orders_df = pd.DataFrame(); config.MAX_START_DAY = 0
             return # 无法继续初始化

        # 更新 time_range (可能因为转换失败而改变)
        self.time_range = (orders_df['timestamp'].min(), orders_df['timestamp'].max())

        start_timestamp = self.time_range[0].normalize()
        orders_df['relative_day'] = (orders_df['timestamp'] - start_timestamp).dt.days
        hours = orders_df['timestamp'].dt.hour; minutes = orders_df['timestamp'].dt.minute
        minutes_from_midnight = hours * 60 + minutes
        orders_df['time_slice'] = (minutes_from_midnight // config.MACRO_STATISTICS_STEP_MINUTES).clip(0, 143) # 使用宏观时间片分组
        self.orders_by_day_and_slice = {}
        for (day, time_slice), group in orders_df.groupby(['relative_day', 'time_slice']):
            key = (int(day), int(time_slice))
            group_sorted = group.sort_values('timestamp')
            orders_list = group_sorted[
                ['order_id', 'departure_time', 'fee', 'grid_index', 'dest_grid_index', 'timestamp']].to_dict('records')
            self.orders_by_day_and_slice[key] = orders_list
        self.total_days = int(orders_df['relative_day'].max()) + 1 if not orders_df.empty else 0
        elapsed = time.time() - start_time
        print(f" ✓ ({elapsed:.1f}秒)")
        print(f"    数据集天数: {self.total_days}天")
        if config.MAX_START_DAY is None:
            config.MAX_START_DAY = max(0, self.total_days - config.EPISODE_DAYS)
            print(f"    可选开始日期范围: 0-{config.MAX_START_DAY}天")
        self.orders_df = orders_df


    def _load_orders_for_macro_step(self, current_day, current_time_slice):
        """(内部函数) 获取这个 10 分钟宏观时间片的所有订单"""
        key = (current_day, current_time_slice); orders_in_slice = self.orders_by_day_and_slice.get(key, [])
        new_orders = [o.copy() for o in orders_in_slice]; [o.update({'status': 'pending'}) for o in new_orders]
        return new_orders # (已按时间排序)

    def get_day_count(self): return self.total_days


# ========== VehicleManager Class (V5.2) ==========
# (与 V5.1 相同，无需修改)
class VehicleManager:
    """车辆管理器 (V5.2 - 逻辑不变, 但依赖 10s/30s Tick 调用)"""
    def __init__(self, config):
        self.config = config; self.vehicles = {}; self.initialize_vehicles()
    def initialize_vehicles(self):
        total_vehicles = self.config.TOTAL_VEHICLES; seed = getattr(self.config, 'SEED', None)
        rng = np.random.default_rng(seed); positions = rng.integers(0, self.config.NUM_GRIDS, total_vehicles)
        print(f"  车辆初始化 ({total_vehicles}辆): 均匀随机分布")
        self.vehicles = { i: {'id': i, 'current_grid': int(positions[i]), 'status': 'idle', 'idle_since': None,
                              'dispatch_target': None, 'dispatch_start_time': None, 'assigned_order': None,
                              'pending_dispatch_experience': None} for i in range(total_vehicles) }
    def reset(self): self.initialize_vehicles()
    def update_dispatching_vehicles(self, current_time):
        vehicles_arrived = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'dispatching' and vehicle['dispatch_start_time'] is not None:
                try:
                    elapsed_seconds = (current_time - vehicle['dispatch_start_time']).total_seconds()
                    travel_time_seconds = self._calculate_travel_time(vehicle['current_grid'], vehicle['dispatch_target']) * 60.0
                    if elapsed_seconds >= travel_time_seconds: vehicles_arrived.append(vehicle_id)
                except TypeError as te:
                    print(f"警告: update_dispatching_vehicles 时间比较错误 (可能时区不匹配?): {te}. Vehicle ID: {vehicle_id}")
                    # 尝试强制设置 idle_since 让车辆恢复
                    vehicle['idle_since'] = current_time
                    vehicle['status'] = 'idle'
                    vehicle['dispatch_start_time'] = None

        for vehicle_id in vehicles_arrived:
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]; target_grid = vehicle['dispatch_target']
                if isinstance(target_grid, (int, np.integer)) and 0 <= target_grid < self.config.NUM_GRIDS: vehicle['current_grid'] = target_grid
                vehicle['status'] = 'idle'; vehicle['dispatch_target'] = None; vehicle['dispatch_start_time'] = None; vehicle['idle_since'] = current_time

    def _calculate_travel_time(self, from_grid, to_grid):
        try: from_grid_int = int(from_grid); to_grid_int = int(to_grid)
        except (ValueError, TypeError): return 1.0 # Default time
        if not (0 <= from_grid_int < self.config.NUM_GRIDS and 0 <= to_grid_int < self.config.NUM_GRIDS): return 1.0 # Default time for invalid grids
        if from_grid_int == to_grid_int: return 0.0
        grid_cols = self.config.GRID_SIZE[1]; from_row, from_col = divmod(from_grid_int, grid_cols); to_row, to_col = divmod(to_grid_int, grid_cols)
        grid_distance = abs(from_row - to_row) + abs(from_col - to_col); avg_speed = self.config.AVG_SPEED_KMH
        time_per_grid = 60.0 / avg_speed if avg_speed > 0 else 2.0; time_minutes = grid_distance * time_per_grid
        return max(0.1, time_minutes) # 返回分钟

    # ===== ★★★ V5.4 修复 start_dispatching UnboundLocalError (最终版) ★★★ =====
    def start_dispatching(self, vehicle_id, target_grid, current_time):
        """将空闲车辆状态变更为调度中"""
        vehicle = None # ★★★ V5.4 初始化 ★★★
        try:
            # 1. 安全地获取 vehicle 对象
            vehicle = self.vehicles.get(vehicle_id)

            # 2. 检查车辆是否存在且状态是否为 idle
            if vehicle is None or vehicle.get('status') != 'idle':
                return False

            # 3. 验证 target_grid
            target_grid_int = int(target_grid) # 移到 try 内部
            if not (0 <= target_grid_int < self.config.NUM_GRIDS):
                 # print(f"警告: start_dispatching 收到无效 target_grid: {target_grid}") # 减少打印
                 return False # 无效目标网格

            # 4. 更新车辆状态 (现在可以安全使用 vehicle 变量)
            vehicle['status'] = 'dispatching'
            vehicle['dispatch_target'] = target_grid_int
            vehicle['dispatch_start_time'] = current_time
            vehicle['idle_since'] = None
            return True
        except (ValueError, TypeError) as e:
            # print(f"警告: start_dispatching 验证 target_grid 时出错: {e}") # 减少打印
            return False
        except Exception as e:
             print(f"❌ start_dispatching 发生意外错误: {e}")
             traceback.print_exc()
             return False
    # ========================================================

    def assign_order(self, vehicle_id, order):
        if vehicle_id not in self.vehicles: return False; vehicle = self.vehicles[vehicle_id]
        if vehicle['status'] != 'idle': return False
        vehicle['status'] = 'serving'; vehicle['assigned_order'] = order; vehicle['idle_since'] = None; vehicle['pending_dispatch_experience'] = None
        return True
    def complete_service(self, vehicle_id, order, destination_grid, current_time):
        if vehicle_id not in self.vehicles: return False; vehicle = self.vehicles[vehicle_id]; current_grid_fallback = vehicle.get('current_grid', 0)
        try: dest_grid_int = int(destination_grid); assert 0 <= dest_grid_int < self.config.NUM_GRIDS
        except (ValueError, TypeError, AssertionError): dest_grid_int = current_grid_fallback
        vehicle['current_grid'] = dest_grid_int; vehicle['status'] = 'idle'; vehicle['assigned_order'] = None
        vehicle['dispatch_target'] = None; vehicle['dispatch_start_time'] = None; vehicle['idle_since'] = current_time
        return True
    def get_long_idle_vehicles(self, current_time, threshold_seconds):
        long_idle_ids = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'idle':
                if vehicle['idle_since'] is None: vehicle['idle_since'] = current_time
                if isinstance(vehicle['idle_since'], pd.Timestamp):
                    try:
                        # 确保时区匹配或可比较
                        idle_since_time = vehicle['idle_since']
                        current_tz = getattr(current_time, 'tzinfo', None)
                        idle_tz = getattr(idle_since_time, 'tzinfo', None)
                        if idle_tz is None and current_tz is not None: idle_since_time = idle_since_time.tz_localize(current_tz)
                        elif idle_tz is not None and current_tz is None: current_time = current_time.tz_localize(idle_tz) # 理论上不应发生
                        elif idle_tz != current_tz: idle_since_time = idle_since_time.tz_convert(current_tz)

                        idle_duration_seconds = (current_time - idle_since_time).total_seconds()
                        if idle_duration_seconds >= threshold_seconds: long_idle_ids.append(vehicle_id)
                    except Exception as e: # 更广泛地捕获可能的时区或类型错误
                        # print(f"警告: 计算 idle duration 时出错: {e}. Vehicle ID: {vehicle_id}") # 减少打印
                        vehicle['idle_since'] = current_time # 重置时间戳
                else: vehicle['idle_since'] = current_time
            else: vehicle['idle_since'] = None
        return long_idle_ids
    def get_statistics(self):
        stats = defaultdict(int)
        for v in self.vehicles.values():
             stats[v.get('status', 'unknown')] += 1
        return dict(stats)
    def get_idle_distribution(self):
        distribution = np.zeros(self.config.NUM_GRIDS, dtype=int)
        for vehicle in self.vehicles.values():
            if vehicle.get('status') == 'idle':
                grid = vehicle.get('current_grid');
                if isinstance(grid, (int, np.integer)) and 0 <= grid < self.config.NUM_GRIDS: distribution[grid] += 1
        return distribution
    def get_busy_distribution(self):
        distribution = np.zeros(self.config.NUM_GRIDS, dtype=int)
        for vehicle in self.vehicles.values():
            status = vehicle.get('status')
            if status == 'serving' or status == 'dispatching':
                grid = vehicle.get('current_grid');
                if isinstance(grid, (int, np.integer)) and 0 <= grid < self.config.NUM_GRIDS: distribution[grid] += 1
        return distribution


# ========== OrderMatcher Class (V5.2) ==========
# (与 V5.1 相同，无需修改)
class OrderMatcher:
    """订单匹配器 (V5.2 - 逻辑不变, 但被 Tick 调用)"""
    def __init__(self, config):
        self.config = config; self.search_radius = getattr(config, 'MATCHER_SEARCH_RADIUS', 2)
        print(f"  OrderMatcher 初始化 (半径: {self.search_radius})")
    def match_orders(self, pending_orders, vehicle_manager, current_time):
        matches = []; unmatched_orders = list(pending_orders); idle_vehicles = []
        for v_id, vehicle in vehicle_manager.vehicles.items():
            if vehicle['status'] == 'idle':
                grid = vehicle.get('current_grid');
                if isinstance(grid, (int, np.integer)) and 0 <= grid < self.config.NUM_GRIDS: idle_vehicles.append({'id': v_id, 'grid': grid})
        if not idle_vehicles: return [], unmatched_orders
        still_unmatched = []; available_idle_vehicles = {v['id']: v for v in idle_vehicles}; grid_cols = self.config.GRID_SIZE[1]
        random.shuffle(unmatched_orders)
        for order in unmatched_orders:
            try: order_grid = int(order['grid_index']); assert 0 <= order_grid < self.config.NUM_GRIDS
            except (ValueError, TypeError, KeyError, AssertionError): still_unmatched.append(order); continue
            order_row, order_col = divmod(order_grid, grid_cols); best_match_vehicle_id = None; min_travel_time = float('inf')
            vehicles_to_check_in_radius = []
            vehicle_items = list(available_idle_vehicles.items())
            random.shuffle(vehicle_items)
            for v_id, v_data in vehicle_items:
                v_row, v_col = divmod(v_data['grid'], grid_cols); dist = abs(v_row - order_row) + abs(v_col - order_col)
                if dist <= self.search_radius:
                     travel_time = vehicle_manager._calculate_travel_time(v_data['grid'], order_grid)
                     if travel_time < min_travel_time:
                         min_travel_time = travel_time
                         best_match_vehicle_id = v_id
            if best_match_vehicle_id is not None:
                assign_success = vehicle_manager.assign_order(best_match_vehicle_id, order)
                if assign_success:
                    matches.append({'order': order, 'vehicle_id': best_match_vehicle_id, 'distance': min_travel_time})
                    if best_match_vehicle_id in available_idle_vehicles: del available_idle_vehicles[best_match_vehicle_id]
                else: still_unmatched.append(order)
            else: still_unmatched.append(order)
        return matches, still_unmatched


# ========== RewardCalculator Class (V5.3) ==========
# (与 V5.2 相同，无需修改)
class RewardCalculator:
    """奖励计算器 - V5.3 (降级：仅用于评估和统计)"""
    def __init__(self, config):
        self.config = config
        self.use_episode_bonus = getattr(config, 'USE_EPISODE_BONUS', False)
        self.episode_bonus_thresholds = EPISODE_BONUS_PARAMS if self.use_episode_bonus else {}
        self.reset(); self._print_config()
    def _print_config(self):
        print(f"\n{'=' * 70}")
        print(f"奖励计算器配置 (V5.3 - 仅用于评估)")
        t0_val = 'N/A'
        if hasattr(self.config, 'REWARD_FORMULA_V4') and isinstance(self.config.REWARD_FORMULA_V4, dict):
            t0_val = self.config.REWARD_FORMULA_V4.get('T_CHARACTERISTIC', 'N/A')
        print(f"  (事件驱动奖励: exp(-wait_time / T0), T0={t0_val})")
        print(f"{'=' * 70}\n")
    def reset(self):
        self.completed_orders = 0; self.cancelled_orders = 0; self.waiting_times = []
        self.dispatch_success_count = 0; self.dispatch_total_count = 0;
        self.step_count = 0; self.total_revenue = 0.0
    def update(self, step_info):
        self.step_count += 1; self.completed_orders += step_info.get('matched_orders', 0)
        self.cancelled_orders += step_info.get('cancelled_orders', 0)
        self.waiting_times.extend(step_info.get('waiting_times', []))
        self.dispatch_success_count += step_info.get('dispatch_success', 0)
        self.dispatch_total_count += step_info.get('dispatch_total', 0)
        self.total_revenue += step_info.get('revenue', 0.0)
    def calculate_step_reward(self, step_info): return 0.0
    def get_metrics(self, calculate_total=False):
        total_p = self.completed_orders + self.cancelled_orders
        metrics = {'completed_orders': self.completed_orders, 'cancelled_orders': self.cancelled_orders,
                   'total_revenue': self.total_revenue,
                   'completion_rate': self.completed_orders / total_p if total_p > 0 else 0.0,
                   'cancel_rate': self.cancelled_orders / total_p if total_p > 0 else 0.0, }
        if self.waiting_times:
            metrics.update({'avg_waiting_time': np.mean(self.waiting_times), 'max_waiting_time': np.max(self.waiting_times),
                           'min_waiting_time': np.min(self.waiting_times), 'std_waiting_time': np.std(self.waiting_times)})
        else: metrics.update({k: 0.0 for k in ['avg_waiting_time', 'max_waiting_time', 'min_waiting_time', 'std_waiting_time']})
        return metrics
    def print_summary(self):
        metrics = self.get_metrics()
        print(f"\nEpisode Summary (Ticks: {self.step_count})")
        print(f"  总收入: {metrics['total_revenue']:.2f}")
        print(f"  订单: 完成={metrics['completed_orders']:,} ({metrics['completion_rate']:.1%}), Cancelled={metrics['cancelled_orders']:,} ({metrics['cancel_rate']:.1%})")
        print(f"  等待 (秒): Avg={metrics['avg_waiting_time']:.1f}, Std={metrics['std_waiting_time']:.1f}, Min={metrics['min_waiting_time']:.1f}, Max={metrics['max_waiting_time']:.1f}")
    def calculate_episode_bonus(self, episode_summary):
        if not self.use_episode_bonus: return 0.0
        completion_rate = episode_summary.get('reward_metrics', {}).get('completion_rate', 0.0)
        for threshold in sorted(self.episode_bonus_thresholds.keys(), reverse=True):
            if completion_rate >= threshold: return self.episode_bonus_thresholds[threshold]
        return 0.0


# ========== RideHailingEnvironment Class (V5.4 - Policy Switch) ==========
class RideHailingEnvironment:
    """网约车环境 - (V5.4 - 支持策略切换)"""
    # ===== ★★★ V5.4 修改 ★★★ =====
    def __init__(self, config, data_processor, orders_df, dispatch_policy='drl'): # 添加 dispatch_policy 参数
        self.config = config; self.data_processor = data_processor
        if orders_df.empty: print("警告：初始化 RideHailingEnvironment 时 orders_df 为空!")
        self.order_generator = OrderGenerator(config, orders_df)
        self.vehicle_manager = VehicleManager(config)
        self.order_matcher = OrderMatcher(config)
        self.reward_calculator = RewardCalculator(config)
        self._load_action_mapping(); self.pending_orders = deque(); self.event_queue = []
        self.buffered_orders = deque(); self.current_macro_slice_key = None
        try: self.simulation_time = pd.Timestamp.now(tz='Asia/Shanghai')
        except Exception: self.simulation_time = pd.Timestamp.now().tz_localize('UTC').tz_convert('Asia/Shanghai')
        self.current_time = self.simulation_time
        self.episode_start_day = 0; self.current_day = 0; self.current_time_slice = 0
        self.episode_step = 0; self.episode_stats = {}
        # V5.4: Add daily results tracking
        self.daily_stats = defaultdict(lambda: {'matched': 0, 'cancelled': 0, 'wait_times': [], 'revenue': 0.0, 'dispatches': 0})
        self.last_day_processed = -1

        self.model = None; self.replay_buffer = None; self.device = 'cpu'
        self.T0 = 300.0; self.reward_scale = 1.0
        if hasattr(self.config, 'REWARD_FORMULA_V4') and isinstance(self.config.REWARD_FORMULA_V4, dict):
             self.T0 = self.config.REWARD_FORMULA_V4.get('T_CHARACTERISTIC', 300.0)
             self.reward_scale = self.config.REWARD_SCALE_FACTOR
        if self.T0 <= 0: self.T0 = 300.0
        # --- V5.4 存储策略 ---
        self.dispatch_policy = dispatch_policy.lower()
        print(f"✓ 环境初始化完成 (Tick {config.TICK_DURATION_SEC}s, T0={self.T0}s, Scale={self.reward_scale}, Policy='{self.dispatch_policy}')")
    # ==========================

        # V5.4: Initialize daily stats
        self.daily_stats = defaultdict(lambda: {'matched': 0, 'cancelled': 0, 'wait_times': [], 'revenue': 0.0, 'dispatches': 0})
        self.last_day_processed = -1

    def _load_action_mapping(self):
        mapping_file = os.path.join(self.config.PROCESSED_DATA_PATH, 'action_mapping.json')
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f: mapping = json.load(f)
                self.action_to_grid = {int(k): int(v) for k, v in mapping.items()}
                # (V5.5: 存储热点网格列表)
                self.hotspot_grids = list(self.action_to_grid.values())
                print(f"  动作映射加载成功 ({len(self.action_to_grid)} actions -> {len(self.hotspot_grids)} hotspots)")
            else: raise FileNotFoundError(f"文件不存在: {mapping_file}")
        except Exception as e:
            print(f"  ❌ 加载动作映射文件失败: {e}. 使用默认映射.")
            self.action_to_grid = {i: i % self.config.NUM_GRIDS for i in range(self.config.NUM_ACTIONS)}
            self.hotspot_grids = list(range(self.config.NUM_GRIDS)) # 备用：所有网格都是热点
        if len(self.action_to_grid) != self.config.NUM_ACTIONS: print(f"  ⚠ 警告: NUM_ACTIONS ({self.config.NUM_ACTIONS}) 与映射大小不匹配!")
        if not self.hotspot_grids: # 如果热点列表为空，则使用所有网格
            print("  ⚠ 警告: 热点网格列表为空! 使用所有网格作为随机调度目标。")
            self.hotspot_grids = list(range(self.config.NUM_GRIDS))

    def reset(self):
        available_days = self.order_generator.get_day_count()
        max_start_day = max(0, available_days - self.config.EPISODE_DAYS)
        self.episode_start_day = random.randint(0, max_start_day)
        self.current_day = self.episode_start_day
        self.current_time_slice = 0
        self.episode_step = 0
        if hasattr(self.order_generator, 'time_range') and self.order_generator.time_range[0] != pd.Timestamp.min:
            base_time = self.order_generator.time_range[0].normalize()
        else:
            base_time = pd.Timestamp(self.config.DATA_START_DATE, tz='Asia/Shanghai').normalize()
        self.simulation_time = base_time + pd.Timedelta(days=self.current_day)
        if self.simulation_time.tzinfo is None: 
            self.simulation_time = self.simulation_time.tz_localize('Asia/Shanghai')
        self.current_time = self.simulation_time
        self.pending_orders.clear()
        self.event_queue.clear()
        self.buffered_orders.clear()
        self.current_macro_slice_key = None
        self.episode_stats = {'total_orders_generated': 0, 'total_orders_matched': 0, 'total_orders_cancelled': 0,
                              'total_dispatches': 0, 'total_revenue': 0.0}
        # V5.4: Reset daily stats
        self.daily_stats.clear()
        self.last_day_processed = -1
        
        self.vehicle_manager.reset()
        self.reward_calculator.reset()
        return self._get_state()

    def step(self, current_epsilon=0.1):
        """ (V5.3) 执行一个微观 Tick """
        self.episode_step += 1
        step_info = {'matched_orders': 0, 'cancelled_orders': 0, 'waiting_times': [], 'dispatch_success': 0,
                     'dispatch_total': 0, 'new_orders': 0, 'revenue': 0.0}
        try:
            # --- T 时刻开始 ---
            # 1. 处理到期事件
            self._process_events(self.simulation_time, step_info)
            # 2. 更新车辆移动
            self.vehicle_manager.update_dispatching_vehicles(self.simulation_time)
            # 3. 取消超时订单
            cancelled_this_tick = self._cancel_timeout_orders(self.simulation_time)
            step_info['cancelled_orders'] = cancelled_this_tick
            # 4. 匹配订单
            if self.pending_orders:
                matches, still_pending = self.order_matcher.match_orders(list(self.pending_orders),
                                                                         self.vehicle_manager, self.simulation_time)
                self.pending_orders = deque(still_pending)
                step_info['matched_orders'] = len(matches)
                self.episode_stats['total_orders_matched'] += len(matches)
                for match in matches:
                    order = match['order']
                    order['status'] = 'matched'
                    order['matched_time'] = self.simulation_time
                    try:
                        wait_to_match_sec = (self.simulation_time - order[
                            'generated_at']).total_seconds()
                        assert wait_to_match_sec >= 0
                    except (TypeError, AssertionError, KeyError) as e:
                        # 添加详细日志记录来诊断问题
                        print(f"警告: wait_to_match_sec计算异常: {e}")
                        print(f"  simulation_time: {self.simulation_time}")
                        print(f"  order generated_at: {order.get('generated_at', 'MISSING')}")
                        print(f"  order keys: {list(order.keys())}")
                        
                        # 使用更合理的fallback值：假设刚刚匹配，等待时间为10秒
                        wait_to_match_sec = 10.0
                    wait_for_pickup_min = match.get('distance', 0.0)
                    wait_for_pickup_sec = wait_for_pickup_min * 60.0
                    total_wait_time_sec = wait_to_match_sec + wait_for_pickup_sec
                    step_info['waiting_times'].append(total_wait_time_sec)
                    order['total_wait_time_sec'] = total_wait_time_sec
                    self._schedule_order_completion(match, self.simulation_time)
            # 5. 生成 *本 Tick* 新订单
            new_orders = self._load_orders_for_tick()
            [o.update({'generated_at': self.simulation_time}) for o in new_orders]
            self.pending_orders.extend(new_orders)
            step_info['new_orders'] = len(new_orders)
            self.episode_stats['total_orders_generated'] += len(new_orders)
            # 6. 执行主动调度
            dispatch_info = self._execute_proactive_dispatch(current_epsilon)
            step_info.update(dispatch_info)
            self.episode_stats['total_dispatches'] += dispatch_info.get('dispatch_total', 0)
            # 7. 更新评估指标
            self.reward_calculator.update(step_info)
            # 8. 推进时间
            self.simulation_time += pd.Timedelta(seconds=self.config.TICK_DURATION_SEC)
            self.current_time = self.simulation_time
            time_since_start_days = (
                        self.current_time.normalize() - self.order_generator.time_range[0].normalize()).days
            self.current_day = time_since_start_days
            minutes_today = self.current_time.hour * 60 + self.current_time.minute
            self.current_time_slice = min(minutes_today // self.config.MACRO_STATISTICS_STEP_MINUTES, 143)
            # 9. 检查结束
            done = self.episode_step >= self.config.MAX_TICKS_PER_EPISODE
            # 10. 获取下一状态
            next_state = self._get_state()
            # 11. 返回
            info = {'step_reward': 0.0, 'pending': len(self.pending_orders), 'step_info': step_info}
            return next_state, 0.0, done, info

        except Exception as e:
            print(f"❌ env.step() 内部发生严重错误 (Tick {self.episode_step}): {e}")
            traceback.print_exc()
            # 返回一个表示错误的终止状态
            return self._get_state(), 0.0, True, {'error': str(e)}

    def _load_orders_for_tick(self):
        key = (self.current_day, self.current_time_slice)
        if key != self.current_macro_slice_key:
            all_slice_orders = self.order_generator._load_orders_for_macro_step(self.current_day,
                                                                                self.current_time_slice)
            self.buffered_orders = deque(all_slice_orders)
            self.current_macro_slice_key = key
        new_orders_for_tick = []
        tick_end_time = self.simulation_time + pd.Timedelta(seconds=self.config.TICK_DURATION_SEC)
        while self.buffered_orders:
            order = self.buffered_orders[0]
            order_time = order['timestamp']
            # 确保时区一致
            if order_time.tzinfo is None and tick_end_time.tzinfo is not None:
                order_time = order_time.tz_localize(tick_end_time.tzinfo)
            elif order_time.tzinfo is not None and tick_end_time.tzinfo is None:
                tick_end_time = tick_end_time.tz_localize(order_time.tzinfo)
            # 处理时区不匹配的情况（例如，一个有时区一个没有）
            elif order_time.tzinfo != tick_end_time.tzinfo:
                try:
                    order_time = order_time.tz_convert(tick_end_time.tzinfo)
                except Exception as tz_e:
                    print(f"警告: 时区转换失败 {tz_e}")
                    break  # 无法比较，退出

            if order_time < tick_end_time:
                new_orders_for_tick.append(self.buffered_orders.popleft())
            else:
                break
        return new_orders_for_tick

    def _cancel_timeout_orders(self, current_time):
        cancelled_count = 0
        still_pending = deque()
        try:
            cutoff_time = current_time - pd.Timedelta(seconds=self.config.MAX_WAITING_TIME)
            # 确保 cutoff_time 也有时区 (如果 current_time 有)
            if current_time.tzinfo is not None and cutoff_time.tzinfo is None:
                cutoff_time = cutoff_time.tz_localize(current_time.tzinfo)
        except OverflowError:
            cutoff_time = pd.Timestamp.min.tz_localize(current_time.tzinfo) if current_time.tzinfo else pd.Timestamp.min
        while self.pending_orders:
            order = self.pending_orders.popleft()
            gen_time = order.get('generated_at')
            if isinstance(gen_time, pd.Timestamp):
                # 确保时区一致
                if gen_time.tzinfo is None and cutoff_time.tzinfo is not None:
                    gen_time = gen_time.tz_localize(cutoff_time.tzinfo)
                elif gen_time.tzinfo is not None and cutoff_time.tzinfo is None:
                    cutoff_time = cutoff_time.tz_localize(gen_time.tzinfo)
                elif gen_time.tzinfo != cutoff_time.tzinfo:
                    try:
                        gen_time = gen_time.tz_convert(cutoff_time.tzinfo)
                    except Exception:
                        pass  # 转换失败则跳过比较

                if gen_time <= cutoff_time:
                    order['status'] = 'cancelled'
                    self.episode_stats['total_orders_cancelled'] += 1
                    cancelled_count += 1
                else:
                    still_pending.append(order)
            else:
                still_pending.append(order)  # 保留没有生成时间的订单
        self.pending_orders = still_pending
        return cancelled_count

    def _execute_proactive_dispatch(self, epsilon):
        """ (V5.2 - 核心) 在 Tick 执行调度决策，获取 S_t 并包含真实位置 """
        if self.model is None: 
            return {'dispatch_success': 0, 'dispatch_total': 0}  # 评估模式下 buffer 为 None
        # (检查 buffer 仅在需要 push 时进行)

        idle_vehicle_ids = self.vehicle_manager.get_long_idle_vehicles(self.simulation_time,
                                                                       self.config.IDLE_THRESHOLD_SEC)
        if not idle_vehicle_ids: 
            return {'dispatch_success': 0, 'dispatch_total': 0}
        dispatch_total = len(idle_vehicle_ids)
        dispatch_success = 0

        # (优化：如果车辆过多，可以考虑只处理一部分，或并行处理)
        # random.shuffle(idle_vehicle_ids) # 可选：随机化处理顺序

        for vehicle_id in idle_vehicle_ids:
            # (在循环开始时获取车辆，确保状态最新)
            vehicle = self.vehicle_manager.vehicles.get(vehicle_id)
            # (再次检查状态，因为匹配可能改变了它)
            if not vehicle or vehicle['status'] != 'idle': 
                continue

            current_grid = vehicle['current_grid']
            try:
                S_micro = self._get_state(vehicle_location_override=current_grid)
                node_features_gpu = S_micro['node_features'].to(self.device)
                day_of_week_gpu = torch.tensor([S_micro['day_of_week']], dtype=torch.long, device=self.device)
                vehicle_loc_gpu = torch.tensor([current_grid], dtype=torch.long, device=self.device)
            except Exception as state_e:
                print(f"错误: 获取状态失败 (车辆 {vehicle_id}): {state_e}")
                continue  # 跳过这辆车

            action = -1  # 初始化 action
            if random.random() < epsilon:
                action = random.randint(0, self.config.NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    try:
                        q_values = self.model(node_features_gpu.unsqueeze(0), vehicle_loc_gpu, day_of_week_gpu)
                        action = q_values.squeeze(0).argmax().item()
                    except Exception as e:
                        print(f"警告: 模型推理失败 (车辆 {vehicle_id}): {e}. 随机动作.")
                        action = random.randint(0, self.config.NUM_ACTIONS - 1)

            if action == -1: 
                continue  # 如果动作选择失败，跳过

            target_grid = self._action_to_hotspot_grid(action)

            # 暂存 (S, A) - 确保只在 replay_buffer 存在时暂存
            if self.replay_buffer is not None:
                vehicle['pending_dispatch_experience'] = (S_micro, action)
            else:
                vehicle['pending_dispatch_experience'] = None  # 评估模式清除

            success = self.vehicle_manager.start_dispatching(vehicle_id, target_grid, self.simulation_time)
            if success: 
                dispatch_success += 1
            # (如果 start_dispatching 失败，pending_exp 会在下次检查时被覆盖或清除)

        return {'dispatch_success': dispatch_success, 'dispatch_total': dispatch_total}

    def _action_to_hotspot_grid(self, action):
        grid_id = self.action_to_grid.get(action)
        if grid_id is None: 
            return action % self.config.NUM_GRIDS
        return max(0, min(self.config.NUM_GRIDS - 1, grid_id))

    def _schedule_order_completion(self, match, current_time):
        order = match.get('order')
        vehicle_id = match.get('vehicle_id')
        if order is None or vehicle_id is None: 
            return
        try:
            pickup_grid = int(order['grid_index'])
            dest_grid = int(order.get('dest_grid_index', pickup_grid))
        except (ValueError, TypeError, KeyError):
            print(f"警告: _schedule... 缺少网格信息: {order}")
            return
        if not (0 <= dest_grid < self.config.NUM_GRIDS): 
            dest_grid = pickup_grid
        if not (0 <= pickup_grid < self.config.NUM_GRIDS): 
            pickup_grid = 0
        service_time_minutes = self.vehicle_manager._calculate_travel_time(pickup_grid, dest_grid)
        pickup_time_minutes = match.get('distance', 0.0)
        total_duration_minutes = pickup_time_minutes + service_time_minutes
        service_duration = pd.Timedelta(minutes=total_duration_minutes)
        min_duration = pd.Timedelta(seconds=self.config.TICK_DURATION_SEC)
        completion_time = current_time + max(min_duration, service_duration)
        event = {'time': completion_time, 'type': 'order_completion', 'vehicle_id': int(vehicle_id), 'order': order,
                 'destination_grid': dest_grid}
        # (V5.1 优化: 使用 bisect 保持排序, 兼容 Python 3.9)
        # Python 3.10+ 支持 key 参数，3.9 及以下版本不支持
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda e: e['time'])

    def _process_events(self, current_time, step_info):
        """ (V5.3 - 核心 / 再次修复 UnboundLocalError) 处理到期事件, PUSH 基于等待时间的奖励 (S, A, R, S') 到 Buffer """
        due_events_indices = []
        processed_count = 0
        # (假设 event_queue 已通过 bisect 排序 或 在这里排序)
        # self.event_queue.sort(key=lambda e: e['time']) # 如果 _schedule_order_completion 没用 bisect

        for i, event in enumerate(self.event_queue):
            try:
                # 检查 event 是否为 None 或缺少 'time' 键
                if event is None or 'time' not in event:
                    continue  # 跳过无效事件

                event_time = event.get('time')
                # 确保 event_time 是 Timestamp
                if not isinstance(event_time, pd.Timestamp):
                    continue

                # 确保时区一致或可比较
                if event_time.tzinfo is None and current_time.tzinfo is not None:
                    event_time = event_time.tz_localize(current_time.tzinfo)
                elif event_time.tzinfo is not None and current_time.tzinfo is None:
                    current_time = current_time.tz_localize(event_time.tzinfo)

                if event_time <= current_time:
                    due_events_indices.append(i)
                else:
                    break  # 由于排序，后续事件都不会到期
            except Exception as e:
                print(f"❌ 检查事件时间时发生错误: {e}. Event: {event}")
                continue

        # 从后往前删除，避免索引变化
        for i in reversed(due_events_indices):
            event = self.event_queue.pop(i)
            processed_count += 1
            
            try:
                if event['type'] == 'order_completion':
                    vehicle_id = event['vehicle_id']
                    order = event['order']
                    destination_grid = event['destination_grid']
                    
                    # 完成服务
                    self.vehicle_manager.complete_service(vehicle_id, order, destination_grid, current_time)
                    
                    # ===== 添加经验存储逻辑 =====
                    vehicle = self.vehicle_manager.vehicles.get(vehicle_id)
                    if (vehicle and 'pending_dispatch_experience' in vehicle and 
                        vehicle['pending_dispatch_experience'] is not None and 
                        self.replay_buffer is not None):
                        
                        S_t, A_t = vehicle['pending_dispatch_experience']
                        o_data = order
                        new_grid = destination_grid

                        # 计算等待时间奖励
                        try:
                            generated_at = o_data.get('generated_at')
                            matched_time = o_data.get('matched_time')
                            if generated_at and matched_time:
                                wait_to_match_sec = max(0, (matched_time - generated_at).total_seconds())
                            else:
                                wait_to_match_sec = 60.0  # 1分钟

                            # 估算接驾时间
                            pickup_time_minutes_est = self.vehicle_manager._calculate_travel_time(
                                S_t['vehicle_location'], o_data['grid_index'])
                            wait_for_pickup_sec_est = pickup_time_minutes_est * 60.0
                            total_wait_time_sec = wait_to_match_sec + wait_for_pickup_sec_est
                        except Exception:
                            total_wait_time_sec = 300  # Fallback

                        wait_score = np.exp(-max(0.0, total_wait_time_sec) / self.T0)
                        R_t = wait_score * self.reward_scale

                        # 获取包含新位置的 S_{t+1}
                        S_t_plus_1 = self._get_state(vehicle_location_override=new_grid)
                        done = False

                        # PUSH 经验
                        try:
                            if isinstance(S_t, dict) and isinstance(S_t_plus_1, dict):
                                self.replay_buffer.push(S_t, A_t, R_t, S_t_plus_1, done)
                            else:
                                print(f"错误: 尝试 PUSH 非字典状态.")
                        except Exception as push_e:
                            print(f"❌ Replay Buffer PUSH 失败: {push_e}")

                        # 清除暂存的经验
                        vehicle['pending_dispatch_experience'] = None
                    
                    # 计算收入
                    revenue = order.get('fare', 0.0)
                    step_info['revenue'] += revenue
                    self.episode_stats['total_revenue'] += revenue

            except KeyError as ke:
                print(f"❌ 处理事件时发生 KeyError: {ke}. Event: {event}")
            except Exception as e:
                print(f"❌ 处理事件时发生意外错误: {e}. Event: {event}")
                traceback.print_exc()

        return processed_count

def _calculate_daily_metrics(daily_step_infos, start_day):
    """
    (V5.2 新增) 计算每个模拟日的指标
    Args:
        daily_step_infos (dict): {day_index: [step_info_tick1, step_info_tick2, ...]}
        start_day (int): Episode 的起始日期索引
    Returns:
        list[dict]: 包含每天指标的字典列表
    """
    daily_metrics_list = []
    # 按天排序处理
    for day_index in sorted(daily_step_infos.keys()):
        step_infos_today = daily_step_infos[day_index]

        total_matched = sum(info.get('matched_orders', 0) for info in step_infos_today)
        total_cancelled = sum(info.get('cancelled_orders', 0) for info in step_infos_today)
        all_waiting_times = [wt for info in step_infos_today for wt in info.get('waiting_times', [])]
        total_revenue = sum(info.get('revenue', 0.0) for info in step_infos_today)
        total_dispatches = sum(info.get('dispatch_total', 0) for info in step_infos_today)
        total_new_orders = sum(info.get('new_orders', 0) for info in step_infos_today)

        total_processed = total_matched + total_cancelled
        completion_rate = total_matched / total_processed if total_processed > 0 else 0.0
        cancel_rate = total_cancelled / total_processed if total_processed > 0 else 0.0
        avg_waiting_time = np.mean(all_waiting_times) if all_waiting_times else 0.0

        daily_metrics_list.append({
            'day_index': day_index,  # 相对天数 (e.g., 0, 1, ...)
            'actual_day': start_day + day_index,  # 数据集中的实际天数
            'total_revenue': round(total_revenue, 2),
            'completed_orders': total_matched,
            'cancelled_orders': total_cancelled,
            'completion_rate': round(completion_rate, 4),
            'cancel_rate': round(cancel_rate, 4),
            'avg_waiting_time': round(avg_waiting_time, 1),
            'total_dispatches': total_dispatches,
            'total_new_orders': total_new_orders,
        })
    return daily_metrics_list

    # ===== V5.1 修复 =====
    def _get_state(self, current_time=None, vehicle_location_override=None):
        """获取状态 (V5.1 - 修复 vehicle_location)"""
        time_to_use = current_time if current_time is not None else self.current_time
        idle_vehicle_dist = self.vehicle_manager.get_idle_distribution()
        busy_vehicle_dist = self.vehicle_manager.get_busy_distribution()
        node_features = None # 初始化
        day_of_week = 0 # 初始化
        if not hasattr(self, 'data_processor') or self.data_processor is None:
            node_features = torch.zeros((self.config.NUM_GRIDS, self.config.INPUT_DIM))
        else:
            try:
                node_features = self.data_processor.prepare_mgcn_input(self.order_generator.orders_df, time_to_use, idle_vehicle_dist=idle_vehicle_dist, busy_vehicle_dist=busy_vehicle_dist)
            except Exception as e: print(f"错误: prepare_mgcn_input 失败: {e}"); node_features = torch.zeros((self.config.NUM_GRIDS, self.config.INPUT_DIM))
            # 确保 time_to_use 是 Timestamp
            if isinstance(time_to_use, pd.Timestamp):
                 day_of_week = time_to_use.dayofweek
            else:
                 # 尝试转换或使用默认值
                 try: day_of_week = pd.to_datetime(time_to_use).dayofweek
                 except: day_of_week = 0 # Fallback

        final_vehicle_location = 0
        if vehicle_location_override is not None:
             try: final_vehicle_location = int(vehicle_location_override)
             except (ValueError, TypeError): final_vehicle_location = 0 # Fallback
        state_dict = {'node_features': node_features.cpu(), 'vehicle_location': final_vehicle_location, 'day_of_week': day_of_week}
        return state_dict
    # =====================

    def get_episode_summary(self):
        metrics = self.reward_calculator.get_metrics()
        waiting_stats = {k: metrics.get(k, 0.0) for k in ['avg_waiting_time', 'max_waiting_time', 'min_waiting_time', 'std_waiting_time']}
        waiting_stats['count'] = len(getattr(self.reward_calculator, 'waiting_times', []))
        vehicle_stats = self.vehicle_manager.get_statistics(); total_vehicles = sum(vehicle_stats.values())
        if total_vehicles > 0: metrics.update({k: vehicle_stats.get(s, 0) / total_vehicles for k, s in [('vehicle_utilization', 'serving'), ('idle_rate', 'idle'), ('dispatching_rate', 'dispatching')]})
        else: metrics.update({k: 0.0 for k in ['vehicle_utilization', 'idle_rate', 'dispatching_rate']})
        total_gen = self.episode_stats.get('total_orders_generated', 0); processed = metrics.get('completed_orders', 0) + metrics.get('cancelled_orders', 0)
        metrics['processing_rate'] = processed / total_gen if total_gen > 0 else 0.0; metrics['total_revenue'] = self.episode_stats.get('total_revenue', 0.0)
        # V5.4: Add daily results from internal tracking
        daily_summary = _calculate_daily_metrics(self.daily_stats, self.episode_start_day) # Use internal daily_stats
        return {'episode_stats': self.episode_stats.copy(),
                'reward_metrics': metrics,
                'waiting_time_stats': waiting_stats,
                'daily_summary': daily_summary # V5.4 Add daily summary here
               }


    # ===== V5.1 修复 =====
    def set_model_and_buffer(self, model, replay_buffer, device):
        """ (V5.1) 设置模型, Buffer 引用, 和设备 """
        self.model = model; self.replay_buffer = replay_buffer; self.device = device
        if model is not None:
            print(f"✓ 模型已设置到环境 (设备: {self.device})")
            if replay_buffer is not None: print("✓ Replay Buffer 已设置到环境")
            else: print("⚠ Replay Buffer 未设置 (评估模式?)")
        else: print("模型已从环境移除。")
    # =====================

