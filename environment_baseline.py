# environment_baseline.py
"""
基准测试环境 - 专门为 Baselines 准备
支持多种调度策略进行公平对比：
- 'random_walk': 随机游走（上下左右或停留）
- 'random_dispatch': 随机调度到热点网格
- 'sarsa': SARSA-SAA 策略（由外部 Agent 控制）
- 'hmarl': H-MARL 策略（由外部 Agent 控制）

基于 environment_back.py (V5.4) 简化而来，去除了 DRL 相关的复杂逻辑
"""

import numpy as np
import pandas as pd
import torch
from collections import deque, defaultdict
import json
import os
import random
import traceback
import time


# ========== OrderGenerator Class ==========
class OrderGenerator:
    """订单生成器"""
    def __init__(self, config, orders_df):
        self.config = config
        print(f"  初始化订单生成器（{len(orders_df):,}条订单）...", end='', flush=True)
        start_time = time.time()

        if orders_df.empty:
            print("\n错误：传入 OrderGenerator 的 orders_df 为空！")
            self.time_range = (pd.Timestamp.min, pd.Timestamp.max)
            self.orders_by_day_and_slice = {}
            self.total_days = 0
            self.orders_df = pd.DataFrame()
            config.MAX_START_DAY = 0
            return

        self.time_range = (orders_df['timestamp'].min(), orders_df['timestamp'].max())
        orders_df = orders_df.sort_values('timestamp').reset_index(drop=True)

        # 处理时间戳
        if 'timestamp' not in orders_df.columns:
            if 'departure_time' in orders_df.columns:
                try:
                    example_time = orders_df['departure_time'].iloc[0]
                    unit = 'ms' if example_time > 1e10 else 's'
                    orders_df['timestamp_utc'] = pd.to_datetime(orders_df['departure_time'], unit=unit, utc=True)
                    orders_df['timestamp'] = orders_df['timestamp_utc'].dt.tz_convert('Asia/Shanghai')
                except Exception as e:
                    print(f"\n  错误: 时间转换失败: {e}")
                    orders_df['timestamp'] = pd.NaT
                    orders_df = orders_df.dropna(subset=['timestamp'])
            else:
                print("\n错误: 订单数据缺少时间列")
                return

        self.time_range = (orders_df['timestamp'].min(), orders_df['timestamp'].max())

        # 分组订单
        start_timestamp = self.time_range[0].normalize()
        orders_df['relative_day'] = (orders_df['timestamp'] - start_timestamp).dt.days
        hours = orders_df['timestamp'].dt.hour
        minutes = orders_df['timestamp'].dt.minute
        minutes_from_midnight = hours * 60 + minutes
        orders_df['time_slice'] = (minutes_from_midnight // config.MACRO_STATISTICS_STEP_MINUTES).clip(0, 143)

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
        """获取指定时间片的订单"""
        key = (current_day, current_time_slice)
        orders_in_slice = self.orders_by_day_and_slice.get(key, [])
        new_orders = [o.copy() for o in orders_in_slice]
        for o in new_orders:
            o['status'] = 'pending'
        return new_orders

    def get_day_count(self):
        return self.total_days


# ========== VehicleManager Class ==========
class VehicleManager:
    """车辆管理器"""
    def __init__(self, config):
        self.config = config
        self.vehicles = {}
        self.initialize_vehicles()

    def initialize_vehicles(self):
        total_vehicles = self.config.TOTAL_VEHICLES
        seed = getattr(self.config, 'SEED', None)
        rng = np.random.default_rng(seed)
        positions = rng.integers(0, self.config.NUM_GRIDS, total_vehicles)
        print(f"  车辆初始化 ({total_vehicles}辆): 均匀随机分布")

        self.vehicles = {
            i: {
                'id': i,
                'current_grid': int(positions[i]),
                'status': 'idle',
                'idle_since': None,
                'dispatch_target': None,
                'dispatch_start_time': None,
                'assigned_order': None,
            }
            for i in range(total_vehicles)
        }

    def reset(self):
        self.initialize_vehicles()

    def update_dispatching_vehicles(self, current_time):
        """更新正在调度的车辆状态"""
        vehicles_arrived = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'dispatching' and vehicle['dispatch_start_time'] is not None:
                try:
                    elapsed_seconds = (current_time - vehicle['dispatch_start_time']).total_seconds()
                    travel_time_seconds = self._calculate_travel_time(
                        vehicle['current_grid'], vehicle['dispatch_target']
                    ) * 60.0
                    if elapsed_seconds >= travel_time_seconds:
                        vehicles_arrived.append(vehicle_id)
                except TypeError:
                    vehicle['status'] = 'idle'
                    vehicle['dispatch_start_time'] = None

        for vehicle_id in vehicles_arrived:
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]
                target_grid = vehicle['dispatch_target']
                if isinstance(target_grid, (int, np.integer)) and 0 <= target_grid < self.config.NUM_GRIDS:
                    vehicle['current_grid'] = target_grid
                vehicle['status'] = 'idle'
                vehicle['dispatch_target'] = None
                vehicle['dispatch_start_time'] = None
                vehicle['idle_since'] = current_time

    def _calculate_travel_time(self, from_grid, to_grid):
        """计算两个网格间的旅行时间（分钟）"""
        try:
            from_grid_int = int(from_grid)
            to_grid_int = int(to_grid)
        except (ValueError, TypeError):
            return 1.0

        if not (0 <= from_grid_int < self.config.NUM_GRIDS and 0 <= to_grid_int < self.config.NUM_GRIDS):
            return 1.0

        if from_grid_int == to_grid_int:
            return 0.0

        grid_cols = self.config.GRID_SIZE[1]
        from_row, from_col = divmod(from_grid_int, grid_cols)
        to_row, to_col = divmod(to_grid_int, grid_cols)
        grid_distance = abs(from_row - to_row) + abs(from_col - to_col)

        avg_speed = self.config.AVG_SPEED_KMH
        time_per_grid = 60.0 / avg_speed if avg_speed > 0 else 2.0
        time_minutes = grid_distance * time_per_grid

        return max(0.1, time_minutes)

    def start_dispatching(self, vehicle_id, target_grid, current_time):
        """将车辆状态改为调度中"""
        try:
            vehicle = self.vehicles.get(vehicle_id)
            if vehicle is None or vehicle.get('status') != 'idle':
                return False

            target_grid_int = int(target_grid)
            if not (0 <= target_grid_int < self.config.NUM_GRIDS):
                return False

            vehicle['status'] = 'dispatching'
            vehicle['dispatch_target'] = target_grid_int
            vehicle['dispatch_start_time'] = current_time
            vehicle['idle_since'] = None
            return True
        except (ValueError, TypeError):
            return False

    def assign_order(self, vehicle_id, order):
        """分配订单给车辆"""
        if vehicle_id not in self.vehicles:
            return False

        vehicle = self.vehicles[vehicle_id]
        if vehicle['status'] != 'idle':
            return False

        vehicle['status'] = 'serving'
        vehicle['assigned_order'] = order
        vehicle['idle_since'] = None
        return True

    def complete_service(self, vehicle_id, destination_grid, current_time):
        """完成服务"""
        if vehicle_id not in self.vehicles:
            return False

        vehicle = self.vehicles[vehicle_id]
        try:
            dest_grid_int = int(destination_grid)
            if not (0 <= dest_grid_int < self.config.NUM_GRIDS):
                dest_grid_int = vehicle.get('current_grid', 0)
        except (ValueError, TypeError):
            dest_grid_int = vehicle.get('current_grid', 0)

        vehicle['current_grid'] = dest_grid_int
        vehicle['status'] = 'idle'
        vehicle['assigned_order'] = None
        vehicle['dispatch_target'] = None
        vehicle['dispatch_start_time'] = None
        vehicle['idle_since'] = current_time
        return True

    def get_long_idle_vehicles(self, current_time, threshold_seconds):
        """获取空闲时间超过阈值的车辆"""
        long_idle_ids = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'idle':
                if vehicle['idle_since'] is None:
                    vehicle['idle_since'] = current_time

                try:
                    idle_since_time = vehicle['idle_since']
                    current_tz = getattr(current_time, 'tzinfo', None)
                    idle_tz = getattr(idle_since_time, 'tzinfo', None)

                    if idle_tz is None and current_tz is not None:
                        idle_since_time = idle_since_time.tz_localize(current_tz)
                    elif idle_tz != current_tz and idle_tz is not None:
                        idle_since_time = idle_since_time.tz_convert(current_tz)

                    idle_duration_seconds = (current_time - idle_since_time).total_seconds()
                    if idle_duration_seconds >= threshold_seconds:
                        long_idle_ids.append(vehicle_id)
                except Exception:
                    vehicle['idle_since'] = current_time
            else:
                vehicle['idle_since'] = None

        return long_idle_ids

    def get_statistics(self):
        """获取车辆统计信息"""
        stats = defaultdict(int)
        for v in self.vehicles.values():
            stats[v.get('status', 'unknown')] += 1
        return dict(stats)

    def get_idle_distribution(self):
        """获取空闲车辆分布"""
        distribution = np.zeros(self.config.NUM_GRIDS, dtype=int)
        for vehicle in self.vehicles.values():
            if vehicle.get('status') == 'idle':
                grid = vehicle.get('current_grid')
                if isinstance(grid, (int, np.integer)) and 0 <= grid < self.config.NUM_GRIDS:
                    distribution[grid] += 1
        return distribution


# ========== OrderMatcher Class ==========
class OrderMatcher:
    """订单匹配器"""
    def __init__(self, config):
        self.config = config
        self.search_radius = getattr(config, 'MATCHER_SEARCH_RADIUS', 2)
        print(f"  OrderMatcher 初始化 (半径: {self.search_radius})")

    def match_orders(self, pending_orders, vehicle_manager, current_time):
        """匹配订单和车辆"""
        matches = []
        unmatched_orders = list(pending_orders)
        idle_vehicles = []

        for v_id, vehicle in vehicle_manager.vehicles.items():
            if vehicle['status'] == 'idle':
                grid = vehicle.get('current_grid')
                if isinstance(grid, (int, np.integer)) and 0 <= grid < self.config.NUM_GRIDS:
                    idle_vehicles.append({'id': v_id, 'grid': grid})

        if not idle_vehicles:
            return [], unmatched_orders

        still_unmatched = []
        available_idle_vehicles = {v['id']: v for v in idle_vehicles}
        grid_cols = self.config.GRID_SIZE[1]

        random.shuffle(unmatched_orders)
        for order in unmatched_orders:
            try:
                order_grid = int(order['grid_index'])
                assert 0 <= order_grid < self.config.NUM_GRIDS
            except (ValueError, TypeError, KeyError, AssertionError):
                still_unmatched.append(order)
                continue

            order_row, order_col = divmod(order_grid, grid_cols)
            best_match_vehicle_id = None
            min_travel_time = float('inf')

            vehicle_items = list(available_idle_vehicles.items())
            random.shuffle(vehicle_items)

            for v_id, v_data in vehicle_items:
                v_row, v_col = divmod(v_data['grid'], grid_cols)
                dist = abs(v_row - order_row) + abs(v_col - order_col)

                if dist <= self.search_radius:
                    travel_time = vehicle_manager._calculate_travel_time(v_data['grid'], order_grid)
                    if travel_time < min_travel_time:
                        min_travel_time = travel_time
                        best_match_vehicle_id = v_id

            if best_match_vehicle_id is not None:
                assign_success = vehicle_manager.assign_order(best_match_vehicle_id, order)
                if assign_success:
                    matches.append({'order': order, 'vehicle_id': best_match_vehicle_id, 'distance': min_travel_time})
                    if best_match_vehicle_id in available_idle_vehicles:
                        del available_idle_vehicles[best_match_vehicle_id]
                else:
                    still_unmatched.append(order)
            else:
                still_unmatched.append(order)

        return matches, still_unmatched


# ========== RewardCalculator Class ==========
class RewardCalculator:
    """奖励计算器（仅用于统计）"""
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.completed_orders = 0
        self.cancelled_orders = 0
        self.waiting_times = []
        self.total_revenue = 0.0

    def update(self, step_info):
        """更新统计信息"""
        self.completed_orders += step_info.get('matched_orders', 0)
        self.cancelled_orders += step_info.get('cancelled_orders', 0)
        self.waiting_times.extend(step_info.get('waiting_times', []))
        self.total_revenue += step_info.get('revenue', 0.0)

    def get_metrics(self):
        """获取指标"""
        total_p = self.completed_orders + self.cancelled_orders
        metrics = {
            'completed_orders': self.completed_orders,
            'cancelled_orders': self.cancelled_orders,
            'total_revenue': self.total_revenue,
            'completion_rate': self.completed_orders / total_p if total_p > 0 else 0.0,
            'cancel_rate': self.cancelled_orders / total_p if total_p > 0 else 0.0,
        }

        if self.waiting_times:
            metrics.update({
                'avg_waiting_time': np.mean(self.waiting_times),
                'max_waiting_time': np.max(self.waiting_times),
                'min_waiting_time': np.min(self.waiting_times),
                'std_waiting_time': np.std(self.waiting_times)
            })
        else:
            metrics.update({
                'avg_waiting_time': 0.0,
                'max_waiting_time': 0.0,
                'min_waiting_time': 0.0,
                'std_waiting_time': 0.0
            })

        return metrics


# ========== BaselineEnvironment Class ==========
class BaselineEnvironment:
    """基准测试环境"""
    def __init__(self, config, data_processor, orders_df, dispatch_policy='random_walk'):
        self.config = config
        self.data_processor = data_processor

        if orders_df.empty:
            print("警告：初始化 BaselineEnvironment 时 orders_df 为空!")

        self.order_generator = OrderGenerator(config, orders_df)
        self.vehicle_manager = VehicleManager(config)
        self.order_matcher = OrderMatcher(config)
        self.reward_calculator = RewardCalculator(config)

        self.pending_orders = deque()
        self.event_queue = []
        self.buffered_orders = deque()
        self.current_macro_slice_key = None

        try:
            self.simulation_time = pd.Timestamp.now(tz='Asia/Shanghai')
        except Exception:
            self.simulation_time = pd.Timestamp.now().tz_localize('UTC').tz_convert('Asia/Shanghai')

        self.current_time = self.simulation_time
        self.episode_start_day = 0
        self.current_day = 0
        self.current_time_slice = 0
        self.episode_step = 0
        self.episode_stats = {}
        self.daily_stats = defaultdict(lambda: {'matched': 0, 'cancelled': 0, 'wait_times': [], 'revenue': 0.0})
        self.last_day_processed = -1

        self.dispatch_policy = dispatch_policy.lower()
        print(f"✓ BaselineEnvironment 初始化完成 (Policy='{self.dispatch_policy}')")

    def reset(self, start_day=None):
        """重置环境"""
        available_days = self.order_generator.get_day_count()
        max_start_day = max(0, available_days - self.config.EPISODE_DAYS)

        if start_day is not None:
            self.episode_start_day = start_day
        else:
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

        self.episode_stats = {
            'total_orders_generated': 0,
            'total_orders_matched': 0,
            'total_orders_cancelled': 0,
            'total_dispatches': 0,
            'total_revenue': 0.0
        }

        self.daily_stats.clear()
        self.last_day_processed = -1

        self.vehicle_manager.reset()
        self.reward_calculator.reset()

    def step(self):
        """执行一个时间步"""
        self.episode_step += 1
        step_info = {
            'matched_orders': 0,
            'cancelled_orders': 0,
            'waiting_times': [],
            'dispatch_success': 0,
            'dispatch_total': 0,
            'new_orders': 0,
            'revenue': 0.0
        }

        try:
            # 1. 更新车辆状态
            self.vehicle_manager.update_dispatching_vehicles(self.current_time)

            # 2. 加载新订单
            new_orders = self.order_generator._load_orders_for_macro_step(self.current_day, self.current_time_slice)
            self.pending_orders.extend(new_orders)
            step_info['new_orders'] = len(new_orders)
            self.episode_stats['total_orders_generated'] += len(new_orders)

            # 3. 匹配订单
            matches, unmatched = self.order_matcher.match_orders(
                self.pending_orders, self.vehicle_manager, self.current_time
            )
            self.pending_orders = deque(unmatched)

            step_info['matched_orders'] = len(matches)
            self.episode_stats['total_orders_matched'] += len(matches)

            for match in matches:
                step_info['revenue'] += match['order'].get('fee', 0.0)
                step_info['waiting_times'].append(0.0)  # 匹配时等待时间为 0

            self.episode_stats['total_revenue'] += step_info['revenue']

            # 4. 处理超时订单
            cancelled = self._cancel_timeout_orders()
            step_info['cancelled_orders'] = cancelled
            self.episode_stats['total_orders_cancelled'] += cancelled

            # 5. 执行调度策略
            if self.dispatch_policy == 'random_walk':
                dispatch_info = self._execute_random_walk_dispatch()
            elif self.dispatch_policy == 'random_dispatch':
                dispatch_info = self._execute_random_dispatch()
            else:
                dispatch_info = {'dispatch_success': 0, 'dispatch_total': 0}

            step_info['dispatch_success'] = dispatch_info.get('dispatch_success', 0)
            step_info['dispatch_total'] = dispatch_info.get('dispatch_total', 0)
            self.episode_stats['total_dispatches'] += step_info['dispatch_success']

            # 6. 更新时间
            self.current_time_slice += 1
            if self.current_time_slice >= self.config.NUM_TIME_SLICES:
                self.current_time_slice = 0
                self.current_day += 1

            self.simulation_time += pd.Timedelta(seconds=self.config.TICK_DURATION_SEC)
            self.current_time = self.simulation_time

            # 7. 检查是否完成
            done = self.episode_step >= self.config.MAX_TICKS_PER_EPISODE

            self.reward_calculator.update(step_info)

            return {}, 0.0, done, {'step_info': step_info}

        except Exception as e:
            print(f"错误: step() 失败: {e}")
            traceback.print_exc()
            return {}, 0.0, True, {}

    def _cancel_timeout_orders(self):
        """取消超时订单"""
        cancelled_count = 0
        still_pending = []

        for order in self.pending_orders:
            if 'timestamp' in order:
                try:
                    gen_time = order['timestamp']
                    if isinstance(gen_time, str):
                        gen_time = pd.to_datetime(gen_time)

                    if gen_time.tzinfo is None and self.current_time.tzinfo is not None:
                        gen_time = gen_time.tz_localize(self.current_time.tzinfo)

                    wait_time_sec = (self.current_time - gen_time).total_seconds()

                    if wait_time_sec > self.config.MAX_WAITING_TIME:
                        order['status'] = 'cancelled'
                        self.episode_stats['total_orders_cancelled'] += 1
                        cancelled_count += 1
                    else:
                        still_pending.append(order)
                except Exception:
                    still_pending.append(order)
            else:
                still_pending.append(order)

        self.pending_orders = deque(still_pending)
        return cancelled_count

    def _execute_random_walk_dispatch(self):
        """随机游走调度策略"""
        idle_vehicle_ids = self.vehicle_manager.get_long_idle_vehicles(
            self.current_time, self.config.IDLE_THRESHOLD_SEC
        )

        if not idle_vehicle_ids:
            return {'dispatch_success': 0, 'dispatch_total': 0}

        dispatch_total = len(idle_vehicle_ids)
        dispatch_success = 0

        for vehicle_id in idle_vehicle_ids:
            vehicle = self.vehicle_manager.vehicles.get(vehicle_id)
            if vehicle is None or vehicle['status'] != 'idle':
                continue

            current_grid = vehicle['current_grid']
            grid_rows = self.config.GRID_SIZE[0]
            grid_cols = self.config.GRID_SIZE[1]
            row, col = divmod(current_grid, grid_cols)

            # 随机选择：上下左右或停留
            choices = [
                (row - 1, col),      # 上
                (row + 1, col),      # 下
                (row, col - 1),      # 左
                (row, col + 1),      # 右
                (row, col),          # 停留
            ]

            target_row, target_col = random.choice(choices)

            # 边界检查
            if target_row < 0 or target_row >= grid_rows or target_col < 0 or target_col >= grid_cols:
                target_row, target_col = row, col

            target_grid = target_row * grid_cols + target_col

            success = self.vehicle_manager.start_dispatching(vehicle_id, target_grid, self.current_time)
            if success:
                dispatch_success += 1

        return {'dispatch_success': dispatch_success, 'dispatch_total': dispatch_total}

    def _execute_random_dispatch(self):
        """随机调度到热点网格"""
        idle_vehicle_ids = self.vehicle_manager.get_long_idle_vehicles(
            self.current_time, self.config.IDLE_THRESHOLD_SEC
        )

        if not idle_vehicle_ids:
            return {'dispatch_success': 0, 'dispatch_total': 0}

        # 获取热点网格（订单最多的地方）
        hotspot_grids = self._get_hotspot_grids()
        if not hotspot_grids:
            hotspot_grids = list(range(self.config.NUM_GRIDS))

        dispatch_total = len(idle_vehicle_ids)
        dispatch_success = 0

        for vehicle_id in idle_vehicle_ids:
            vehicle = self.vehicle_manager.vehicles.get(vehicle_id)
            if vehicle is None or vehicle['status'] != 'idle':
                continue

            target_grid = random.choice(hotspot_grids)
            success = self.vehicle_manager.start_dispatching(vehicle_id, target_grid, self.current_time)
            if success:
                dispatch_success += 1

        return {'dispatch_success': dispatch_success, 'dispatch_total': dispatch_total}

    def _get_hotspot_grids(self, top_k=20):
        """获取订单最多的网格"""
        try:
            grid_counts = np.zeros(self.config.NUM_GRIDS)
            for order in self.pending_orders:
                try:
                    grid = int(order.get('grid_index', 0))
                    if 0 <= grid < self.config.NUM_GRIDS:
                        grid_counts[grid] += 1
                except (ValueError, TypeError):
                    pass

            if grid_counts.sum() == 0:
                return []

            top_indices = np.argsort(grid_counts)[-top_k:]
            return top_indices.tolist()
        except Exception:
            return []

    def get_episode_summary(self):
        """获取 Episode 总结"""
        metrics = self.reward_calculator.get_metrics()
        waiting_stats = {
            'avg_waiting_time': metrics.get('avg_waiting_time', 0.0),
            'max_waiting_time': metrics.get('max_waiting_time', 0.0),
            'min_waiting_time': metrics.get('min_waiting_time', 0.0),
            'std_waiting_time': metrics.get('std_waiting_time', 0.0),
        }

        return {
            'episode_stats': self.episode_stats.copy(),
            'reward_metrics': metrics,
            'waiting_time_stats': waiting_stats,
        }

