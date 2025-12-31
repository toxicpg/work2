# environment.py
"""
网约车调度环境 - 完整版 (V5.3 - 10秒 Tick / 事件驱动 / 等待时间奖励 / 再次修复 UnboundLocalError)
核心修改:
1. ... (V5.2 修改不变) ...
9. (V5.3) 在 _process_events 中再次加强 vehicle 变量的检查逻辑。
"""

import numpy as np
import pandas as pd
import torch
import time
import torch.nn.functional as F
from collections import deque, defaultdict
import json
import os
import random
import bisect
import traceback

# ===== ★★★ V5.5 K-D Tree 优化：添加导入 ★★★ =====
try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    print("="*70)
    print("警告: 未找到 'scipy' 库 (pip install scipy)。")
    print("OrderMatcher 将回退到 O(N*M) 的低效暴力搜索模式。")
    print("强烈建议安装 scipy 以大幅提升模拟速度。")
    print("="*70)
    SCIPY_AVAILABLE = False


# ... (OrderGenerator, VehicleManager, OrderMatcher, RewardCalculator 类代码保持 V5.2 不变) ...
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
            self.time_range = (pd.Timestamp.min, pd.Timestamp.max);
            self.orders_by_day_and_slice = {}
            self.total_days = 0;
            self.orders_df = pd.DataFrame();
            config.MAX_START_DAY = 0
            return
        self.time_range = (orders_df['timestamp'].min(), orders_df['timestamp'].max())
        orders_df = orders_df.sort_values('timestamp').reset_index(drop=True)
        if 'timestamp' not in orders_df.columns:
            if orders_df['departure_time'].iloc[0] > 1e10:
                orders_df['timestamp_utc'] = pd.to_datetime(orders_df['departure_time'], unit='ms').dt.tz_localize(
                    'UTC')
            else:
                orders_df['timestamp_utc'] = pd.to_datetime(orders_df['departure_time'], unit='s').dt.tz_localize('UTC')
            try:
                orders_df['timestamp'] = orders_df['timestamp_utc'].dt.tz_convert('Asia/Shanghai')
                print(f"\n  已将时间戳转换为 'Asia/Shanghai' 时区。")
            except Exception as e:
                print(f"\n  错误: 转换为 'Asia/Shanghai' 时区失败: {e}. (请 'pip install pytz')")
                orders_df['timestamp'] = orders_df['timestamp_utc']
        start_timestamp = self.time_range[0].normalize()
        orders_df['relative_day'] = (orders_df['timestamp'] - start_timestamp).dt.days
        hours = orders_df['timestamp'].dt.hour;
        minutes = orders_df['timestamp'].dt.minute
        minutes_from_midnight = hours * 60 + minutes
        orders_df['time_slice'] = (minutes_from_midnight // config.MACRO_STATISTICS_STEP_MINUTES).clip(0,143)  # 使用宏观时间片分组
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
        key = (current_day, current_time_slice)
        orders_in_slice = self.orders_by_day_and_slice.get(key, [])
        new_orders = [o.copy() for o in orders_in_slice]
        [o.update({'status': 'pending'}) for o in new_orders]
        return new_orders  # (已按时间排序)

    def get_day_count(self):
        return self.total_days


# ========== VehicleManager Class (V5.4 - 修复 Loss=0 Bug) ==========
# (与 V5.3 相同，但修复了 assign_order)
class VehicleManager:
    """车辆管理器 (V5.4 - 修复 assign_order 导致的 Loss=0 Bug)"""

    def __init__(self, config):
        self.config = config;
        self.vehicles = {};
        self.initialize_vehicles()

    def initialize_vehicles(self):
        total_vehicles = self.config.TOTAL_VEHICLES;
        seed = getattr(self.config, 'SEED', None)
        rng = np.random.default_rng(seed);
        positions = rng.integers(0, self.config.NUM_GRIDS, total_vehicles)
        print(f"  车辆初始化 ({total_vehicles}辆): 均匀随机分布")
        self.vehicles = {i: {'id': i, 'current_grid': int(positions[i]), 'status': 'idle', 'idle_since': None,
                             'dispatch_target': None, 'dispatch_start_time': None, 'assigned_order': None,
                             'pending_dispatch_experience': None} for i in range(total_vehicles)}

    def reset(self):
        self.initialize_vehicles()

    def update_dispatching_vehicles(self, current_time):
        vehicles_arrived = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'dispatching' and vehicle['dispatch_start_time'] is not None:
                try:
                    elapsed_seconds = (current_time - vehicle['dispatch_start_time']).total_seconds()
                    travel_time_seconds = self._calculate_travel_time(vehicle['current_grid'],
                                                                      vehicle['dispatch_target']) * 60.0
                    if elapsed_seconds >= travel_time_seconds: vehicles_arrived.append(vehicle_id)
                except TypeError as te:
                    # (处理时区不匹配等错误)
                    print(f"警告: update_dispatching_vehicles 时间比较错误: {te}. Vehicle ID: {vehicle_id}")
                    vehicle['status'] = 'idle' # 强制恢复
                    vehicle['idle_since'] = current_time
                    vehicle['dispatch_start_time'] = None


        for vehicle_id in vehicles_arrived:
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id];
                target_grid = vehicle['dispatch_target']
                if isinstance(target_grid, (int, np.integer)) and 0 <= target_grid < self.config.NUM_GRIDS: vehicle[
                    'current_grid'] = target_grid
                vehicle['status'] = 'idle';
                vehicle['dispatch_target'] = None;
                vehicle['dispatch_start_time'] = None;
                vehicle['idle_since'] = current_time

    def _calculate_travel_time(self, from_grid, to_grid):
        try:
            from_grid_int = int(from_grid); to_grid_int = int(to_grid)
        except (ValueError, TypeError):
            return 1.0
        if not (0 <= from_grid_int < self.config.NUM_GRIDS and 0 <= to_grid_int < self.config.NUM_GRIDS): return 1.0
        if from_grid_int == to_grid_int: return 0.0
        grid_cols = self.config.GRID_SIZE[1];
        from_row, from_col = divmod(from_grid_int, grid_cols);
        to_row, to_col = divmod(to_grid_int, grid_cols)
        grid_distance = abs(from_row - to_row) + abs(from_col - to_col);
        avg_speed = self.config.AVG_SPEED_KMH
        time_per_grid = 60.0 / avg_speed if avg_speed > 0 else 2.0;
        time_minutes = grid_distance * time_per_grid
        return max(0.1, time_minutes)

    # (V5.3 修复版 start_dispatching - 保持不变)
    def start_dispatching(self, vehicle_id, target_grid, current_time):
        """将空闲车辆状态变更为调度中"""
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None or vehicle.get('status') != 'idle':
            return False
        try:
            target_grid_int = int(target_grid)
            if not (0 <= target_grid_int < self.config.NUM_GRIDS):
                print(f"警告: start_dispatching 收到无效 target_grid: {target_grid}")
                return False
        except (ValueError, TypeError):
            print(f"警告: start_dispatching 收到无法转换的 target_grid: {target_grid}")
            return False
        vehicle['status'] = 'dispatching'
        vehicle['dispatch_target'] = target_grid_int
        vehicle['dispatch_start_time'] = current_time
        vehicle['idle_since'] = None
        return True

    # ===== ★★★ V5.4 修复 Loss=0 Bug ★★★ =====
    def assign_order(self, vehicle_id, order):
        if vehicle_id not in self.vehicles: 
            return False
            
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None: # 增加安全检查
             return False

        if vehicle['status'] != 'idle': 
            return False
            
        vehicle['status'] = 'serving'
        vehicle['assigned_order'] = order
        vehicle['idle_since'] = None
        
        # [核心修复]
        # 移除: vehicle['pending_dispatch_experience'] = None
        # 理由:
        # 1. OrderMatcher 只匹配 'idle' 状态的车辆。
        # 2. 车辆在到达调度目标点后，状态会从 'dispatching' 变回 'idle'。
        # 3. 此时，它暂存的 (S_t, A_t) 经验仍然有效，等待被这个 'idle' 状态下匹配到的
        #    第一个订单（即调度的结果）来奖励。
        # 4. 如果我们在这里清除 'pending_dispatch_experience'，
        #    那么 _process_events 将永远找不到 (S, A) 来关联奖励 R。
        #    这将导致 Replay Buffer 中没有调度经验，Loss 恒为 0。
        
        return True
    # ========================================

    def complete_service(self, vehicle_id, order, destination_grid, current_time):
        if vehicle_id not in self.vehicles:
            return False
        
        vehicle = self.vehicles[vehicle_id]
        current_grid_fallback = vehicle.get('current_grid', 0)
        try:
            dest_grid_int = int(destination_grid); assert 0 <= dest_grid_int < self.config.NUM_GRIDS
        except (ValueError, TypeError, AssertionError):
            dest_grid_int = current_grid_fallback
        vehicle['current_grid'] = dest_grid_int;
        vehicle['status'] = 'idle';
        vehicle['assigned_order'] = None
        vehicle['dispatch_target'] = None;
        vehicle['dispatch_start_time'] = None;
        vehicle['idle_since'] = current_time
        return True

    def get_long_idle_vehicles(self, current_time, threshold_seconds):
        long_idle_ids = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['status'] == 'idle':
                if vehicle['idle_since'] is None: vehicle['idle_since'] = current_time
                if isinstance(vehicle['idle_since'], pd.Timestamp):
                    try:
                        # (V5.4 修复: 确保时区可比较)
                        idle_since_time = vehicle['idle_since']
                        current_tz = getattr(current_time, 'tzinfo', None)
                        idle_tz = getattr(idle_since_time, 'tzinfo', None)
                        if idle_tz is None and current_tz is not None: idle_since_time = idle_since_time.tz_localize(current_tz)
                        elif idle_tz is not None and current_tz is None: current_time = current_time.tz_localize(idle_tz)
                        elif idle_tz != current_tz: idle_since_time = idle_since_time.tz_convert(current_tz)

                        idle_duration_seconds = (current_time - idle_since_time).total_seconds()
                        if idle_duration_seconds >= threshold_seconds: long_idle_ids.append(vehicle_id)
                    except Exception as e:
                        # print(f"警告: 计算 idle duration 时出错: {e}. Vehicle ID: {vehicle_id}") # 减少打印
                        vehicle['idle_since'] = current_time;
                else:
                    vehicle['idle_since'] = current_time
            else:
                vehicle['idle_since'] = None
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



# ========== OrderMatcher Class (V5.6 - K-D Tree 优化 & Bug修复) ==========
class OrderMatcher:
    """订单匹配器 (V5.6 - 使用 K-D 树进行 O(N*k) 高效匹配, 修复了 distance Bug)"""

    def __init__(self, config):
        self.config = config
        # (V5.5: 使用 K-NN 搜索)
        self.k_to_search = getattr(config, 'MATCHER_KNN_K', 10) # 从 config 读取 K
        self.travel_time_weight = getattr(config, 'MATCHER_TRAVEL_TIME_WEIGHT', 1.0)
        self.euclidean_weight = getattr(config, 'MATCHER_EUCLIDEAN_WEIGHT', 0.0)
        
        if SCIPY_AVAILABLE:
            print(f"  OrderMatcher 初始化 (K-D Tree 模式, k={self.k_to_search})")
        else:
            # (V5.5: 保留旧的半径作为回退)
            self.search_radius = getattr(config, 'MATCHER_SEARCH_RADIUS', 20)
            print(f"  OrderMatcher 初始化 (回退到 O(N*M) 暴力搜索模式, 半径={self.search_radius})")

    def match_orders(self, pending_orders, vehicle_manager, current_time):
        """ (V5.6) 使用 K-D 树 (或回退) 执行匹配, 修复了 distance Bug """
        
        matches = []
        unmatched_orders = list(pending_orders)
        
        # 1. 收集所有空闲车辆 (V5.5: 收集坐标和ID)
        idle_vehicles_data = []
        grid_cols = self.config.GRID_SIZE[1] # (在循环外获取)

        for v_id, vehicle in vehicle_manager.vehicles.items():
            if vehicle['status'] == 'idle':
                grid = vehicle.get('current_grid')
                if isinstance(grid, (int, np.integer)) and 0 <= grid < self.config.NUM_GRIDS:
                    # (V5.5: 计算 K-D 树所需的 2D 坐标)
                    row, col = divmod(grid, grid_cols)
                    # 存储 (坐标, 车辆ID, 原始网格ID)
                    idle_vehicles_data.append( ([row, col], v_id, grid) )

        if not idle_vehicles_data: 
            return [], unmatched_orders # 没有空闲车

        # 2. 准备订单和可用车辆
        still_unmatched = []
        # (V5.5: 使用 Set 跟踪可用车辆 ID，O(1) 移除)
        available_vehicle_ids = set(data[1] for data in idle_vehicles_data)
        random.shuffle(unmatched_orders)

        # 3. ★★★ K-D 树优化路径 (如果 scipy 可用) ★★★
        if SCIPY_AVAILABLE and len(idle_vehicles_data) > 0:
            try:
                # 提取所有坐标点
                points = [data[0] for data in idle_vehicles_data]
                # 提取所有 ID (与 points 顺序一致)
                point_ids = [data[1] for data in idle_vehicles_data]
                
                # ★★★ 构建 K-D 树 (O(M log M)) ★★★
                vehicle_tree = KDTree(points)
                
                # 确定 k 值 (不能超过树中点的数量)
                k_actual = min(self.k_to_search, len(points))
                if k_actual == 0: # 如果 K-D 树为空
                    return [], unmatched_orders

                # ★★★ 循环 N 个订单 (N * O(k log M)) ★★★
                for order in unmatched_orders:
                    try: 
                        order_grid = int(order['grid_index'])
                        assert 0 <= order_grid < self.config.NUM_GRIDS
                    except (ValueError, TypeError, KeyError, AssertionError):
                        still_unmatched.append(order); continue
                    
                    order_row, order_col = divmod(order_grid, grid_cols)
                    order_coords = [order_row, order_col]
                    
                    # ★★★ 查询 K-D 树 (O(log M) 或 O(k log M)) ★★★
                    try:
                        distances, neighbor_indices = vehicle_tree.query(order_coords, k=k_actual, p=2) # p=2: 欧几里得距离
                    except ValueError as ve:
                         # (查询失败，例如 k > 树中节点数，或树为空)
                         # print(f"警告: KDTree query 失败: {ve}") # 减少打印
                         still_unmatched.append(order)
                         continue

                    # (确保 k=1 时返回的是列表)
                    if k_actual == 1:
                        if not isinstance(neighbor_indices, (list, np.ndarray)): neighbor_indices = [neighbor_indices]
                        if not isinstance(distances, (list, np.ndarray)): distances = [distances]
                    
                    # ★★★ 在 K 个结果中决策 (O(K)) ★★★
                    best_match_vehicle_id = None
                    min_cost = float('inf') 
                    
                    # ===== [V5.6 修复: 必须存储获胜者的真实旅行时间] =====
                    min_travel_time_for_best_vehicle = float('inf') 
                    # =================================================

                    for i, index in enumerate(neighbor_indices):
                        if index >= len(point_ids):
                            continue
                        vehicle_id = point_ids[index]
                        if vehicle_id in available_vehicle_ids:
                            euclidean_distance = distances[i]
                            vehicle_grid = vehicle_manager.vehicles[vehicle_id]['current_grid']
                            travel_time = vehicle_manager._calculate_travel_time(vehicle_grid, order_grid)
                            cost = self.euclidean_weight * euclidean_distance + self.travel_time_weight * travel_time
                            if cost < min_cost:
                                min_cost = cost
                                best_match_vehicle_id = vehicle_id
                                min_travel_time_for_best_vehicle = travel_time

                    # 7. 分配 (与之前相同)
                    if best_match_vehicle_id is not None:
                        assign_success = vehicle_manager.assign_order(best_match_vehicle_id, order)
                        if assign_success:
                            # ===== [V5.6 修复: 存储 'distance' 为真实的分钟数] =====
                            matches.append({'order': order, 
                                            'vehicle_id': best_match_vehicle_id, 
                                            'distance': min_travel_time_for_best_vehicle}) # <-- 必须是分钟
                            # ===================================================
                            available_vehicle_ids.remove(best_match_vehicle_id) # 从可用集合中移除
                        else:
                            still_unmatched.append(order)
                    else:
                        still_unmatched.append(order) # K个邻居都不可用或已被分配
                
                return matches, still_unmatched # (返回 K-D 树路径的结果)

            except Exception as e:
                 print(f"❌ K-D Tree 匹配失败: {e}. 本次 step 无匹配。")
                 traceback.print_exc()
                 return [], unmatched_orders # 出错则本轮不匹配

        # 4. ★★★ 回退到 O(N*M) 暴力搜索 (如果 scipy 不可用) ★★★
        else:
            # (这是您之前的 O(N*M) 逻辑，稍作修改以使用 available_vehicle_ids)
            available_idle_vehicles_dict = {data[1]: data[2] for data in idle_vehicles_data} # {id: grid}
            
            for order in unmatched_orders:
                try: order_grid = int(order['grid_index']); assert 0 <= order_grid < self.config.NUM_GRIDS
                except (ValueError, TypeError, KeyError, AssertionError): still_unmatched.append(order); continue
                
                order_row, order_col = divmod(order_grid, grid_cols);
                best_match_vehicle_id = None; min_travel_time = float('inf')

                vehicle_items = list(available_idle_vehicles_dict.items()) # {id: grid}
                random.shuffle(vehicle_items)
                
                for v_id, v_grid in vehicle_items:
                    v_row, v_col = divmod(v_grid, grid_cols);
                    dist = abs(v_row - order_row) + abs(v_col - order_col)
                    
                    if dist <= self.search_radius: # (使用旧的半径逻辑)
                        travel_time = vehicle_manager._calculate_travel_time(v_grid, order_grid)
                        if travel_time < min_travel_time:
                            min_travel_time = travel_time
                            best_match_vehicle_id = v_id

                if best_match_vehicle_id is not None:
                    assign_success = vehicle_manager.assign_order(best_match_vehicle_id, order)
                    if assign_success:
                        matches.append({'order': order, 'vehicle_id': best_match_vehicle_id, 'distance': min_travel_time})
                        if best_match_vehicle_id in available_idle_vehicles_dict:
                             del available_idle_vehicles_dict[best_match_vehicle_id]
                    else:
                        still_unmatched.append(order)
                else:
                    still_unmatched.append(order)
            
            return matches, still_unmatched



class RewardCalculator:
    """奖励计算器 - V5.2 (降级：仅用于评估和统计)"""

    def __init__(self, config):
        self.config = config
        self.use_episode_bonus = getattr(config, 'USE_EPISODE_BONUS', False)
        self.episode_bonus_thresholds = EPISODE_BONUS_PARAMS if self.use_episode_bonus else {}
        self.reset();
        self._print_config()

    def _print_config(self):
        print(f"\n{'=' * 70}")
        print(f"奖励计算器配置 (V5.2 - 仅用于评估)")
        t0_val = 'N/A'
        if hasattr(self.config, 'REWARD_FORMULA_V4') and isinstance(self.config.REWARD_FORMULA_V4, dict):
            t0_val = self.config.REWARD_FORMULA_V4.get('T_CHARACTERISTIC', 'N/A')
        print(f"  (事件驱动奖励: exp(-wait_time / T0), T0={t0_val})")
        print(f"{'=' * 70}\n")

    def reset(self):
        self.completed_orders = 0;
        self.cancelled_orders = 0;
        self.matched_orders_total = 0;
        self.waiting_times = []
        self.dispatch_success_count = 0;
        self.dispatch_total_count = 0;
        self.step_count = 0;
        self.total_revenue = 0.0

    def update(self, step_info):
        self.step_count += 1;
        # 记录本tick的匹配与完成、取消数
        self.completed_orders += step_info.get('completed_orders', 0)
        self.matched_orders_total += step_info.get('matched_orders', 0)
        self.cancelled_orders += step_info.get('cancelled_orders', 0)
        self.waiting_times.extend(step_info.get('waiting_times', []))
        self.dispatch_success_count += step_info.get('dispatch_success', 0)
        self.dispatch_total_count += step_info.get('dispatch_total', 0)
        self.total_revenue += step_info.get('revenue', 0.0)

    def calculate_step_reward(self, step_info):
        """计算当前步骤的奖励"""
        reward = 0.0
        
        # 基于匹配订单数的奖励
        matched_orders = step_info.get('matched_orders', 0)
        reward += matched_orders * 10.0  # 每匹配一个订单奖励10分
        
        # 基于取消订单数的惩罚
        cancelled_orders = step_info.get('cancelled_orders', 0)
        reward -= cancelled_orders * 5.0  # 每取消一个订单惩罚5分
        
        # 基于完成订单数的奖励
        completed_orders = step_info.get('completed_orders', 0)
        reward += completed_orders * 20.0  # 每完成一个订单奖励20分
        
        return reward

    def get_metrics(self, calculate_total=False):
        total_p = self.completed_orders + self.cancelled_orders
        metrics = {'completed_orders': self.completed_orders, 'cancelled_orders': self.cancelled_orders,
                   'total_revenue': self.total_revenue,
                   'completion_rate': self.completed_orders / total_p if total_p > 0 else 0.0,
                   'cancel_rate': self.cancelled_orders / total_p if total_p > 0 else 0.0,
                   'matched_orders_total': self.matched_orders_total,
                   'match_rate': self.matched_orders_total / (self.matched_orders_total + self.cancelled_orders)
                   if (self.matched_orders_total + self.cancelled_orders) > 0 else 0.0}
        if self.waiting_times:
            metrics.update(
                {'avg_waiting_time': np.mean(self.waiting_times), 'max_waiting_time': np.max(self.waiting_times),
                 'min_waiting_time': np.min(self.waiting_times), 'std_waiting_time': np.std(self.waiting_times)})
        else:
            metrics.update(
                {k: 0.0 for k in ['avg_waiting_time', 'max_waiting_time', 'min_waiting_time', 'std_waiting_time']})
        return metrics

    def print_summary(self):
        metrics = self.get_metrics()
        print(f"\nEpisode Summary (Ticks: {self.step_count})")
        print(f"  总收入: {metrics['total_revenue']:.2f}")
        print(
            f"  订单: 完成={metrics['completed_orders']:,} ({metrics['completion_rate']:.1%}), Cancelled={metrics['cancelled_orders']:,} ({metrics['cancel_rate']:.1%})")
        print(
            f"  等待 (秒): Avg={metrics['avg_waiting_time']:.1f}, Std={metrics['std_waiting_time']:.1f}, Min={metrics['min_waiting_time']:.1f}, Max={metrics['max_waiting_time']:.1f}")

    def calculate_episode_bonus(self, episode_summary):
        if not self.use_episode_bonus: return 0.0
        completion_rate = episode_summary.get('reward_metrics', {}).get('completion_rate', 0.0)
        for threshold in sorted(self.episode_bonus_thresholds.keys(), reverse=True):
            if completion_rate >= threshold: return self.episode_bonus_thresholds[threshold]
        return 0.0


class RideHailingEnvironment:

    def __init__(self, config, data_processor, orders_df):
        self.config = config;
        self.data_processor = data_processor
        if orders_df.empty: print("警告：初始化 RideHailingEnvironment 时 orders_df 为空!")
        self.order_generator = OrderGenerator(config, orders_df)
        self.vehicle_manager = VehicleManager(config)
        self.order_matcher = OrderMatcher(config)
        self.reward_calculator = RewardCalculator(config)
        self._load_action_mapping();
        self.pending_orders = deque();
        self.event_queue = []
        self.buffered_orders = deque();
        self.current_macro_slice_key = None
        # 尝试使用 UTC 以避免时区问题，或确保所有 Timestamp 都有时区
        try:
            self.simulation_time = pd.Timestamp.now(tz='Asia/Shanghai')
        except Exception:
            self.simulation_time = pd.Timestamp.now().tz_localize('UTC').tz_convert('Asia/Shanghai')  # Fallback
        self.current_time = self.simulation_time
        self.start_time = self.simulation_time  # 添加start_time属性
        self.episode_start_day = 0;
        self.current_day = 0;
        self.current_time_slice = 0
        self.episode_step = 0;
        self.episode_stats = {}
        self.model = None;
        self.replay_buffer = None;
        self.device = 'cpu'
        self.T0 = 300.0  # Default
        self.reward_scale = 1.0  # Default
        if hasattr(self.config, 'REWARD_FORMULA_V4') and isinstance(self.config.REWARD_FORMULA_V4, dict):
            self.T0 = self.config.REWARD_FORMULA_V4.get('T_CHARACTERISTIC', 300.0)
            self.reward_scale = self.config.REWARD_SCALE_FACTOR
        if self.T0 <= 0: self.T0 = 300.0
        print(f"✓ 环境初始化完成 (每 Tick {config.TICK_DURATION_SEC} 秒, T0={self.T0}s, Scale={self.reward_scale})")

    def _load_action_mapping(self):
        mapping_file = os.path.join(self.config.PROCESSED_DATA_PATH, 'action_mapping.json')
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                self.action_to_grid = {int(k): int(v) for k, v in mapping.items()}
                print(f"  动作映射加载成功 ({len(self.action_to_grid)} actions)")
            else:
                raise FileNotFoundError(f"文件不存在: {mapping_file}")
        except Exception as e:
            print(f"  ❌ 加载动作映射文件失败: {e}. 使用默认映射.")
            self.action_to_grid = {i: i % self.config.NUM_GRIDS for i in range(self.config.NUM_ACTIONS)}
        if len(self.action_to_grid) != self.config.NUM_ACTIONS: print(
            f"  ⚠ 警告: NUM_ACTIONS ({self.config.NUM_ACTIONS}) 与映射大小不匹配!")

    def _action_to_hotspot_grid(self, action):
        if not hasattr(self, 'action_to_grid') or not self.action_to_grid: return action % self.config.NUM_GRIDS
        grid_id = self.action_to_grid.get(action)
        if grid_id is None: return action % self.config.NUM_GRIDS
        return max(0, min(self.config.NUM_GRIDS - 1, grid_id))

    def reset(self):
        available_days = self.order_generator.get_day_count();
        max_start_day = max(0, available_days - self.config.EPISODE_DAYS)
        self.episode_start_day = random.randint(0, max_start_day);
        self.current_day = self.episode_start_day
        self.current_time_slice = 0;
        self.episode_step = 0
        if hasattr(self.order_generator, 'time_range') and self.order_generator.time_range[0] != pd.Timestamp.min:
            base_time = self.order_generator.time_range[0].normalize()
        else:
            base_time = pd.Timestamp(self.config.DATA_START_DATE, tz='Asia/Shanghai').normalize()
        self.simulation_time = base_time + pd.Timedelta(days=self.current_day)
        if self.simulation_time.tzinfo is None: self.simulation_time = self.simulation_time.tz_localize('Asia/Shanghai')
        self.current_time = self.simulation_time
        self.start_time = self.simulation_time  # 重置start_time
        self.pending_orders.clear();
        self.event_queue.clear();
        self.buffered_orders.clear();
        self.current_macro_slice_key = None
        self.episode_stats = {'total_orders_generated': 0, 'total_orders_matched': 0, 'total_orders_cancelled': 0,
                              'total_dispatches': 0, 'total_revenue': 0.0}
        self.vehicle_manager.reset();
        self.reward_calculator.reset()
        return self._get_state()

    def step(self, current_epsilon=0.1):
        """ (V5.3) 执行一个微观 Tick """
        self.episode_step += 1
        step_info = {'matched_orders': 0, 'cancelled_orders': 0, 'completed_orders': 0, 'waiting_times': [], 'dispatch_success': 0,
                     'dispatch_total': 0, 'new_orders': 0, 'revenue': 0.0}
        
        # 添加调试输出
        if getattr(self.config, 'VERBOSE', False) and self.episode_step <= 5:
            print(f"DEBUG Step {self.episode_step}: Starting step at time {self.simulation_time}")
        
        try:
            # --- T 时刻开始 ---
            # 1. 处理到期事件
            self._process_events(self.simulation_time, step_info)
            # 2. 更新车辆移动
            self.vehicle_manager.update_dispatching_vehicles(self.simulation_time)
            # 3. 取消超时订单
            cancelled_this_tick = self._cancel_timeout_orders(self.simulation_time);
            step_info['cancelled_orders'] = cancelled_this_tick
            # 4. 匹配订单
            if self.pending_orders:
                matches, still_pending = self.order_matcher.match_orders(list(self.pending_orders),
                                                                         self.vehicle_manager, self.simulation_time)
                
                # 添加调试输出
                if hasattr(self, '_debug_match_count'):
                    self._debug_match_count += 1
                else:
                    self._debug_match_count = 1
                    
                if getattr(self.config, 'VERBOSE', False) and self._debug_match_count <= 10:
                    print(f"DEBUG Match - Count {self._debug_match_count}:")
                    print(f"  Pending orders: {len(self.pending_orders)}")
                    print(f"  Matches made: {len(matches)}")
                
                self.pending_orders = deque(still_pending);
                step_info['matched_orders'] = len(matches);
                self.episode_stats['total_orders_matched'] += len(matches)
                for match in matches:
                    order = match['order'];
                    order['status'] = 'matched';
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
                    wait_for_pickup_min = match.get('distance', 0.0);
                    wait_for_pickup_sec = wait_for_pickup_min * 60.0
                    total_wait_time_sec = wait_to_match_sec + wait_for_pickup_sec
                    step_info['waiting_times'].append(total_wait_time_sec)
                    order['total_wait_time_sec'] = total_wait_time_sec
                    self._schedule_order_completion(match, self.simulation_time)
            # 5. 生成 *本 Tick* 新订单
            new_orders = self._load_orders_for_tick();
            [o.update({'generated_at': self.simulation_time}) for o in new_orders]
            self.pending_orders.extend(new_orders);
            step_info['new_orders'] = len(new_orders);
            self.episode_stats['total_orders_generated'] += len(new_orders)
            # 6. 执行主动调度
            dispatch_info = self._execute_proactive_dispatch(current_epsilon);
            step_info.update(dispatch_info);
            self.episode_stats['total_dispatches'] += dispatch_info.get('dispatch_total', 0)
            # 7. 更新评估指标
            self.reward_calculator.update(step_info)
            # 计算当前步骤的奖励
            step_reward = self.reward_calculator.calculate_step_reward(step_info)
            # 8. 推进时间
            self.simulation_time += pd.Timedelta(seconds=self.config.TICK_DURATION_SEC);
            self.current_time = self.simulation_time
            time_since_start_days = (
                        self.current_time.normalize() - self.order_generator.time_range[0].normalize()).days;
            self.current_day = time_since_start_days
            minutes_today = self.current_time.hour * 60 + self.current_time.minute;
            self.current_time_slice = min(minutes_today // self.config.MACRO_STATISTICS_STEP_MINUTES, 143)
            # 9. 检查结束
            done = self.episode_step >= self.config.MAX_TICKS_PER_EPISODE
            # 10. 获取下一状态
            next_state = self._get_state()
            # 11. 返回
            info = {'step_reward': step_reward, 'pending': len(self.pending_orders), 'step_info': step_info}
            return next_state, step_reward, done, info

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
            self.buffered_orders = deque(all_slice_orders);
            self.current_macro_slice_key = key
        new_orders_for_tick = [];
        tick_end_time = self.simulation_time + pd.Timedelta(seconds=self.config.TICK_DURATION_SEC)
        while self.buffered_orders:
            order = self.buffered_orders[0];
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
                    print(f"警告: 时区转换失败 {tz_e}"); break  # 无法比较，退出

            if order_time < tick_end_time:
                new_orders_for_tick.append(self.buffered_orders.popleft())
            else:
                break
        return new_orders_for_tick

    def _cancel_timeout_orders(self, current_time):
        cancelled_count = 0;
        still_pending = deque()
        try:
            cutoff_time = current_time - pd.Timedelta(seconds=self.config.MAX_WAITING_TIME)
            # 确保 cutoff_time 也有时区 (如果 current_time 有)
            if current_time.tzinfo is not None and cutoff_time.tzinfo is None:
                cutoff_time = cutoff_time.tz_localize(current_time.tzinfo)
        except OverflowError:
            cutoff_time = pd.Timestamp.min.tz_localize(current_time.tzinfo) if current_time.tzinfo else pd.Timestamp.min
        while self.pending_orders:
            order = self.pending_orders.popleft();
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
                    order['status'] = 'cancelled'; self.episode_stats[
                        'total_orders_cancelled'] += 1; cancelled_count += 1
                else:
                    still_pending.append(order)
            else:
                still_pending.append(order)  # 保留没有生成时间的订单
        self.pending_orders = still_pending
        return cancelled_count

    def _execute_proactive_dispatch(self, epsilon):
        """ (V5.2 - 核心) 在 Tick 执行调度决策，获取 S_t 并包含真实位置 """
        if self.model is None: return {'dispatch_success': 0, 'dispatch_total': 0}  # 评估模式下 buffer 为 None
        # (检查 buffer 仅在需要 push 时进行)

        idle_vehicle_ids = self.vehicle_manager.get_long_idle_vehicles(self.simulation_time,
                                                                       self.config.IDLE_THRESHOLD_SEC)
        
        # 添加调试输出
        if hasattr(self, '_debug_dispatch_count'):
            self._debug_dispatch_count += 1
        else:
            self._debug_dispatch_count = 1
            
        if getattr(self.config, 'VERBOSE', False) and self._debug_dispatch_count <= 10:
            total_idle = sum(1 for v in self.vehicle_manager.vehicles.values() if v['status'] == 'idle')
            elapsed_seconds = (self.simulation_time - self.start_time).total_seconds()
            print(f"DEBUG Dispatch [Step {self.episode_step}]: Time={self.simulation_time.strftime('%H:%M:%S')}, "
                  f"Elapsed={elapsed_seconds:.0f}s, Threshold={self.config.IDLE_THRESHOLD_SEC}s")
            print(f"  Total_idle={total_idle}, Long_idle={len(idle_vehicle_ids)}, epsilon={epsilon:.4f}")
        
        if not idle_vehicle_ids: return {'dispatch_success': 0, 'dispatch_total': 0}
        dispatch_total = len(idle_vehicle_ids);
        dispatch_success = 0

        # (优化：如果车辆过多，可以考虑只处理一部分，或并行处理)
        # random.shuffle(idle_vehicle_ids) # 可选：随机化处理顺序

        for vehicle_id in idle_vehicle_ids:
            # (在循环开始时获取车辆，确保状态最新)
            vehicle = self.vehicle_manager.vehicles.get(vehicle_id)
            # (再次检查状态，因为匹配可能改变了它)
            if not vehicle or vehicle['status'] != 'idle': continue

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

            if action == -1: continue  # 如果动作选择失败，跳过

            target_grid = self._action_to_hotspot_grid(action)

            # 暂存 (S, A) - 确保只在 replay_buffer 存在时暂存
            if self.replay_buffer is not None:
                vehicle['pending_dispatch_experience'] = (S_micro, action)
            else:
                vehicle['pending_dispatch_experience'] = None  # 评估模式清除

            success = self.vehicle_manager.start_dispatching(vehicle_id, target_grid, self.simulation_time)
            if success: dispatch_success += 1
            # (如果 start_dispatching 失败，pending_exp 会在下次检查时被覆盖或清除)

        return {'dispatch_success': dispatch_success, 'dispatch_total': dispatch_total}

    def _schedule_order_completion(self, match, current_time):
        order = match.get('order');
        vehicle_id = match.get('vehicle_id');
        if order is None or vehicle_id is None: return
        try:
            pickup_grid = int(order['grid_index']); dest_grid = int(order.get('dest_grid_index', pickup_grid));
        except (ValueError, TypeError, KeyError):
            print(f"警告: _schedule... 缺少网格信息: {order}"); return
        if not (0 <= dest_grid < self.config.NUM_GRIDS): dest_grid = pickup_grid
        if not (0 <= pickup_grid < self.config.NUM_GRIDS): pickup_grid = 0
        service_time_minutes = self.vehicle_manager._calculate_travel_time(pickup_grid, dest_grid)
        pickup_time_minutes = match.get('distance', 0.0)
        total_duration_minutes = pickup_time_minutes + service_time_minutes
        service_duration = pd.Timedelta(minutes=total_duration_minutes)
        min_duration = pd.Timedelta(seconds=self.config.TICK_DURATION_SEC)
        completion_time = current_time + max(min_duration, service_duration)
        
        # 添加调试输出
        if hasattr(self, '_debug_schedule_count'):
            self._debug_schedule_count += 1
        else:
            self._debug_schedule_count = 1
            
        if getattr(self.config, 'VERBOSE', False) and self._debug_schedule_count <= 10:
            print(f"DEBUG Schedule - Count {self._debug_schedule_count}:")
            print(f"  Vehicle {vehicle_id}: pickup_time={pickup_time_minutes:.2f}min, service_time={service_time_minutes:.2f}min")
            print(f"  Total duration: {total_duration_minutes:.2f}min")
            print(f"  Completion time: {completion_time}")
            print(f"  Current time: {current_time}")
        
        event = {'time': completion_time, 'type': 'order_completion', 'vehicle_id': int(vehicle_id), 'order': order,
                 'destination_grid': dest_grid}
        # (V5.1 优化: 使用 bisect 保持排序, 兼容 Python 3.9)
        # Python 3.10+ 支持 key 参数，3.9 及以下版本不支持
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda e: e['time'])

    # ===== ★★★ V5.3 再次修复 UnboundLocalError ★★★ =====
    def _process_events(self, current_time, step_info):
        """ (V5.3 - 核心 / 再次修复 UnboundLocalError) 处理到期事件, PUSH 基于等待时间的奖励 (S, A, R, S') 到 Buffer """
        due_events_indices = [];
        processed_count = 0
        # (假设 event_queue 已通过 bisect 排序 或 在这里排序)
        # self.event_queue.sort(key=lambda e: e['time']) # 如果 _schedule_order_completion 没用 bisect

        for i, event in enumerate(self.event_queue):
            try:
                # 检查 event 是否为 None 或缺少 'time' 键
                if event is None or 'time' not in event:
                    # print(f"警告: event_queue 中发现无效事件 (None 或缺少 time): {event}") # 减少打印
                    continue  # 跳过无效事件

                event_time = event.get('time')
                # 确保 event_time 是 Timestamp
                if not isinstance(event_time, pd.Timestamp):
                    # print(f"警告: event_queue 中事件时间类型错误: {type(event_time)}, Event: {event}")
                    continue

                # 确保时区一致或可比较
                if event_time.tzinfo is None and current_time.tzinfo is not None:
                    event_time = event_time.tz_localize(current_time.tzinfo)
                elif event_time.tzinfo is not None and current_time.tzinfo is None:
                    current_time = current_time.tz_localize(event_time.tzinfo)
                elif event_time.tzinfo != current_time.tzinfo:
                    try:
                        event_time = event_time.tz_convert(current_time.tzinfo)
                    except Exception:
                        # print(f"警告: 事件时间时区转换失败: {event}")
                        continue  # 无法比较，跳过

                if event_time <= current_time:
                    due_events_indices.append(i)
                else:
                    # (由于队列已排序, 我们可以提前停止)
                    break
            except Exception as ex:
                # 捕获其他可能的异常
                print(f"警告: 检查事件时间时出错: {event}. Error: {ex}")
                continue  # 跳过这个事件

        if not due_events_indices:  # 如果没有到期事件，直接返回
            return

        # 从后往前处理并移除，避免索引问题
        for i in reversed(due_events_indices):
            event = None  # 初始化 event 变量 outside try block
            try:
                # 确保索引有效 before pop
                if i < len(self.event_queue):
                    event = self.event_queue.pop(i)
                    processed_count += 1
                else:
                    # print(f"警告: 尝试 pop 无效索引 {i} (队列长度 {len(self.event_queue)})") # 减少打印
                    continue  # 跳过无效索引

                if event is None:  # 再次检查 pop 的结果
                    continue

                event_type = event.get('type')

                if event_type == 'order_completion':
                    v_id = event.get('vehicle_id');
                    o_data = event.get('order');
                    d_grid = event.get('destination_grid')

                    if v_id is None or o_data is None or d_grid is None:
                        # print(f"警告: 'order_completion' 事件缺少关键信息: {event}") # 减少打印
                        continue

                    # ★★★ V5.3 核心修复逻辑 ★★★
                    # 1. 先获取 vehicle 对象
                    vehicle = self.vehicle_manager.vehicles.get(v_id)
                    if not vehicle:
                        # (车辆可能已下线或 reset, 跳过即可)
                        continue

                    # 2. 调用 complete_service
                    service_completed = self.vehicle_manager.complete_service(v_id, o_data, d_grid, current_time)
                    if not service_completed:
                        # (理论上不应失败，除非 v_id 在 complete_service 内部失效)
                        continue

                    # --- 现在 vehicle 对象肯定存在且 service 已完成 ---
                    new_grid = vehicle['current_grid']  # 获取新位置

                    # 3. 检查是否有暂存经验
                    pending_exp = vehicle.get('pending_dispatch_experience')
                    if (pending_exp is not None and self.replay_buffer is not None):
                        # 4. 构建 (S, A, R, S')
                        (S_t, A_t) = vehicle.pop('pending_dispatch_experience')  # 取出并清除

                        # === V5.2 奖励计算 ===
                        total_wait_time_sec = o_data.get('total_wait_time_sec', -1.0)
                        if total_wait_time_sec < 0:
                            # 尝试备用计算
                            try:
                                generated_at = o_data.get('generated_at')
                                matched_time = o_data.get('matched_time')
                                if generated_at and matched_time:
                                    wait_to_match_sec = max(0, (matched_time - generated_at).total_seconds())
                                else:
                                    # 使用更合理的fallback值：假设平均等待时间
                                    wait_to_match_sec = 60.0  # 1分钟

                                # 估算接驾时间 (使用 S_t 中的位置)
                                pickup_time_minutes_est = self.vehicle_manager._calculate_travel_time(
                                    S_t['vehicle_location'], o_data['grid_index'])
                                wait_for_pickup_sec_est = pickup_time_minutes_est * 60.0
                                total_wait_time_sec = wait_to_match_sec + wait_for_pickup_sec_est
                            except Exception:
                                total_wait_time_sec = 300  # Fallback

                        wait_score = np.exp(-max(0.0, total_wait_time_sec) / self.T0)
                        w = getattr(self.config, 'REWARD_WEIGHTS', None)
                        if isinstance(w, dict):
                            w_match = float(w.get('W_MATCH', 1.0))
                            w_wait = float(w.get('W_WAIT', 1.0))
                            w_wait_score = float(w.get('W_WAIT_SCORE', 0.0))
                        else:
                            w_match = 1.0
                            w_wait = 1.0
                            w_wait_score = 0.0
                        R_t = (w_match + w_wait_score * wait_score - w_wait * (max(0.0, total_wait_time_sec) / self.T0)) * self.reward_scale
                        
                        # 调试输出奖励计算
                        if hasattr(self, '_debug_reward_count'):
                            self._debug_reward_count += 1
                        else:
                            self._debug_reward_count = 1
                            
                        if getattr(self.config, 'VERBOSE', False) and self._debug_reward_count <= 10:
                            print(f"DEBUG Reward - Count {self._debug_reward_count}:")
                            print(f"  Wait time: {total_wait_time_sec:.2f}s")
                            print(f"  Wait score: {wait_score:.6f}")
                            print(f"  R_t: {R_t:.6f}")
                            print(f"  T0: {self.T0}, Scale: {self.reward_scale}")

                        # (V5.1) 获取包含 *新位置* 的 S_{t+1}
                        S_t_plus_1 = self._get_state(vehicle_location_override=new_grid)
                        done = False

                        # 5. PUSH 经验
                        try:
                            if self.replay_buffer is not None and isinstance(S_t, dict) and isinstance(S_t_plus_1, dict):
                                self.replay_buffer.push(S_t, A_t, R_t, S_t_plus_1, done)
                            elif self.replay_buffer is None:
                                pass  # 评估模式，不存储经验
                            else:
                                print(f"错误: 尝试 PUSH 非字典状态.")
                        except Exception as push_e:
                            print(f"❌ Replay Buffer PUSH 失败: {push_e}")

                        # 6. 记录统计 (收入和完成订单数)
                        revenue = o_data.get('fee', 0.0)
                        step_info['revenue'] += revenue
                        self.episode_stats['total_revenue'] += revenue
                        
                        # 更新完成订单计数
                        if 'completed_orders' not in step_info:
                            step_info['completed_orders'] = 0
                        step_info['completed_orders'] += 1
                        self.episode_stats['total_orders_completed'] = self.episode_stats.get('total_orders_completed', 0) + 1
                    else:
                        # (匹配订单完成, 记录收入和完成订单数)
                        revenue = o_data.get('fee', 0.0)
                        step_info['revenue'] += revenue
                        self.episode_stats['total_revenue'] += revenue
                        
                        # 更新完成订单计数
                        if 'completed_orders' not in step_info:
                            step_info['completed_orders'] = 0
                        step_info['completed_orders'] += 1
                        self.episode_stats['total_orders_completed'] = self.episode_stats.get('total_orders_completed', 0) + 1
                # --- 其他 event_type ---

            except KeyError as ke:
                print(f"❌ 处理事件时发生 KeyError: {ke}. Event: {event}")
            except Exception as e:
                print(f"❌ 处理事件时发生意外错误: {e}. Event: {event}"); traceback.print_exc()

    # ===============================

    # ===== V5.1 修复 =====
    def _get_state(self, current_time=None, vehicle_location_override=None):
        """获取状态 (V5.1 - 修复 vehicle_location)"""
        time_to_use = current_time if current_time is not None else self.current_time
        idle_vehicle_dist = self.vehicle_manager.get_idle_distribution()
        busy_vehicle_dist = self.vehicle_manager.get_busy_distribution()
        if not hasattr(self, 'data_processor') or self.data_processor is None:
            node_features = torch.zeros((self.config.NUM_GRIDS, self.config.INPUT_DIM));
            day_of_week = 0
        else:
            try:
                node_features = self.data_processor.prepare_mgcn_input(self.order_generator.orders_df, time_to_use,
                                                                       idle_vehicle_dist=idle_vehicle_dist,
                                                                       busy_vehicle_dist=busy_vehicle_dist)
            except Exception as e:
                print(f"错误: prepare_mgcn_input 失败: {e}"); node_features = torch.zeros(
                    (self.config.NUM_GRIDS, self.config.INPUT_DIM))
            day_of_week = time_to_use.dayofweek
        final_vehicle_location = 0
        if vehicle_location_override is not None:
            try:
                final_vehicle_location = int(vehicle_location_override)
            except (ValueError, TypeError):
                final_vehicle_location = 0  # Fallback
        state_dict = {'node_features': node_features.cpu(), 'vehicle_location': final_vehicle_location,
                      'day_of_week': day_of_week}
        return state_dict

    # =====================

    def get_episode_summary(self):
        metrics = self.reward_calculator.get_metrics()
        waiting_stats = {k: metrics.get(k, 0.0) for k in
                         ['avg_waiting_time', 'max_waiting_time', 'min_waiting_time', 'std_waiting_time']}
        waiting_stats['count'] = len(getattr(self.reward_calculator, 'waiting_times', []))
        vehicle_stats = self.vehicle_manager.get_statistics();
        total_vehicles = sum(vehicle_stats.values())
        if total_vehicles > 0:
            metrics.update({k: vehicle_stats.get(s, 0) / total_vehicles for k, s in
                            [('vehicle_utilization', 'serving'), ('idle_rate', 'idle'),
                             ('dispatching_rate', 'dispatching')]})
        else:
            metrics.update({k: 0.0 for k in ['vehicle_utilization', 'idle_rate', 'dispatching_rate']})
        total_gen = self.episode_stats.get('total_orders_generated', 0);
        processed = metrics.get('completed_orders', 0) + metrics.get('cancelled_orders', 0)
        metrics['processing_rate'] = processed / total_gen if total_gen > 0 else 0.0;
        metrics['total_revenue'] = self.episode_stats.get('total_revenue', 0.0)
        return {'episode_stats': self.episode_stats.copy(), 'reward_metrics': metrics,
                'waiting_time_stats': waiting_stats}

    # ===== V5.1 修复 =====
    def set_model_and_buffer(self, model, replay_buffer, device):
        """ (V5.1) 设置模型, Buffer 引用, 和设备 """
        self.model = model;
        self.replay_buffer = replay_buffer;
        self.device = device
        if model is not None:
            print(f"✓ 模型已设置到环境 (设备: {self.device})")
            if replay_buffer is not None:
                print("✓ Replay Buffer 已设置到环境")
            else:
                print("⚠ Replay Buffer 未设置 (评估模式?)")
        else:
            print("模型已从环境移除。")
    # =====================

