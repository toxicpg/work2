
import unittest
import pandas as pd
import numpy as np
from collections import deque
from config import Config
from environment_baseline import BaselineEnvironment, VehicleManager, OrderMatcher

class TestBaselineMatchingFix(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.MAX_WAITING_TIME = 300  # 300秒
        self.config.AVG_SPEED_KMH = 40      # 40 km/h -> 1.5 min/grid
        self.config.GRID_SIZE = (20, 20)
        self.config.NUM_GRIDS = 400

        # Mock DataProcessor and Orders
        self.mock_orders_df = pd.DataFrame({
            'order_id': ['test_1'],
            'timestamp': [pd.Timestamp('2016-11-01 00:00:00', tz='Asia/Shanghai')],
            'departure_time': [1477929600],
            'grid_index': [0],
            'dest_grid_index': [1],
            'fee': [10.0],
            'relative_day': [0],
            'time_slice': [0]
        })

        # Initialize Environment
        # We pass a dummy data_processor as None since we won't use it for this test
        self.env = BaselineEnvironment(self.config, None, self.mock_orders_df, dispatch_policy='none')
        self.env.reset()

    def test_timeout_prediction_in_matching(self):
        """测试场景 1：预判超时"""
        print("\n=== 测试场景 1：预判超时 ===")

        # 1. 设置当前时间
        base_time = pd.Timestamp('2016-11-01 00:00:00', tz='Asia/Shanghai')
        self.env.current_time = base_time + pd.Timedelta(seconds=250) # 已经过了250秒

        # 2. 创建一个订单，生成时间为 base_time (T=0)
        order = {
            'order_id': 'order_timeout_test',
            'timestamp': base_time,
            'grid_index': 0,  # 在 Grid 0
            'dest_grid_index': 1,
            'status': 'pending'
        }

        # 3. 创建一个车辆，在 Grid 10
        # 假设 Grid 0 和 Grid 10 的距离导致接驾时间 > 50秒
        # Grid 0 = (0, 0), Grid 10 = (0, 10) -> 距离 10
        # 速度 40km/h -> 1.5 min/grid -> 10 * 1.5 = 15分钟 = 900秒
        # 肯定超时

        # 为了精确控制，我们找一个刚好超时的
        # 剩余时间预算 = 300 - 250 = 50秒 = 0.83分钟
        # 1 grid = 1.5分钟 = 90秒 > 50秒
        # 所以只要距离 >= 1 grid，就应该超时

        vehicle_id = 0
        self.env.vehicle_manager.vehicles[vehicle_id] = {
            'id': vehicle_id,
            'current_grid': 1, # 距离 Grid 0 为 1 (0,1) -> (0,0) dist=1
            'status': 'idle',
            'idle_since': self.env.current_time
        }

        # 4. 尝试匹配
        pending_orders = [order]
        matches, unmatched = self.env.order_matcher.match_orders(
            pending_orders, self.env.vehicle_manager, self.env.current_time
        )

        # 5. 验证
        # 预期：不匹配，因为 250 + 90 = 340 > 300
        print(f"  当前等待: 250s")
        print(f"  接驾距离: 1 grid -> 预计接驾时间: {1.5 * 60}s = 90s")
        print(f"  预计总时间: 340s > 300s")
        print(f"  匹配结果数量: {len(matches)}")

        self.assertEqual(len(matches), 0, "应该因为预判超时而不匹配")
        self.assertEqual(len(unmatched), 1, "订单应该保持未匹配")

    def test_position_update_on_cancel(self):
        """测试场景 2：取消后位置更新"""
        print("\n=== 测试场景 2：取消后位置更新 ===")

        # 1. 设置时间 T=0
        base_time = pd.Timestamp('2016-11-01 00:00:00', tz='Asia/Shanghai')
        self.env.current_time = base_time

        # 2. 创建订单和车辆
        order = {
            'order_id': 'order_cancel_test',
            'timestamp': base_time,
            'grid_index': 100,  # 接驾点在 Grid 100
            'dest_grid_index': 101,
            'status': 'pending'
        }

        vehicle_id = 0
        start_grid = 0 # 车辆初始在 Grid 0
        self.env.vehicle_manager.vehicles[vehicle_id] = {
            'id': vehicle_id,
            'current_grid': start_grid,
            'status': 'idle',
            'idle_since': base_time
        }

        # 3. 强制分配订单 (跳过 match_orders 的预判，直接测试 update_serving_vehicles)
        # 计算接驾时间
        travel_time = self.env.vehicle_manager._calculate_travel_time(start_grid, 100)
        self.env.vehicle_manager.assign_order(vehicle_id, order, base_time, travel_time)

        vehicle = self.env.vehicle_manager.vehicles[vehicle_id]
        print(f"  分配后状态: {vehicle['status']}")
        print(f"  分配后位置: {vehicle['current_grid']} (应为 {start_grid})")
        self.assertEqual(vehicle['status'], 'picking_up')
        self.assertEqual(vehicle['current_grid'], start_grid)

        # 4. 推进时间到超时 (T=301)
        self.env.current_time = base_time + pd.Timedelta(seconds=301)

        # 5. 更新服务状态
        completed, cancelled = self.env.vehicle_manager.update_serving_vehicles(self.env.current_time)

        # 6. 验证
        vehicle = self.env.vehicle_manager.vehicles[vehicle_id]
        print(f"  超时后状态: {vehicle['status']}")
        print(f"  超时后位置: {vehicle['current_grid']} (预期为 100)")

        self.assertEqual(len(cancelled), 1, "订单应该被取消")
        self.assertEqual(vehicle['status'], 'idle', "车辆应该变回 idle")
        self.assertEqual(vehicle['current_grid'], 100, "车辆位置应该更新为接驾点 (Grid 100)")

if __name__ == '__main__':
    unittest.main()

