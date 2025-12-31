# utils/data_process.py
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import pickle
import os

# (Config 的导入依赖于 config.py 在根目录)
try:
    from config import Config
except ImportError:
    print("警告 (data_process.py): 无法从根目录导入 Config。")


    # 定义一个临时的 Config 类以防单独运行文件时出错
    class Config:
        NUM_GRIDS = 400
        INPUT_DIM = 5
        NUM_TIME_SLICES = 144
        TIME_WINDOW_MINUTES = 15
        TOTAL_VEHICLES = 1000
        GRID_SIZE = (20, 20)
        SEED = 42
        RAW_DATA_PATH = 'data/raw'
        ORDER_FILE = 'orders.csv'
        DATA_START_DATE = '2016-11-01 00:00:00'
        DATA_END_DATE = '2016-11-30 00:00:00'
        PROCESSED_DATA_PATH = 'data/processed'
        NEIGHBOR_ADJ_FILE = 'neighbor_adj.pt'
        POI_ADJ_FILE = 'poi_adj.pt'
        NUM_ACTIONS = 179


class VehicleSimulator:
    """车辆分布仿真器 (用于 prepare_mgcn_input 的默认值)"""

    def __init__(self, config):
        self.config = config

    def initialize_vehicles(self, order_density=None):
        # (这个函数在 V4 逻辑中几乎不再使用，但保留)
        return self._initialize_normal_distribution()

    def _initialize_normal_distribution(self):
        rows, cols = self.config.GRID_SIZE
        total_grids = self.config.NUM_GRIDS
        total_vehicles = int(self.config.TOTAL_VEHICLES)
        std_grid = getattr(self.config, 'INIT_STD_GRID', 60)
        std_row = max(0.5, std_grid / max(cols, 1))
        std_col = max(0.5, std_grid / max(rows, 1))
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0
        seed = getattr(self.config, 'SEED', None)
        rng = np.random.default_rng(seed)
        if total_vehicles <= 0: return np.zeros(total_grids, dtype=int)
        row_pos = rng.normal(loc=center_row, scale=std_row, size=total_vehicles)
        col_pos = rng.normal(loc=center_col, scale=std_col, size=total_vehicles)
        row_idx = np.clip(np.rint(row_pos), 0, rows - 1).astype(int)
        col_idx = np.clip(np.rint(col_pos), 0, cols - 1).astype(int)
        grid_ids = row_idx * cols + col_idx
        vehicle_grid_counts = np.bincount(grid_ids, minlength=total_grids).astype(int)
        # print("VehicleSimulator generated counts:", vehicle_grid_counts) # (减少打印)
        return vehicle_grid_counts


class FeatureExtractor:
    """特征提取器 - (代码不变)"""

    def __init__(self, config):
        self.config = config

    def process_orders(self, order_df):
        """处理订单数据 (不变)"""
        # print(f"处理订单数据，原始数据量: {len(order_df)}") # (减少打印)
        if 'departure_time' not in order_df.columns:
            raise ValueError("订单数据缺少 'departure_time' 列")

        # (假设 timestamp 已在 load_and_process_orders 中转换)
        if 'timestamp' not in order_df.columns:
            print("警告: process_orders 期望 'timestamp' 列已存在。")
            # (添加转换作为备用)
            if order_df['departure_time'].iloc[0] > 1e10:
                order_df['timestamp'] = pd.to_datetime(order_df['departure_time'], unit='ms', utc=True).dt.tz_convert(
                    'Asia/Shanghai')
            else:
                order_df['timestamp'] = pd.to_datetime(order_df['departure_time'], unit='s', utc=True).dt.tz_convert(
                    'Asia/Shanghai')

        invalid_grids = (order_df['grid_index'] < 0) | (order_df['grid_index'] >= self.config.NUM_GRIDS)
        if invalid_grids.sum() > 0:
            print(f"发现 {invalid_grids.sum()} 个无效网格索引，将被过滤")
            order_df = order_df[~invalid_grids].copy()

        # print(f"有效数据量: {len(order_df)}") # (减少打印)
        return order_df

    def extract_time_window_features(self, order_df, current_time):
        """提取时间窗口特征 (不变)"""
        # current_time 应该是一个带时区的 timestamp
        start_time = current_time - pd.Timedelta(minutes=self.config.TIME_WINDOW_MINUTES)

        # 确保 mask 比较在相容的时区下进行
        # (order_df['timestamp'] 已经是带时区的 'Asia/Shanghai')
        mask = (order_df['timestamp'] >= start_time) & (order_df['timestamp'] <= current_time)
        window_orders = order_df[mask]

        grid_counts = window_orders.groupby('grid_index').size()
        # 使用 .reindex 填充所有网格，比循环更快
        order_features_series = grid_counts.reindex(range(self.config.NUM_GRIDS), fill_value=0)
        order_features = order_features_series.values
        return order_features

    def get_historical_demand_density(self, order_df):
        """获取历史需求密度 (不变)"""
        grid_counts = order_df.groupby('grid_index').size()
        demand_density_series = grid_counts.reindex(range(self.config.NUM_GRIDS), fill_value=0)
        return demand_density_series.values


class DataProcessor:
    """主数据处理器 - (修改 load_and_process_orders 和 prepare_mgcn_input)"""

    def __init__(self, config):
        self.config = config
        self.vehicle_simulator = VehicleSimulator(config)
        self.feature_extractor = FeatureExtractor(config)

    def load_and_process_orders(self, file_path=None):
        """加载和处理订单数据 (修复时区问题)"""
        if file_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(project_root, self.config.RAW_DATA_PATH, self.config.ORDER_FILE)

        print(f"加载订单数据: {file_path}")
        dtype = {'order_id': str, 'departure_time': int, 'fee': float, 'grid_index': int, 'dest_grid_index': int}

        try:
            order_df = pd.read_csv(file_path, dtype=dtype)
        except FileNotFoundError:
            print(f"错误: 订单文件未找到: {file_path}")
            raise

        print(f"  原始订单数: {len(order_df):,}条")

        # ===== ★★★ 关键修复：时区转换 ★★★ =====
        print(f"  转换Unix时间戳 (UTC)...")
        # 1. 转换Unix时间戳 (默认解析为 UTC)
        if order_df['departure_time'].iloc[0] > 1e10:  # 毫秒
            # pd.to_datetime 默认创建的是 naive timestamp，我们需要先 localize 到 UTC
            order_df['timestamp_utc'] = pd.to_datetime(order_df['departure_time'], unit='ms').dt.tz_localize('UTC')
        else:  # 秒
            order_df['timestamp_utc'] = pd.to_datetime(order_df['departure_time'], unit='s').dt.tz_localize('UTC')

        # 2. 关键修复：将时间转换为上海时区 (UTC+8)
        print(f"  转换为 'Asia/Shanghai' (UTC+8) 时区...")
        try:
            order_df['timestamp'] = order_df['timestamp_utc'].dt.tz_convert('Asia/Shanghai')
        except Exception as e:
            print(f"错误: 转换为 'Asia/Shanghai' 时区失败: {e}")
            print("请确保已安装 'pytz' 库: pip install pytz")
            order_df['timestamp'] = order_df['timestamp_utc']  # Fallback

        print(f"  原始UTC时间范围: {order_df['timestamp_utc'].min()} 到 {order_df['timestamp_utc'].max()}")
        print(f"  转换后上海时间范围: {order_df['timestamp'].min()} 到 {order_df['timestamp'].max()}")
        # =======================================

        # 时间过滤
        if hasattr(self.config, 'DATA_START_DATE') and self.config.DATA_START_DATE:
            # 3. 确保过滤日期也是带时区的
            start_date = pd.Timestamp(self.config.DATA_START_DATE, tz='Asia/Shanghai')
            end_date = pd.Timestamp(self.config.DATA_END_DATE, tz='Asia/Shanghai')

            print(f"\n  应用时间过滤 (上海时间): [{start_date}] 到 [{end_date}]")
            before_count = len(order_df)
            order_df = order_df[
                (order_df['timestamp'] >= start_date) &
                (order_df['timestamp'] < end_date)
                ].copy()
            after_count = len(order_df)
            print(f"    过滤后: {after_count:,}条 (移除了 {before_count - after_count:,} 条)")
            if after_count == 0:
                raise ValueError("时间过滤后没有数据！请检查 config.py 中的 DATA_START_DATE 和 DATA_END_DATE。")

        order_df = order_df.reset_index(drop=True)

        # 4. 调用 process_orders (现在只做网格验证)
        processed_orders = self.feature_extractor.process_orders(order_df)
        print(f"  最终订单数: {len(processed_orders):,}条\n")
        return processed_orders

    # ===== ★★★ 关键修改：拆分车辆特征 ★★★ =====
    def prepare_mgcn_input(self, order_df, current_time,
                           idle_vehicle_dist=None,
                           busy_vehicle_dist=None):
        """
        准备MGCN输入数据 (V-Final-Embedding: INPUT_DIM=5)
        接收 idle_vehicle_dist 和 busy_vehicle_dist
        """
        # 1. 订单特征 (历史)
        order_features = self.feature_extractor.extract_time_window_features(order_df, current_time)

        # 2. 车辆特征 (来自环境)
        if idle_vehicle_dist is None or busy_vehicle_dist is None:
            # (这个分支理论上不应该在训练中被调用)
            print("警告: prepare_mgcn_input 未收到车辆分布! 使用模拟器生成默认分布。")
            total_vehicle_dist = self.vehicle_simulator.initialize_vehicles(None)
            idle_vehicle_dist = (total_vehicle_dist * 0.5).astype(int)
            busy_vehicle_dist = total_vehicle_dist - idle_vehicle_dist

        # 3. 验证 Config.py
        if self.config.INPUT_DIM != 5:
            raise ValueError("Config.INPUT_DIM 必须为 5 ([订单, 空闲车辆, 繁忙车辆, time_sin, time_cos])")

        # 构造节点特征矩阵 (400, 5)
        node_features = np.zeros((self.config.NUM_GRIDS, self.config.INPUT_DIM))

        # 4. 归一化特征 (特征 0, 1, 2)
        order_max = np.max(order_features) if np.max(order_features) > 0 else 1.0
        idle_max = np.max(idle_vehicle_dist) if np.max(idle_vehicle_dist) > 0 else 1.0
        busy_max = np.max(busy_vehicle_dist) if np.max(busy_vehicle_dist) > 0 else 1.0

        node_features[:, 0] = order_features / order_max
        node_features[:, 1] = idle_vehicle_dist / idle_max  # <-- 空闲车辆
        node_features[:, 2] = busy_vehicle_dist / busy_max  # <-- 繁忙车辆 (serving + dispatching)

        # 5. 计算 "一天中的时间" 特征 (特征 3, 4)
        time_slice = self._get_time_slice(current_time)
        time_angle = (2 * np.pi * time_slice) / self.config.NUM_TIME_SLICES

        node_features[:, 3] = np.sin(time_angle)  # time_sin
        node_features[:, 4] = np.cos(time_angle)  # time_cos

        return torch.FloatTensor(node_features)  # 返回 CPU Tensor

    # ===============================

    def _get_time_slice(self, timestamp):
        """获取时间片ID (0-143) (不变)"""
        # timestamp 现在是带时区的，.hour 和 .minute 会自动使用本地时区 (上海)
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        if timestamp.tzinfo is None:
            # print("警告: _get_time_slice 收到一个 naive timestamp") # 减少打印
            pass

        minutes_in_day = timestamp.hour * 60 + timestamp.minute
        time_slice = min(minutes_in_day // 10, self.config.NUM_TIME_SLICES - 1)
        return time_slice

    def identify_hotspots(self, order_df, top_k=None):
        """识别热点区域 (不变)"""
        if top_k is None:
            top_k = self.config.NUM_ACTIONS
        grid_counts = order_df.groupby('grid_index').size().sort_values(ascending=False)
        hotspot_grids = grid_counts.head(top_k).index.tolist()
        print(f"识别出 {len(hotspot_grids)} 个热点区域 (Top-K={top_k})")
        return hotspot_grids, grid_counts

    def save_processed_data(self, data, filename):
        """保存处理后的数据 (不变)"""
        os.makedirs(self.config.PROCESSED_DATA_PATH, exist_ok=True)
        filepath = os.path.join(self.config.PROCESSED_DATA_PATH, filename)
        try:
            if filename.endswith('.pkl'):
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            elif filename.endswith('.pt'):
                torch.save(data, filepath)
            else:
                raise ValueError(f"不支持的文件格式: {filename}")
            print(f"数据已保存到: {filepath}")
        except Exception as e:
            print(f"保存处理数据时出错: {e}")

    def load_processed_data(self, filename):
        """加载处理后的数据 (不变)"""
        filepath = os.path.join(self.config.PROCESSED_DATA_PATH, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        if filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filename.endswith('.pt'):
            return torch.load(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")


    def split_data_by_time(self, orders_df, train_ratio=0.7, val_ratio=0.15):
        """按时间划分数据 (V5.1 - 固定最后 7 天为测试集)"""
        orders_df = orders_df.sort_values('timestamp').reset_index(drop=True)
        min_date = orders_df['timestamp'].min().date()
        max_date = orders_df['timestamp'].max().date()
        total_days = (max_date - min_date).days + 1
        print(f"\n数据时间范围 (本地时间): {min_date} 到 {max_date} (共 {total_days} 天)")

        orders_df['date'] = orders_df['timestamp'].dt.date
        all_dates = sorted(orders_df['date'].unique())
        if len(all_dates) < total_days:
            print(f"警告: 数据中实际有订单的天数 ({len(all_dates)}) 少于总天数 ({total_days})")
            total_days = len(all_dates)  # 使用实际有订单的天数进行计算

        # ===== ★★★ 核心修改：固定测试集为最后 7 天 ★★★ =====
        test_days = 7
        if total_days < test_days + 2:  # 至少需要 1 天训练 + 1 天验证 + 7 天测试
            raise ValueError(f"数据集总天数 ({total_days}) 不足以支持固定的 7 天测试集以及至少 1 天训练和 1 天验证。")

        # 从剩余天数中计算验证集天数 (至少 1 天)
        remaining_days = total_days - test_days
        val_days = max(1, int(remaining_days * val_ratio))  # 基于剩余天数按比例计算验证天数

        # 训练集天数为剩下的天数
        train_days = remaining_days - val_days

        # 再次检查，确保训练天数也至少为 1
        if train_days < 1:
            # 如果训练天数不足，尝试减少验证天数（如果验证天数>1）
            if val_days > 1:
                val_days -= 1
                train_days += 1
            else:  # 如果验证天数也无法减少，则数据确实太少
                raise ValueError(f"数据集天数 ({total_days}) 在固定 7 天测试集后，不足以分配至少 1 天训练和 1 天验证。")
        # =======================================================

        print(f"划分方案: 训练 {train_days}天, 验证 {val_days}天, 测试 {test_days}天 (固定)")

        # (日期切片逻辑保持不变，但基于新的天数计算)
        train_dates = all_dates[:train_days]
        val_dates = all_dates[train_days: train_days + val_days]
        test_dates = all_dates[train_days + val_days:]  # 这将是最后 7 天

        train_orders = orders_df[orders_df['date'].isin(train_dates)].copy()
        val_orders = orders_df[orders_df['date'].isin(val_dates)].copy()
        test_orders = orders_df[orders_df['date'].isin(test_dates)].copy()

        for df in [train_orders, val_orders, test_orders]:
            if 'date' in df.columns:
                df.drop('date', axis=1, inplace=True)

        print(f"最终数据量: 训练集={len(train_orders):,}, 验证集={len(val_orders):,}, 测试集={len(test_orders):,}")

        # 验证测试集是否确实是最后 7 天的数据
        if len(test_dates) != test_days:
            print(f"警告: 实际测试集天数 ({len(test_dates)}) 与预期的 ({test_days}) 不符，请检查数据日期是否连续。")

        if test_orders.empty:
            raise ValueError("测试集为空，请检查数据量和划分逻辑。")
        if train_orders.empty:
            raise ValueError("训练集为空，请检查数据量和划分逻辑。")
        if val_orders.empty:
            raise ValueError("验证集为空，请检查数据量和划分逻辑。")

        return train_orders, val_orders, test_orders