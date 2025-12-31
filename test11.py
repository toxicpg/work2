# check_time_format.py
"""检查订单数据的时间格式"""

import pandas as pd
import os

# 读取订单文件
project_root = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(project_root, 'data', 'raw', 'orders.csv')  # 修改为你的实际路径

print(f"读取文件: {file_path}")

dtype = {
    'order_id': str,
    'departure_time': int,
    'fee': float,
    'grid_index': int,
    'dest_grid_index': int
}

df = pd.read_csv(file_path, dtype=dtype, nrows=10)  # 只读前10行

print(f"\n前10行数据:")
print(df)

print(f"\ndeparture_time 列:")
print(df['departure_time'].head())

# 尝试转换
print(f"\n转换为datetime:")
df['timestamp'] = pd.to_datetime(df['departure_time'], unit='s')
print(df[['departure_time', 'timestamp']].head())

print(f"\n时间范围:")
print(f"  最小: {df['timestamp'].min()}")
print(f"  最大: {df['timestamp'].max()}")