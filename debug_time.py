import pandas as pd
import datetime

# 加载数据
df = pd.read_csv('data/raw/orders.csv')
print(f'原始订单数: {len(df)}')

# 转换时间戳
df['timestamp_utc'] = pd.to_datetime(df['departure_time'], unit='s').dt.tz_localize('UTC')
df['timestamp'] = df['timestamp_utc'].dt.tz_convert('Asia/Shanghai')

print(f'UTC时间范围: {df["timestamp_utc"].min()} 到 {df["timestamp_utc"].max()}')
print(f'上海时间范围: {df["timestamp"].min()} 到 {df["timestamp"].max()}')

# 应用时间过滤
start_date = pd.Timestamp('2016-11-01 00:00:00', tz='Asia/Shanghai')
end_date = pd.Timestamp('2016-11-30 23:59:59', tz='Asia/Shanghai')

print(f'过滤范围: {start_date} 到 {end_date}')

filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]
print(f'过滤后订单数: {len(filtered_df)}')

# 检查最后一天的数据
last_day_data = df[df['timestamp'].dt.date == pd.Timestamp('2016-11-29').date()]
print(f'2016-11-29的订单数: {len(last_day_data)}')

# 检查是否有2016-11-30的数据
nov30_data = df[df['timestamp'].dt.date == pd.Timestamp('2016-11-30').date()]
print(f'2016-11-30的订单数: {len(nov30_data)}')