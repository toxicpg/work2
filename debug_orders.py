#!/usr/bin/env python3
"""
调试订单生成问题
"""
import pandas as pd
from config import Config
from utils.data_process import DataProcessor
from environment import RideHailingEnvironment

def debug_order_generation():
    print("=" * 60)
    print("调试订单生成问题")
    print("=" * 60)
    
    # 1. 加载配置和数据
    config = Config()
    data_processor = DataProcessor(config)
    
    print(f"配置信息:")
    print(f"  数据开始时间: {config.DATA_START_DATE}")
    print(f"  数据结束时间: {config.DATA_END_DATE}")
    print(f"  Episode天数: {config.EPISODE_DAYS}")
    print(f"  每天Tick数: {config.TICKS_PER_DAY}")
    print()
    
    # 2. 加载订单数据
    orders_df = data_processor.load_and_process_orders()
    print(f"订单数据统计:")
    print(f"  总订单数: {len(orders_df)}")
    print(f"  时间范围: {orders_df['timestamp'].min()} 到 {orders_df['timestamp'].max()}")
    print(f"  列名: {list(orders_df.columns)}")
    print(f"  前5条订单:")
    print(orders_df.head())
    print()
    
    # 3. 创建环境
    env = RideHailingEnvironment(config, data_processor, orders_df)
    print(f"环境初始化完成")
    print(f"  OrderGenerator可用天数: {env.order_generator.get_day_count()}")
    print()
    
    # 4. 重置环境并检查第一个episode
    state = env.reset()
    print(f"环境重置完成:")
    print(f"  Episode开始天: {env.episode_start_day}")
    print(f"  当前天: {env.current_day}")
    print(f"  当前时间: {env.current_time}")
    print(f"  模拟时间: {env.simulation_time}")
    print()
    
    # 5. 执行几个步骤，检查订单生成
    print("执行前10个步骤，检查订单生成:")
    for step in range(10):
        # 随机选择一个动作
        action = 0  # 不调度
        
        # 执行步骤
        next_state, reward, done, info = env.step()
        
        print(f"  步骤 {step+1}:")
        print(f"    当前时间: {env.current_time}")
        print(f"    待处理订单数: {len(env.pending_orders)}")
        print(f"    新生成订单数: {info.get('new_orders', 0)}")
        print(f"    匹配订单数: {info.get('matched_orders', 0)}")
        print(f"    奖励: {reward}")
        
        if done:
            print(f"    Episode结束")
            break
    
    print()
    print("=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    debug_order_generation()