#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
from environment import RideHailingEnvironment
from utils.data_process import DataProcessor
from config import Config

def test_completion_rate():
    # 使用配置类
    config = Config()

    # 创建数据处理器和环境
    data_processor = DataProcessor(config)
    orders_df = data_processor.load_orders()
    env = RideHailingEnvironment(config, data_processor, orders_df)

    # 重置环境
    env.reset()

    # 运行几个步骤
    print('运行测试步骤...')
    for i in range(10):
        state, reward, done, info = env.step()
        metrics = env.reward_calculator.get_metrics()
        matched = info.get('matched_orders', 0)
        completed = info.get('completed_orders', 0)
        completion_rate = metrics['completion_rate']
        
        print(f'Step {i+1}: 匹配={matched}, 完成={completed}, 完成率={completion_rate:.3f}')
        
        if done:
            break

    print('测试完成')
    
    # 打印最终统计
    final_metrics = env.reward_calculator.get_metrics()
    print(f'\n最终统计:')
    print(f'总完成订单: {final_metrics["completed_orders"]}')
    print(f'总取消订单: {final_metrics["cancelled_orders"]}')
    print(f'完成率: {final_metrics["completion_rate"]:.3f}')

if __name__ == '__main__':
    test_completion_rate()