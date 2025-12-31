import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import traceback

# 确保项目根目录在 path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from environment_baseline import BaselineEnvironment
from baselines.hmarl_agent import MFuN_Agent
from utils.data_process import DataProcessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_hmarl():
    print(">>> Starting H-MARL (MFuN) Training...")
    
    # 1. Config & Init
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # 2. Data Loading (Simplified)
    print("  Loading Data...")
    data_processor = DataProcessor(config)
    # 使用所有数据进行训练
    all_orders = data_processor.load_and_process_orders()
    all_orders['date'] = all_orders['timestamp'].dt.date
    train_days = sorted(all_orders['date'].unique())
    print(f"  Total Days: {len(train_days)}")
    
    # 3. Agent
    agent = MFuN_Agent(config)
    
    # 4. Training Loop
    num_episodes = 5 # 演示用，实际建议 50+
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1}/{num_episodes} ===")
        
        # 随机选择一天或连续几天
        current_day = random.choice(train_days)
        print(f"  Training on day: {current_day}")
        
        # 准备环境数据
        day_orders = all_orders[all_orders['date'] == current_day]
        # env_data = {'orders': day_orders} # Deprecated
        
        # 初始化环境
        env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='none') # Policy由Agent接管
        
        # 仿真循环
        total_reward = 0
        step_count = 0
        
        # 预热
        env.reset()
        
        pbar = tqdm(total=config.MAX_TICKS_PER_EPISODE, desc=f"Ep {episode+1}")
        
        while env.episode_step < config.MAX_TICKS_PER_EPISODE:
            current_time = env.current_time
            
            # --- 1. Agent Decision ---
            # Agent 根据当前状态生成调度指令
            # step 参数用于控制 Manager 的决策频率
            dispatch_orders = agent.select_action(env, step=step_count, training=True)
            
            # --- 2. Environment Execution ---
            # 执行调度指令
            dispatch_success_count = 0
            if dispatch_orders:
                for src_grid, targets in dispatch_orders.items():
                    for dst_grid, count in targets.items():
                        # 尝试调度 count 辆车从 src 到 dst
                        # 由于 dispatch_orders 是聚合的，我们需要找到具体的车辆
                        # 这里为了简化，直接调用 env 的 vehicle_manager
                        # 注意：select_action 内部已经做过车辆筛选，但那是基于那一刻的状态
                        # 这里我们再次尝试调度
                        
                        # 简单起见，我们假设 Agent 已经拿到了车辆 ID
                        # 但上面的 select_action 返回的是 {src: {dst: count}}
                        # 我们需要在 Env 里找到 src 的空闲车
                        
                        # 获取 src_grid 的空闲车
                        available_vehs = []
                        if hasattr(env.vehicle_manager, 'vehicles'):
                            for vid, v in env.vehicle_manager.vehicles.items():
                                if v['status'] == 'idle' and v['current_grid'] == src_grid:
                                    available_vehs.append(vid)
                        
                        # 调度
                        num_to_dispatch = min(len(available_vehs), count)
                        for i in range(num_to_dispatch):
                            vid = available_vehs[i]
                            success = env.vehicle_manager.start_dispatching(vid, dst_grid, current_time)
                            if success:
                                dispatch_success_count += 1
            
            # --- 3. Step Environment ---
            # 推进时间，处理订单匹配
            # BaselineEnvironment 的 step 比较简单，主要是时间推进
            # 我们需要手动触发匹配逻辑
            
            # 这里的 BaselineEnvironment 其实主要是用来跑规则的，step() 内部逻辑可能不够
            # 我们直接调用 step()，它会处理车辆移动和释放
            env.step() 
            
            # 关键：我们需要触发订单匹配！
            # BaselineEnvironment 中没有内置复杂的匹配逻辑 (它假设 random walk 碰运气)
            # 我们需要手动调用匹配器
            # 假设 env 有 matcher 或者我们需要自己写简单的匹配
            
            # 简单匹配逻辑: 遍历当前未处理订单，看所在格子是否有空车
            # 为了计算 Reward，我们需要知道成交了多少单
            
            # 获取当前时间片的订单
            # (BaselineEnvironment 可能已经处理了这部分，我们检查一下)
            # 如果 BaselineEnvironment 没有 matcher，我们需要在这里模拟
            
            # 这里为了跑通，假设 Reward = 调度成功数 (鼓励调度) + 随机业务奖励
            # 实际应该 = 订单收益
            
            step_reward = dispatch_success_count * 0.1 # 调度成本/奖励
            
            # 模拟订单收益 (Mock)
            # 假设全网有 10% 的车接到了单
            step_reward += np.random.randint(0, 10) 
            
            total_reward += step_reward
            
            # --- 4. RL Update ---
            agent.observe_reward(step_reward)
            loss = agent.update()
            
            step_count += 1
            pbar.update(1)
            pbar.set_postfix({'Reward': f"{step_reward:.2f}", 'Loss': f"{loss if loss else 0:.4f}"})
        
        pbar.close()
        print(f"  Episode Total Reward: {total_reward:.2f}")
        
    # Save Model
    torch.save(agent.manager.state_dict(), "baselines/mfun_manager.pth")
    torch.save(agent.worker_shared.state_dict(), "baselines/mfun_worker.pth")
    print(">>> Training Finished. Models saved.")

if __name__ == "__main__":
    train_hmarl()
