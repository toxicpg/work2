import os
import random
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
    """
    H-MARL (MFuN) 训练流程
    按照 Si et al. 论文实现：
    1. Manager 每5分钟决策一次（输出子目标）
    2. Worker 每30秒执行一次（具体调度）
    3. 内部奖励 + 外部奖励联合训练
    """
    print(">>> Starting H-MARL (MFuN) Training...")
    print("=" * 70)
    print("架构说明:")
    print("  - Manager: 全局协调，每5分钟输出子目标")
    print("  - Worker: 局部执行，每30秒调度车辆（参数共享）")
    print("  - 奖励机制: 外部奖励（订单收益）+ 内部奖励（完成子目标）")
    print("=" * 70)

    # 1. Config & Init
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[配置]")
    print(f"  Device: {device}")
    print(f"  Grid Size: {config.GRID_SIZE}")
    print(f"  Total Vehicles: {config.TOTAL_VEHICLES}")
    print(f"  Tick Duration: {config.TICK_DURATION_SEC}秒")

    # 2. Data Loading
    print(f"\n[数据加载]")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()

    # 按时间划分数据集：训练/验证/测试
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )

    # 获取训练天数
    train_orders['date'] = train_orders['timestamp'].dt.date
    train_days = sorted(train_orders['date'].unique())
    print(f"  训练天数: {len(train_days)}")
    print(f"  训练日期范围: {train_days[0]} ~ {train_days[-1]}")

    # 获取验证天数
    val_orders['date'] = val_orders['timestamp'].dt.date
    val_days = sorted(val_orders['date'].unique())
    print(f"  验证天数: {len(val_days)}")
    print(f"  验证日期范围: {val_days[0]} ~ {val_days[-1]}")

    # 3. Agent
    print(f"\n[智能体初始化]")
    agent = MFuN_Agent(config)
    print(f"  Manager 参数量: {sum(p.numel() for p in agent.manager.parameters()):,}")
    print(f"  Worker 参数量: {sum(p.numel() for p in agent.worker_shared.parameters()):,}")

    # 4. Training Loop
    num_episodes = 10  # 论文建议50+，这里演示用10

    # 早停参数
    best_reward = float('-inf')
    early_stopping_counter = 0
    early_stopping_patience = config.EARLY_STOPPING_PATIENCE

    # Manager 决策频率（论文：每5分钟）
    # 假设 TICK_DURATION_SEC = 60，则 5分钟 = 5 ticks
    manager_decision_interval = max(1, int(5 * 60 / config.TICK_DURATION_SEC))
    print(f"  Manager 决策间隔: {manager_decision_interval} ticks ({manager_decision_interval * config.TICK_DURATION_SEC / 60:.1f}分钟)")

    print(f"\n{'='*70}")
    print(f"开始训练 (共 {num_episodes} episodes)")
    print(f"{'='*70}\n")

    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode+1}/{num_episodes}")
        print(f"{'='*70}")

        # 随机选择训练天
        current_day = random.choice(train_days)
        print(f"  训练日期: {current_day}")

        # 准备环境数据
        day_orders = train_orders[train_orders['date'] == current_day]
        print(f"  订单数量: {len(day_orders)}")

        if len(day_orders) == 0:
            print(f"  ⚠️ 警告：该天没有订单，跳过")
            continue

        # 初始化环境（使用 'none' 策略，由 Agent 完全接管）
        env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='none')
        # 触发冷启动机制（start_day=0即可，因为day_orders只有一天数据）
        env.reset(start_day=0)

        # 打印调试信息
        print(f"  环境初始化完成:")
        print(f"    - 订单生成器天数: {env.order_generator.total_days}")
        print(f"    - 当前day: {env.current_day}, time_slice: {env.current_time_slice}")
        print(f"    - 订单时间范围: {env.order_generator.time_range[0]} ~ {env.order_generator.time_range[1]}")

        # 重置 Agent 状态
        agent.reset_episode()

        # 仿真循环
        total_reward = 0
        external_reward_sum = 0
        intrinsic_reward_sum = 0
        step_count = 0
        total_dispatched = 0

        pbar = tqdm(total=config.MAX_TICKS_PER_EPISODE, desc=f"Ep {episode+1}")
        
        try:
            # 每10个tick才调度一次，大幅加速
            dispatch_interval = 10

            while env.episode_step < config.MAX_TICKS_PER_EPISODE:
                current_time = env.current_time

                # --- 1. Agent Decision（每10 ticks一次）---
                if step_count % dispatch_interval == 0:
                    dispatch_orders = agent.select_action(env, step=step_count, training=True)

                    # --- 2. Environment Execution ---
                    dispatch_success_count = 0
                    if dispatch_orders:
                        for src_grid, targets in dispatch_orders.items():
                            for dst_grid, count in targets.items():
                                # 获取 src_grid 的空闲车辆
                                available_vehs = []
                                if hasattr(env.vehicle_manager, 'vehicles'):
                                    for vid, v in env.vehicle_manager.vehicles.items():
                                        if v['status'] == 'idle' and v['current_grid'] == src_grid:
                                            available_vehs.append(vid)

                                # 调度车辆
                                num_to_dispatch = min(len(available_vehs), count)
                                for i in range(num_to_dispatch):
                                    vid = available_vehs[i]
                                    success = env.vehicle_manager.start_dispatching(vid, dst_grid, current_time)
                                    if success:
                                        dispatch_success_count += 1

                    total_dispatched += dispatch_success_count
                else:
                    dispatch_orders = {}
                    dispatch_success_count = 0

                # --- 3. Step Environment ---
                _, _, _, info = env.step()
                step_info = info.get('step_info', {})

                # --- 4. Reward Calculation ---
                # 外部奖励：基于环境的实际业务指标
                tick_completed = step_info.get('completed_orders', 0)
                tick_cancelled = step_info.get('cancelled_orders', 0)
                tick_matched = step_info.get('matched_orders', 0)

                # 外部奖励计算
                external_reward = (
                    tick_completed * 10
                    - tick_cancelled * 5
                    + tick_matched * 1
                )

                # 内部奖励（只在调度时计算）
                if dispatch_orders:
                    intrinsic_reward = agent.compute_intrinsic_reward(env, dispatch_orders)
                else:
                    intrinsic_reward = 0

                step_reward = external_reward + intrinsic_reward
                total_reward += step_reward
                external_reward_sum += external_reward
                intrinsic_reward_sum += intrinsic_reward

                # --- 5. RL Update（每30个tick更新一次）---
                if step_count % 30 == 0:
                    agent.observe_reward(step_reward)
                    loss = agent.update()
                else:
                    loss = None

                step_count += 1

                # 每50 ticks更新一次进度条
                if step_count % 50 == 0:
                    pbar.update(50)
                    pbar.set_postfix({
                        'Step': step_count,
                        'New': step_info.get('new_orders', 0),
                        'Matched': tick_matched,
                        'Pending': len(env.pending_orders),
                        'Dispatch': total_dispatched
                    })

                # 前10个tick打印详细信息
                if step_count <= 10:
                    print(f"\n  Tick {step_count}: new_orders={step_info.get('new_orders', 0)}, "
                          f"matched={tick_matched}, cancelled={tick_cancelled}, "
                          f"pending={len(env.pending_orders)}, "
                          f"current_day={env.current_day}, time_slice={env.current_time_slice}")

        except Exception as e:
            print(f"\n❌ Episode {episode+1} 执行出错: {e}")
            traceback.print_exc()
            continue
        finally:
            pbar.close()

        # Episode 统计
        metrics = env.reward_calculator.get_metrics(total_orders_generated=len(day_orders))
        print(f"\n{'='*70}")
        print(f"Episode {episode+1} 结果:")
        print(f"{'='*70}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  外部奖励: {external_reward_sum:.2f}")
        print(f"  内部奖励: {intrinsic_reward_sum:.2f}")
        print(f"  调度次数: {total_dispatched}")
        print(f"  匹配率: {metrics['match_rate']:.2%}")
        print(f"  完成率: {metrics.get('completion_rate', 0):.2%}")
        print(f"  平均等待时间: {metrics.get('avg_wait_time', 0):.1f}秒")
        print(f"{'='*70}")

        # 早停检查
        if total_reward > best_reward:
            best_reward = total_reward
            early_stopping_counter = 0
            print(f"  ✓ 新的最佳奖励: {best_reward:.2f}")

            # 保存最佳模型
            os.makedirs("baselines/checkpoints", exist_ok=True)
            torch.save(agent.manager.state_dict(), "baselines/checkpoints/mfun_manager_best.pth")
            torch.save(agent.worker_shared.state_dict(), "baselines/checkpoints/mfun_worker_best.pth")
        else:
            early_stopping_counter += 1
            print(f"  早停计数: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            print(f"\n早停触发！连续 {early_stopping_patience} 个episode奖励未提升。")
            print(f"最佳奖励: {best_reward:.2f}")
            break

    # Save Final Model
    os.makedirs("baselines", exist_ok=True)
    torch.save(agent.manager.state_dict(), "baselines/mfun_manager.pth")
    torch.save(agent.worker_shared.state_dict(), "baselines/mfun_worker.pth")
    print(f"\n{'='*70}")
    print(">>> Training Finished!")
    print(f"  模型已保存至: baselines/mfun_manager.pth, baselines/mfun_worker.pth")
    print(f"  最佳模型: baselines/checkpoints/mfun_manager_best.pth")
    print(f"{'='*70}")

if __name__ == "__main__":
    train_hmarl()
