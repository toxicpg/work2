import sys
import os

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.trainer import MGCNTrainer
from evaluate import _calculate_daily_metrics

# 导入绘图函数
try:
    from plot_training_curves import plot_training_curves, plot_waiting_time_curve, load_training_stats
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入绘图模块: {e}")
    PLOTTING_AVAILABLE = False

def run_validation(trainer, data_processor, val_orders, config):
    print("\n--- 开始验证 ---")
    val_env = RideHailingEnvironment(config, data_processor, val_orders)
    if hasattr(val_env, 'set_model_and_buffer'):
        val_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        val_env.model = trainer.main_net
        val_env.replay_buffer = None
        val_env.device = config.DEVICE

    all_daily_infos = {}
    num_val_days = val_env.order_generator.get_day_count()
    if num_val_days == 0:
        print("警告: 验证集中没有完整的天数可供评估。")
        return 0.0

    for day_index in range(num_val_days):
        # 每次重置环境以模拟新的一天
        val_env.reset()
        # ★ 强制从指定的day开始，避免随机
        val_env.episode_start_day = day_index
        val_env.current_day = day_index
        # 重置时间到指定天
        if hasattr(val_env.order_generator, 'time_range') and val_env.order_generator.time_range[0] != pd.Timestamp.min:
            base_time = val_env.order_generator.time_range[0].normalize()
        else:
            base_time = pd.Timestamp(config.DATA_START_DATE, tz='Asia/Shanghai').normalize()
        val_env.simulation_time = base_time + pd.Timedelta(days=day_index)
        val_env.current_time = val_env.simulation_time
        val_env.start_time = val_env.simulation_time
        val_env.current_time_slice = 0

        pbar = tqdm(range(config.TICKS_PER_DAY), desc=f"验证中 (Day {day_index+1}/{num_val_days})")

        daily_infos = []
        ticks = 0
        while ticks < config.TICKS_PER_DAY:
            # 在 step 内部，环境会使用设置好的模型进行决策
            _, _, _, info = val_env.step(current_epsilon=0.0) # 使用贪婪策略进行评估
            daily_infos.append(info.get('step_info', {}))
            ticks += 1
            pbar.update(1)
        pbar.close()
        all_daily_infos[day_index] = daily_infos

    daily_metrics = _calculate_daily_metrics(all_daily_infos, 0)
    if not daily_metrics:
        print("--- 验证完成: 未能计算任何指标 ---")
        return 0.0

    df = pd.DataFrame(daily_metrics)
    avg_completion_rate = df['completion_rate'].mean()
    print(f"--- 验证完成: 平均完成率 = {avg_completion_rate:.4f} ---")
    return avg_completion_rate

def main():
    """主训练循环 (V5.4)"""
    config = Config()
    if not config.validate_config():
        return

    # 1. 加载数据和图
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, _ = data_processor.split_data_by_time(all_orders, config.TRAIN_RATIO, config.VAL_RATIO)

    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()

    # 2. 初始化训练器和环境
    trainer = MGCNTrainer(config, neighbor_adj, poi_adj)
    # 初始化训练所需的额外属性
    trainer.best_validation_metric = 0.0
    trainer.early_stopping_counter = 0
    env = RideHailingEnvironment(config, data_processor, train_orders)

    # 3. 设置模型与经验池到环境（兼容无 set_model_and_buffer 的版本）
    if hasattr(env, 'set_model_and_buffer'):
        env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, config.DEVICE)
    else:
        env.model = trainer.main_net
        env.replay_buffer = trainer.replay_buffer
        env.device = config.DEVICE

    print("--- 开始训练 ---")
    for episode in range(1, config.NUM_EPISODES + 1):
        print(f"\n===== Episode {episode}/{config.NUM_EPISODES} =====")

        # 训练一个 episode (环境内部会使用 trainer 的模型和经验池)
        reward, loss = trainer.train_episode(env, episode)
        trainer.log_message(f"Episode {episode}: Total Reward={reward:.2f}, Avg Loss={loss:.4f}, Epsilon={trainer.epsilon:.4f}")

        # 验证、保存和早停
        if episode % config.VALIDATION_INTERVAL == 0:
            # 传递 data_processor 以确保验证环境使用正确的、已加载的数据
            val_completion_rate = run_validation(trainer, data_processor, val_orders, config)

            if val_completion_rate > trainer.best_validation_metric:
                print(f"发现新的最佳模型！完成率: {val_completion_rate:.4f} > {trainer.best_validation_metric:.4f}")
                trainer.best_validation_metric = val_completion_rate
                trainer.early_stopping_counter = 0
                trainer.save_checkpoint(episode)
            else:
                trainer.early_stopping_counter += 1
                print(f"验证性能未提升. 早停计数: {trainer.early_stopping_counter}/{config.EARLY_STOPPING_PATIENCE}")
                # 即使性能没有提升，也保存当前模型作为普通检查点
                trainer.save_checkpoint(episode)

            if trainer.early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                print("早停触发！连续多个验证周期性能未提升。")
                break
        else:
             # 非验证轮次，按频率保存普通模型
             if episode % config.SAVE_FREQ == 0:
                trainer.save_checkpoint(episode)

    print("--- 训练结束 ---")

    # 自动绘制训练曲线
    if PLOTTING_AVAILABLE:
        print("\n" + "="*70)
        print("📊 开始绘制训练曲线...")
        print("="*70)
        try:
            # 构建统计数据字典
            stats_dict = {
                'total_rewards': trainer.total_rewards,
                'losses': trainer.losses,
                'epsilon_history': trainer.epsilon_history,
                'completion_rates': trainer.completion_rates,
                'avg_waiting_times': trainer.avg_waiting_times,
                'match_rates': trainer.match_rates,
                'cancel_rates': trainer.cancel_rates
            }

            # 绘制曲线
            plot_training_curves(stats_dict)
            plot_waiting_time_curve(stats_dict)

            print("✓ 训练曲线已保存到 results/plots/")
            print("  - training_curves.png")
            print("  - waiting_time_curve.png")
        except Exception as e:
            print(f"⚠ 绘制曲线时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ 跳过绘图（绘图模块不可用）")
        print("  提示: 训练完成后可以手动运行: python plot_training_curves.py")

if __name__ == '__main__':
    main()
