"""
快速验证脚本 - 用极小配置测试整个训练流程是否能跑通
目标: 1-2分钟内完成，验证没有致命错误

使用策略:
1. 只使用1天的数据
2. 只训练2个Episode
3. 极小的batch size和buffer size
4. 频繁保存以测试保存逻辑
"""

import os
import sys

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.trainer import MGCNTrainer

# 导入绘图函数
try:
    from plot_training_curves import plot_training_curves, plot_waiting_time_curve
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入绘图模块: {e}")
    PLOTTING_AVAILABLE = False


class QuickTestConfig(Config):
    """快速测试配置 - 覆盖默认配置"""

    def __init__(self):
        super().__init__()

        # ========== 极小化数据配置 ==========
        self.EPISODE_DAYS = 1  # 只用1天数据
        self.MAX_TICKS_PER_EPISODE = self.TICKS_PER_DAY * self.EPISODE_DAYS  # 2880 ticks

        # ========== 极小化训练配置 ==========
        self.NUM_EPISODES = 2  # 只训练2个episode
        self.BATCH_SIZE = 16  # 极小batch
        self.REPLAY_BUFFER_SIZE = 1000  # 极小buffer
        self.MIN_REPLAY_SIZE = 100  # 更快开始训练

        # ========== 更频繁的训练和保存 ==========
        self.TRAIN_EVERY_N_TICKS = 50  # 每50个tick训练一次(原来可能是100+)
        self.TARGET_UPDATE_FREQ = 50  # 更频繁更新target网络
        self.VALIDATION_INTERVAL = 1  # 每个episode都验证
        self.SAVE_FREQ = 999  # 不频繁保存，加快速度（只在验证时保存）

        # ========== 进度显示 ==========
        self.SHOW_PROGRESS_EVERY_N_TICKS = 100  # 每100个tick显示一次进度

        # ========== 禁用早停(测试用) ==========
        self.EARLY_STOPPING_PATIENCE = 999  # 禁用早停

        # ========== 加快epsilon衰减 ==========
        self.EPSILON_START = 0.5  # 降低初始探索
        self.EPSILON_END = 0.1
        self.EPSILON_DECAY = 0.9  # 更快衰减

        # ========== 日志配置 ==========
        self.VERBOSE = True  # 显示详细日志
        self.LOG_SAVE_PATH = 'logs/quick_test/'

        # ========== 保存路径 ==========
        self.CHECKPOINT_PATH = 'checkpoints/quick_test/'

        print("\n" + "="*70)
        print("⚡ 快速测试配置已加载")
        print("="*70)
        print(f"  Episode天数: {self.EPISODE_DAYS}天")
        print(f"  总Episode数: {self.NUM_EPISODES}")
        print(f"  每Episode Ticks: {self.MAX_TICKS_PER_EPISODE}")
        print(f"  预计运行时间: 1-3分钟")
        print(f"  Batch Size: {self.BATCH_SIZE}")
        print(f"  Buffer Size: {self.REPLAY_BUFFER_SIZE}")
        print(f"  日志保存路径: {self.LOG_SAVE_PATH}")
        print(f"  模型保存路径: {self.CHECKPOINT_PATH}")
        print("="*70 + "\n")


def run_quick_validation(trainer, data_processor, val_orders, config):
    """快速验证 - 只跑1天，只跑500个tick"""
    print("\n--- 快速验证 (只跑500 ticks) ---")

    # 检查验证数据
    if len(val_orders) == 0:
        print("⚠ 警告: 验证集为空！")
        return 0.0

    val_env = RideHailingEnvironment(config, data_processor, val_orders)
    if hasattr(val_env, 'set_model_and_buffer'):
        val_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        val_env.model = trainer.main_net
        val_env.replay_buffer = None
        val_env.device = config.DEVICE

    # ★ 强制从第0天开始，避免随机到没有数据的时间段
    val_env.reset()
    val_env.episode_start_day = 0  # 强制从验证集的第一天开始
    val_env.current_day = 0
    # 重置时间到第一天
    if hasattr(val_env.order_generator, 'time_range') and val_env.order_generator.time_range[0] != pd.Timestamp.min:
        base_time = val_env.order_generator.time_range[0].normalize()
    else:
        base_time = pd.Timestamp(config.DATA_START_DATE, tz='Asia/Shanghai').normalize()
    val_env.simulation_time = base_time
    val_env.current_time = base_time
    val_env.start_time = base_time
    val_env.current_time_slice = 0

    print(f"  ✓ 验证开始时间: {val_env.current_time}")
    print(f"  ✓ 验证集数据范围: {val_env.order_generator.time_range[0]} 到 {val_env.order_generator.time_range[1]}")
    print(f"  ✓ 验证集总天数: {val_env.order_generator.get_day_count()}天")

    # 只跑500个tick（或者完整一天如果配置的天数少）
    max_val_ticks = min(500, config.MAX_TICKS_PER_EPISODE)
    daily_infos = []

    for tick in tqdm(range(max_val_ticks), desc="快速验证"):
        _, _, _, info = val_env.step(current_epsilon=0.0)
        step_info = info.get('step_info', {})
        daily_infos.append(step_info)

        # 每100个tick输出一次进度（调试用）
        if tick > 0 and tick % 100 == 0:
            temp_total = sum(s.get('orders_generated', 0) for s in daily_infos)
            temp_matched = sum(s.get('orders_matched', 0) for s in daily_infos)
            print(f"  Tick {tick}: 已生成 {temp_total} 订单, 已匹配 {temp_matched} 订单")

    # 计算简单指标
    total_orders = sum(info.get('orders_generated', 0) for info in daily_infos)
    matched_orders = sum(info.get('orders_matched', 0) for info in daily_infos)
    match_rate = matched_orders / total_orders if total_orders > 0 else 0.0

    print(f"--- 快速验证完成: 匹配率 = {match_rate:.4f} ({matched_orders}/{total_orders}) ---")

    # 如果匹配率为0，输出更多调试信息
    if match_rate == 0.0 and total_orders == 0:
        print("⚠ 警告: 验证期间没有生成订单！")
        print(f"  验证集订单数: {len(val_orders)}")
        print(f"  验证tick数: {max_val_ticks}")
        print(f"  建议: 增加验证tick数或检查验证数据时间范围")

    return match_rate


def main():
    """快速测试主函数"""
    print("\n" + "🚀"*35)
    print("开始快速测试 - 验证训练流程是否可以正常运行")
    print("🚀"*35 + "\n")

    # 1. 加载快速测试配置
    config = QuickTestConfig()

    # 验证配置
    if hasattr(config, 'validate_config'):
        if not config.validate_config():
            print("❌ 配置验证失败！")
            return

    try:
        # 2. 加载数据和图
        print("\n[步骤 1/6] 加载数据...")
        data_processor = DataProcessor(config)
        all_orders = data_processor.load_and_process_orders()
        print(f"  ✓ 加载了 {len(all_orders)} 条订单")

        # 只使用很少的数据进行训练
        print(f"\n[步骤 2/6] 划分数据集...")
        train_orders, val_orders, test_orders = data_processor.split_data_by_time(
            all_orders, config.TRAIN_RATIO, config.VAL_RATIO
        )
        print(f"  ✓ 训练集: {len(train_orders)} 条")
        print(f"  ✓ 验证集: {len(val_orders)} 条")
        print(f"  ✓ 测试集: {len(test_orders)} 条")

        # 3. 加载图
        print(f"\n[步骤 3/6] 构建图结构...")
        gb = GraphBuilder(config)
        neighbor_adj, poi_adj = gb.load_graphs_pt()
        print(f"  ✓ 图已加载")

        # 4. 初始化训练器
        print(f"\n[步骤 4/6] 初始化模型...")
        trainer = MGCNTrainer(config, neighbor_adj, poi_adj)
        trainer.best_validation_metric = 0.0  # 初始化最佳验证指标
        print(f"  ✓ 模型已初始化")
        print(f"  ✓ 模型参数量: {sum(p.numel() for p in trainer.main_net.parameters()):,}")

        # 5. 初始化环境
        print(f"\n[步骤 5/6] 初始化训练环境...")
        env = RideHailingEnvironment(config, data_processor, train_orders)

        if hasattr(env, 'set_model_and_buffer'):
            env.set_model_and_buffer(trainer.main_net, trainer.replay_buffer, config.DEVICE)
        else:
            env.model = trainer.main_net
            env.replay_buffer = trainer.replay_buffer
            env.device = config.DEVICE
        print(f"  ✓ 环境已初始化")

        # 6. 快速训练循环
        print(f"\n[步骤 6/6] 开始快速训练...")
        print("="*70)
        print(f"💡 提示: Replay Buffer 需要先收集 {config.MIN_REPLAY_SIZE} 条经验才会开始训练")
        print(f"         在此之前可能看起来\"卡住\"，但实际上正在收集数据")
        print("="*70)

        for episode in range(1, config.NUM_EPISODES + 1):
            print(f"\n{'='*70}")
            print(f"Episode {episode}/{config.NUM_EPISODES}")
            print(f"{'='*70}")

            try:
                # 在训练前检查 buffer 大小
                buffer_size_before = len(trainer.replay_buffer)
                print(f"⏳ 当前 Replay Buffer 大小: {buffer_size_before}/{config.MIN_REPLAY_SIZE}")
                if buffer_size_before < config.MIN_REPLAY_SIZE:
                    print(f"   还需收集 {config.MIN_REPLAY_SIZE - buffer_size_before} 条经验才开始训练...")

                # 训练一个episode
                reward, loss = trainer.train_episode(env, episode)

                # 训练后再次检查
                buffer_size_after = len(trainer.replay_buffer)
                print(f"✓ Episode 结束后 Buffer 大小: {buffer_size_after}")

                print(f"\n✓ Episode {episode} 完成:")
                print(f"  总奖励: {reward:.2f}")
                print(f"  平均Loss: {loss:.4f}")
                print(f"  当前Epsilon: {trainer.epsilon:.4f}")
                print(f"  Buffer大小: {len(trainer.replay_buffer)}")

                # 验证
                if episode % config.VALIDATION_INTERVAL == 0:
                    try:
                        val_match_rate = run_quick_validation(trainer, data_processor, val_orders, config)

                        if val_match_rate > trainer.best_validation_metric:
                            print(f"✓ 发现更好的模型！匹配率: {val_match_rate:.4f}")
                            trainer.best_validation_metric = val_match_rate
                            # 只在配置允许时保存模型
                            if episode % config.SAVE_FREQ == 0:
                                trainer.save_checkpoint(episode)
                                print(f"  ✓ 已保存最佳模型 (Episode {episode})")
                        else:
                            print(f"  当前匹配率: {val_match_rate:.4f} (最佳: {trainer.best_validation_metric:.4f})")
                            # 定期保存
                            if episode % config.SAVE_FREQ == 0:
                                trainer.save_checkpoint(episode)
                                print(f"  ✓ 已保存检查点 (Episode {episode})")
                    except Exception as e:
                        print(f"⚠ 验证过程出错 (非致命): {e}")
                        import traceback
                        traceback.print_exc()
                        # 即使验证失败，也尝试保存模型
                        if episode % config.SAVE_FREQ == 0:
                            try:
                                trainer.save_checkpoint(episode)
                            except Exception as save_err:
                                print(f"❌ 保存模型失败: {save_err}")

            except Exception as e:
                print(f"\n❌ Episode {episode} 训练出错:")
                print(f"  错误信息: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n⚠ 停止训练以避免进一步错误")
                break

        print("\n" + "="*70)
        print("✅ 快速测试完成!")
        print("="*70)
        print(f"  ✓ 成功完成 {episode} 个 Episode")
        print(f"  ✓ 日志保存在: {config.LOG_SAVE_PATH}")
        print(f"  ✓ 模型保存在: {config.CHECKPOINT_PATH}")

        # 自动绘制训练曲线
        if PLOTTING_AVAILABLE and len(trainer.total_rewards) > 0:
            print("\n" + "="*70)
            print("📊 开始绘制快速测试训练曲线...")
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

                # 绘制曲线到单独的目录
                plot_training_curves(stats_dict, save_dir='results/plots/quick_test/')
                plot_waiting_time_curve(stats_dict, save_dir='results/plots/quick_test/')

                print("✓ 训练曲线已保存到 results/plots/quick_test/")
                print("  - training_curves.png")
                print("  - waiting_time_curve.png")
            except Exception as e:
                print(f"⚠ 绘制曲线时出错: {e}")
                import traceback
                traceback.print_exc()

        print("\n💡 提示:")
        print("  如果看到这条消息，说明快速测试成功!")
        print("  现在可以运行完整训练: python train.py")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ 快速测试失败!")
        print(f"{'='*70}")
        print(f"错误信息: {e}")
        print(f"\n详细错误堆栈:")
        import traceback
        traceback.print_exc()
        print(f"\n{'='*70}")
        print(f"💡 调试建议:")
        print(f"  1. 检查数据文件是否存在")
        print(f"  2. 检查GPU内存是否充足 (如果使用GPU)")
        print(f"  3. 检查Python环境和依赖包版本")
        print(f"  4. 查看上方的详细错误信息")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

