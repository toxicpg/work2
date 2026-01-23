"""
测试训练好的模型
使用方法:
    python test_model.py
"""
import os
import sys

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.trainer import MGCNTrainer
from evaluate import evaluate_model, print_evaluation_results, save_benchmark_results


def test_trained_model(model_path=None, num_test_episodes=7):
    """
    测试训练好的模型

    Args:
        model_path: 模型检查点路径，如果为None则使用最新的模型
        num_test_episodes: 测试的episode数量 (默认7，覆盖测试集所有天)
    """
    print("\n" + "="*80)
    print("🧪 开始测试训练好的模型")
    print("="*80 + "\n")

    # 1. 加载配置
    config = Config()
    if not config.validate_config():
        print("❌ 配置验证失败!")
        return

    print(f"✓ 配置加载完成")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  模型路径: {config.MODEL_SAVE_PATH}")
    print(f"  测试Episodes: {num_test_episodes}")

    # 2. 加载数据
    print(f"\n[1/5] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    print(f"  ✓ 总订单数: {len(all_orders)}")

    # 划分数据集
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )
    print(f"  ✓ 测试集: {len(test_orders)} 条订单")

    # 3. 加载图
    print(f"\n[2/5] 加载图结构...")
    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()
    print(f"  ✓ 图加载完成")

    # 4. 初始化训练器和环境
    print(f"\n[3/5] 初始化模型...")
    trainer = MGCNTrainer(config, neighbor_adj, poi_adj)

    # 5. 加载训练好的模型
    print(f"\n[4/5] 加载训练好的模型...")

    if model_path is None:
        # 自动查找最新的模型
        import glob
        model_files = glob.glob(os.path.join(config.MODEL_SAVE_PATH, '*.pt'))
        if not model_files:
            print(f"❌ 在 {config.MODEL_SAVE_PATH} 中未找到模型文件!")
            print(f"提示: 请先训练模型或指定模型路径")
            return

        # 按修改时间排序，选择最新的
        model_path = max(model_files, key=os.path.getmtime)
        print(f"  → 自动选择最新模型: {os.path.basename(model_path)}")

    try:
        episode = trainer.load_checkpoint(model_path)
        print(f"  ✓ 模型加载成功 (训练到Episode {episode})")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 6. 创建测试环境
    print(f"\n[5/5] 开始测试...")
    test_env = RideHailingEnvironment(config, data_processor, test_orders)

    # 设置模型
    if hasattr(test_env, 'set_model_and_buffer'):
        test_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        test_env.model = trainer.main_net
        test_env.replay_buffer = None
        test_env.device = config.DEVICE

    # 7. 评估模型
    print(f"\n{'='*80}")
    print(f"开始在测试集上评估 (共 {num_test_episodes} 个Episodes)")
    print(f"{'='*80}\n")

    avg_results, daily_results_df = evaluate_model(
        trainer, test_env, num_test_episodes, config, verbose=True
    )

    # 8. 打印结果
    print_evaluation_results(
        avg_results,
        daily_results_df,
        title=f"📊 测试集评估结果 (车辆数: {config.TOTAL_VEHICLES})"
    )

    # 8.5 详细打印每天的结果
    if not daily_results_df.empty:
        print("\n" + "="*80)
        print("📅 测试集每日详细结果 (共 {} 天)".format(len(daily_results_df)))
        print("="*80)

        # 创建汇总表格
        print("\n【每日指标汇总表】")
        print("-"*95)
        print(f"{'天数':^8} {'收入':>10} {'完成':>8} {'取消':>8} {'完成率':>10} {'等待时间':>12} {'调度次数':>10}")
        print("-"*95)

        for idx, row in daily_results_df.iterrows():
            day_label = f"第{row['day_index']+1}天"
            print(f"{day_label:^8} "
                  f"{row['total_revenue']:>10.2f} "
                  f"{row['completed_orders']:>8d} "
                  f"{row['cancelled_orders']:>8d} "
                  f"{row['completion_rate']*100:>9.2f}% "
                  f"{row['avg_waiting_time']:>11.1f}s "
                  f"{row['total_dispatches']:>10d}")

        print("-"*95)

        # 计算和显示平均值
        print(f"{'平均':^8} "
              f"{daily_results_df['total_revenue'].mean():>10.2f} "
              f"{daily_results_df['completed_orders'].mean():>8.1f} "
              f"{daily_results_df['cancelled_orders'].mean():>8.1f} "
              f"{daily_results_df['completion_rate'].mean()*100:>9.2f}% "
              f"{daily_results_df['avg_waiting_time'].mean():>11.1f}s "
              f"{daily_results_df['total_dispatches'].mean():>10.1f}")
        print("="*95)

        # 详细的逐天信息
        print("\n【逐天详细信息】")
        for idx, row in daily_results_df.iterrows():
            print(f"\n▶ 第 {row['day_index']+1} 天 (测试集第 {row['actual_day']+1} 天)")
            print(f"  ├─ 总收入: {row['total_revenue']:.2f} 元")
            print(f"  ├─ 完成订单: {row['completed_orders']} 单")
            print(f"  ├─ 取消订单: {row['cancelled_orders']} 单")
            print(f"  ├─ 完成率: {row['completion_rate']*100:.2f}%")
            print(f"  ├─ 取消率: {row['cancel_rate']*100:.2f}%")
            print(f"  ├─ 平均等待: {row['avg_waiting_time']:.1f} 秒")
            print(f"  ├─ 总调度次数: {row['total_dispatches']}")
            print(f"  └─ 新订单数: {row['total_new_orders']}")

        print("\n" + "="*80)

    # 9. 保存结果
    save_benchmark_results(avg_results, daily_results_df, config)

    # 10. 额外的性能总结
    print("\n" + "="*80)
    print("🎯 关键性能指标总结")
    print("="*80)
    print(f"  匹配率 (完成率): {avg_results.get('completion_rate', 0.0):.2%}")
    print(f"  取消率: {avg_results.get('cancel_rate', 0.0):.2%}")
    print(f"  平均等待时间: {avg_results.get('avg_waiting_time', 0.0):.1f} 秒")
    print(f"  车辆利用率: {avg_results.get('vehicle_utilization', 0.0):.2%}")
    print(f"  总收入: {avg_results.get('avg_total_revenue', 0.0):.2f}")
    print("="*80 + "\n")

    return avg_results, daily_results_df


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径 (默认: 自动选择最新模型)')
    parser.add_argument('--episodes', type=int, default=7,
                       help='测试的episode数量 (默认: 7，覆盖测试集所有天)')
    args = parser.parse_args()

    test_trained_model(
        model_path=args.model,
        num_test_episodes=args.episodes
    )


if __name__ == '__main__':
    main()

