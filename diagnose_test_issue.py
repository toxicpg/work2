#!/usr/bin/env python3
"""
诊断测试效果差的问题
检查所有可能导致测试效果差的原因
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())

from config import Config
from utils.data_process import DataProcessor
from utils.graph_builder import GraphBuilder
from environment import RideHailingEnvironment
from models.trainer import MGCNTrainer

def diagnose_test_issue(model_path=None):
    """
    诊断测试问题的完整流程
    """
    print("\n" + "="*80)
    print("🔍 主实验测试问题诊断")
    print("="*80 + "\n")

    # 1. 加载配置
    config = Config()
    print(f"✓ 配置加载完成")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  空闲阈值: {config.IDLE_THRESHOLD_SEC}秒")
    print(f"  Tick间隔: {config.TICK_DURATION_SEC}秒")
    print(f"  每天Ticks: {config.TICKS_PER_DAY}")
    print(f"  匹配搜索半径: {getattr(config, 'MATCHER_SEARCH_RADIUS', 10)}")
    print(f"  最大等待时间: {config.MAX_WAITING_TIME}秒")

    # 2. 加载数据
    print(f"\n[1/6] 加载数据...")
    data_processor = DataProcessor(config)
    all_orders = data_processor.load_and_process_orders()
    train_orders, val_orders, test_orders = data_processor.split_data_by_time(
        all_orders, config.TRAIN_RATIO, config.VAL_RATIO
    )
    print(f"  ✓ 训练集: {len(train_orders)} 条")
    print(f"  ✓ 验证集: {len(val_orders)} 条")
    print(f"  ✓ 测试集: {len(test_orders)} 条")

    # 3. 加载图
    print(f"\n[2/6] 加载图结构...")
    gb = GraphBuilder(config)
    neighbor_adj, poi_adj = gb.load_graphs_pt()
    print(f"  ✓ 图加载完成")

    # 4. 初始化训练器
    print(f"\n[3/6] 初始化模型...")
    trainer = MGCNTrainer(config, neighbor_adj, poi_adj)

    # 5. 检查模型状态
    print(f"\n[4/6] 检查模型状态...")

    if model_path is None:
        import glob
        model_files = glob.glob(os.path.join(config.MODEL_SAVE_PATH, '*.pt'))
        if not model_files:
            print(f"  ❌ 没有找到模型文件!")
            print(f"  说明: 这就是问题所在 - 你需要先训练模型!")
            return False
        model_path = max(model_files, key=os.path.getmtime)
        print(f"  → 自动选择最新模型: {os.path.basename(model_path)}")

    try:
        episode = trainer.load_checkpoint(model_path)
        print(f"  ✓ 模型加载成功 (训练到Episode {episode})")

        # 检查模型参数
        total_params = sum(p.numel() for p in trainer.main_net.parameters())
        trainable_params = sum(p.numel() for p in trainer.main_net.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

        # 检查模型是否全是零或者未训练
        first_layer_mean = None
        first_layer_std = None
        for name, param in trainer.main_net.named_parameters():
            if param.requires_grad:
                first_layer_mean = param.data.mean().item()
                first_layer_std = param.data.std().item()
                print(f"  第一层参数({name}): mean={first_layer_mean:.6f}, std={first_layer_std:.6f}")
                break

        if first_layer_mean is not None and abs(first_layer_mean) < 1e-6 and first_layer_std < 1e-6:
            print(f"  ⚠️  警告: 模型参数接近零,可能未训练!")
            return False

    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 创建测试环境并运行短期测试
    print(f"\n[5/6] 创建测试环境...")
    test_env = RideHailingEnvironment(config, data_processor, test_orders)

    # 设置模型
    if hasattr(test_env, 'set_model_and_buffer'):
        test_env.set_model_and_buffer(trainer.main_net, None, config.DEVICE)
    else:
        test_env.model = trainer.main_net
        test_env.replay_buffer = None
        test_env.device = config.DEVICE

    print(f"  ✓ 环境初始化完成")

    # 7. 运行短期诊断测试
    print(f"\n[6/6] 运行诊断测试 (100 ticks)...")
    test_env.reset()

    # 记录统计信息
    stats = {
        'matched_orders': 0,
        'cancelled_orders': 0,
        'new_orders': 0,
        'dispatch_success': 0,
        'dispatch_total': 0,
        'idle_vehicles': [],
        'pending_orders': [],
        'waiting_times': []
    }

    for tick in range(100):
        # 使用epsilon=0.0进行贪婪策略测试
        next_state, _, done, info = test_env.step(current_epsilon=0.0)
        step_info = info.get('step_info', {})

        stats['matched_orders'] += step_info.get('matched_orders', 0)
        stats['cancelled_orders'] += step_info.get('cancelled_orders', 0)
        stats['new_orders'] += step_info.get('new_orders', 0)
        stats['dispatch_success'] += step_info.get('dispatch_success', 0)
        stats['dispatch_total'] += step_info.get('dispatch_total', 0)
        stats['waiting_times'].extend(step_info.get('waiting_times', []))

        # 记录车辆状态
        idle_count = sum(1 for v in test_env.vehicle_manager.vehicles.values() if v['status'] == 'idle')
        stats['idle_vehicles'].append(idle_count)
        stats['pending_orders'].append(len(test_env.pending_orders))

        if done:
            break

    # 8. 分析结果
    print(f"\n" + "="*80)
    print(f"📊 诊断结果")
    print(f"="*80)

    total_processed = stats['matched_orders'] + stats['cancelled_orders']
    completion_rate = stats['matched_orders'] / total_processed if total_processed > 0 else 0.0
    cancel_rate = stats['cancelled_orders'] / total_processed if total_processed > 0 else 0.0

    print(f"\n订单统计:")
    print(f"  生成订单: {stats['new_orders']}")
    print(f"  匹配订单: {stats['matched_orders']}")
    print(f"  取消订单: {stats['cancelled_orders']}")
    print(f"  完成率: {completion_rate:.2%}")
    print(f"  取消率: {cancel_rate:.2%}")

    print(f"\n调度统计:")
    print(f"  调度尝试: {stats['dispatch_total']}")
    print(f"  调度成功: {stats['dispatch_success']}")
    print(f"  调度成功率: {stats['dispatch_success']/stats['dispatch_total']*100 if stats['dispatch_total'] > 0 else 0:.1f}%")

    print(f"\n车辆状态:")
    print(f"  平均空闲车辆: {np.mean(stats['idle_vehicles']):.1f} / {config.TOTAL_VEHICLES}")
    print(f"  空闲率: {np.mean(stats['idle_vehicles'])/config.TOTAL_VEHICLES*100:.1f}%")

    print(f"\n订单队列:")
    print(f"  平均待匹配订单: {np.mean(stats['pending_orders']):.1f}")

    if stats['waiting_times']:
        print(f"\n等待时间:")
        print(f"  平均: {np.mean(stats['waiting_times']):.1f}秒")
        print(f"  最大: {np.max(stats['waiting_times']):.1f}秒")

    # 9. 问题诊断
    print(f"\n" + "="*80)
    print(f"🔍 问题诊断")
    print(f"="*80 + "\n")

    issues_found = []

    # 检查1: 匹配率过低
    if completion_rate < 0.3:
        issues_found.append(f"❌ 严重问题: 完成率过低 ({completion_rate:.2%} < 30%)")
        print(f"❌ 严重问题: 完成率过低 ({completion_rate:.2%})")
        print(f"   可能原因:")
        print(f"   - 模型未充分训练")
        print(f"   - 模型参数损坏")
        print(f"   - 环境配置错误")
    elif completion_rate < 0.7:
        issues_found.append(f"⚠️  问题: 完成率偏低 ({completion_rate:.2%} < 70%)")
        print(f"⚠️  问题: 完成率偏低 ({completion_rate:.2%})")
        print(f"   可能原因:")
        print(f"   - 模型训练不充分")
        print(f"   - 测试环境与训练环境不一致")
    else:
        print(f"✅ 完成率正常 ({completion_rate:.2%})")

    # 检查2: 调度率过低
    dispatch_rate = stats['dispatch_total'] / 100  # 每tick平均调度次数
    if dispatch_rate < 1:
        issues_found.append(f"⚠️  调度频率过低 ({dispatch_rate:.2f} 次/tick)")
        print(f"⚠️  调度频率过低: {dispatch_rate:.2f} 次/tick")
        print(f"   可能原因:")
        print(f"   - IDLE_THRESHOLD_SEC ({config.IDLE_THRESHOLD_SEC}秒) 过高")
        print(f"   - 车辆很少达到空闲阈值")
    else:
        print(f"✅ 调度频率正常 ({dispatch_rate:.2f} 次/tick)")

    # 检查3: 空闲车辆过多
    idle_rate = np.mean(stats['idle_vehicles']) / config.TOTAL_VEHICLES
    if idle_rate > 0.8:
        issues_found.append(f"⚠️  空闲车辆过多 ({idle_rate:.1%})")
        print(f"⚠️  空闲车辆过多: {idle_rate:.1%}")
        print(f"   说明调度策略不积极或订单量不足")
    elif idle_rate < 0.2:
        print(f"✅ 车辆利用率高 (空闲率: {idle_rate:.1%})")
    else:
        print(f"✅ 空闲率正常 ({idle_rate:.1%})")

    # 检查4: 待匹配订单堆积
    avg_pending = np.mean(stats['pending_orders'])
    if avg_pending > 100:
        issues_found.append(f"❌ 订单堆积严重 (平均{avg_pending:.0f}个)")
        print(f"❌ 订单堆积严重: 平均{avg_pending:.0f}个待匹配订单")
        print(f"   说明匹配效率低或车辆不足")
    elif avg_pending > 50:
        issues_found.append(f"⚠️  订单有所堆积 (平均{avg_pending:.0f}个)")
        print(f"⚠️  订单有所堆积: 平均{avg_pending:.0f}个")
    else:
        print(f"✅ 订单队列正常 (平均{avg_pending:.0f}个)")

    # 总结
    print(f"\n" + "="*80)
    print(f"📋 诊断总结")
    print(f"="*80 + "\n")

    if not issues_found:
        print(f"✅ 未发现明显问题,模型测试效果应该是正常的")
        print(f"   如果你认为效果差,请:")
        print(f"   1. 检查'效果好'时的完成率是多少")
        print(f"   2. 对比训练时的验证集完成率")
        print(f"   3. 确认使用的模型是否正确")
    else:
        print(f"发现 {len(issues_found)} 个问题:\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")

        print(f"\n💡 建议:")
        if completion_rate < 0.3:
            print(f"  1. 检查模型是否真的训练过 (查看训练日志)")
            print(f"  2. 重新训练模型: python train.py")
            print(f"  3. 确认模型文件路径是否正确")
        elif completion_rate < 0.7:
            print(f"  1. 增加训练轮数")
            print(f"  2. 检查训练时的验证集完成率")
            print(f"  3. 调整超参数(学习率、epsilon等)")

        if dispatch_rate < 1:
            print(f"  4. 降低 IDLE_THRESHOLD_SEC (当前{config.IDLE_THRESHOLD_SEC}秒)")
            print(f"     建议: config.py中设置为60秒")

    print(f"\n" + "="*80 + "\n")

    return len(issues_found) == 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='诊断测试效果差的问题')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    args = parser.parse_args()

    success = diagnose_test_issue(args.model)
    sys.exit(0 if success else 1)

