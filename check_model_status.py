#!/usr/bin/env python3
"""
快速检查模型状态的诊断工具
"""
import glob
import os

from config import Config


def check_model_status():
    """检查模型文件状态"""
    print("\n" + "="*80)
    print("🔍 模型状态检查")
    print("="*80 + "\n")

    config = Config()

    # 检查配置
    print(f"📋 当前配置:")
    print(f"  车辆数: {config.TOTAL_VEHICLES}")
    print(f"  模型保存路径: {config.MODEL_SAVE_PATH}")
    print(f"  消融实验路径: {config.ABLATION_SAVE_PATH}")
    print(f"  Baseline路径: {config.BENCHMARK_SAVE_PATH}")

    # 检查目录是否存在
    print(f"\n📁 目录状态:")
    for path_name, path in [
        ("模型目录", config.MODEL_SAVE_PATH),
        ("消融目录", config.ABLATION_SAVE_PATH),
        ("Baseline目录", config.BENCHMARK_SAVE_PATH),
    ]:
        exists = os.path.exists(path)
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"  {path_name}: {status}")
        if exists:
            files = os.listdir(path)
            print(f"    文件数: {len(files)}")

    # 查找所有模型文件
    print(f"\n🔎 搜索所有模型文件 (.pt):")
    all_pt_files = []
    search_paths = [
        'results/',
        'models/',
        '.',
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            pattern = os.path.join(search_path, '**/*.pt')
            files = glob.glob(pattern, recursive=True)
            all_pt_files.extend(files)

    if all_pt_files:
        print(f"  找到 {len(all_pt_files)} 个模型文件:")
        for i, f in enumerate(sorted(all_pt_files, key=os.path.getmtime, reverse=True), 1):
            size_mb = os.path.getsize(f) / 1024 / 1024
            mtime = os.path.getmtime(f)
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"    {i}. {f}")
            print(f"       大小: {size_mb:.2f} MB, 修改时间: {mtime_str}")
    else:
        print("  ❌ 没有找到任何模型文件!")
        print("\n💡 建议:")
        print("  1. 运行主实验训练: python train.py")
        print("  2. 或运行消融实验: python run_ablation_simple.py")
        print("  3. 训练完成后再运行测试脚本")

    # 检查是否有训练记录
    print(f"\n📊 训练记录:")
    log_files = glob.glob(os.path.join(config.LOG_SAVE_PATH, '**/*.json'), recursive=True) + \
                glob.glob(os.path.join(config.LOG_SAVE_PATH, '**/*.csv'), recursive=True)
    if log_files:
        print(f"  找到 {len(log_files)} 个日志文件")
        for f in sorted(log_files, key=os.path.getmtime, reverse=True)[:5]:
            print(f"    - {f}")
    else:
        print("  ❌ 没有找到训练日志")

    print("\n" + "="*80)

    # 给出明确建议
    if not all_pt_files:
        print("\n⚠️  警告: 没有找到任何训练好的模型!")
        print("\n📝 操作步骤:")
        print("  步骤1: 训练主实验模型")
        print("    命令: python train.py")
        print("    说明: 这会训练MGCN-D3QN模型并保存到 results/vehicles_1800/models/")
        print()
        print("  步骤2: 训练完成后测试模型")
        print("    命令: python test_model.py")
        print("    说明: 这会使用训练好的模型在测试集上评估")
        print()
        print("  如果需要运行消融实验:")
        print("    命令: python run_ablation_simple.py")
        print()
    else:
        print("\n✅ 找到模型文件,可以运行测试!")
        print("\n测试命令:")
        print(f"  python test_model.py")
        print(f"  或指定模型: python test_model.py --model {all_pt_files[0]}")

    print("="*80 + "\n")

if __name__ == '__main__':
    check_model_status()

