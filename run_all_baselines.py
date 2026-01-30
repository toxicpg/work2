"""
统一运行所有Baseline实验
包括：Random Walk, Random Dispatch, SARSA-SAA, H-MARL, CNN-DDQN
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json

# 确保在项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

BASELINE_SCRIPTS = {
    'random_walk': 'baselines/random_walk.py',
    'random_dispatch': 'baselines/random_dispatch.py',
    'sarsa_saa': 'baselines/sarsa_saa.py',
    'hmarl': 'baselines/train_hmarl.py',
    'cnn_ddqn': 'baselines/cnn_ddqn.py'
}


def run_baseline(name, script_path):
    """运行单个baseline实验"""
    print(f"\n{'='*80}")
    print(f"开始运行: {name.upper()}")
    print(f"脚本: {script_path}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            cwd=project_root
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {name.upper()} 完成！用时: {elapsed_time/60:.1f}分钟")
            return True, elapsed_time
        else:
            print(f"\n✗ {name.upper()} 失败！返回码: {result.returncode}")
            return False, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {name.upper()} 运行出错: {e}")
        return False, elapsed_time


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"运行所有Baseline实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # 检查脚本是否存在
    print("检查脚本文件...")
    for name, script_path in BASELINE_SCRIPTS.items():
        if not os.path.exists(script_path):
            print(f"  ✗ {name}: {script_path} 不存在！")
        else:
            print(f"  ✓ {name}: {script_path}")

    print("\n选择运行模式:")
    print("  1. 运行所有baseline")
    print("  2. 选择特定baseline运行")
    print("  3. 退出")

    choice = input("\n请选择 (1-3): ").strip()

    if choice == '3':
        print("退出。")
        return
    elif choice == '2':
        print("\n可用的baseline:")
        for i, name in enumerate(BASELINE_SCRIPTS.keys(), 1):
            print(f"  {i}. {name}")

        selected = input("\n请输入要运行的baseline序号 (逗号分隔): ").strip()
        try:
            indices = [int(x.strip()) for x in selected.split(',')]
            baselines_to_run = [list(BASELINE_SCRIPTS.keys())[i-1] for i in indices]
        except:
            print("输入无效，退出。")
            return
    else:
        baselines_to_run = list(BASELINE_SCRIPTS.keys())

    # 运行选定的baseline
    results = {}
    total_start_time = time.time()

    for name in baselines_to_run:
        script_path = BASELINE_SCRIPTS[name]
        success, elapsed_time = run_baseline(name, script_path)
        results[name] = {
            'success': success,
            'elapsed_time': elapsed_time
        }

    total_elapsed_time = time.time() - total_start_time

    # 打印总结
    print(f"\n\n{'='*80}")
    print(f"所有实验完成！")
    print(f"总用时: {total_elapsed_time/60:.1f}分钟")
    print(f"{'='*80}\n")

    print("实验结果总结:")
    for name, result in results.items():
        status = "✓ 成功" if result['success'] else "✗ 失败"
        time_str = f"{result['elapsed_time']/60:.1f}分钟"
        print(f"  {name:20s} {status:10s} {time_str}")

    # 保存结果摘要
    summary_file = f'baseline_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': total_elapsed_time,
            'results': results
        }, f, indent=2)

    print(f"\n✓ 结果摘要已保存到: {summary_file}")


if __name__ == '__main__':
    main()

