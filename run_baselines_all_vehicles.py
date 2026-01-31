"""
批量运行所有Baseline实验 - 针对不同车辆数量
运行1800、2000、2200三种车辆配置，每种配置运行所有5个baseline方法
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

# 确保在项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# 车辆数量配置
VEHICLE_COUNTS = [1800, 2000, 2200]

# Baseline方法
BASELINE_SCRIPTS = {
    'random_walk': 'baselines/random_walk.py',
    'random_dispatch': 'baselines/random_dispatch.py',
    'sarsa_saa': 'baselines/sarsa_saa.py',
    'hmarl': 'baselines/train_hmarl.py',
    'cnn_ddqn': 'baselines/cnn_ddqn.py'
}


def modify_config_vehicles(vehicle_count):
    """临时修改config.py中的车辆数量"""
    config_path = 'config.py'

    # 读取config.py
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 找到TOTAL_VEHICLES行并修改
    modified = False
    for i, line in enumerate(lines):
        if line.strip().startswith('TOTAL_VEHICLES') and '=' in line:
            # 保持原有的缩进
            indent = len(line) - len(line.lstrip())
            lines[i] = ' ' * indent + f'TOTAL_VEHICLES = {vehicle_count}  # 车辆总数\n'
            modified = True
            break

    if not modified:
        print(f"警告: 未找到TOTAL_VEHICLES配置行")
        return False

    # 写回config.py
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"✓ 已将车辆数量设置为: {vehicle_count}")
    return True


def run_baseline(name, script_path, vehicle_count):
    """运行单个baseline实验"""
    print(f"\n{'='*80}")
    print(f"运行: {name.upper()} (车辆数: {vehicle_count})")
    print(f"脚本: {script_path}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=project_root
        )

        elapsed_time = time.time() - start_time

        # 打印输出
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"\n✓ {name.upper()} 完成！用时: {elapsed_time/60:.1f}分钟")
            return True, elapsed_time, None
        else:
            error_msg = f"返回码: {result.returncode}"
            print(f"\n✗ {name.upper()} 失败！{error_msg}")
            return False, elapsed_time, error_msg

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        print(f"\n✗ {name.upper()} 运行出错: {error_msg}")
        return False, elapsed_time, error_msg


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"批量运行Baseline实验 - 多车辆配置")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"车辆配置: {VEHICLE_COUNTS}")
    print(f"Baseline方法: {list(BASELINE_SCRIPTS.keys())}")
    print(f"{'='*80}\n")

    # 备份原始config.py
    print("备份原始config.py...")
    import shutil
    shutil.copy('config.py', 'config.py.backup')
    print("✓ 备份完成: config.py.backup\n")

    # 总结果记录
    all_results = {}
    total_start_time = time.time()

    try:
        # 遍历每种车辆配置
        for vehicle_count in VEHICLE_COUNTS:
            print(f"\n{'#'*80}")
            print(f"# 开始运行车辆数量: {vehicle_count}")
            print(f"{'#'*80}\n")

            # 修改config.py
            if not modify_config_vehicles(vehicle_count):
                print(f"✗ 跳过车辆数量 {vehicle_count}")
                continue

            # 等待一下确保文件写入
            time.sleep(1)

            # 运行所有baseline
            vehicle_results = {}

            for name, script_path in BASELINE_SCRIPTS.items():
                success, elapsed_time, error_msg = run_baseline(name, script_path, vehicle_count)

                vehicle_results[name] = {
                    'success': success,
                    'elapsed_time': elapsed_time,
                    'error': error_msg
                }

                # 每个实验之间暂停一下
                time.sleep(2)

            all_results[vehicle_count] = vehicle_results

            # 打印本次配置的总结
            print(f"\n{'='*80}")
            print(f"车辆数量 {vehicle_count} 完成总结:")
            print(f"{'='*80}")
            for name, result in vehicle_results.items():
                status = "✓ 成功" if result['success'] else "✗ 失败"
                time_str = f"{result['elapsed_time']/60:.1f}分钟"
                print(f"  {name:20s} {status:10s} {time_str}")

    finally:
        # 恢复原始config.py
        print(f"\n{'='*80}")
        print("恢复原始config.py...")
        shutil.copy('config.py.backup', 'config.py')
        os.remove('config.py.backup')
        print("✓ 已恢复原始配置")

    total_elapsed_time = time.time() - total_start_time

    # 打印最终总结
    print(f"\n\n{'='*80}")
    print(f"所有实验完成！")
    print(f"总用时: {total_elapsed_time/3600:.1f}小时")
    print(f"{'='*80}\n")

    # 详细总结
    for vehicle_count, vehicle_results in all_results.items():
        print(f"\n车辆数量: {vehicle_count}")
        print("-" * 60)
        for name, result in vehicle_results.items():
            status = "✓ 成功" if result['success'] else "✗ 失败"
            time_str = f"{result['elapsed_time']/60:.1f}分钟"
            error_str = f" (错误: {result['error']})" if result['error'] else ""
            print(f"  {name:20s} {status:10s} {time_str}{error_str}")

    # 保存总结果
    summary_file = f'baseline_all_vehicles_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_hours': total_elapsed_time / 3600,
            'vehicle_counts': VEHICLE_COUNTS,
            'baselines': list(BASELINE_SCRIPTS.keys()),
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\n✓ 完整结果摘要已保存到: {summary_file}")

    # 统计成功率
    total_experiments = len(VEHICLE_COUNTS) * len(BASELINE_SCRIPTS)
    successful_experiments = sum(
        1 for vehicle_results in all_results.values()
        for result in vehicle_results.values()
        if result['success']
    )

    print(f"\n总实验数: {total_experiments}")
    print(f"成功: {successful_experiments}")
    print(f"失败: {total_experiments - successful_experiments}")
    print(f"成功率: {successful_experiments/total_experiments*100:.1f}%")


if __name__ == '__main__':
    main()

