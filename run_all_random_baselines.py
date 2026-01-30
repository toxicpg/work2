"""
一键运行所有Random Walk和Random Dispatch实验
包括1800、2000、2200三种车辆配置
"""
import subprocess
import sys
import time
from datetime import datetime

# 定义要运行的脚本列表
scripts = [
    'run_random_walk_1800.py',
    'run_random_walk_2000.py',
    'run_random_walk_2200.py',
    'run_random_dispatch_1800.py',
    'run_random_dispatch_2000.py',
    'run_random_dispatch_2200.py',
]

def main():
    print("="*80)
    print("开始批量运行Random Baseline实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    results = {}
    total_start = time.time()

    for i, script in enumerate(scripts, 1):
        print(f"\n{'#'*80}")
        print(f"# [{i}/{len(scripts)}] 运行: {script}")
        print(f"{'#'*80}\n")

        start = time.time()

        try:
            # 运行脚本
            result = subprocess.run(
                [sys.executable, script],
                capture_output=False,  # 实时显示输出
                text=True
            )

            elapsed = time.time() - start

            if result.returncode == 0:
                print(f"\n✓ {script} 成功！用时: {elapsed/60:.1f}分钟")
                results[script] = {'success': True, 'time': elapsed}
            else:
                print(f"\n✗ {script} 失败！返回码: {result.returncode}")
                results[script] = {'success': False, 'time': elapsed}

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n✗ {script} 运行出错: {e}")
            results[script] = {'success': False, 'time': elapsed, 'error': str(e)}

        print(f"\n{'-'*80}\n")
        time.sleep(2)  # 短暂暂停

    # 总结
    total_time = time.time() - total_start

    print("\n" + "="*80)
    print("所有实验完成!")
    print(f"总用时: {total_time/3600:.2f}小时 ({total_time/60:.1f}分钟)")
    print("="*80)

    # 详细结果
    success_count = sum(1 for r in results.values() if r['success'])
    print(f"\n成功: {success_count}/{len(scripts)}")
    print("\n详细结果:")
    for script, result in results.items():
        status = "✓" if result['success'] else "✗"
        time_str = f"{result['time']/60:.1f}分钟"
        print(f"  {status} {script:35s} {time_str}")

    print("\n" + "="*80)

    if success_count == len(scripts):
        print("🎉 所有实验都成功完成！")
    else:
        print(f"⚠️  有 {len(scripts) - success_count} 个实验失败")

    print("="*80)

if __name__ == '__main__':
    main()

