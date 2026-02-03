#!/bin/bash
# 快速执行分布对比测试脚本

echo "========================================"
echo "  🚗 Baseline 车辆分布对比测试"
echo "========================================"
echo ""

# 检查是否在项目根目录
if [ ! -f "config.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 找不到 Python"
    exit 1
fi

echo "✓ 检查环境通过"
echo ""

# 执行测试
echo "开始测试..."
echo "测试配置:"
echo "  - Baseline: Random Walk, Random Dispatch"
echo "  - 分布: 均匀 + 正态(σ=1,3,5,7)"
echo "  - 天数: 1天（快速模式）"
echo ""

python test_baselines_distribution.py "$@"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  ✅ 测试完成！"
    echo "========================================"
    echo ""
    echo "查看结果："
    echo "  CSV: results/vehicles_*/distribution_tests/*.csv"
    echo "  报告: results/vehicles_*/distribution_tests/*.txt"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  ❌ 测试失败"
    echo "========================================"
    echo ""
    exit 1
fi

