#!/bin/bash
# 灵活运行baseline实验的Shell脚本
# 可以快速切换车辆数量和baseline方法

echo "=========================================="
echo "Baseline实验运行脚本"
echo "=========================================="
echo ""
echo "使用方法示例:"
echo "  1. 运行单个baseline (车辆2000):"
echo "     TOTAL_VEHICLES=2000 python baselines/random_walk.py"
echo ""
echo "  2. 运行所有车辆配置 (1800, 2000, 2200):"
echo "     python run_baselines_all_vehicles.py"
echo ""
echo "  3. 手动逐个运行:"
echo "     # 修改config.py中的TOTAL_VEHICLES"
echo "     # 然后运行: python baselines/xxx.py"
echo ""
echo "=========================================="
echo ""

# 选择模式
echo "请选择运行模式:"
echo "  1. 运行所有车辆配置 (1800, 2000, 2200) x 所有baseline"
echo "  2. 运行单个配置"
echo "  3. 查看帮助"
echo ""
read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "运行所有配置..."
        python run_baselines_all_vehicles.py
        ;;
    2)
        echo ""
        echo "可用车辆数量: 1800, 2000, 2200"
        read -p "请输入车辆数量: " vehicles

        echo ""
        echo "可用Baseline方法:"
        echo "  1. random_walk"
        echo "  2. random_dispatch"
        echo "  3. sarsa_saa"
        echo "  4. hmarl"
        echo "  5. cnn_ddqn"
        read -p "请输入方法编号 (1-5): " method_num

        case $method_num in
            1) method="random_walk" ;;
            2) method="random_dispatch" ;;
            3) method="sarsa_saa" ;;
            4) method="hmarl" ;;
            5) method="cnn_ddqn" ;;
            *) echo "无效选择"; exit 1 ;;
        esac

        echo ""
        echo "=========================================="
        echo "配置: 车辆数=$vehicles, 方法=$method"
        echo "=========================================="
        echo ""
        echo "请手动修改 config.py 中的 TOTAL_VEHICLES = $vehicles"
        echo "然后运行: python baselines/${method}.py"
        echo ""
        read -p "是否现在运行? (y/n): " run_now

        if [ "$run_now" = "y" ] || [ "$run_now" = "Y" ]; then
            python baselines/${method}.py
        fi
        ;;
    3)
        echo ""
        echo "=========================================="
        echo "详细帮助"
        echo "=========================================="
        echo ""
        echo "1. 完整批量运行 (推荐):"
        echo "   python run_baselines_all_vehicles.py"
        echo "   - 自动运行所有车辆配置和所有baseline"
        echo "   - 自动切换配置文件"
        echo "   - 生成完整报告"
        echo ""
        echo "2. 单独运行某个baseline:"
        echo "   步骤1: 编辑 config.py, 设置 TOTAL_VEHICLES"
        echo "   步骤2: python baselines/<方法名>.py"
        echo "   例如: python baselines/cnn_ddqn.py"
        echo ""
        echo "3. 查看结果:"
        echo "   结果保存在: results/vehicles_<数量>/baselines/"
        echo ""
        echo "4. 所有可用的baseline方法:"
        echo "   - random_walk.py       (无需训练, 快)"
        echo "   - random_dispatch.py   (无需训练, 快)"
        echo "   - sarsa_saa.py         (需要训练, 中)"
        echo "   - train_hmarl.py       (需要训练, 慢)"
        echo "   - cnn_ddqn.py          (需要训练, 慢)"
        echo ""
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

