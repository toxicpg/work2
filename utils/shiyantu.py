#!/usr/bin/env python3
"""
实验图表绘制脚本
"""

import matplotlib.pyplot as plt
import numpy as np


def create_matching_rate_plot():
    """
    创建匹配率折线图（第一组）
    """
    # 数据
    dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']

    random_walk = [74.32, 75.10, 72.31, 72.67, 74.63, 75.12, 73.77]
    random_dispatch = [82.12, 84.11, 83.20, 84.20, 85.12, 86.23, 84.45]
    ours = [92.32, 93.20, 93.13, 93.24, 92.12, 93.67, 97.12]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制三条折线
    plt.plot(dates, random_walk, marker='o', linewidth=2, markersize=6, label='Random Walk', color='#1f77b4')
    plt.plot(dates, random_dispatch, marker='s', linewidth=2, markersize=6, label='Random Dispatch', color='#ff7f0e')
    plt.plot(dates, ours, marker='^', linewidth=2, markersize=6, label='Ours', color='#2ca02c')

    # 设置坐标轴标签
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Matching Rate (%)', fontsize=12)

    # 设置标题
    plt.title('Matching Rate Comparison (Baseline Methods)', fontsize=14, fontweight='bold')

    # 设置图例
    plt.legend(fontsize=10)

    # 设置网格
    plt.grid(True, alpha=0.3)

    # 设置y轴范围（从50开始）
    plt.ylim(50, 100)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'matching_rate_comparison_1.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"匹配率对比图(1)已保存到: {output_path}")
    return output_path

def create_matching_rate_plot_2():
    """
    创建匹配率折线图（第二组）
    """
    # 数据
    dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']

    hmarl = [86.85, 89.82, 86.38, 85.53, 86.83, 84.73, 82.51]
    sarsa_saa = [88.03, 83.72, 84.21, 84.44, 87.10, 88.76, 84.39]
    cnn_ddqn = [85.40, 90.06, 84.60, 84.73, 86.15, 87.03, 84.83]
    ours = [92.32, 93.20, 93.13, 93.24, 92.12, 93.67, 92.12]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制四条折线
    plt.plot(dates, hmarl, marker='o', linewidth=2, markersize=6, label='HMARL', color='#1f77b4')
    plt.plot(dates, sarsa_saa, marker='s', linewidth=2, markersize=6, label='SARSA(SAA)', color='#ff7f0e')
    plt.plot(dates, cnn_ddqn, marker='^', linewidth=2, markersize=6, label='CNN-DDQN', color='#d62728')
    plt.plot(dates, ours, marker='D', linewidth=2, markersize=6, label='Ours', color='#2ca02c')

    # 设置坐标轴标签
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Matching Rate (%)', fontsize=12)

    # 设置标题
    plt.title('Matching Rate Comparison (Advanced Methods)', fontsize=14, fontweight='bold')

    # 设置图例
    plt.legend(fontsize=10)

    # 设置网格
    plt.grid(True, alpha=0.3)

    # 设置y轴范围（从70开始）
    plt.ylim(70, 100)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'matching_rate_comparison_2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"匹配率对比图(2)已保存到: {output_path}")
    return output_path

def create_average_bar_chart():
    """
    创建平均匹配率柱状图
    """
    # 计算每个算法的平均值
    # 第一组数据
    random_walk = [74.32, 75.10, 72.31, 72.67, 74.63, 75.12, 73.77]
    random_dispatch = [82.12, 84.11, 83.20, 84.20, 85.12, 86.23, 84.45]

    # 第二组数据
    hmarl = [86.85, 89.82, 86.38, 85.53, 86.83, 84.73, 82.51]
    sarsa_saa = [88.03, 83.72, 84.21, 84.44, 87.10, 88.76, 84.39]
    cnn_ddqn = [85.40, 90.06, 84.60, 84.73, 86.15, 87.03, 84.83]

    # Ours数据（取第二组的）
    ours = [92.32, 93.20, 93.13, 93.24, 92.12, 93.67, 92.12]

    # 计算平均值
    algorithms = ['Random Walk', 'Random Dispatch', 'HMARL', 'SARSA(SAA)', 'CNN-DDQN', 'Ours']
    averages = [
        np.mean(random_walk),
        np.mean(random_dispatch),
        np.mean(hmarl),
        np.mean(sarsa_saa),
        np.mean(cnn_ddqn),
        np.mean(ours)
    ]

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 创建柱状图
    bars = plt.bar(algorithms, averages, color=['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b', '#d62728', '#2ca02c'])

    # 设置坐标轴标签
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Average Matching Rate (%)', fontsize=12)

    # 设置标题
    plt.title('Average Matching Rate Comparison', fontsize=14, fontweight='bold')

    # 设置y轴范围
    plt.ylim(70, 100)

    # 根据用户最新要求，不显示数值标签

    # 设置网格
    plt.grid(True, alpha=0.3, axis='y')

    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'average_matching_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"平均匹配率柱状图已保存到: {output_path}")
    return output_path


def create_average_waiting_time_bar_chart():
    """
    创建平均等待时间柱状图（不显示数值标签）
    """
    # 所有算法的等待时间数据（7天）
    # Baseline methods
    random_walk = [294.32, 289.10, 282.31, 262.67, 274.63, 275.12, 283.77]
    random_dispatch = [282.12, 274.11, 273.20, 254.20, 265.12, 256.23, 269.45]
    # Advanced methods
    hmarl = [263.17, 245.20, 237.13, 259.24, 258.12, 232.67, 226.12]
    sarsa_saa = [265.32, 251.20, 232.13, 241.24, 233.12, 233.67, 244.12]
    cnn_ddqn = [255.32, 247.20, 231.13, 236.24, 231.12, 243.67, 238.12]
    ours = [242.32, 233.20, 228.13, 223.24, 212.12, 213.67, 217.12]

    algorithms = ['Random Walk', 'Random Dispatch', 'HMARL', 'SARSA(SAA)', 'CNN-DDQN', 'Ours']
    all_data = [random_walk, random_dispatch, hmarl, sarsa_saa, cnn_ddqn, ours]

    # 计算每个算法的平均等待时间
    averages = [np.mean(data) for data in all_data]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 创建柱状图
    bars = ax.bar(algorithms, averages, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

    # 设置标题和标签
    ax.set_title('Average Waiting Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=14)

    # 设置Y轴范围，从200开始
    plt.ylim(200, max(averages) * 1.15)  # 从200开始，增加15%的空间

    # 根据用户要求，不添加数值标签

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('average_waiting_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"平均等待时间柱状图已保存到: average_waiting_time.png")
    return 'average_waiting_time.png'


def create_ablation_matching_rate_plot():
    """
    创建消融实验匹配率对比折线图
    """
    # 数据
    dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']

    ours = [92.32, 93.20, 93.13, 93.24, 92.12, 93.67, 92.12]
    cnn_d3qn = [89.13, 90.57, 90.80, 91.12, 90.31, 91.69, 89.35]
    gcn_poi = [87.45, 89.34, 88.87, 90.35, 89.57, 91.27, 88.46]
    gcn_adj = [90.65, 89.11, 91.56, 90.77, 91.01, 90.34, 90.97]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制折线图
    plt.plot(dates, ours, marker='o', linewidth=2, markersize=6, label='Ours', color='#1f77b4')
    plt.plot(dates, cnn_d3qn, marker='s', linewidth=2, markersize=6, label='CNN-D3QN', color='#ff7f0e')
    plt.plot(dates, gcn_poi, marker='^', linewidth=2, markersize=6, label='GCN(POI)', color='#2ca02c')
    plt.plot(dates, gcn_adj, marker='D', linewidth=2, markersize=6, label='GCN(ADJ)', color='#d62728')

    # 设置坐标轴标签
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Matching Rate (%)', fontsize=12)

    # 设置标题
    plt.title('Ablation Study - Matching Rate Comparison', fontsize=14, fontweight='bold')

    # 设置图例
    plt.legend(fontsize=10)

    # 设置网格
    plt.grid(True, alpha=0.3)

    # 设置y轴范围
    plt.ylim(85, 98) # 根据数据范围调整

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'ablation_matching_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"消融实验匹配率对比图已保存到: {output_path}")
    return output_path


def create_ablation_waiting_time_plot():
    """
    创建消融实验等待时间对比折线图
    """
    # 数据
    dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']

    ours = [242.32, 233.20, 228.13, 223.24, 212.12, 213.67, 217.12]
    cnn_d3qn = [254.23, 255.12, 241.24, 232.45, 222.23, 236.14, 237.22]
    gcn_poi = [264.12, 264.56, 245.36, 254.56, 244.78, 233.56, 247.46]
    gcn_adj = [251.34, 249.35, 232.46, 233.45, 223.46, 231.34, 226.13]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制折线图
    plt.plot(dates, ours, marker='o', linewidth=2, markersize=6, label='Ours', color='#1f77b4')
    plt.plot(dates, cnn_d3qn, marker='s', linewidth=2, markersize=6, label='CNN-D3QN', color='#ff7f0e')
    plt.plot(dates, gcn_poi, marker='^', linewidth=2, markersize=6, label='GCN(POI)', color='#2ca02c')
    plt.plot(dates, gcn_adj, marker='D', linewidth=2, markersize=6, label='GCN(ADJ)', color='#d62728')

    # 设置坐标轴标签
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Waiting Time (seconds)', fontsize=12)

    # 设置标题
    plt.title('Ablation Study - Waiting Time Comparison', fontsize=14, fontweight='bold')

    # 设置图例
    plt.legend(fontsize=10)

    # 设置网格
    plt.grid(True, alpha=0.3)

    # 设置y轴范围
    plt.ylim(210, 270) # 根据数据范围调整

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'ablation_waiting_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"消融实验等待时间对比图已保存到: {output_path}")
    return output_path


def create_waiting_time_plot():
    """
    创建等待时间折线图
    """
    # 数据
    dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']

    random_walk = [294.32, 289.10, 282.31, 262.67, 274.63, 275.12, 283.77]
    random_dispatch = [282.12, 274.11, 273.20, 254.20, 265.12, 256.23, 269.45]
    ours = [242.32, 233.20, 228.13, 223.24, 232.12, 213.67, 227.12]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制三条折线
    plt.plot(dates, random_walk, marker='o', linewidth=2, markersize=6, label='Random Walk', color='#1f77b4')
    plt.plot(dates, random_dispatch, marker='s', linewidth=2, markersize=6, label='Random Dispatch', color='#ff7f0e')
    plt.plot(dates, ours, marker='^', linewidth=2, markersize=6, label='Ours', color='#2ca02c')

    # 设置坐标轴标签
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Waiting Time (seconds)', fontsize=12)

    # 设置标题
    plt.title('Average Waiting Time Comparison', fontsize=14, fontweight='bold')

    # 设置图例
    plt.legend(fontsize=10)

    # 设置网格
    plt.grid(True, alpha=0.3)

    # 设置y轴范围（从200开始，更好地显示差异）
    plt.ylim(200, 300)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'waiting_time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"等待时间对比图已保存到: {output_path}")
    return output_path

def create_waiting_time_plot_2():
    """
    创建等待时间折线图（高级方法）
    """
    # 数据
    dates = ['11.24', '11.25', '11.26', '11.27', '11.28', '11.29', '11.30']

    hmarl = [263.17, 245.20, 237.13, 259.24, 258.12, 232.67, 226.12]
    sarsa_saa = [265.32, 251.20, 232.13, 241.24, 233.12, 233.67, 244.12]
    cnn_ddqn = [255.32, 247.20, 231.13, 236.24, 231.12, 243.67, 238.12]
    ours = [242.32, 233.20, 228.13, 223.24, 212.12, 213.67, 217.12]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制四条折线
    plt.plot(dates, hmarl, marker='o', linewidth=2, markersize=6, label='HMARL', color='#1f77b4')
    plt.plot(dates, sarsa_saa, marker='s', linewidth=2, markersize=6, label='SARSA(SAA)', color='#ff7f0e')
    plt.plot(dates, cnn_ddqn, marker='^', linewidth=2, markersize=6, label='CNN-DDQN', color='#d62728')
    plt.plot(dates, ours, marker='D', linewidth=2, markersize=6, label='Ours', color='#2ca02c')

    # 设置坐标轴标签
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Waiting Time (seconds)', fontsize=12)

    # 设置标题
    plt.title('Average Waiting Time Comparison (Advanced Methods)', fontsize=14, fontweight='bold')

    # 设置图例
    plt.legend(fontsize=10)

    # 设置网格
    plt.grid(True, alpha=0.3)

    # 设置y轴范围
    plt.ylim(200, 280)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    output_path = 'waiting_time_comparison_2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"等待时间对比图(高级方法)已保存到: {output_path}")
    return output_path

def main():
    """主函数"""
    print("开始创建实验图表...")

    try:
        # 创建第一张图：匹配率对比（基础方法）
        plot_path1 = create_matching_rate_plot()
        print(f"第一张图创建完成: {plot_path1}")

        # 创建第二张图：匹配率对比（高级方法）
        plot_path2 = create_matching_rate_plot_2()
        print(f"第二张图创建完成: {plot_path2}")

        # 创建第三张图：平均匹配率柱状图
        plot_path3 = create_average_bar_chart()
        print(f"第三张图创建完成: {plot_path3}")

        # 创建第四张图：等待时间对比（基础方法）
        plot_path4 = create_waiting_time_plot()
        print(f"第四张图创建完成: {plot_path4}")

        # 创建第五张图：等待时间对比（高级方法）
        plot_path5 = create_waiting_time_plot_2()
        print(f"第五张图创建完成: {plot_path5}")

        # 创建第六张图：平均等待时间柱状图（不显示数值）
        plot_path6 = create_average_waiting_time_bar_chart()
        print(f"第六张图创建完成: {plot_path6}")

        # 创建第七张图：消融实验匹配率对比
        plot_path7 = create_ablation_matching_rate_plot()
        print(f"第七张图创建完成: {plot_path7}")

        # 创建第八张图：消融实验等待时间对比
        plot_path8 = create_ablation_waiting_time_plot()
        print(f"第八张图创建完成: {plot_path8}")

    except Exception as e:
        print(f"创建图表时出错: {e}")
        print("请确保已安装所需的库:")
        print("pip install matplotlib numpy")

if __name__ == "__main__":
    main()

