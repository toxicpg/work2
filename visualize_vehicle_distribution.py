"""
可视化车辆初始分布密度
展示均匀分布和不同标准差的正态分布 (σ=1,3,5,7)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def generate_vehicle_positions(num_vehicles, grid_size, distribution_type='uniform', std=4.0, seed=42):
    """
    生成车辆位置

    Args:
        num_vehicles: 车辆数量
        grid_size: 网格大小 (rows, cols)
        distribution_type: 'uniform' 或 'normal'
        std: 正态分布标准差
        seed: 随机种子

    Returns:
        positions: 车辆所在的格子ID数组
    """
    rows, cols = grid_size
    rng = np.random.default_rng(seed)

    if distribution_type == 'normal':
        # 中心点
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0

        # 生成二维正态分布
        row_pos = rng.normal(loc=center_row, scale=std, size=num_vehicles)
        col_pos = rng.normal(loc=center_col, scale=std, size=num_vehicles)

        # 裁剪到网格范围
        row_idx = np.clip(np.rint(row_pos), 0, rows-1).astype(int)
        col_idx = np.clip(np.rint(col_pos), 0, cols-1).astype(int)
        positions = row_idx * cols + col_idx
    else:
        # 均匀分布
        positions = rng.integers(0, rows * cols, num_vehicles)

    return positions


def positions_to_density_grid(positions, grid_size):
    """
    将车辆位置转换为密度网格

    Args:
        positions: 车辆所在格子ID数组
        grid_size: (rows, cols)

    Returns:
        density_grid: (rows, cols) 的密度矩阵
    """
    rows, cols = grid_size
    num_grids = rows * cols

    # 统计每个格子的车辆数
    counts = np.bincount(positions, minlength=num_grids)

    # 转换为二维网格
    density_grid = counts.reshape(rows, cols)

    return density_grid


def plot_single_distribution(ax, positions, grid_size, title):
    """
    绘制单个分布的散点图（每辆车一个点）

    Args:
        ax: matplotlib axis
        positions: 车辆所在格子ID数组
        grid_size: (rows, cols)
        title: 图标题
    """
    rows, cols = grid_size

    # 将格子ID转换为坐标，并在格子内随机偏移（避免重叠）
    row_coords = positions // cols
    col_coords = positions % cols

    # 在每个格子内添加随机偏移，使同一格子内的车辆不重叠
    rng = np.random.default_rng(42)
    row_coords = row_coords + rng.uniform(-0.4, 0.4, size=len(positions))
    col_coords = col_coords + rng.uniform(-0.4, 0.4, size=len(positions))

    # 白色背景
    ax.set_facecolor('white')

    # 绘制网格线
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)

    # 绘制车辆散点（红色）- 每个点代表一辆车
    ax.scatter(col_coords, row_coords, c='red', s=3, alpha=0.7, edgecolors='none')

    # 设置标题和标签
    ax.set_title(title, fontsize=20, fontweight='bold', pad=10)
    ax.set_xlabel('列 (X)', fontsize=20)
    ax.set_ylabel('行 (Y)', fontsize=20)

    # 设置坐标轴范围和刻度
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # 反转Y轴，使原点在左上角
    ax.set_xticks(np.arange(0, cols, 2))
    ax.set_yticks(np.arange(0, rows, 2))
    ax.tick_params(axis='both', which='major', labelsize=20)  # 设置刻度标签字体大小
    ax.set_aspect('equal')

    # 添加统计信息
    total_vehicles = len(positions)
    unique_grids = len(np.unique(positions))

    stats_text = f'总车辆: {total_vehicles}\n' \
                 f'覆盖格子: {unique_grids}/{rows*cols}\n' \
                 f'覆盖率: {unique_grids/(rows*cols)*100:.1f}%'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=20, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    return ax


def visualize_all_distributions(num_vehicles=2000, grid_size=(20, 20), seed=42):
    """
    可视化所有分布类型（散点图）
    """
    distributions = [
        ('uniform', 0.0, '均匀分布'),
        ('normal', 1.0, '正态分布 (σ=1)'),
        ('normal', 3.0, '正态分布 (σ=3)'),
        ('normal', 5.0, '正态分布 (σ=5)'),
        ('normal', 7.0, '正态分布 (σ=7)'),
    ]

    # 生成所有分布的车辆位置
    all_positions = []
    for dist_type, std, label in distributions:
        positions = generate_vehicle_positions(num_vehicles, grid_size, dist_type, std, seed)
        all_positions.append((positions, label))

    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'车辆初始分布可视化 ({num_vehicles}辆车, {grid_size[0]}×{grid_size[1]}格子)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 绘制5个子图
    for idx, (positions, label) in enumerate(all_positions):
        ax = plt.subplot(2, 3, idx + 1)
        plot_single_distribution(ax, positions, grid_size, label)

    # 第6个子图添加说明
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # 添加说明文字
    explanation = (
        '说明：\n\n'
        '• 每个红点代表一辆车\n'
        '• 均匀分布：车辆随机分布在所有格子\n'
        '• 正态分布：车辆集中在中心区域\n\n'
        '分布参数：\n'
        '• σ=1: 高度集中在中心\n'
        '• σ=3: 中等集中\n'
        '• σ=5: 适度分散\n'
        '• σ=7: 较为分散'
    )
    ax6.text(0.1, 0.7, explanation, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_comparison_histogram(num_vehicles=2000, grid_size=(20, 20), seed=42):
    """
    绘制不同分布的密度直方图对比
    """
    distributions = [
        ('uniform', 0.0, '均匀'),
        ('normal', 1.0, 'σ=1'),
        ('normal', 3.0, 'σ=3'),
        ('normal', 5.0, 'σ=5'),
        ('normal', 7.0, 'σ=7'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('车辆分布密度统计对比', fontsize=14, fontweight='bold')

    # 左图：密度分布直方图
    ax1 = axes[0]
    for dist_type, std, label in distributions:
        positions = generate_vehicle_positions(num_vehicles, grid_size, dist_type, std, seed)
        density_grid = positions_to_density_grid(positions, grid_size)

        # 统计每个格子的车辆数分布
        unique, counts = np.unique(density_grid, return_counts=True)
        ax1.plot(unique, counts, marker='o', label=label, linewidth=2)

    ax1.set_xlabel('每格车辆数', fontsize=11)
    ax1.set_ylabel('格子数量', fontsize=11)
    ax1.set_title('密度分布直方图', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图：覆盖率对比
    ax2 = axes[1]
    labels = []
    coverage_rates = []
    max_densities = []

    for dist_type, std, label in distributions:
        positions = generate_vehicle_positions(num_vehicles, grid_size, dist_type, std, seed)
        density_grid = positions_to_density_grid(positions, grid_size)

        non_zero_grids = np.sum(density_grid > 0)
        total_grids = grid_size[0] * grid_size[1]
        coverage_rate = non_zero_grids / total_grids * 100
        max_density = density_grid.max()

        labels.append(label)
        coverage_rates.append(coverage_rate)
        max_densities.append(max_density)

    x = np.arange(len(labels))
    width = 0.35

    ax2.bar(x - width/2, coverage_rates, width, label='覆盖率 (%)', color='skyblue')
    ax2.bar(x + width/2, max_densities, width, label='最大密度', color='coral')

    ax2.set_xlabel('分布类型', fontsize=11)
    ax2.set_ylabel('数值', fontsize=11)
    ax2.set_title('覆盖率与最大密度对比', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


if __name__ == '__main__':
    import os

    print("="*80)
    print("车辆初始分布密度可视化")
    print("="*80)

    # 参数设置
    NUM_VEHICLES = 2000
    GRID_SIZE = (20, 20)
    SEED = 42

    print(f"\n配置:")
    print(f"  车辆数: {NUM_VEHICLES}")
    print(f"  网格: {GRID_SIZE[0]} × {GRID_SIZE[1]}")
    print(f"  分布类型: 均匀, 正态(σ=1,3,5,7)")

    # 创建输出目录
    output_dir = 'results/visualizations/'
    os.makedirs(output_dir, exist_ok=True)

    # 定义所有分布类型
    distributions = [
        ('uniform', 0.0, '均匀分布', 'uniform'),
        ('normal', 1.0, '正态分布 (σ=1)', 'normal_std1'),
        ('normal', 3.0, '正态分布 (σ=3)', 'normal_std3'),
        ('normal', 5.0, '正态分布 (σ=5)', 'normal_std5'),
        ('normal', 7.0, '正态分布 (σ=7)', 'normal_std7'),
    ]

    # 分别生成每个分布的图片
    print("\n生成独立的分布图...")
    for dist_type, std, label, filename in distributions:
        # 生成车辆位置
        positions = generate_vehicle_positions(NUM_VEHICLES, GRID_SIZE, dist_type, std, SEED)

        # 创建单独的图形
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.patch.set_facecolor('white')

        # 绘制散点图
        plot_single_distribution(ax, positions, GRID_SIZE, label)

        # 调整布局，增加边距使网格居中
        plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)

        # 保存PNG格式 (便于预览)
        output_path_png = os.path.join(output_dir, f'vehicle_dist_{filename}.png')
        fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
        print(f"  ✓ {label}(PNG): {output_path_png}")

        # 保存TIFF格式 (适合论文发表)
        output_path_tiff = os.path.join(output_dir, f'vehicle_dist_{filename}.tiff')
        fig.savefig(output_path_tiff, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5, format='tiff')
        print(f"  ✓ {label}(TIFF): {output_path_tiff}")

        plt.close(fig)

    # 生成统计对比图
    print("\n生成统计对比图...")
    fig2 = plot_comparison_histogram(NUM_VEHICLES, GRID_SIZE, SEED)

    # 保存PNG格式
    output_path2_png = os.path.join(output_dir, 'vehicle_distribution_stats.png')
    fig2.savefig(output_path2_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存到(PNG): {output_path2_png}")

    # 保存TIFF格式
    output_path2_tiff = os.path.join(output_dir, 'vehicle_distribution_stats.tiff')
    fig2.savefig(output_path2_tiff, dpi=300, bbox_inches='tight', format='tiff')
    print(f"  ✓ 保存到(TIFF): {output_path2_tiff}")

    plt.close(fig2)

    print("\n" + "="*80)
    print("✅ 可视化完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - vehicle_dist_uniform.png (均匀分布)")
    print(f"  - vehicle_dist_normal_std1.png (σ=1)")
    print(f"  - vehicle_dist_normal_std3.png (σ=3)")
    print(f"  - vehicle_dist_normal_std5.png (σ=5)")
    print(f"  - vehicle_dist_normal_std7.png (σ=7)")
    print(f"  - vehicle_distribution_stats.png (统计对比)")
    print()

