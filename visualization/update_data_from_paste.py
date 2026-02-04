"""
快速更新数据脚本
从粘贴的表格数据更新 data_config.py

使用方法：
1. 复制你的Excel/表格数据（3组，每组2行×7列）
2. 将数据粘贴到下方的 PASTE_DATA 变量中
3. 运行此脚本: python visualization/update_data_from_paste.py
"""

# ==================== 在这里粘贴你的数据 ====================
PASTE_DATA = """
0.8321  0.8476  0.8635  0.8774  0.8890  0.8981  0.9051
247.35  240.12  232.89  226.54  220.78  216.32  212.45

0.8701  0.8833  0.8956  0.9065  0.9158  0.9236  0.9298
218.67  212.34  206.78  201.92  197.56  193.87  190.65

0.8912  0.9021  0.9118  0.9203  0.9277  0.9340  0.9393
198.45  193.21  188.56  184.32  180.67  177.43  174.56
"""

# ============================================================


def parse_pasted_data(data_str):
    """解析粘贴的数据"""
    lines = [line.strip() for line in data_str.strip().split('\n') if line.strip()]

    groups = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            # 第1行：匹配率
            completion = [float(x) for x in lines[i].split()]
            # 第2行：等待时间
            waiting = [float(x) for x in lines[i+1].split()]

            groups.append({
                'completion_rate': completion,
                'waiting_time': waiting
            })
            i += 2
        else:
            i += 1

    return groups


def generate_data_config(groups):
    """生成 data_config.py 的内容"""

    vehicle_nums = [1800, 2000, 2200]

    config_content = '"""\n实验数据配置文件\n自动生成时间: {}\n"""\n\n'.format(
        __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    config_content += "# ==================== 主实验数据 ====================\n"
    config_content += "# 格式：每组第1行=匹配率(小数), 第2行=等待时间(秒)\n\n"

    for i, (group, vnum) in enumerate(zip(groups, vehicle_nums)):
        config_content += f"# {vnum}辆车的数据\n"
        config_content += f"data_{vnum} = {{\n"
        config_content += f"    'completion_rate': {group['completion_rate']},\n"
        config_content += f"    'waiting_time': {group['waiting_time']}\n"
        config_content += "}\n\n"

    config_content += """
# ==================== 消融实验数据 ====================
# TODO: 请填入你的消融实验数据
ablation_data = {
    'full_model': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    },
    'cnn': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    },
    'no_mgcn': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    }
}


# ==================== 初始分布实验数据 ====================
# TODO: 请填入你的初始分布实验数据
distribution_data = {
    'uniform': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    },
    'normal_std3': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    }
}


# ==================== 每日订单数 ====================
daily_orders = [5234, 5678, 6012, 5892, 5745, 6123, 6234]  # TODO: 替换
"""

    return config_content


def main():
    print('=' * 60)
    print('解析粘贴数据...')
    print('=' * 60)

    try:
        groups = parse_pasted_data(PASTE_DATA)
        print(f'✓ 成功解析 {len(groups)} 组数据')

        for i, group in enumerate(groups, 1):
            print(f'\n组 {i}:')
            print(f'  匹配率: {len(group["completion_rate"])} 个值')
            print(f'  等待时间: {len(group["waiting_time"])} 个值')

        # 生成配置文件
        config_content = generate_data_config(groups)

        # 保存
        output_path = 'visualization/data_config.py'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f'\n✓ 已更新: {output_path}')
        print('\n' + '=' * 60)
        print('✅ 数据更新完成！现在可以运行绘图脚本了')
        print('运行命令: python visualization/plot_main_results.py')
        print('=' * 60)

    except Exception as e:
        print(f'\n❌ 错误: {e}')
        print('\n请检查数据格式：')
        print('  - 每组2行（第1行=匹配率，第2行=等待时间）')
        print('  - 每行7个数值（用空格或制表符分隔）')
        print('  - 组与组之间空一行')


if __name__ == '__main__':
    main()

