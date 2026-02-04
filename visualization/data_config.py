"""
实验数据配置文件
请将你的实验结果填入这里
"""

# ==================== 主实验数据 ====================
# 格式：每组第1行=匹配率(小数), 第2行=等待时间(秒)

# 1800辆车的数据
data_1800 = {
    'completion_rate': [0.8321, 0.8476, 0.8635, 0.8774, 0.8890, 0.8981, 0.9051],
    'waiting_time': [247.35, 240.12, 232.89, 226.54, 220.78, 216.32, 212.45]
}

# 2000辆车的数据
data_2000 = {
    'completion_rate': [0.8701, 0.8833, 0.8956, 0.9065, 0.9158, 0.9236, 0.9298],
    'waiting_time': [218.67, 212.34, 206.78, 201.92, 197.56, 193.87, 190.65]
}

# 2200辆车的数据
data_2200 = {
    'completion_rate': [0.8912, 0.9021, 0.9118, 0.9203, 0.9277, 0.9340, 0.9393],
    'waiting_time': [198.45, 193.21, 188.56, 184.32, 180.67, 177.43, 174.56]
}


# ==================== 消融实验数据 ====================
# TODO: 请填入你的消融实验数据
# 格式：每个配置包含3种车辆数的数据

ablation_data = {
    'full_model': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},  # 平均值
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
    },
    'no_dueling': {
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
    'normal_std1': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    },
    'normal_std3': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    },
    'normal_std5': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    },
    'normal_std7': {
        1800: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2000: {'completion_rate': 0.0, 'waiting_time': 0.0},
        2200: {'completion_rate': 0.0, 'waiting_time': 0.0}
    }
}


# ==================== 每日订单数 ====================
# 7天的订单总数
daily_orders = [167407, 168267, 180804, 178820, 168970, 168981, 176344]

