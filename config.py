# config.py
import torch
import os
import json


class Config:
    # ================== 数据配置 ==================
    GRID_SIZE = (20, 20)
    NUM_GRIDS = 400
    TIME_WINDOW_MINUTES = 10 # 用于 prepare_mgcn_input 计算近期订单
    # 10分钟的时间片，用于 OrderGenerator 批量加载订单
    MACRO_STATISTICS_STEP_MINUTES = 10
    DATA_START_DATE = '2016-11-01 00:00:00'
    DATA_END_DATE = '2016-11-29 23:59:59' # 确保包含最后一天的数据
    # ... (LAT_RANGE, LON_RANGE 不变) ...

    # ================== 模拟器时间配置 (V5.2 - 优化版) ==================
    # (NUM_TIME_SLICES 用于 data_process.py 计算时间特征)
    NUM_TIME_SLICES = (24 * 60) // MACRO_STATISTICS_STEP_MINUTES # 每天 144 个宏观时间片
    TICK_DURATION_SEC = 60  # 模拟器的"心跳"间隔 (60秒 = 1分钟，加快2倍)
    TICKS_PER_DAY = (24 * 60 * 60) // TICK_DURATION_SEC  # 每天 1440 个 Ticks

    EPISODE_DAYS = 1  # 优化: 从2天改为1天，再加快2倍
    MAX_TICKS_PER_EPISODE = EPISODE_DAYS * TICKS_PER_DAY  # 每个 Episode 的最大 Ticks 数

    MAX_START_DAY = None  # 将在 OrderGenerator 中动态设置

    # ================== 模型配置 (INPUT_DIM=5) ==================
    # [订单数, 空闲车辆数, 繁忙车辆数, time_sin, time_cos]
    INPUT_DIM = 5
    # =========================================================

    HIDDEN_DIMS = [64, 32]
    MGCN_FUSION_TYPE = 'attention'
    GRAPH_MODE = 'neighbor_only'  # 可选: 'both' | 'neighbor_only' | 'poi_only'

    # Embedding配置
    POSITION_EMB_DIM = 16
    DAY_EMB_DIM = 8

    # 融合网络配置
    FUSION_HIDDEN_DIM = 128
    FINAL_HIDDEN_DIM = 64
    DROPOUT_RATE = 0.5  # 大幅增加 Dropout 防止过拟合
    USE_BATCH_NORM = True  # 添加 Batch Normalization 提高稳定性

    # 输出配置
    PROCESSED_DATA_PATH = 'data/processed/'

    # 动态加载动作数
    _mapping_file = os.path.join(PROCESSED_DATA_PATH, 'action_mapping.json')
    if os.path.exists(_mapping_file):
        with open(_mapping_file, 'r') as f:
            _mapping = json.load(f)
        NUM_ACTIONS = len(_mapping)
        print(f"自动设置 NUM_ACTIONS = {NUM_ACTIONS}")
    else:
        print(f"警告: 动作映射文件 {_mapping_file} 不存在, 使用默认值 179")
        NUM_ACTIONS = 179

    # 图配置
    GCN_KERNEL_TYPE = 'localpool'
    GCN_K = 1

    # ================== 设备配置 ==================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("=" * 70);
        print("GPU优化配置:");
        try:
            print(f"  设备: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        except Exception as e:
            print(f"  无法获取 GPU 详细信息: {e}")
        print("=" * 70)

    # ================== 训练配置 ==================
    LEARNING_RATE = 1e-5  # 进一步降低学习率，防止过快过拟合
    WEIGHT_DECAY = 5e-4  # 大幅增加正则化，更强的防过拟合
    GAMMA = 0.92  # 进一步降低折扣因子，减少Q值积累
    TARGET_UPDATE_FREQ = 200  # 更频繁更新目标网络，提高稳定性
    GRAD_CLIP_NORM = 1.0  # 梯度裁剪，防止梯度爆炸

    EPSILON_START = 0.6
    EPSILON_END = 0.05  # 优化: 从 0.1 改为 0.05，保留更多探索空间
    EPSILON_DECAY = 0.95  # 优化: 从 0.85 改为 0.95，更平缓的衰减

    REPLAY_BUFFER_SIZE = 30000  # 优化: 从50000减少到30000，加快训练
    MIN_REPLAY_SIZE = 2000  # 优化: 从5000降到2000，更快开始训练
    BATCH_SIZE = 256  # 优化: 从128增加到256，充分利用GPU，加快训练

    # V5 训练循环配置 (优化版本 - 参考 TRAINING_ACCELERATION_GUIDE.md)
    TRAIN_EVERY_N_TICKS = 10  # 优化: 从 30 改为 10 (配合60秒tick，每10分钟训练一次)
    TRAIN_LOOPS_PER_BATCH = 8  # 优化: 从 4 改为 8 (增加单次训练深度，补偿训练频率降低)

    # 进度显示配置
    SHOW_PROGRESS_EVERY_N_TICKS = 50  # 每50个tick(50分钟)显示一次进度

    # ================== PER配置 ==================
    PER_ALPHA = 0.4
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 100000

    # ================== 训练过程配置 ==================
    NUM_EPISODES = 50
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    VALIDATION_INTERVAL = 1  # 每1个episode验证，更及时发现过拟合
    VAL_EPISODES = 2
    TEST_EPISODES = 1
    SAVE_FREQ = 4
    LOG_FREQ = 2

    EARLY_STOPPING_PATIENCE = 5  # 增加patience，给模型更多收敛机会
    RAW_DATA_PATH = 'data/raw/'
    ORDER_FILE = 'orders.csv'
    NEIGHBOR_ADJ_FILE = 'neighbor_adj.pt'
    POI_ADJ_FILE = 'poi_adj.pt'

    # ================== 仿真配置 ==================
    TOTAL_VEHICLES = 1800  # 减少到1800辆，增加调度挑战性（供需比0.65）

    # ================== 保存路径配置 (基于车辆数) ==================
    MODEL_SAVE_PATH = f'results/models/vehicles_{TOTAL_VEHICLES}/'  # 按车辆数分目录
    LOG_SAVE_PATH = f'results/logs/vehicles_{TOTAL_VEHICLES}/'      # 日志也分目录
    IDLE_THRESHOLD_SEC = 120
    AVG_SPEED_KMH = 40
    MAX_WAITING_TIME = 300  # 订单等待 600 秒 (10分钟) 后取消
    MATCHER_SEARCH_RADIUS = 10      # 订单匹配半径 (V5.5 新增)
    MATCHER_KNN_K = 30
    MATCHER_TRAVEL_TIME_WEIGHT = 1.5
    MATCHER_EUCLIDEAN_WEIGHT = 0.1


    # ================== 奖励函数配置 (V5.3 - 优化版) ==================
    # 核心改进:
    # 1. 缩短T0使奖励信号更明显
    # 2. 平衡各项权重,避免某一项主导
    # 3. 增强等待时间的重要性
    REWARD_FORMULA_V4 = {
        'T_CHARACTERISTIC': 180.0  # 从240降到180,使exp函数变化更明显
    }
    # (可以选择性地使用缩放因子调整奖励大小)
    REWARD_SCALE_FACTOR = 1.0

    # ================== 多阶段奖励权重 (方案 B - 优化版) ==================
    # 匹配奖励: 订单成功匹配时立即给予
    # 完成奖励: 订单完成时根据总等待时间给予
    # 取消惩罚: 订单因超时取消时给予惩罚
    #
    # 优化原则:
    # 1. 降低completion基础权重,避免"只要完成就好"的策略
    # 2. 提高wait相关权重,让模型更关注等待时间
    # 3. 增加wait_score权重,奖励快速服务
    REWARD_WEIGHTS = {
        'W_MATCH': 0.8,        # 匹配奖励基础权重 (降低,避免过度重视匹配)
        'W_WAIT': 3.0,         # 等待时间惩罚权重 (大幅提升,强化时间敏感性)
        'W_CANCEL': 2.5,       # 取消惩罚权重 (提高,避免让订单超时)
        'W_WAIT_SCORE': 1.5,   # 等待时间评分权重 (提高,奖励快速服务)
        'W_COMPLETION': 1.0,   # 订单完成奖励权重 (降低,避免主导策略)
        'W_MATCH_SPEED': 0.8   # 快速匹配奖励权重 (稍降,平衡各项)
    }
    # =======================================================================

    # ================== 调试配置 ==================
    DEBUG = False  # 正式训练时设为 False
    VERBOSE = False
    SEED = 42

    @classmethod
    def validate_config(cls):
        """验证配置参数的合理性 (已更新为 V5.2)"""
        errors = []
        if cls.NUM_GRIDS != cls.GRID_SIZE[0] * cls.GRID_SIZE[1]: errors.append("网格数量不匹配")

        # ===== V5 验证 =====
        expected_input_dim = 5
        if cls.INPUT_DIM != expected_input_dim: errors.append(f"输入维度应为 {expected_input_dim}")
        if cls.MACRO_STATISTICS_STEP_MINUTES <= 0: errors.append("MACRO_STATISTICS_STEP_MINUTES 必须大于 0")
        if cls.TICK_DURATION_SEC <= 0: errors.append("TICK_DURATION_SEC 必须大于 0")
        if cls.MAX_TICKS_PER_EPISODE != cls.EPISODE_DAYS * (24 * 60 * 60) // cls.TICK_DURATION_SEC: errors.append(f"MAX_TICKS_PER_EPISODE 计算错误")
        if cls.IDLE_THRESHOLD_SEC < cls.TICK_DURATION_SEC: errors.append("空闲阈值必须大于 Tick 间隔")
        # ======================

        # ===== V5.2 奖励验证 =====
        if not hasattr(cls, 'REWARD_FORMULA_V4'): errors.append("缺少 REWARD_FORMULA_V4 奖励配置字典")
        elif not isinstance(cls.REWARD_FORMULA_V4, dict): errors.append("REWARD_FORMULA_V4 必须是一个字典")
        elif 'T_CHARACTERISTIC' not in cls.REWARD_FORMULA_V4: errors.append("REWARD_FORMULA_V4 缺少 'T_CHARACTERISTIC' 参数")
        elif cls.REWARD_FORMULA_V4['T_CHARACTERISTIC'] <= 0: errors.append("T_CHARACTERISTIC 必须大于 0")
        # =========================

        if errors:
            print("❌ 配置验证失败:")
            for error in errors: print(f"  - {error}")
            return False
        else:
            print("✓ 配置验证通过!")
            return True
