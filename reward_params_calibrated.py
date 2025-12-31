# reward_params_calibrated.py
# 自动生成的奖励参数配置
# 生成时间: 2025-10-18 17:34:19.339809

# ==================== 推荐方案 ====================

# 方案1: 对称奖励（最简单）
REWARD_PARAMS_SIMPLE = {
    'order_matched': 0.005470,
    'order_cancelled': -0.005470,
    'base_reward': 0.000000,
}

# 方案2: 带基础奖励（推荐）
REWARD_PARAMS_RECOMMENDED = {
    'order_matched': 0.004923 ,
    'order_cancelled': -0.004923 ,
    'base_reward': 0.434028,
}

# 方案3: 非对称奖励（更激进）
REWARD_PARAMS_ASYMMETRIC = {
    'order_matched': 0.004447,
    'order_cancelled': -0.002223,
    'base_reward': 0.434028,
}

# ==================== 时间质量奖励（可选）====================
TIME_BONUS_PARAMS = {
    'fast_bonus': 0.001969,    # <120秒
    'medium_bonus': 0.000985,  # 120-300秒
    'slow_bonus': 0.0,      # >300秒
}

# ==================== Episode奖励（可选）====================
EPISODE_BONUS_PARAMS = {
    0.85: 300,   # 完成率>85%
    0.75: 100,   # 完成率>75%
    0.65: 0,     # 完成率<75%
}

# ==================== 使用建议 ====================
"""
第一阶段（基础训练）:
    使用 REWARD_PARAMS_SIMPLE 或 REWARD_PARAMS_RECOMMENDED
    不使用 TIME_BONUS 和 EPISODE_BONUS
    目标：让完成率从60%提升到70%+

第二阶段（添加时间优化）:
    保持第一阶段的参数
    添加 TIME_BONUS_PARAMS
    目标：在保持完成率的同时，降低平均等待时间

第三阶段（整体优化）:
    保持前两阶段的参数
    添加 EPISODE_BONUS_PARAMS
    目标：完成率冲击80-85%
"""
