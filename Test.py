# reward_calculator.py
"""
奖励计算器 - 整合自动校准参数
支持分阶段训练：基础 → 时间优化 → Episode优化
"""

import numpy as np
from reward_params_calibrated import (
    REWARD_PARAMS_RECOMMENDED,
    TIME_BONUS_PARAMS,
    EPISODE_BONUS_PARAMS
)


class RewardCalculator:
    """奖励计算器 - 使用自动校准的参数"""

    def __init__(self, config):
        self.config = config

        # ==================== 阶段控制开关 ====================
        # 第一阶段：只用基础奖励
        self.use_time_bonus = True  # 时间质量奖励（第二阶段打开）
        self.use_episode_bonus = True  # Episode完成率奖励（第三阶段打开）

        # ==================== 核心参数（自动校准）====================
        self.order_matched_reward = REWARD_PARAMS_RECOMMENDED['order_matched']
        self.order_cancelled_penalty = REWARD_PARAMS_RECOMMENDED['order_cancelled']
        self.base_reward = REWARD_PARAMS_RECOMMENDED['base_reward']

        # ==================== 时间质量参数 ====================
        if self.use_time_bonus:
            self.fast_bonus = TIME_BONUS_PARAMS['fast_bonus']  # <120秒
            self.medium_bonus = TIME_BONUS_PARAMS['medium_bonus']  # 120-300秒
            self.slow_bonus = TIME_BONUS_PARAMS['slow_bonus']  # >300秒
        else:
            self.fast_bonus = 0.0
            self.medium_bonus = 0.0
            self.slow_bonus = 0.0

        # ==================== Episode奖励参数 ====================
        self.episode_bonus_thresholds = EPISODE_BONUS_PARAMS if self.use_episode_bonus else {}

        # ==================== 其他参数（可选）====================
        self.idle_penalty_threshold = 0.6  # 空闲率>60%开始惩罚
        self.idle_penalty_weight = 0.5

        self.backlog_threshold = 800  # 积压>800开始惩罚
        self.backlog_penalty_weight = 0.01

        # 统计信息
        self.reset()

        # 打印配置
        self._print_config()

    def _print_config(self):
        """打印奖励配置"""
        print(f"\n{'=' * 70}")
        print(f"奖励计算器配置")
        print(f"{'=' * 70}")
        print(f"\n📊 核心参数（自动校准）:")
        print(f"  订单匹配奖励: +{self.order_matched_reward:.6f}")
        print(f"  订单取消惩罚: {self.order_cancelled_penalty:.6f}")
        print(f"  基础奖励/步: +{self.base_reward:.6f}")

        print(f"\n⏱️  时间质量奖励: {'✓ 启用' if self.use_time_bonus else '✗ 禁用'}")
        if self.use_time_bonus:
            print(f"  快速(<120秒): +{self.fast_bonus:.6f}")
            print(f"  中等(120-300秒): +{self.medium_bonus:.6f}")
            print(f"  慢速(>300秒): {self.slow_bonus:.6f}")

        print(f"\n🎯 Episode完成率奖励: {'✓ 启用' if self.use_episode_bonus else '✗ 禁用'}")
        if self.use_episode_bonus:
            for threshold, bonus in sorted(self.episode_bonus_thresholds.items(), reverse=True):
                print(f"  完成率 ≥{threshold:.0%}: +{bonus:.0f}")

        print(f"\n💡 训练建议:")
        if not self.use_time_bonus and not self.use_episode_bonus:
            print(f"  当前：第一阶段（基础训练）")
            print(f"  目标：完成率从60%提升到70%+")
            print(f"  下一步：完成后设置 use_time_bonus = True")
        elif self.use_time_bonus and not self.use_episode_bonus:
            print(f"  当前：第二阶段（时间优化）")
            print(f"  目标：降低平均等待时间")
            print(f"  下一步：稳定后设置 use_episode_bonus = True")
        else:
            print(f"  当前：第三阶段（整体优化）")
            print(f"  目标：完成率冲击80-85%")

        print(f"{'=' * 70}\n")

    def reset(self):
        """重置统计"""
        self.total_orders = 0
        self.completed_orders = 0
        self.cancelled_orders = 0
        self.waiting_times = []
        self.match_times = []  # 订单从生成到匹配的时间
        self.idle_vehicle_count = 0
        self.dispatch_success_count = 0
        self.dispatch_total_count = 0

    def update(self, step_info):
        """
        更新统计信息（每步调用）

        Args:
            step_info: 包含本步的统计信息
                - new_orders: 新生成的订单数
                - matched_orders: 本步匹配的订单数
                - cancelled_orders: 本步取消的订单数
                - waiting_times: 本步完成订单的等待时间列表
                - match_times: 订单从生成到匹配的时间列表
                - idle_vehicles: 当前空闲车辆数
                - dispatch_success: 本步成功调度数
                - dispatch_total: 本步总调度数
        """
        self.total_orders += step_info.get('new_orders', 0)
        self.completed_orders += step_info.get('matched_orders', 0)
        self.cancelled_orders += step_info.get('cancelled_orders', 0)

        if 'waiting_times' in step_info:
            self.waiting_times.extend(step_info['waiting_times'])

        if 'match_times' in step_info:
            self.match_times.extend(step_info['match_times'])

        self.idle_vehicle_count = step_info.get('idle_vehicles', 0)
        self.dispatch_success_count += step_info.get('dispatch_success', 0)
        self.dispatch_total_count += step_info.get('dispatch_total', 0)

    def calculate_step_reward(self, step_info):
        """
        计算单步即时奖励（使用自动校准的参数）

        Args:
            step_info: dict, 包含本步的统计信息
                - matched_orders: 本步匹配的订单数
                - cancelled_orders: 本步取消的订单数
                - waiting_times: 本步完成订单的等待时间列表（秒）
                - match_times: 订单从生成到匹配的时间列表（秒）
                - idle_vehicles: 当前空闲车辆数
                - pending_orders: 当前待处理订单数
                - dispatch_success: 本步成功调度数
                - dispatch_total: 本步总调度数

        Returns:
            float: 本步奖励
        """
        reward = 0.0

        # ===== 1. 基础奖励（每步都有）=====
        reward += self.base_reward

        # ===== 2. 订单匹配奖励（核心）=====
        matched = step_info.get('matched_orders', 0)
        cancelled = step_info.get('cancelled_orders', 0)

        if matched > 0:
            reward += matched * self.order_matched_reward

        if cancelled > 0:
            reward += cancelled * self.order_cancelled_penalty  # 注意：这个已经是负数

        # ===== 3. 时间质量奖励（第二阶段）=====
        if self.use_time_bonus and matched > 0:
            if 'waiting_times' in step_info:
                waiting_times = step_info['waiting_times']
                for wt in waiting_times:
                    if wt < 120:
                        reward += self.fast_bonus
                    elif wt < 300:
                        reward += self.medium_bonus
                    else:
                        reward += self.slow_bonus  # 通常是0

            # 或者用 match_times（从生成到匹配的时间）
            elif 'match_times' in step_info:
                match_times = step_info['match_times']
                for mt in match_times:
                    if mt < 120:
                        reward += self.fast_bonus
                    elif mt < 300:
                        reward += self.medium_bonus
                    else:
                        reward += self.slow_bonus

        # ===== 4. 车辆空闲惩罚（可选）=====
        idle_vehicles = step_info.get('idle_vehicles', 0)
        idle_rate = idle_vehicles / self.config.TOTAL_VEHICLES

        if idle_rate > self.idle_penalty_threshold:
            # 空闲率过高，轻微惩罚
            excess_idle = idle_rate - self.idle_penalty_threshold
            reward -= excess_idle * self.idle_penalty_weight

        # ===== 5. 订单积压惩罚（可选）=====
        pending = step_info.get('pending_orders', 0)
        if pending > self.backlog_threshold:
            # 积压过多，轻微惩罚
            excess_backlog = pending - self.backlog_threshold
            reward -= excess_backlog * self.backlog_penalty_weight

        # ===== 6. 调度效率奖励（可选）=====
        dispatch_success = step_info.get('dispatch_success', 0)
        dispatch_total = step_info.get('dispatch_total', 0)

        if dispatch_total > 0:
            efficiency = dispatch_success / dispatch_total
            # 小奖励：效率高说明匹配质量好
            reward += efficiency * 0.5 * abs(self.order_matched_reward)

        return reward

    def calculate_episode_bonus(self, episode_summary):
        """
        计算Episode完成率奖励（第三阶段使用）

        Args:
            episode_summary: dict, 包含episode的统计
                - reward_metrics: dict
                    - completion_rate: 完成率
                    - cancel_rate: 取消率

        Returns:
            float: Episode奖励（如果不启用则返回0）
        """
        if not self.use_episode_bonus:
            return 0.0

        completion_rate = episode_summary['reward_metrics']['completion_rate']

        # 根据完成率阈值给奖励
        for threshold in sorted(self.episode_bonus_thresholds.keys(), reverse=True):
            if completion_rate >= threshold:
                bonus = self.episode_bonus_thresholds[threshold]
                print(f"  🎁 Episode完成率奖励: {completion_rate:.1%} ≥ {threshold:.0%} → +{bonus:.0f}")
                return bonus

        return 0.0

    def calculate_episode_reward(self):
        """
        计算整个episode的累积奖励（用于最终评估和统计）

        注意：这个不用于训练，只用于显示和记录
        训练用的是 calculate_step_reward 的累加

        Returns:
            float: episode总奖励（标准化评分）
        """
        if self.total_orders == 0:
            return 0.0

        # 1. 完成率得分（0-100分）
        completion_rate = self.completed_orders / max(1, self.total_orders)
        completion_score = completion_rate * 100

        # 2. 取消率扣分（0到-50分）
        cancel_rate = self.cancelled_orders / max(1, self.total_orders)
        cancel_penalty = -cancel_rate * 50

        # 3. 等待时间得分（0-50分）
        if len(self.waiting_times) > 0:
            avg_wait = np.mean(self.waiting_times)
            target_wait = 120  # 2分钟

            if avg_wait <= target_wait:
                wait_score = 50
            else:
                # 超过目标，线性扣分
                wait_score = max(0, 50 - (avg_wait - target_wait) / 10)
        else:
            wait_score = 0

        # 4. 调度效率得分（0-30分）
        if self.dispatch_total_count > 0:
            efficiency = self.dispatch_success_count / self.dispatch_total_count
            efficiency_score = efficiency * 30
        else:
            efficiency_score = 0

        # 总分（理论范围：-50 到 230）
        total_score = (
                completion_score +
                cancel_penalty +
                wait_score +
                efficiency_score
        )

        return total_score

    def get_metrics(self):
        """
        获取当前统计指标

        Returns:
            dict: 包含各种指标的字典
        """
        metrics = {
            'total_orders': self.total_orders,
            'completed_orders': self.completed_orders,
            'cancelled_orders': self.cancelled_orders,
            'completion_rate': self.completed_orders / max(1, self.total_orders),
            'cancel_rate': self.cancelled_orders / max(1, self.total_orders),
        }

        if len(self.waiting_times) > 0:
            metrics['avg_waiting_time'] = np.mean(self.waiting_times)
            metrics['max_waiting_time'] = np.max(self.waiting_times)
            metrics['min_waiting_time'] = np.min(self.waiting_times)
        else:
            metrics['avg_waiting_time'] = 0
            metrics['max_waiting_time'] = 0
            metrics['min_waiting_time'] = 0

        if self.dispatch_total_count > 0:
            metrics['dispatch_efficiency'] = self.dispatch_success_count / self.dispatch_total_count
        else:
            metrics['dispatch_efficiency'] = 0

        return metrics

    def print_summary(self):
        """打印episode总结"""
        metrics = self.get_metrics()

        print(f"\n{'=' * 70}")
        print(f"Episode奖励总结")
        print(f"{'=' * 70}")
        print(f"\n📊 订单统计:")
        print(f"  总订单: {metrics['total_orders']:,}")
        print(f"  完成: {metrics['completed_orders']:,} ({metrics['completion_rate']:.1%})")
        print(f"  取消: {metrics['cancelled_orders']:,} ({metrics['cancel_rate']:.1%})")

        print(f"\n⏱️  等待时间:")
        print(f"  平均: {metrics['avg_waiting_time']:.1f}秒")
        print(f"  最大: {metrics['max_waiting_time']:.1f}秒")
        print(f"  最小: {metrics['min_waiting_time']:.1f}秒")

        print(f"\n🎯 调度效率: {metrics['dispatch_efficiency']:.1%}")

        episode_score = self.calculate_episode_reward()
        print(f"\n💰 Episode总评分: {episode_score:.2f}")
        print(f"{'=' * 70}\n")


# ==================== 使用示例 ====================
if __name__ == '__main__':
    """测试奖励计算器"""


    class MockConfig:
        TOTAL_VEHICLES = 3000


    config = MockConfig()
    calculator = RewardCalculator(config)

    # 模拟一步
    step_info = {
        'matched_orders': 1000,
        'cancelled_orders': 200,
        'waiting_times': [80, 120, 150, 200, 250],  # 秒
        'idle_vehicles': 1500,
        'pending_orders': 500,
        'dispatch_success': 900,
        'dispatch_total': 1000,
        'new_orders': 1200
    }

    # 更新统计
    calculator.update(step_info)

    # 计算奖励
    step_reward = calculator.calculate_step_reward(step_info)
    print(f"单步奖励: {step_reward:.6f}")

    # 打印指标
    print("\n当前指标:")
    metrics = calculator.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")