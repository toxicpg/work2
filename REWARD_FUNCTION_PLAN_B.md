# 方案 B - 多阶段奖励函数实现指南

## 概述

方案 B 实现了**多阶段奖励** (Multi-stage Reward) 设计，在订单生命周期的**三个关键节点**分别给予反馈，以更好地引导强化学习模型优化**乘客等待时间**和**订单匹配率**。

---

## 核心设计

### 1. 三个关键节点

#### 节点 1: 匹配奖励 (Matching Reward)
- **触发时机**: 订单成功匹配时
- **奖励公式**:
  ```
  R_match = W_MATCH + W_MATCH_SPEED * exp(-wait_to_match_sec / T0)
  ```
  - `W_MATCH`: 匹配基础奖励权重 (默认 1.2)
  - `W_MATCH_SPEED`: 快速匹配奖励权重 (默认 0.5)
  - `wait_to_match_sec`: 从下单到匹配的等待时间
  - `T0`: 特征时间常数 (默认 240 秒)

- **目的**: 鼓励快速匹配，减少乘客初始等待时间

#### 节点 2: 完成奖励 (Completion Reward)
- **触发时机**: 订单完成时
- **奖励公式**:
  ```
  R_completion = W_COMPLETION + W_WAIT_SCORE * exp(-total_wait_sec / T0) - W_WAIT * (total_wait_sec / T0)
  ```
  - `W_COMPLETION`: 完成基础奖励权重 (默认 2.0)
  - `W_WAIT_SCORE`: 等待时间评分权重 (默认 0.4)
  - `W_WAIT`: 等待时间惩罚权重 (默认 1.8)
  - `total_wait_sec`: 理论等待时间 = 匹配等待时间 + 预计接驾时间

- **目的**: 根据总等待时间给予最终反馈，平衡完成和等待时间优化

#### 节点 3: 取消惩罚 (Cancellation Penalty)
- **触发时机**: 订单因超时而取消时
- **惩罚公式**:
  ```
  R_cancel = -W_CANCEL * (wait_time_sec / MAX_WAITING_TIME)
  ```
  - `W_CANCEL`: 取消惩罚权重 (默认 1.0)
  - `wait_time_sec`: 订单已等待的时间
  - `MAX_WAITING_TIME`: 最大等待时间阈值 (默认 300 秒)

- **目的**: 惩罚未能及时匹配的情况，鼓励提高匹配率

---

## 实现细节

### 文件修改

#### 1. `config.py` - 奖励权重配置

```python
REWARD_WEIGHTS = {
    'W_MATCH': 1.2,        # 匹配奖励基础权重
    'W_WAIT': 1.8,         # 等待时间惩罚权重
    'W_CANCEL': 1.0,       # 取消惩罚权重
    'W_WAIT_SCORE': 0.4,   # 等待时间评分权重
    'W_COMPLETION': 2.0,   # 订单完成奖励权重 (新增)
    'W_MATCH_SPEED': 0.5   # 快速匹配奖励权重 (新增)
}
```

#### 2. `environment.py` - 多阶段奖励计算

**匹配节点** (在 `step()` 方法中):
```python
# 在订单匹配时计算快速匹配奖励
match_speed_score = np.exp(-wait_to_match_sec / self.T0)
match_reward = w_match + w_match_speed * match_speed_score
order['match_reward'] = match_reward
```

**取消节点** (在 `_cancel_timeout_orders()` 方法中):
```python
# 在订单取消时计算惩罚
wait_time_sec = (current_time - gen_time).total_seconds()
cancel_penalty = -w_cancel * (wait_time_sec / self.config.MAX_WAITING_TIME)
order['cancel_penalty'] = cancel_penalty
```

**完成节点** (在 `_process_events()` 方法中):
```python
# 在订单完成时计算最终奖励
wait_score = np.exp(-max(0.0, total_wait_time_sec) / self.T0)
R_t = (w_completion + w_wait_score * wait_score - w_wait * (max(0.0, total_wait_time_sec) / self.T0)) * self.reward_scale
```

#### 3. `models/trainer.py` - 训练曲线记录

新增指标记录:
```python
self.completion_rates = []      # 订单完成率
self.avg_waiting_times = []     # 平均等待时间
self.match_rates = []           # 订单匹配率
self.cancel_rates = []          # 订单取消率
```

在每个 Episode 结束时记录:
```python
episode_summary = env.get_episode_summary()
reward_metrics = episode_summary.get('reward_metrics', {})

self.completion_rates.append(reward_metrics.get('completion_rate', 0.0))
self.avg_waiting_times.append(reward_metrics.get('avg_waiting_time', 0.0))
self.match_rates.append(reward_metrics.get('match_rate', 0.0))
self.cancel_rates.append(reward_metrics.get('cancel_rate', 0.0))
```

---

## 理论等待时间定义

**理论等待时间** = 匹配等待时间 + 预计接驾时间

```
total_wait_time_sec = wait_to_match_sec + wait_for_pickup_sec

其中:
  - wait_to_match_sec: 从下单到司机接单的时间
  - wait_for_pickup_sec: 司机从当前位置到乘客位置的预计时间
```

这个定义确保了:
1. **公平性**: 同一订单在不同调度决策下可比较
2. **稳定性**: 不依赖于实际接驾时间的变化
3. **可优化性**: 模型可以通过优化调度位置来减少接驾时间

---

## 强化学习训练曲线

### 关键指标

#### 1. Loss (训练损失)
- **含义**: 模型预测 Q 值与目标 Q 值之间的误差
- **期望**: 应该逐渐下降，表示模型收敛
- **观察**: 如果 Loss 始终很高或不下降，可能是学习率过高或奖励信号不稳定

#### 2. Reward (累积奖励)
- **含义**: 每个 Episode 的总奖励（通常是总收入）
- **期望**: 应该上升或稳定在高水平，表示策略改进
- **观察**: 如果 Reward 下降，可能是模型过拟合或探索不足

#### 3. Completion Rate (完成率)
- **含义**: 已完成订单数 / (已完成订单数 + 已取消订单数)
- **期望**: 应该上升并稳定在高水平 (理想 > 90%)
- **观察**: 这是优化目标之一，应该持续改进

#### 4. Average Waiting Time (平均等待时间)
- **含义**: 所有已完成订单的平均等待时间
- **期望**: 应该下降，表示乘客等待时间减少
- **观察**: 这是主要优化目标，应该持续下降

#### 5. Match Rate (匹配率)
- **含义**: 已匹配订单数 / (已匹配订单数 + 已取消订单数)
- **期望**: 应该上升并稳定在高水平
- **观察**: 反映调度策略的有效性

#### 6. Epsilon (探索率)
- **含义**: 随机探索的概率
- **期望**: 应该逐渐衰减 (从 0.6 → 0.1)
- **观察**: 衰减过快可能导致探索不足，衰减过慢可能导致学习缓慢

---

## 超参数调整指南

### 1. 奖励权重调整

**增加 `W_MATCH`** (默认 1.2):
- 效果: 更强调快速匹配
- 适用场景: 当匹配率低时

**增加 `W_COMPLETION`** (默认 2.0):
- 效果: 更强调订单完成
- 适用场景: 当完成率低时

**增加 `W_WAIT_SCORE`** (默认 0.4):
- 效果: 更奖励短等待时间
- 适用场景: 当等待时间不下降时

**增加 `W_WAIT`** (默认 1.8):
- 效果: 更惩罚长等待时间
- 适用场景: 当等待时间仍然很长时

**增加 `W_CANCEL`** (默认 1.0):
- 效果: 更严厉地惩罚取消
- 适用场景: 当取消率高时

### 2. 时间常数调整

**`T_CHARACTERISTIC`** (默认 240 秒):
- 含义: 指数衰减的特征时间
- 增大: 对长等待时间的惩罚减轻
- 减小: 对长等待时间的惩罚加重

### 3. 探索率调整

**`EPSILON_START`** (默认 0.6):
- 含义: 初始探索概率
- 增大: 初期探索更充分，但可能导致训练不稳定

**`EPSILON_DECAY`** (默认 0.85):
- 含义: 每个 Episode 的衰减因子
- 增大 (接近 1.0): 探索衰减缓慢，训练时间更长
- 减小 (接近 0.5): 探索衰减快速，可能过早收敛

---

## 使用训练曲线可视化工具

### 运行可视化脚本

```bash
python plot_training_curves.py
```

### 输出

脚本会生成以下图表，保存在 `results/plots/` 目录:

1. **training_curves.png**: 包含 4 个子图
   - 累积奖励曲线 (Total Reward)
   - 训练损失曲线 (Loss)
   - 探索率衰减曲线 (Epsilon)
   - 业务指标曲线 (Completion Rate, Match Rate, Cancel Rate)

2. **waiting_time_curve.png**: 平均等待时间曲线

### 解读曲线

**理想的训练曲线应该表现为**:
- Loss: 逐渐下降，最终稳定在较低水平
- Reward: 逐渐上升或稳定在高水平
- Epsilon: 平滑衰减到 0.1
- Completion Rate: 快速上升到 85%+ 并保持
- Average Waiting Time: 逐渐下降
- Match Rate: 快速上升到 90%+ 并保持

---

## 训练建议

### 阶段 1: 初期训练 (前 10-20 个 Episode)
- **目标**: 快速提高匹配率和完成率
- **监控**: 完成率应快速上升到 80%+
- **调整**: 如果完成率不升，增加 `W_MATCH` 或 `W_COMPLETION`

### 阶段 2: 中期优化 (20-40 个 Episode)
- **目标**: 在保持高完成率的同时，降低等待时间
- **监控**: 等待时间应稳定下降
- **调整**: 如果等待时间不下降，增加 `W_WAIT_SCORE` 或减小 `T_CHARACTERISTIC`

### 阶段 3: 后期微调 (40+ 个 Episode)
- **目标**: 达到稳定的最优策略
- **监控**: 所有指标应稳定在理想水平
- **调整**: 微调权重以达到最佳平衡

---

## 常见问题

### Q1: Loss 始终很高怎么办?
**A**:
- 检查奖励信号是否稳定（应该在合理范围内）
- 尝试降低学习率
- 检查 Replay Buffer 大小是否足够

### Q2: 完成率不上升怎么办?
**A**:
- 增加 `W_MATCH` 或 `W_COMPLETION`
- 检查匹配器是否正常工作
- 增加训练轮数

### Q3: 等待时间不下降怎么办?
**A**:
- 增加 `W_WAIT_SCORE` 或 `W_WAIT`
- 减小 `T_CHARACTERISTIC` 以加强对长等待的惩罚
- 检查调度策略是否合理

### Q4: 模型过拟合怎么办?
**A**:
- 增加 `EPSILON_START` 以保持更多探索
- 减小 `EPSILON_DECAY` 以缓慢衰减探索率
- 增加训练数据多样性

---

## 参考文献

- Dueling DQN: https://arxiv.org/abs/1511.06581
- Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
- Multi-task Learning: https://arxiv.org/abs/1707.08114

---

**版本**: 方案 B (V1.0)
**最后更新**: 2026-01-01

