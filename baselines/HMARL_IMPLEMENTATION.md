# H-MARL (MFuN) 实现说明

## 论文来源
**Si et al. - Hierarchical Multi-Agent RL for Intercity Ridepooling**

---

## 核心架构

### 1. 两层分层框架

```
┌─────────────────────────────────────────────┐
│  上层（战略层）：Manager Network              │
│  决策频率：每 5 分钟                          │
│  输出：子目标 (Sub-goals)                    │
│  内容：各grid期望调入的车辆数                 │
└─────────────────┬───────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────┐
│  下层（战术层）：Worker Network (参数共享)    │
│  决策频率：每 30 秒 (每个tick)                │
│  输出：具体调度动作                           │
│  内容：从哪个grid调车到哪个grid               │
└─────────────────────────────────────────────┘
```

### 2. 核心创新点

#### (1) 封建制度的层级结构
- **Manager（管理者）**：全局视角，协调整体资源分配
- **Worker（工人）**：局部视角，执行具体调度任务
- **参数共享**：所有Worker共享网络参数，减少训练复杂度

#### (2) 内部奖励机制（Intrinsic Reward）
```python
# 鼓励Worker完成Manager的子目标
r_intrinsic = -||实际调度 - 子目标||²

# 例如：Manager要求调入3辆车，Worker实际调入2辆
r_intrinsic = -(3-2)² = -1
```

#### (3) 双重奖励系统
- **外部奖励（External Reward）**：订单收益、完成率等业务指标
- **内部奖励（Intrinsic Reward）**：完成Manager子目标的程度
- **总奖励**：`R_total = R_external + α * R_intrinsic`

---

## 网络结构详解

### Manager Network

**输入：全局状态**
```python
global_state = [
    # 对每个grid (共400个):
    idle_vehicles,    # 空闲车辆数
    busy_vehicles,    # 繁忙车辆数
    pending_orders,   # 待处理订单数
    time_sin,         # 时间特征 (sin)
    time_cos          # 时间特征 (cos)
]
# 总维度: 400 grids × 5 features = 2000
```

**网络结构：**
```python
Input (2000)
  → GRU(256, 2 layers)
  → FC(512) → ReLU → Dropout
  → FC(256) → ReLU → Dropout
  → Sub-goals Output (400)  # 每个grid的期望调入车辆数
  → Value Output (1)         # 状态价值（用于Actor-Critic）
```

**输出：子目标**
```python
sub_goals = [sg_0, sg_1, ..., sg_399]  # 每个grid的目标
# sg_i > 0: 期望调入 sg_i 辆车
# sg_i < 0: 期望调出 |sg_i| 辆车
# sg_i ≈ 0: 保持当前状态
```

---

### Worker Network

**输入：局部状态 + 子目标**
```python
local_state = [
    idle_vehicles,    # 当前grid的空闲车辆数
    busy_vehicles,    # 当前grid的繁忙车辆数
    pending_orders,   # 当前grid的订单数
    time_sin,         # 时间特征
    time_cos
]
sub_goal = sg_i       # Manager分配给当前grid的子目标

combined_input = concat(local_state, sub_goal)  # 维度: 6
```

**网络结构：**
```python
Input (6)
  → GRU(128, 1 layer)
  → FC(128) → ReLU → Dropout
  → FC(64) → ReLU → Dropout
  → Action Logits (400)  # 调度到各grid的概率分布
  → Value Output (1)      # 状态价值
```

**输出：动作分布**
```python
action_logits = [a_0, a_1, ..., a_399]
action_probs = softmax(action_logits)
# 选择概率最高的grid作为调度目标
```

---

## 训练流程

### 1. Episode 循环

```python
for episode in range(num_episodes):
    # 1. 随机选择训练日期
    current_day = random.choice(train_days)
    day_orders = load_orders(current_day)

    # 2. 初始化环境
    env = BaselineEnvironment(config, day_orders, policy='none')
    env.reset()

    # 3. 重置Agent状态
    agent.reset_episode()

    # 4. 仿真循环
    for tick in range(MAX_TICKS):
        # Step 1: Manager决策（每5分钟）
        if tick % manager_interval == 0:
            sub_goals = manager.forward(global_state)

        # Step 2: Worker执行（每tick）
        for grid_id in range(num_grids):
            local_state = get_local_state(env, grid_id)
            action = worker.forward(local_state, sub_goals[grid_id])
            dispatch(action)

        # Step 3: 环境推进
        env.step()

        # Step 4: 计算奖励
        external_reward = compute_business_metrics(env)
        intrinsic_reward = compute_goal_completion(dispatch_actions, sub_goals)
        total_reward = external_reward + intrinsic_reward

        # Step 5: 更新网络
        manager.update()
        worker.update()
```

### 2. 奖励计算

**外部奖励（业务指标）：**
```python
external_reward = (
    completed_orders * 10      # 完成订单奖励
    - cancelled_orders * 5     # 取消订单惩罚
    - empty_distance * 0.5     # 空驶惩罚
)
```

**内部奖励（目标完成度）：**
```python
intrinsic_reward = 0
for grid_id in range(num_grids):
    # 统计实际调度变化
    actual_change = (调入车辆数 - 调出车辆数)

    # 与子目标对比
    target_change = sub_goals[grid_id]

    # 惩罚偏差
    intrinsic_reward -= (actual_change - target_change) ** 2

# 缩放
intrinsic_reward *= 0.01
```

### 3. 早停机制

```python
if episode_reward > best_reward:
    best_reward = episode_reward
    early_stopping_counter = 0
    save_best_model()
else:
    early_stopping_counter += 1
    if early_stopping_counter >= patience:
        print("Early stopping triggered!")
        break
```

---

## 与主算法的区别

| 特性 | 主算法 (DDQN + MGCN) | H-MARL (MFuN) |
|------|---------------------|---------------|
| **架构** | 单层强化学习 | 两层分层强化学习 |
| **决策粒度** | 车辆级别 | Grid级别（聚合） |
| **状态表示** | MGCN图神经网络 | GRU时序编码 |
| **动作空间** | 179维离散动作 | 400维调度分布 |
| **奖励机制** | 单一外部奖励 | 外部 + 内部双重奖励 |
| **训练复杂度** | 高（大状态空间） | 中（参数共享） |
| **适用场景** | 精细化调度 | 大规模宏观调度 |

---

## 关键超参数

```python
# Manager
MANAGER_LR = 0.001
MANAGER_HIDDEN_SIZE = 256
MANAGER_UPDATE_FREQ = 5 * 60  # 5分钟

# Worker
WORKER_LR = 0.0005
WORKER_HIDDEN_SIZE = 128
WORKER_UPDATE_FREQ = 30  # 30秒

# 训练
NUM_EPISODES = 10  # 演示用，论文建议50+
GAMMA = 0.99
EARLY_STOPPING_PATIENCE = 3

# 奖励
EXTERNAL_REWARD_SCALE = 1.0
INTRINSIC_REWARD_SCALE = 0.01
```

---

## 代码文件

1. **`hmarl_agent.py`** - Agent实现
   - `ManagerNetwork`: Manager网络
   - `WorkerNetwork`: Worker网络
   - `MFuN_Agent`: 完整Agent类

2. **`train_hmarl.py`** - 训练脚本
   - 数据加载
   - 训练循环
   - 奖励计算
   - 早停机制

---

## 使用方法

### 训练

```bash
python baselines/train_hmarl.py
```

### 关键输出

```
Episode 1/10 结果:
======================================================================
  总奖励: 1234.56
  外部奖励: 1200.00
  内部奖励: 34.56
  调度次数: 150
  匹配率: 68.50%
  完成率: 65.20%
  平均等待时间: 185.3秒
======================================================================
```

### 模型保存

- **训练中最佳模型**：`baselines/checkpoints/mfun_manager_best.pth`
- **最终模型**：`baselines/mfun_manager.pth`, `baselines/mfun_worker.pth`

---

## 论文与实现的对应关系

| 论文组件 | 实现位置 |
|---------|---------|
| Manager Network | `ManagerNetwork` 类 |
| Worker Network | `WorkerNetwork` 类 |
| Sub-goal Generation | `manager.forward()` |
| Action Selection | `worker.forward()` |
| Intrinsic Reward | `agent.compute_intrinsic_reward()` |
| Hierarchical Control | `agent.select_action()` |
| Parameter Sharing | `worker_shared` 网络 |

---

## 实验建议

### 1. 对比实验
- **H-MARL vs. Random Walk**：体现层级决策的优势
- **H-MARL vs. DDQN**：对比分层与单层的性能

### 2. 消融实验
- **去除内部奖励**：验证双重奖励机制的有效性
- **不共享Worker参数**：验证参数共享的重要性
- **固定子目标**：验证Manager的必要性

### 3. 超参数敏感性
- Manager决策频率：3分钟 vs. 5分钟 vs. 10分钟
- 内部奖励权重：0.001 vs. 0.01 vs. 0.1
- GRU层数：1层 vs. 2层 vs. 3层

---

## 注意事项

1. **内存占用**：GRU会维护隐藏状态，注意显存管理
2. **训练时间**：分层训练比单层慢，建议使用GPU
3. **收敛速度**：Manager和Worker的学习率需要平衡
4. **状态同步**：确保Manager和Worker使用一致的状态表示

---

## ✅ 已实现的论文组件

### 1. MILP求解器（Worker动作映射）

**位置**: `hmarl_agent.py -> solve_milp_dispatch()`

**功能**:
- 将Worker的调度意图（概率分布）转化为具体的车辆分配方案
- 考虑需求预测、调度成本、Manager子目标约束

**公式**:
```python
max  Σ x[v,g] * (demand[g] * prob[g] - distance[v,g] * 0.1)
s.t. Σ x[v,g] ≤ 1  (每辆车最多去一个grid)
     Σ x[v,g] ∈ [0.8*target, 1.2*target]  (满足子目标)
```

### 2. ALNS优化器（下层路径优化）

**位置**: `hmarl_agent.py -> alns_optimize_routes()`

**功能**:
- 破坏算子：随机移除20%的调度
- 修复算子：贪婪插入到需求最高且距离适中的grid

**流程**:
```python
1. Destroy: 移除部分车辆的调度
2. Repair: 重新分配到最优位置
3. Evaluate: 计算 score = demand*2.0 - distance*0.5
```

---

## 未来改进方向

1. ✅ **ALNS集成** - 已完成简化版
2. ✅ **MILP辅助** - 已集成到Worker决策
3. **通信机制**：增加Worker之间的信息交换
4. **注意力机制**：在Manager中引入Attention机制
5. **课程学习**：从简单场景逐步过渡到复杂场景
6. **完整ALNS**：实现多种破坏/修复算子，自适应权重调整

---

**作者**: CatPaw AI Assistant
**日期**: 2026-01-31
**版本**: 1.0

