# 第 3 章 方法论（续）- 3.4 到 3.5

## 3.4 环境模拟器

### 3.4.1 模拟器的设计理念与架构

网约车调度问题的训练和评估需要一个**真实、高效、可控**的环境模拟器。我们设计的模拟器具有以下特点：

#### 3.4.1.1 设计原则

**原则 1：事件驱动架构**

传统的时间步驱动模拟器在每个固定时间间隔都进行状态更新，即使没有事件发生也会消耗计算资源。我们采用**事件驱动架构**：

- **订单事件**：新订单到达、订单匹配、订单完成、订单超时取消
- **车辆事件**：车辆抵达目的地、车辆状态转换
- **调度事件**：系统做出调度决策、车辆被调度到新位置

只有当这些事件发生时，模拟器才进行相应的状态更新。这种设计能够显著提高模拟效率，使得我们能够快速模拟数天甚至数周的运营数据。

**原则 2：真实性与可控性的平衡**

模拟器需要：
- **真实性**：基于真实数据集（如北京滴滴数据），反映真实的订单分布、用户行为
- **可控性**：允许调整关键参数（如车辆数量、订单生成速率），进行对照实验

我们通过以下方式实现这种平衡：
- 使用真实数据集作为基础
- 提供参数化接口，允许调整模拟参数
- 支持多种订单生成策略（真实数据回放、合成数据生成）

**原则 3：计算效率**

模拟器需要能够在合理的时间内完成大规模模拟。关键优化包括：
- 使用 K-D 树进行高效的空间搜索（订单匹配）
- 使用网格化表示而非精确坐标
- 使用 NumPy 向量化操作而非 Python 循环
- 使用 C++ 或 Cython 加速关键路径

#### 3.4.1.2 模拟器的总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  模拟器 (Simulator)                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  事件队列 (Event Queue)                              │   │
│  │  ├─ 订单事件：(时间, 订单ID, 事件类型)              │   │
│  │  ├─ 车辆事件：(时间, 车辆ID, 事件类型)              │   │
│  │  └─ 调度事件：(时间, 调度ID, 目标网格)              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  订单管理器 (OrderManager)                            │   │
│  │  ├─ 订单库：所有订单的状态和信息                     │   │
│  │  ├─ 订单生成：根据数据集生成订单                     │   │
│  │  └─ 订单匹配：基于车辆位置和时间预测                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  车辆管理器 (VehicleManager)                          │   │
│  │  ├─ 车辆库：1800 辆车的状态和位置                    │   │
│  │  ├─ 状态转换：空闲 → 调度中 → 执行中 → 空闲        │   │
│  │  └─ 位置更新：基于速度和时间的位置推进              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  调度决策执行器 (DispatchExecutor)                    │   │
│  │  ├─ 接收 RL 模型的调度决策                           │   │
│  │  ├─ 验证决策的合法性                                 │   │
│  │  └─ 执行调度操作                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  状态观测器 (StateObserver)                           │   │
│  │  ├─ 聚合网格级别的信息                               │   │
│  │  ├─ 生成 RL 模型的输入状态                           │   │
│  │  └─ 时间编码                                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  奖励计算器 (RewardCalculator)                        │   │
│  │  ├─ 计算多阶段奖励                                   │   │
│  │  ├─ 统计业务指标                                     │   │
│  │  └─ 记录训练曲线数据                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4.2 订单管理器 (OrderManager)

#### 3.4.2.1 订单的生命周期

每个订单在系统中经历以下状态转换：

```
┌─────────┐
│ 创建    │ (订单在系统中生成)
└────┬────┘
     │ (等待匹配)
     ▼
┌─────────┐
│ 待匹配  │ (在缓冲区中等待司机接单)
└────┬────┘
     │ (被司机接单)
     ▼
┌─────────┐
│ 已匹配  │ (司机正在前往乘客位置)
└────┬────┘
     │ (司机到达乘客位置，乘客上车)
     ▼
┌─────────┐
│ 执行中  │ (司机正在送乘客到目的地)
└────┬────┘
     │ (到达目的地)
     ▼
┌─────────┐
│ 已完成  │ (订单成功完成)
└─────────┘

或者

┌─────────┐
│ 待匹配  │
└────┬────┘
     │ (超时，无司机接单)
     ▼
┌─────────┐
│ 已取消  │ (订单被取消)
└─────────┘
```

#### 3.4.2.2 订单数据结构

每个订单包含以下信息：

```python
class Order:
    def __init__(self):
        # 基本信息
        self.order_id: int              # 订单唯一 ID
        self.created_time: float        # 订单创建时间（秒）
        self.origin_grid: int           # 出发地网格 ID (0-399)
        self.destination_grid: int      # 目的地网格 ID (0-399)

        # 乘客信息
        self.passenger_id: int          # 乘客 ID

        # 状态信息
        self.status: str                # 订单状态：created, waiting, matched, executing, completed, cancelled
        self.matched_time: float = None # 匹配时间
        self.matched_vehicle_id: int = None  # 匹配的车辆 ID
        self.completed_time: float = None    # 完成时间
        self.cancelled_time: float = None    # 取消时间

        # 时间相关
        self.timeout_threshold: float = 600  # 超时阈值（秒）

        # 奖励相关
        self.reward_accumulated: float = 0.0  # 累积奖励
```

#### 3.4.2.3 订单生成过程

订单生成基于真实数据集，具体过程如下：

**步骤 1：数据预处理**

从原始数据集中提取订单信息：
- 订单的时间戳（精确到秒）
- 订单的起点和终点坐标
- 将坐标映射到 $20 \times 20$ 网格

**步骤 2：时间对齐**

将所有订单的时间戳对齐到模拟时间轴：
- 假设原始数据集的第一个订单时间为 $t_0$
- 模拟时间轴从 $t = 0$ 开始
- 每个订单的相对时间 $t_i = t_i^{\text{original}} - t_0$

**步骤 3：订单加载**

在模拟初始化时，将所有订单加载到内存中，按时间排序。

**步骤 4：订单到达**

在模拟运行过程中，当模拟时间达到订单的创建时间时，订单被激活，进入"待匹配"状态。

#### 3.4.2.4 订单匹配算法

订单匹配是一个关键的操作，直接影响模拟的真实性和效率。

**匹配策略**

我们采用**贪心匹配策略**：

1. **候选车辆选择**：找到距离订单起点最近的 $K$ 个空闲车辆（通常 $K = 10$）
2. **可达性检查**：检查车辆是否能在超时时间内到达乘客位置
3. **最优车辆选择**：在可达的车辆中，选择预计到达时间最短的车辆

**数学表述**

对于订单 $o$，其候选车辆集合为：

$$\mathcal{V}_{\text{candidate}} = \arg\min_v \text{distance}(v_{\text{loc}}, o_{\text{origin}}) \quad (|\mathcal{V}_{\text{candidate}}| = K)$$

对于每个候选车辆 $v$，计算其预计到达时间：

$$t_{\text{arrival}}(v, o) = \frac{\text{distance}(v_{\text{loc}}, o_{\text{origin}})}{\text{average\_speed}}$$

可达性条件：

$$t_{\text{arrival}}(v, o) < o_{\text{timeout}}$$

最优车辆选择：

$$v^* = \arg\min_v t_{\text{arrival}}(v, o) \quad (v \in \mathcal{V}_{\text{candidate}}, \text{ 满足可达性条件})$$

**高效实现：K-D 树**

为了高效地找到距离最近的 $K$ 个车辆，我们使用 **K-D 树** (K-Dimensional Tree) 数据结构：

- **时间复杂度**：$O(\log N)$（其中 $N$ 是车辆总数）
- **相比于线性搜索**：从 $O(N)$ 降低到 $O(\log N)$，在 $N = 1800$ 时，加速比约为 270 倍

K-D 树的构建和查询过程：

```
构建 K-D 树：
  输入：所有车辆的位置 {(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)}
  输出：平衡的 K-D 树

  步骤：
    1. 选择中位数作为根节点
    2. 递归地在左右子树中重复过程
    3. 在不同维度之间交替分割

查询最近的 K 个点：
  输入：查询点 (x_q, y_q)，K 值
  输出：最近的 K 个点

  步骤：
    1. 从根节点开始递归搜索
    2. 维护一个优先队列，存储最近的 K 个点
    3. 使用距离阈值进行剪枝
    4. 返回优先队列中的 K 个点
```

### 3.4.3 车辆管理器 (VehicleManager)

#### 3.4.3.1 车辆状态模型

每个车辆在系统中可能处于以下状态之一：

```
┌──────────┐
│ 空闲     │ (没有任务，可以接单或被调度)
└────┬─────┘
     │ (被调度到目标网格)
     ▼
┌──────────┐
│ 调度中   │ (正在前往目标网格)
└────┬─────┘
     │ (到达目标网格)
     ▼
┌──────────┐
│ 等待中   │ (在目标网格等待订单)
└────┬─────┘
     │ (接到订单)
     ▼
┌──────────┐
│ 执行中   │ (正在送乘客)
└────┬─────┘
     │ (完成订单)
     ▼
┌──────────┐
│ 空闲     │ (回到空闲状态)
└──────────┘
```

#### 3.4.3.2 车辆数据结构

```python
class Vehicle:
    def __init__(self):
        # 基本信息
        self.vehicle_id: int            # 车辆唯一 ID (0-1799)
        self.current_grid: int          # 当前所在网格 ID
        self.target_grid: int = None    # 目标网格 ID（调度时设置）

        # 状态信息
        self.status: str                # 车辆状态：idle, dispatching, waiting, executing
        self.status_changed_time: float # 最后一次状态改变的时间

        # 任务信息
        self.current_order: Order = None  # 当前执行的订单
        self.order_start_time: float = None  # 订单开始时间

        # 位置信息
        self.x: float                   # 精确的 X 坐标（用于计算距离）
        self.y: float                   # 精确的 Y 坐标

        # 统计信息
        self.total_orders_completed: int = 0  # 完成的订单数
        self.total_distance_traveled: float = 0.0  # 行驶总距离
        self.total_time_idle: float = 0.0  # 空闲总时间
```

#### 3.4.3.3 车辆位置更新

在模拟过程中，车辆的位置需要根据其状态进行更新。

**平均速度设定**

假设城市中的平均速度为 $v_{\text{avg}} = 10$ m/s（相当于 36 km/h）。

**位置更新公式**

对于处于"调度中"或"执行中"状态的车辆，其位置在时间间隔 $\Delta t$ 内更新为：

$$\vec{p}(t + \Delta t) = \vec{p}(t) + v_{\text{avg}} \cdot \Delta t \cdot \frac{\vec{p}_{\text{target}} - \vec{p}(t)}{|\vec{p}_{\text{target}} - \vec{p}(t)|}$$

其中：
- $\vec{p}(t)$ 是当前位置
- $\vec{p}_{\text{target}}$ 是目标位置
- 分数项是单位方向向量

**到达判定**

当车辆与目标的距离小于一个阈值时（通常 50 米），认为车辆已到达目标：

$$|\vec{p}(t) - \vec{p}_{\text{target}}| < \text{threshold}$$

此时，车辆的状态转换为"等待中"或"执行中"（取决于是否已接单）。

### 3.4.4 调度决策执行器 (DispatchExecutor)

#### 3.4.4.1 调度决策的接收与验证

调度决策执行器接收来自 RL 模型的决策，并将其转换为具体的调度操作。

**决策格式**

RL 模型的输出是一个动作 $a \in \{0, 1, ..., 178\}$，代表一个热点网格。

**决策验证**

在执行调度前，需要进行以下验证：

1. **动作有效性**：检查 $a$ 是否在有效范围内 $(0 \leq a < 179)$
2. **可用车辆检查**：检查是否存在空闲车辆可以被调度
3. **目标网格检查**：检查目标网格是否存在待匹配的订单

#### 3.4.4.2 调度操作的执行

当决策通过验证后，执行器执行以下操作：

**步骤 1：选择调度车辆**

从所有空闲车辆中，选择距离目标网格最近的 $N$ 个车辆（通常 $N = 5$）：

$$\mathcal{V}_{\text{dispatch}} = \arg\min_v \text{distance}(v_{\text{current\_grid}}, a) \quad (|\mathcal{V}_{\text{dispatch}}| = N)$$

**步骤 2：更新车辆状态**

对于每个被调度的车辆 $v \in \mathcal{V}_{\text{dispatch}}$：

```python
v.status = "dispatching"
v.target_grid = a
v.status_changed_time = current_time
```

**步骤 3：记录调度事件**

调度事件被记录到事件队列中，以便后续追踪：

```python
dispatch_event = {
    'time': current_time,
    'action': a,
    'vehicles_dispatched': len(mathcal_V_dispatch),
    'target_grid': a
}
```

### 3.4.5 状态观测器 (StateObserver)

#### 3.4.5.1 网格级别信息的聚合

RL 模型需要的状态是网格级别的聚合信息，而不是个体订单和车辆的信息。状态观测器负责这种聚合。

**网格信息的定义**

对于每个网格 $g_i$，在时刻 $t$ 的信息为：

$$s_i(t) = \{n_{\text{orders}}^i(t), n_{\text{idle}}^i(t), n_{\text{busy}}^i(t)\}$$

其中：
- $n_{\text{orders}}^i(t)$：网格 $i$ 中**待匹配的订单数**
- $n_{\text{idle}}^i(t)$：网格 $i$ 中**空闲车辆的数量**
- $n_{\text{busy}}^i(t)$：网格 $i$ 中**执行任务的车辆数量**

**聚合算法**

```python
def aggregate_state(current_time):
    state = np.zeros((400, 3))  # 400 个网格，每个网格 3 个特征

    # 遍历所有订单
    for order in orders:
        if order.status == "waiting":
            grid_id = order.origin_grid
            state[grid_id, 0] += 1  # 待匹配订单数

    # 遍历所有车辆
    for vehicle in vehicles:
        grid_id = vehicle.current_grid
        if vehicle.status == "idle":
            state[grid_id, 1] += 1  # 空闲车辆数
        elif vehicle.status in ["executing", "dispatching"]:
            state[grid_id, 2] += 1  # 执行任务的车辆数

    return state
```

**时间复杂度**：$O(M + V)$，其中 $M$ 是订单数，$V$ 是车辆数。

#### 3.4.5.2 时间编码

为了捕捉时间的周期性特征，状态观测器添加时间编码：

$$\text{time\_sin} = \sin\left(\frac{2\pi \cdot t}{86400}\right)$$
$$\text{time\_cos} = \cos\left(\frac{2\pi \cdot t}{86400}\right)$$

其中 $t$ 是当前时刻（秒），$86400$ 是一天的秒数。

这种编码方式能够自然地表示时间的循环性，避免时间边界处的不连续性。

#### 3.4.5.3 完整状态的生成

最终的状态是一个 $(400, 5)$ 的矩阵：

$$s_t = [n_{\text{orders}}^1, ..., n_{\text{orders}}^{400}, n_{\text{idle}}^1, ..., n_{\text{idle}}^{400}, n_{\text{busy}}^1, ..., n_{\text{busy}}^{400}, \text{time\_sin}, \text{time\_cos}]$$

### 3.4.6 奖励计算器 (RewardCalculator)

#### 3.4.6.1 多阶段奖励的计算

奖励计算器在订单的不同生命周期阶段计算奖励。

**阶段 1：订单匹配时**

当订单被成功匹配时，立即给予匹配奖励：

$$R_{\text{match}} = W_{\text{match}} = 1.2$$

**阶段 2：订单完成时**

当订单被完成时，根据乘客的等待时间计算完成奖励：

$$R_{\text{completion}} = W_{\text{completion}} \cdot \exp\left(-\frac{w_t}{T_0}\right)$$

其中：
- $w_t = t_{\text{match}} + t_{\text{travel}}$ 是理论等待时间
- $W_{\text{completion}} = 2.0$
- $T_0 = 240$ 秒

**阶段 3：订单超时取消时**

当订单因超时被取消时，给予取消惩罚：

$$R_{\text{cancel}} = -W_{\text{cancel}} = -1.0$$

#### 3.4.6.2 业务指标的统计

除了奖励，奖励计算器还统计以下业务指标：

**1. 匹配率 (Match Rate)**

$$\text{Match Rate} = \frac{\text{Matched Orders}}{\text{Total Orders}} \times 100\%$$

**2. 平均等待时间 (Average Waiting Time)**

$$\bar{w}_t = \frac{1}{M} \sum_{i=1}^{M} w_{t,i}$$

其中 $M$ 是匹配的订单数。

**3. 取消率 (Cancellation Rate)**

$$\text{Cancel Rate} = \frac{\text{Cancelled Orders}}{\text{Total Orders}} \times 100\%$$

**4. 车辆利用率 (Vehicle Utilization Rate)**

$$\text{Utilization Rate} = \frac{\text{Total Time Executing}}{\text{Total Time}} \times 100\%$$

**5. 平均订单收入 (Average Order Revenue)**

这是一个虚拟的指标，用于衡量系统的经济效益：

$$\bar{R}_{\text{order}} = \frac{\text{Total Revenue}}{\text{Completed Orders}}$$

### 3.4.7 模拟循环

#### 3.4.7.1 完整的模拟流程

```python
def run_simulation(num_episodes, max_ticks_per_episode):
    """
    运行完整的模拟

    Args:
        num_episodes: 模拟的 episode 数
        max_ticks_per_episode: 每个 episode 的最大时间步数
    """

    for episode in range(num_episodes):
        # 初始化 episode
        env.reset()
        state = env.get_state()
        episode_reward = 0.0
        episode_metrics = {
            'match_rate': 0.0,
            'avg_wait_time': 0.0,
            'cancel_rate': 0.0
        }

        for tick in range(max_ticks_per_episode):
            # 1. 获取当前状态
            state = env.get_state()

            # 2. RL 模型做出决策
            if random() < epsilon:
                action = random_action()
            else:
                action = model.select_action(state)

            # 3. 执行调度操作
            env.dispatch(action)

            # 4. 推进模拟
            next_state, reward, done, info = env.step()
            episode_reward += reward

            # 5. 存储经验到回放缓冲区
            replay_buffer.push(state, action, reward, next_state, done)

            # 6. 定期训练模型
            if tick % TRAIN_EVERY_N_TICKS == 0:
                for _ in range(TRAIN_LOOPS_PER_BATCH):
                    batch, indices, weights = replay_buffer.sample()
                    train_step(batch, weights, indices)

            # 7. 定期更新目标网络
            if train_step_count % TARGET_UPDATE_FREQ == 0:
                update_target_network()

            # 8. 记录指标
            episode_metrics = env.get_metrics()

            state = next_state
            if done:
                break

        # Episode 结束
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 记录训练曲线
        log_episode_metrics(episode, episode_reward, episode_metrics)
```

#### 3.4.7.2 模拟时间与真实时间的对应

- **模拟时间步长**：每个 tick 对应 30 秒的真实时间
- **每天的 tick 数**：$\lfloor 24 \times 60 \times 60 / 30 \rfloor = 2880$ ticks
- **模拟 1 天**：需要 2880 个 ticks
- **模拟 1 周**：需要 $2880 \times 7 = 20160$ ticks

### 3.4.8 模拟器的性能分析

#### 3.4.8.1 时间复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 订单匹配 | $O(\log V)$ | 使用 K-D 树查询最近的 K 个车辆 |
| 状态聚合 | $O(M + V)$ | 遍历所有订单和车辆 |
| 车辆位置更新 | $O(V)$ | 更新所有车辆的位置 |
| 事件处理 | $O(E \log E)$ | E 是事件总数 |
| 整个 episode | $O(E \cdot (\log V + M + V))$ | E 是 episode 中的事件总数 |

#### 3.4.8.2 空间复杂度分析

| 数据结构 | 空间复杂度 | 说明 |
|---------|-----------|------|
| 订单库 | $O(M)$ | 存储所有订单 |
| 车辆库 | $O(V)$ | 存储所有车辆 |
| K-D 树 | $O(V)$ | 车辆空间索引 |
| 经验回放缓冲区 | $O(B)$ | B 是缓冲区大小 |
| 总计 | $O(M + V + B)$ | 通常 $M \approx 20000$, $V = 1800$, $B = 50000$ |

#### 3.4.8.3 实际性能

在我们的实现中：
- **单个 episode 的模拟时间**：约 1-2 秒（模拟 1 天的数据）
- **训练 500 个 episodes**：约 10-20 分钟
- **内存占用**：约 500 MB - 1 GB

这个性能足以支持快速的原型开发和实验。

---

## 3.5 实验设置与评估指标

### 3.5.1 实验数据集

#### 3.5.1.1 数据来源

我们使用**北京滴滴出行数据集** (Beijing Didi Dataset)，这是一个公开的网约车真实数据集：

- **数据规模**：约 20 万条订单
- **时间跨度**：连续 7 天的数据
- **地理范围**：北京市中心区域，约 50 km × 50 km
- **数据质量**：经过清洗和验证

#### 3.5.1.2 数据预处理

**步骤 1：坐标映射**

将原始的经纬度坐标映射到 $20 \times 20$ 的网格：

$$\text{grid\_id} = \left\lfloor \frac{x - x_{\min}}{(x_{\max} - x_{\min}) / 20} \right\rfloor \times 20 + \left\lfloor \frac{y - y_{\min}}{(y_{\max} - y_{\min}) / 20} \right\rfloor$$

**步骤 2：时间对齐**

将所有订单的时间戳对齐到一个统一的时间轴。

**步骤 3：数据分割**

- **训练集**：前 5 天的数据（约 14 万订单）
- **验证集**：第 6 天的数据（约 3 万订单）
- **测试集**：第 7 天的数据（约 3 万订单）

### 3.5.2 对比算法

#### 3.5.2.1 基线算法 1：随机调度 (Random Dispatch)

**算法描述**

每次调度时，随机选择一个热点网格作为目标。

**伪代码**

```python
def random_dispatch(state):
    action = random.randint(0, 178)  # 随机选择一个热点
    return action
```

**性能特点**
- **优点**：实现简单，无需训练
- **缺点**：性能很差，作为下界基线

#### 3.5.2.2 基线算法 2：启发式算法 (Heuristic Dispatch)

**算法描述**

基于当前的订单分布和车辆分布，使用贪心策略选择调度目标。

**具体规则**

对于每个热点网格 $g_i$，计算一个**吸引力分数** $s_i$：

$$s_i = w_1 \cdot n_{\text{orders}}^i - w_2 \cdot n_{\text{vehicles}}^i$$

其中：
- $n_{\text{orders}}^i$ 是网格 $i$ 的待匹配订单数
- $n_{\text{vehicles}}^i$ 是网格 $i$ 的空闲车辆数
- $w_1 = 1.0$，$w_2 = 0.5$ 是权重

选择得分最高的网格作为调度目标：

$$a^* = \arg\max_i s_i$$

**性能特点**
- **优点**：实现简单，基于常识
- **缺点**：不能学习复杂的时空模式

#### 3.5.2.3 基线算法 3：Q-Learning (表格 Q-Learning)

**算法描述**

使用传统的表格 Q-Learning 方法，将状态空间离散化为有限的状态集合。

**状态离散化**

由于完整的状态空间太大，我们将其简化为：

$$s_{\text{simplified}} = (\text{total\_orders}, \text{total\_vehicles}, \text{hour\_of\_day})$$

其中：
- $\text{total\_orders} \in [0, 100]$（分为 10 个区间）
- $\text{total\_vehicles} \in [0, 1800]$（分为 18 个区间）
- $\text{hour\_of\_day} \in [0, 23]$（24 个小时）

简化状态空间的大小为 $10 \times 18 \times 24 = 4320$。

**Q-Learning 更新规则**

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**性能特点**
- **优点**：能够学习一些基本的调度策略
- **缺点**：状态空间过于简化，丢失了大量的空间信息

### 3.5.3 评估指标

#### 3.5.3.1 主要评估指标

**1. 匹配率 (Match Rate)**

$$\text{Match Rate} = \frac{\text{Matched Orders}}{\text{Total Orders}} \times 100\%$$

- **范围**：0% - 100%
- **目标**：尽可能高（目标 > 90%）
- **业务意义**：直接影响平台的订单完成度和用户满意度

**2. 平均等待时间 (Average Waiting Time)**

$$\bar{w}_t = \frac{1}{M} \sum_{i=1}^{M} w_{t,i}$$

其中 $w_{t,i}$ 是第 $i$ 个订单的理论等待时间（秒）。

- **范围**：0 - 600 秒
- **目标**：尽可能低（目标 < 300 秒）
- **业务意义**：直接影响乘客体验

**3. 取消率 (Cancellation Rate)**

$$\text{Cancel Rate} = \frac{\text{Cancelled Orders}}{\text{Total Orders}} \times 100\%$$

- **范围**：0% - 100%
- **目标**：尽可能低（目标 < 10%）
- **业务意义**：反映系统的可靠性

**4. 车辆利用率 (Vehicle Utilization Rate)**

$$\text{Utilization Rate} = \frac{\sum_v \text{Time Executing}_v}{\sum_v \text{Total Time}_v} \times 100\%$$

- **范围**：0% - 100%
- **目标**：尽可能高（目标 > 60%）
- **业务意义**：反映资源的使用效率

#### 3.5.3.2 辅助评估指标

**5. 平均订单收入 (Average Order Revenue)**

这是一个虚拟的指标，用于衡量系统的经济效益。假设每个订单的基础收入为 15 元，等待时间每增加 60 秒，收入减少 1 元：

$$R_i = 15 - \frac{w_{t,i}}{60}$$

$$\bar{R}_{\text{order}} = \frac{1}{M} \sum_{i=1}^{M} R_i$$

**6. 系统效率指数 (System Efficiency Index)**

综合多个指标的加权指数：

$$\text{SEI} = 0.4 \times \text{Match Rate} + 0.3 \times (1 - \frac{\text{Cancel Rate}}{100\%}) + 0.2 \times \text{Utilization Rate} - 0.1 \times \frac{\bar{w}_t}{600}$$

这个指数综合考虑了匹配率、取消率、利用率和等待时间，值越高表示系统性能越好。

### 3.5.4 实验配置

#### 3.5.4.1 超参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 (LR) | $1 \times 10^{-4}$ | Adam 优化器 |
| 权重衰减 | $1 \times 10^{-5}$ | L2 正则化 |
| 折扣因子 ($\gamma$) | 0.99 | 长期奖励权重 |
| Batch Size | 128 | 每次训练样本数 |
| 缓冲区大小 | 50,000 | 经验回放缓冲区 |
| 训练频率 | 每 30 ticks | 每 15 分钟训练一次 |
| 每次训练循环 | 4 | 每次训练迭代 4 个 batch |
| 目标网络更新 | 每 1000 步 | 硬更新频率 |
| 初始 $\epsilon$ | 0.6 | 初始探索概率 |
| 最小 $\epsilon$ | 0.05 | 最小探索概率 |
| $\epsilon$ 衰减率 | 0.95 | 每个 episode 的衰减率 |
| PER $\alpha$ | 0.4 | 优先级强度 |
| PER $\beta_{\text{start}}$ | 0.4 | 初始补偿权重 |
| PER $\beta_{\text{end}}$ | 1.0 | 最终补偿权重 |

#### 3.5.4.2 训练配置

- **训练 episodes**：500
- **每个 episode 的最大 ticks**：2880（模拟 1 天）
- **总训练时间**：约 10-20 分钟
- **评估频率**：每 50 个 episodes 在验证集上评估一次

### 3.5.5 实验流程

#### 3.5.5.1 完整的实验流程

```
1. 数据准备
   ├─ 加载原始数据集
   ├─ 数据清洗和预处理
   └─ 分割为训练、验证、测试集

2. 环境初始化
   ├─ 初始化模拟器
   ├─ 加载订单数据
   └─ 初始化车辆

3. 模型训练
   ├─ 初始化神经网络
   ├─ 初始化经验回放缓冲区
   ├─ for episode = 1 to 500:
   │  ├─ 重置环境
   │  ├─ for tick = 1 to 2880:
   │  │  ├─ 选择动作（epsilon-greedy）
   │  │  ├─ 执行调度
   │  │  ├─ 推进模拟
   │  │  ├─ 存储经验
   │  │  ├─ 训练模型（定期）
   │  │  ├─ 更新目标网络（定期）
   │  │  └─ 记录指标
   │  ├─ 衰减 epsilon
   │  └─ 在验证集上评估（定期）

4. 模型评估
   ├─ 在测试集上评估最优模型
   ├─ 计算所有评估指标
   └─ 生成性能报告

5. 对比分析
   ├─ 与基线算法对比
   ├─ 生成对比图表
   └─ 进行统计显著性检验
```

---

## 3.6 总结

本章详细介绍了我们提出的网约车调度强化学习方法。主要内容包括：

### 3.6.1 核心贡献

**1. 问题建模与奖励函数设计**

- 将网约车调度问题形式化为 MDP
- 设计了多阶段奖励函数，准确反映乘客体验
- 引入理论等待时间的概念，平衡匹配速度和等待时间

**2. 深度强化学习模型**

- 提出了多图卷积网络 (MGCN)，融合空间邻接关系和功能相似性
- 设计了特征融合层，有效整合全局和上下文信息
- 采用 Dueling DQN 架构，提高学习稳定性和收敛速度

**3. 训练算法**

- 采用优先级经验回放 (PER)，提高样本效率
- 使用双 DQN 缓解高估偏差问题
- 配合 epsilon-greedy 探索策略和学习率调度，确保训练的稳定性

**4. 高效的模拟器**

- 实现了事件驱动的网约车调度模拟器
- 使用 K-D 树等数据结构优化性能
- 支持快速的大规模模拟

### 3.6.2 方法的优势

1. **空间感知**：通过 MGCN 捕捉复杂的空间关系
2. **样本高效**：通过 PER 和 Dueling DQN 提高学习效率
3. **稳定训练**：通过目标网络和梯度裁剪等技术确保训练稳定
4. **可解释性**：模型架构清晰，各个组件的作用明确

### 3.6.3 后续工作

在后续章节中，我们将：

1. **第 4 章 - 实验结果**：详细展示实验结果，包括与基线算法的对比、消融实验等
2. **第 5 章 - 分析与讨论**：深入分析模型的性能、学习过程、失败案例等
3. **第 6 章 - 结论与展望**：总结全文，讨论未来的研究方向

---

**注**：本章方法的实现代码已在 `models/` 目录中提供，具体包括：
- `models/network.py`：神经网络架构
- `models/dqn_agent.py`：DQN agent 实现
- `models/per_buffer.py`：优先级经验回放缓冲区
- `simulator/environment.py`：模拟环境实现
- `simulator/order_manager.py`：订单管理器
- `simulator/vehicle_manager.py`：车辆管理器

