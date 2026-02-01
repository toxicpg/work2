# Baseline 环境更新说明

## 更新时间
2026-02-01

## 更新内容

### 核心修改
已将 `environment_baseline.py` 完整替换为主环境 `environment.py` (V5.3) 的代码，只保留了类名差异。

### 主要变更

1. **完整复制主环境代码**
   - OrderGenerator: 完全一致
   - VehicleManager: 完全一致（包括 V5.4 Loss=0 Bug 修复）
   - OrderMatcher: 完全一致（包括 V5.6 K-D Tree 优化）
   - RewardCalculator: 完全一致
   - BaselineEnvironment: 完全继承 RideHailingEnvironment 的所有逻辑

2. **类名修改**
   - `RideHailingEnvironment` → `BaselineEnvironment`
   - 所有其他类保持原名

3. **订单匹配逻辑**
   - 使用 K-D Tree 进行 O(N*k) 高效匹配
   - 匹配后检查"等待匹配+接驾"总时间是否超时
   - 超时订单会被取消，车辆回滚到 idle 状态
   - 与主环境完全一致的取消机制

4. **车辆状态管理**
   - 使用事件驱动模型处理订单完成
   - 车辆状态: idle, dispatching, serving
   - 移除了之前的 `picking_up` 状态
   - `assign_order` 方法签名与主环境一致

5. **调度策略支持**
   - 通过 `config.DISPATCH_MODE` 控制调度模式
   - 支持 'random_walk' 和 'rl' 模式
   - Baseline 算法可以通过直接调用 `vehicle_manager.start_dispatching()` 进行调度

## 为什么要这样做？

### 之前的问题
1. `environment_baseline.py` 是简化版，存在多处与主环境不一致的地方
2. 匹配率偏低（约10%），因为引入了过度复杂的状态管理和取消逻辑
3. 与主环境的差异导致 baseline 测试结果不公平

### 解决方案
直接复制主环境代码，确保：
- ✅ 订单匹配逻辑完全一致
- ✅ 车辆状态管理完全一致
- ✅ 订单取消机制完全一致
- ✅ K-D Tree 匹配优化
- ✅ 事件驱动模型

## 如何使用

### 1. Random Walk Baseline
```python
from config import Config
from environment_baseline import BaselineEnvironment
from utils.data_process import DataProcessor

config = Config()
config.DISPATCH_MODE = 'random_walk'
data_processor = DataProcessor(config)
orders_df = data_processor.load_and_process_orders()

env = BaselineEnvironment(config, data_processor, orders_df)
env.reset()

# 运行仿真
for step in range(config.MAX_TICKS_PER_EPISODE):
    state, reward, done, info = env.step()
    if done:
        break
```

### 2. SARSA/H-MARL Baseline
```python
config = Config()
config.DISPATCH_MODE = 'rl'  # 或者不设置，默认就是'rl'
env = BaselineEnvironment(config, data_processor, orders_df)
env.reset()

# 在每个 step 前，由外部 Agent 控制调度
for step in range(config.MAX_TICKS_PER_EPISODE):
    # Agent 决策并调度
    idle_vehicles = env.vehicle_manager.get_long_idle_vehicles(
        env.current_time,
        config.IDLE_THRESHOLD_SEC
    )
    for vehicle_id in idle_vehicles:
        action = agent.select_action(state, vehicle_id)
        target_grid = action  # 或者你的动作空间到网格的映射
        env.vehicle_manager.start_dispatching(vehicle_id, target_grid, env.current_time)

    # 环境 step
    state, reward, done, info = env.step()
    if done:
        break
```

### 3. Random Dispatch Baseline
可以在调用 `env.step()` 前手动实现随机调度到热点网格的逻辑，或者在环境内部添加。

## 测试脚本更新

`test_baseline_matching_fix.py` 已更新为新的初始化方式：

```python
config.DISPATCH_MODE = 'random_walk'
env = BaselineEnvironment(config, data_processor, day_orders)
env.reset()
```

## 注意事项

1. **所有 Baseline 算法现在都使用同一套核心逻辑**
   - 订单匹配
   - 车辆管理
   - 订单取消

2. **Baseline 算法的职责**
   - 只负责调度决策（决定空闲车辆去哪个网格）
   - 不需要关心订单匹配、车辆状态更新等细节

3. **与主环境的一致性**
   - 匹配率、完成率、取消率应该在相同量级
   - 只有调度策略的差异会导致结果不同

## 预期结果

修复后，baseline 环境应该表现出与主环境相似的匹配率（>50%），不同的只是调度策略的效果差异。

## 文件清单

- ✅ `environment_baseline.py` - 已更新为主环境 V5.3 代码
- ✅ `test_baseline_matching_fix.py` - 已更新测试脚本
- ✅ `BASELINE_ENV_UPDATE.md` - 本说明文档

## 后续建议

1. **测试验证**: 运行 `test_baseline_matching_fix.py` 确认匹配率恢复正常
2. **更新其他 Baseline 脚本**:
   - `baselines/sarsa_saa.py`
   - `baselines/train_hmarl.py`
   - `baselines/random_dispatch.py`
3. **统一调度接口**: 所有 baseline 都通过 `vehicle_manager.start_dispatching()` 进行调度

---

**总结**: 现在 baseline 环境与主环境共享同一套核心代码，确保了公平对比的基础。

