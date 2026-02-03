# 环境变动影响分析与修复报告

## 📋 环境变动总结

### 核心变动
**BaselineEnvironment 匹配距离限制：从0格放宽至3格（曼哈顿距离）**

修改位置：`environment_baseline.py` 的匹配逻辑
```python
# 修改前
if manhattan_distance > 0:
    continue

# 修改后
if manhattan_distance > 3:
    continue
```

---

## 🔍 各算法影响分析

### 1. Random Walk ✅ 无需修改
- **影响**: 无
- **原因**: Random Walk 不主动调度车辆，只做随机游走
- **结论**: 环境匹配变动不影响此算法

---

### 2. Random Dispatch ✅ 无需修改
- **影响**: 无
- **原因**: Random Dispatch 调度到热点，但不限制距离。环境的匹配距离限制独立于调度距离
- **结论**: 调度逻辑无需修改，环境会自动处理匹配

---

### 3. SARSA-SAA ✅ 已修改
- **影响**: 高
- **原因**: SARSA-SAA 有调度半径参数 `max_dispatch_radius`，原设置为10格
- **修改内容**:
  ```python
  # baselines/sarsa_saa.py
  # Line 53: 修改前
  self.max_dispatch_radius = 10

  # 修改后
  self.max_dispatch_radius = 3
  ```

  ```python
  # Line 244: 修改贪婪算法中的硬编码距离
  # 修改前
  if distance > 10:
      s_idx += 1
      continue

  # 修改后
  if distance > self.max_dispatch_radius:
      s_idx += 1
      continue
  ```
- **测试建议**: 重新运行 SARSA-SAA 以验证性能变化

---

### 4. H-MARL (MFuN) ✅ 已修改
- **影响**: 高
- **原因**: H-MARL可以将车辆调度到任意grid，没有距离限制
- **修改内容**:

  #### 修改1: MILP求解器（降级方案）
  ```python
  # baselines/hmarl_agent.py
  # Line 365-381: 修改前（无距离限制）
  if not self.use_milp:
      dst_grid = torch.argmax(target_grids_probs).item()
      ...

  # 修改后（添加3格限制）
  if not self.use_milp:
      src_row, src_col = grid_id // 20, grid_id % 20
      valid_grids = []
      for g in range(self.num_grids):
          dst_row, dst_col = g // 20, g % 20
          if abs(src_row - dst_row) + abs(src_col - dst_col) <= 3:
              valid_grids.append((g, target_grids_probs[g].item()))
      if not valid_grids:
          return {}
      dst_grid = max(valid_grids, key=lambda x: x[1])[0]
      ...
  ```

  #### 修改2: MILP目标函数
  ```python
  # Line 399-410: 修改前（无距离限制）
  for v_id, v in available_vehicles:
      for g in range(self.num_grids):
          expected_revenue = demand[g] * target_grids_probs[g].item()
          dispatch_cost = abs(src_row - dst_row) + abs(src_col - dst_col)
          objective += x[v_id, g] * (expected_revenue - dispatch_cost * 0.1)

  # 修改后（添加3格限制）
  for v_id, v in available_vehicles:
      for g in range(self.num_grids):
          src_row, src_col = grid_id // 20, grid_id % 20
          dst_row, dst_col = g // 20, g % 20
          dispatch_distance = abs(src_row - dst_row) + abs(src_col - dst_col)

          if dispatch_distance > 3:
              continue  # 跳过超过3格的调度

          expected_revenue = demand[g] * target_grids_probs[g].item()
          dispatch_cost = dispatch_distance * 0.1
          objective += x[v_id, g] * (expected_revenue - dispatch_cost)
  ```

  #### 修改3: ALNS优化器
  ```python
  # Line 507-523: 修改前（无距离限制）
  for g in range(self.num_grids):
      demand = len([o for o in env.order_generator.pending_orders
                  if o.get('origin_grid', -1) == g])
      distance = abs(src_row - dst_row) + abs(src_col - dst_col)
      score = demand * 2.0 - distance * 0.5
      if score > best_score:
          best_score = score
          best_grid = g

  # 修改后（添加3格限制）
  for g in range(self.num_grids):
      src_row, src_col = current_grid // 20, current_grid % 20
      dst_row, dst_col = g // 20, g % 20
      distance = abs(src_row - dst_row) + abs(src_col - dst_col)

      if distance > 3:
          continue  # 跳过超过3格的调度

      demand = len([o for o in env.order_generator.pending_orders
                  if o.get('origin_grid', -1) == g])
      score = demand * 2.0 - distance * 0.5
      if score > best_score:
          best_score = score
          best_grid = g
  ```

  #### 修改4: 异常处理降级方案
  ```python
  # Line 461-468: 修改前
  except Exception as e:
      print(f"MILP求解失败: {e}, 使用降级方案")
      dst_grid = torch.argmax(target_grids_probs).item()
      ...

  # 修改后（添加3格限制）
  except Exception as e:
      print(f"MILP求解失败: {e}, 使用降级方案")
      src_row, src_col = grid_id // 20, grid_id % 20
      valid_grids = []
      for g in range(self.num_grids):
          dst_row, dst_col = g // 20, g % 20
          if abs(src_row - dst_row) + abs(src_col - dst_col) <= 3:
              valid_grids.append((g, target_grids_probs[g].item()))
      if not valid_grids:
          return {}
      dst_grid = max(valid_grids, key=lambda x: x[1])[0]
      ...
  ```

- **测试建议**: 重新训练 H-MARL 以验证收敛性

---

### 5. CNN-DDQN ✅ 无需修改
- **影响**: 低/无
- **原因**:
  1. CNN-DDQN使用专门的`RideHailingEnvironment`（非BaselineEnvironment）
  2. 输出动作空间为179个热点（从`action_mapping.json`加载）
  3. 热点选择已考虑了邻近性和需求
  4. 环境内部的匹配逻辑会自动应用3格限制
- **结论**: 无需修改算法代码

---

## 🔧 H-MARL 训练脚本修复

### 额外修复1: 冷启动机制
```python
# baselines/train_hmarl.py
# Line 98-100: 修改前
env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='none')
env.reset()

# 修改后（触发冷启动）
env = BaselineEnvironment(config, data_processor, day_orders, dispatch_policy='none')
try:
    day_count = env.order_generator.get_day_count()
except Exception:
    day_count = 0
start_day = max(0, day_count - 1)
env.reset(start_day=start_day)
```

### 额外修复2: 奖励计算
```python
# Line 143-150: 修改前（使用累积值）
env.step()
completed_orders = env.reward_calculator.completed_orders
cancelled_orders = env.reward_calculator.cancelled_orders
external_reward = completed_orders * 10 - cancelled_orders * 5

# 修改后（使用step_info获取增量）
_, _, _, info = env.step()
step_info = info.get('step_info', {})
tick_completed = step_info.get('completed_orders', 0)
tick_cancelled = step_info.get('cancelled_orders', 0)
tick_matched = step_info.get('matched_orders', 0)
external_reward = (
    tick_completed * 10
    - tick_cancelled * 5
    + tick_matched * 1
)
```

---

## 📊 预期影响

### 1. SARSA-SAA
- **预期**: 调度成本降低，但可能错过部分远距离高需求区域
- **匹配率**: 可能略有提升（车辆更集中）
- **建议**: 重新运行完整实验

### 2. H-MARL
- **预期**:
  - Manager的子目标设定需要更局部化
  - Worker的动作空间从400个grid减少到每个grid周围最多49个(7x7)有效目标
  - MILP求解速度可能加快（约束更少）
- **挑战**:
  - 原始H-MARL可能依赖长距离调度来平衡供需
  - 需要更多episode才能收敛到新的最优策略
- **建议**:
  1. 增加训练episodes（从10增加到30+）
  2. 监控内部奖励（Intrinsic Reward）的变化
  3. 对比修改前后的子目标分布

### 3. CNN-DDQN
- **预期**: 无显著影响
- **原因**: 环境变动对CNN-DDQN透明

---

## ✅ 验证清单

- [x] Random Walk: 无需修改
- [x] Random Dispatch: 无需修改
- [x] SARSA-SAA: 已修改 `max_dispatch_radius = 3`
- [x] H-MARL: 已修改所有调度逻辑（MILP + ALNS + 降级方案）
- [x] CNN-DDQN: 无需修改
- [x] H-MARL训练脚本: 已修复冷启动和奖励计算

---

## 🚀 下一步行动

1. **重新运行 SARSA-SAA**
   ```bash
   python baselines/sarsa_saa.py
   ```

2. **重新训练 H-MARL**
   ```bash
   python baselines/train_hmarl.py
   ```
   - 预计训练时间：30-60分钟
   - 建议使用GPU加速

3. **对比实验**
   - 记录修改前后的关键指标：
     - 匹配率 (Match Rate)
     - 完成率 (Completion Rate)
     - 平均等待时间 (Avg Waiting Time)
     - 平均调度距离 (Avg Dispatch Distance)

4. **结果分析**
   - 如果H-MARL性能下降明显，考虑：
     - 调整内部奖励权重 (α)
     - 增加Manager决策频率
     - 修改子目标的数值范围

---

## 📝 技术要点

### 曼哈顿距离计算
```python
def manhattan_distance(grid1, grid2):
    row1, col1 = grid1 // 20, grid1 % 20
    row2, col2 = grid2 // 20, grid2 % 20
    return abs(row1 - row2) + abs(col1 - col2)
```

### 3格限制的含义
- **0格**: 同一grid（原始环境）
- **1格**: 上下左右4个邻居
- **2格**: 曼哈顿圆（12个grid）
- **3格**: 曼哈顿圆（24个grid）
- 每个grid最多可调度到 **1 + 4 + 12 + 24 = 41个**目标grid（包括自己）

---

## 🔬 实验建议

### A/B测试
建议对H-MARL进行A/B测试：
- **A组**: 无距离限制（修改前）
- **B组**: 3格距离限制（修改后）

对比指标：
1. 最终匹配率
2. 收敛速度（达到稳定性能的episodes数）
3. 调度效率（单次调度的平均效果）
4. Manager子目标的变化趋势

---

**修改日期**: 2026-02-03
**修改人**: CatPaw AI Assistant
**影响范围**: SARSA-SAA, H-MARL
**风险等级**: 中等（需要重新训练）

