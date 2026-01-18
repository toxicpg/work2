# train.py 检查报告

**检查日期**: 2026-01-18
**检查目的**: 确保正式训练脚本没有问题

---

## ✅ 检查结果：整体良好

你的 `train.py` **可以正常使用**，所有关键Bug都已修复。

---

## 📊 已修复的问题

### 1. ✅ Trainer属性初始化
```python
# train.py 第97-99行
trainer = MGCNTrainer(config, neighbor_adj, poi_adj)
trainer.best_validation_metric = 0.0  # ✓ 已添加
trainer.early_stopping_counter = 0    # ✓ 已添加
```

### 2. ✅ save_checkpoint参数
```python
# train.py 第127, 132, 140行
trainer.save_checkpoint(episode)  # ✓ 移除了is_best参数
```

### 3. ✅ 验证环境初始化
```python
# train.py 第42-56行
for day_index in range(num_val_days):
    val_env.reset()
    # ✓ 强制从指定day开始，避免随机
    val_env.episode_start_day = day_index
    val_env.current_day = day_index
    # 重置时间...
```

### 4. ✅ 自动绘制训练曲线
```python
# train.py 第144-174行
if PLOTTING_AVAILABLE:
    # ✓ 训练结束后自动绘制Loss和Reward曲线
    plot_training_curves(stats_dict)
    plot_waiting_time_curve(stats_dict)
```

---

## 📝 当前配置参数（config.py）

### 关键训练参数

```python
# 数据配置
DATA_START_DATE = '2016-11-01 00:00:00'
DATA_END_DATE = '2016-11-29 23:59:59'
EPISODE_DAYS = 2  # 每个episode训练2天
TICK_DURATION_SEC = 30  # 每个tick 30秒
TICKS_PER_DAY = 2880  # 每天2880个tick
MAX_TICKS_PER_EPISODE = 5760  # 每个episode 5760个tick

# 训练配置
NUM_EPISODES = 50  # 总共50个episode
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 2000  # 开始训练前需要2000条经验
REPLAY_BUFFER_SIZE = 50000
TRAIN_EVERY_N_TICKS = 30  # 每30个tick训练一次
TRAIN_LOOPS_PER_BATCH = 4  # 每次训练4轮

# 验证与保存
VALIDATION_INTERVAL = 10  # 每10个episode验证一次
SAVE_FREQ = 4  # 每4个episode保存一次
EARLY_STOPPING_PATIENCE = 5  # 5次验证不提升则早停

# Epsilon衰减
EPSILON_START = 0.6
EPSILON_END = 0.05
EPSILON_DECAY = 0.95  # 每个episode后 ε = ε × 0.95

# 数据划分
TRAIN_RATIO = 0.70  # 训练集 70%
VAL_RATIO = 0.15    # 验证集 15%
TEST_RATIO = 0.15   # 测试集 15%
```

---

## ⏱️ 预期训练时间

### 单个Episode时间估算

```
每个Episode:
  - 5760 个tick
  - 每30个tick训练1次 = 192次训练机会
  - 每次训练4轮 = 最多768次梯度更新
  - 假设每tick平均0.5秒

单Episode时间 ≈ 5760 × 0.5s = 48分钟（CPU）
               ≈ 5760 × 0.1s = 10分钟（GPU）
```

### 完整训练时间估算

```
50个Episode（GPU）:
  - 前10个episode（验证）: 10 × (10min + 5min验证) = 150分钟
  - 后40个episode: 40 × 10min = 400分钟
  - 总计 ≈ 550分钟 ≈ 9小时

早停情况:
  - 如果在第30个episode触发早停
  - 总时间 ≈ 5-6小时
```

---

## ⚠️ 潜在注意事项

### 1. MIN_REPLAY_SIZE配置

**当前值**: 2000
**初始收集时间**: 约70个tick × 30秒 = 35分钟

**建议**:
- 如果想更快开始训练：降低到1000（训练可能不稳定）
- 如果追求稳定性：保持2000或提高到5000

### 2. 验证时长

**当前**: 验证会运行完整的每一天（2880个tick/天）

**建议**: 如果验证太慢，可以在 `train.py` 的 `run_validation` 中限制验证tick数：

```python
# 第62行改为：
max_val_ticks = min(1000, config.TICKS_PER_DAY)  # 只验证1000个tick
while ticks < max_val_ticks:  # 而不是 TICKS_PER_DAY
```

### 3. 显存使用

**预估GPU显存**:
- 模型参数: ~50MB
- Replay Buffer: ~2GB（50000条经验）
- Batch训练: ~500MB
- **总计**: 约3GB

**建议**:
- 如果显存不足，减小 `REPLAY_BUFFER_SIZE` 到 20000
- 如果仍不足，减小 `BATCH_SIZE` 到 64

### 4. 早停可能过早触发

**当前**: `EARLY_STOPPING_PATIENCE = 5`
**验证间隔**: 10个episode

**问题**: 如果连续5次验证（50个episode）性能不提升就会停止

**建议**:
- 提高到 `EARLY_STOPPING_PATIENCE = 8`（80个episode才停）
- 或缩短验证间隔到 `VALIDATION_INTERVAL = 5`

---

## 🎯 训练流程图

```
开始训练
   ↓
加载数据（11月1-29日，共29天）
   ↓
划分数据集
   ├─ 训练集: 11月1-20日（70%，约20天）
   ├─ 验证集: 11月21-25日（15%，约4天）
   └─ 测试集: 11月26-29日（15%，约4天）
   ↓
初始化模型和环境
   ↓
┌────────────────────────────────┐
│ Episode Loop (1-50)            │
│   ↓                            │
│ 1. 随机选择训练集的2天          │
│ 2. 运行5760个tick               │
│    - 每30个tick训练一次          │
│    - 每100个tick显示进度         │
│ 3. Episode结束，ε衰减            │
│   ↓                            │
│ 4. 每10个episode验证             │
│    - 顺序遍历验证集每一天         │
│    - 计算平均完成率              │
│    - 与历史最佳比较              │
│   ↓                            │
│ 5. 保存检查点                   │
│    - 最佳模型                   │
│    - 定期检查点（每4个episode）   │
│   ↓                            │
│ 6. 检查早停条件                 │
│    - 5次验证不提升 → 停止        │
└────────────────────────────────┘
   ↓
训练结束
   ↓
自动绘制曲线
   ├─ training_curves.png
   └─ waiting_time_curve.png
   ↓
完成！
```

---

## 🐛 目前不存在的问题

### ❌ 不会出现的错误

1. ~~AttributeError: 'MGCNTrainer' object has no attribute 'best_validation_metric'~~
   **已修复** ✓

2. ~~TypeError: save_checkpoint() got an unexpected keyword argument 'is_best'~~
   **已修复** ✓

3. ~~验证时匹配率为0~~
   **已修复**（强制从day 0开始）✓

4. ~~训练时看起来卡住~~
   **已修复**（添加进度显示）✓

---

## 📁 输出文件位置

### 训练过程中生成的文件

```
work2/
├── results/
│   ├── models/              # 模型检查点
│   │   ├── mgcn_dispatcher_episode_4.pt
│   │   ├── mgcn_dispatcher_episode_8.pt
│   │   └── ... (每SAVE_FREQ个episode一个)
│   │
│   ├── logs/                # 训练日志
│   │   └── training_log_20260118_HHMMSS.txt
│   │
│   └── plots/               # 训练曲线（训练结束后自动生成）
│       ├── training_curves.png
│       └── waiting_time_curve.png
```

---

## 🚀 运行命令

### 本地测试（快速验证）
```bash
python quick_test.py
# 预期时间: 2-5分钟
# 验证代码能跑通
```

### 服务器正式训练
```bash
# 方式1: 前台运行（可以看到实时输出）
python train.py

# 方式2: 后台运行（推荐）
nohup python train.py > train.log 2>&1 &

# 查看日志
tail -f train.log

# 查看进度（另一个终端）
watch -n 30 'tail -50 train.log'
```

---

## 📊 监控训练进度

### 从日志中查看的关键指标

```bash
# 查看Episode进度
grep "===== Episode" train.log

# 查看验证结果
grep "验证完成" train.log

# 查看最佳模型
grep "发现新的最佳模型" train.log

# 查看早停状态
grep "早停" train.log

# 查看每100个tick的进度
grep "📊 Tick" train.log | tail -20
```

### 预期的日志输出示例

```
===== Episode 1/50 =====
  📊 Tick 100/5760 (1.7%) | Buffer: 245/2000 | ⏳ 收集经验中...
  📊 Tick 200/5760 (3.5%) | Buffer: 512/2000 | ⏳ 收集经验中...
  ...
  📊 Tick 2100/5760 (36.5%) | Buffer: 2034 | 训练步数: 12 | 平均Loss: 0.4523
  ...

✓ Episode 1 完成:
  总奖励: 15234.56
  平均Loss: 0.3245
  当前Epsilon: 0.5700
  Buffer大小: 3456

===== Episode 10/50 =====
...

--- 开始验证 ---
验证中 (Day 1/4): 100%|████████| 2880/2880
验证中 (Day 2/4): 100%|████████| 2880/2880
验证中 (Day 3/4): 100%|████████| 2880/2880
验证中 (Day 4/4): 100%|████████| 2880/2880
--- 验证完成: 平均完成率 = 0.7823 ---
发现新的最佳模型！完成率: 0.7823 > 0.0000

===== Episode 20/50 =====
...
--- 验证完成: 平均完成率 = 0.8134 ---
发现新的最佳模型！完成率: 0.8134 > 0.7823

===== Episode 30/50 =====
...
--- 验证完成: 平均完成率 = 0.8098 ---
验证性能未提升. 早停计数: 1/5

...

--- 训练结束 ---
📊 开始绘制训练曲线...
✓ 训练曲线已保存到 results/plots/
```

---

## ✅ 结论

### 可以放心运行 ✓

你的 `train.py` **没有严重问题**，可以直接在服务器上运行正式训练。

### 建议的运行流程

1. **先在服务器上运行快速测试**
   ```bash
   python quick_test.py
   ```
   确认环境配置无误（5分钟）

2. **然后运行正式训练**
   ```bash
   nohup python train.py > train.log 2>&1 &
   ```

3. **定期检查进度**
   ```bash
   tail -f train.log  # 实时查看
   # 或
   watch -n 60 'tail -100 train.log'  # 每分钟刷新
   ```

4. **训练结束后查看曲线**
   ```bash
   ls results/plots/
   # 下载 training_curves.png 和 waiting_time_curve.png 查看
   ```

---

**报告生成时间**: 2026-01-18
**状态**: ✅ 通过检查，可以运行

