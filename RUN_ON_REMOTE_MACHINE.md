# 🚀 在另一台机器上运行训练和测试指南

## 📦 1. 同步代码

### 在当前机器上
```bash
cd /Users/qiukuipeng/PycharmProjects/work2

# 提交所有修改
git add .
git commit -m "修复baseline环境：3格距离限制+随机热点调度+训练显示平均等待时间+分布对比测试"
git push origin my-experiments
```

### 在另一台机器上
```bash
cd /path/to/your/project

# 拉取最新代码
git fetch origin
git checkout my-experiments
git pull origin my-experiments

# 验证关键文件是否更新
ls -lh environment_baseline.py
ls -lh baselines/random_dispatch.py
ls -lh test_baselines_distribution.py
```

---

## 🧪 2. 运行 Baseline 分布对比测试（推荐先运行）

### 快速测试（只跑1天，所有baseline × 所有分布）
```bash
python test_baselines_distribution.py
```

这会测试：
- **Baseline**: Random Walk, Random Dispatch
- **分布**: 均匀 + 正态(σ=1, 3, 5, 7)
- **天数**: 只跑最后1天（快速对比）

### 指定测试天数
```bash
# 测试第0天（第一天）
python test_baselines_distribution.py --day 0

# 测试第5天
python test_baselines_distribution.py --day 5
```

### 输出结果
```
results/vehicles_1800/distribution_tests/
├── baseline_distribution_comparison_20260203_143520.csv
└── baseline_distribution_report_20260203_143520.txt
```

**预计运行时间**: 约 10-15 分钟（2个baseline × 5个分布 = 10次测试，每次约1分钟）

---

## 🎯 3. 运行单个 Baseline（完整7天测试）

### Random Walk
```bash
python baselines/random_walk.py
```

### Random Dispatch
```bash
python baselines/random_dispatch.py
```

### SARSA-SAA
```bash
python baselines/sarsa_saa.py
```

### H-MARL
```bash
python baselines/train_hmarl.py
```

**输出结果**:
```
results/vehicles_1800/baselines/
├── random_walk_last7days_*.csv
├── random_dispatch_last7_raw_*.csv
├── sarsa_saa_results_*.json
└── hmarl_results_*.json
```

---

## 🏋️ 4. 主实验训练（RL模型）

### 标准训练
```bash
python train.py
```

### 指定参数训练
```bash
python train.py \
  --episodes 200 \
  --save-interval 10 \
  --eval-interval 5
```

### 多车辆数训练
```bash
# 1800辆车
python train.py

# 或者在代码中修改 config.TOTAL_VEHICLES
```

**输出**:
```
results/vehicles_1800/
├── models/
│   └── mgcn_dqn_ep_*.pt
├── logs/
│   └── training_log_*.csv
└── evaluations/
    └── eval_results_*.json
```

**训练过程中会显示**:
- 每个 Episode 的完成率、取消率
- ⭐ **平均等待时间** (新增！)
- 实时奖励和 Loss

**预计训练时间**:
- 100 Episodes: 约 2-3 小时
- 200 Episodes: 约 4-6 小时

---

## 🔍 5. 验证修复效果

### 5.1 检查 Baseline 匹配率
运行 Random Walk 后，查看匹配率：
```bash
python baselines/random_walk.py | grep "匹配率\|completion_rate"
```

**期望结果**:
- ✅ 匹配率 ~40-60% (3格距离限制)
- ❌ 如果 >90%，说明限制太宽松
- ❌ 如果 <10%，说明限制太严格

### 5.2 检查 Random Dispatch 调度逻辑
```bash
python baselines/random_dispatch.py 2>&1 | grep "随机选择\|热点"
```

**期望输出**:
```
✓ 使用179个热点格子进行随机调度
```

### 5.3 检查训练过程显示
```bash
python train.py | head -100
```

**期望看到**:
```
Episode 1/100: 完成=45.2%, 取消=12.3%, 平均等待=180.5s, 奖励=12345
```

---

## 📊 6. 分析结果

### 对比 Baseline 与 RL
```bash
# 查看所有结果
ls -lh results/vehicles_1800/baselines/
ls -lh results/vehicles_1800/evaluations/

# 使用 pandas 分析（可选）
python << 'EOF'
import pandas as pd
import glob

# 读取 baseline 结果
rw = pd.read_csv(sorted(glob.glob('results/vehicles_1800/baselines/random_walk_*.csv'))[-1])
rd = pd.read_csv(sorted(glob.glob('results/vehicles_1800/baselines/random_dispatch_*.csv'))[-1])

print("Random Walk 平均完成率:", rw['completion_rate'].mean())
print("Random Dispatch 平均完成率:", rd['completion_rate'].mean())
EOF
```

---

## ⚙️ 7. 配置说明

### 关键配置（config.py）
```python
TOTAL_VEHICLES = 1800           # 车辆数
EPISODE_DAYS = 1                # 每个 Episode 跑几天
TICKS_PER_DAY = 1440            # 每天1440个tick (60秒/tick)
IDLE_THRESHOLD_SEC = 300        # 空闲5分钟后触发调度
```

### Baseline 环境配置（environment_baseline.py）
```python
# 匹配距离限制：3格（曼哈顿距离）
if manhattan_distance > 3:
    continue

# 冷启动：初始所有车辆为serving，1-10分钟内随机释放
# 冷却机制：完成订单后需等待当前tick结束才能再次接单
```

---

## 🐛 8. 常见问题

### Q1: ImportError
```bash
# 确保在项目根目录
pwd  # 应该显示 /path/to/work2

# 检查 Python 路径
python -c "import sys; print(sys.path)"
```

### Q2: 匹配率异常
```bash
# 检查环境版本
grep "manhattan_distance > 3" environment_baseline.py

# 应该看到：
# if manhattan_distance > 3:
#     continue
```

### Q3: Random Dispatch 未执行调度
```bash
# 检查方法是否存在
grep "_execute_random_dispatch_to_hotspots" environment_baseline.py

# 应该看到完整的方法定义
```

### Q4: 训练时不显示平均等待时间
```bash
# 检查 trainer.py
grep "avg_wait" models/trainer.py

# 应该看到：
# avg_wait = sum(waiting_times) / len(waiting_times)
```

---

## 📝 9. 实验清单

建议按以下顺序执行：

- [ ] **步骤1**: 拉取最新代码并验证关键文件
- [ ] **步骤2**: 运行分布对比测试 (`test_baselines_distribution.py`)
- [ ] **步骤3**: 运行 Random Walk baseline (7天)
- [ ] **步骤4**: 运行 Random Dispatch baseline (7天)
- [ ] **步骤5**: 运行主实验训练 (`train.py`)
- [ ] **步骤6**: 对比分析所有结果

---

## 💡 10. 高级用法

### 并行运行多个实验（如果有多GPU）
```bash
# Terminal 1: Random Walk
CUDA_VISIBLE_DEVICES=0 python baselines/random_walk.py &

# Terminal 2: Random Dispatch
CUDA_VISIBLE_DEVICES=1 python baselines/random_dispatch.py &

# Terminal 3: 主训练
CUDA_VISIBLE_DEVICES=2 python train.py &
```

### 批量测试多个车辆数
```bash
for vehicles in 1200 1500 1800; do
    echo "测试 $vehicles 辆车..."
    python train.py --vehicles $vehicles --episodes 100
done
```

---

## 📞 需要帮助？

如果遇到问题，检查以下内容：
1. 代码是否是最新版本（git log）
2. 环境依赖是否完整（requirements.txt）
3. 数据文件是否存在（data/processed/）
4. 配置文件是否正确（config.py）

---

**最后修改时间**: 2026-02-03
**关键修复内容**:
- ✅ Baseline 环境 3 格距离限制
- ✅ Random Dispatch 随机热点调度
- ✅ 训练过程显示平均等待时间
- ✅ 车辆分布对比测试脚本

