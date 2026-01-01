# 🚀 快速开始指南 - 方案 B 多阶段奖励

## 5 分钟快速了解

### 问题
- 乘客等待时间长
- 订单匹配率低

### 解决方案
在订单生命周期的**三个节点**分别给予反馈:

```
1️⃣ 匹配时: 快速匹配 → 奖励 ✓
2️⃣ 完成时: 低等待时间 → 奖励 ✓
3️⃣ 取消时: 未能匹配 → 惩罚 ✗
```

### 核心公式

| 节点 | 公式 | 作用 |
|------|------|------|
| 匹配 | `1.2 + 0.5 * exp(-等待时间/240)` | 鼓励快速匹配 |
| 完成 | `2.0 + 0.4 * exp(-总等待/240) - 1.8 * (总等待/240)` | 平衡完成和等待 |
| 取消 | `-1.0 * (等待时间/300)` | 惩罚未匹配 |

---

## 🎯 使用步骤

### Step 1: 训练模型
```bash
# 使用默认配置训练
python -m models.train
```

### Step 2: 查看训练曲线
```bash
# 每隔几个 Episode 运行此命令
python plot_training_curves.py
```

### Step 3: 解读曲线

**好的迹象** ✓:
- Loss 逐渐下降
- Completion Rate 上升到 85%+
- Avg Waiting Time 下降
- Epsilon 平滑衰减

**需要调整** ⚠️:
- Loss 不下降 → 降低学习率
- Completion Rate 低 → 增加 `W_MATCH`
- 等待时间长 → 增加 `W_WAIT_SCORE`
- 取消率高 → 增加 `W_CANCEL`

### Step 4: 调整超参数
```python
# config.py 中修改权重
REWARD_WEIGHTS = {
    'W_MATCH': 1.5,        # 从 1.2 改为 1.5
    'W_COMPLETION': 2.5,   # 从 2.0 改为 2.5
    # ... 其他权重
}
```

### Step 5: 验证改进
```bash
# 重新训练 5-10 个 Episode
python -m models.train

# 查看改进效果
python plot_training_curves.py
```

---

## 📊 监控指标

### 目标值 (理想情况)

| 指标 | 目标 | 范围 |
|------|------|------|
| 完成率 | > 90% | 85-95% |
| 平均等待时间 | < 150s | 100-200s |
| 匹配率 | > 92% | 90-98% |
| 取消率 | < 8% | 5-10% |
| Loss | < 0.01 | 0.001-0.05 |

### 曲线特征

```
Episode 1-10:    快速上升期
  ├─ Completion Rate ↑↑↑
  ├─ Loss ↓↓↓
  └─ Reward 波动

Episode 10-30:   优化期
  ├─ Avg Waiting Time ↓↓
  ├─ Match Rate ↑↑
  └─ Loss 继续 ↓

Episode 30+:     稳定期
  ├─ 所有指标稳定
  ├─ Loss 保持低水平
  └─ Reward 保持高水平
```

---

## 🔧 常见调整

### 问题 1: 完成率低 (< 80%)

```python
# 修改 config.py
REWARD_WEIGHTS = {
    'W_MATCH': 1.5,        # ↑ 增加
    'W_COMPLETION': 2.5,   # ↑ 增加
    'W_MATCH_SPEED': 0.7   # ↑ 增加
}
```

**预期**: 5-10 个 Episode 内完成率上升到 85%+

### 问题 2: 等待时间长 (> 200s)

```python
# 修改 config.py
REWARD_WEIGHTS = {
    'W_WAIT_SCORE': 0.8,   # ↑ 增加
    'W_WAIT': 2.5          # ↑ 增加
}
REWARD_FORMULA_V4 = {
    'T_CHARACTERISTIC': 180  # ↓ 减小
}
```

**预期**: 10-20 个 Episode 内等待时间下降 20-30%

### 问题 3: 取消率高 (> 15%)

```python
# 修改 config.py
REWARD_WEIGHTS = {
    'W_CANCEL': 2.0,       # ↑ 增加
    'W_MATCH': 1.5         # ↑ 增加
}
```

**预期**: 5-10 个 Episode 内取消率下降到 < 10%

### 问题 4: Loss 不下降

```python
# 修改 config.py
LEARNING_RATE = 5e-5      # ↓ 降低
EPSILON_DECAY = 0.90      # ↑ 增加
BATCH_SIZE = 512          # ↑ 增加
```

**预期**: 10-20 个 Episode 内 Loss 开始下降

---

## 📈 曲线解读

### Loss 曲线
```
理想:     ╲╲╲___  (快速下降后稳定)
需调整:   ════════  (始终不变)
过拟合:   ╲╲╲╱╱╱  (下降后上升)
```

### Reward 曲线
```
理想:     ╱╱╱───  (上升后稳定在高水平)
需调整:   ═══╱╱╱  (缓慢上升)
过拟合:   ╱╱╱╲╲╲  (上升后下降)
```

### Completion Rate 曲线
```
理想:     ╱╱╱───  (快速上升到 85%+)
需调整:   ╱╱─────  (缓慢上升)
问题:     ═══════  (保持低水平)
```

### Avg Waiting Time 曲线
```
理想:     ╲╲╲───  (快速下降到 150s)
需调整:   ═══╲╲╲  (缓慢下降)
问题:     ╱╱╱╱╱  (上升)
```

---

## 💡 调整技巧

### 技巧 1: 一次只调一个参数
❌ 不要这样:
```python
W_MATCH = 2.0
W_COMPLETION = 3.0
W_WAIT = 3.0
```

✅ 应该这样:
```python
W_MATCH = 1.5  # 只改这个
# 其他保持不变
# 训练 5-10 个 Episode 后评估效果
```

### 技巧 2: 小步调整
❌ 不要这样:
```python
W_MATCH = 1.2 → 2.5  # 跳跃太大
```

✅ 应该这样:
```python
W_MATCH = 1.2 → 1.3 → 1.4 → 1.5  # 逐步调整
```

### 技巧 3: 记录每次调整
```python
# config.py 中添加注释
# Episode 1-5: W_MATCH=1.2, Completion=75%
# Episode 6-10: W_MATCH=1.5, Completion=85%
# Episode 11-15: W_MATCH=1.5, Completion=88%
```

### 技巧 4: 等待足够的 Episode
❌ 不要这样:
```python
# 调整后立即查看效果
W_MATCH = 1.5
# 运行 1 个 Episode
python plot_training_curves.py  # 太早！
```

✅ 应该这样:
```python
# 调整后运行多个 Episode
W_MATCH = 1.5
# 运行 5-10 个 Episode
python plot_training_curves.py  # 现在可以评估
```

---

## 🎓 学习资源

### 必读文档
1. **IMPLEMENTATION_SUMMARY.md** - 实现总结 (10 分钟)
2. **REWARD_FUNCTION_PLAN_B.md** - 详细设计 (30 分钟)
3. **HYPERPARAMETER_TUNING_GUIDE.md** - 调整指南 (20 分钟)

### 推荐阅读顺序
```
1. 本文档 (QUICK_START.md)
   ↓
2. IMPLEMENTATION_SUMMARY.md
   ↓
3. 运行训练和可视化
   ↓
4. REWARD_FUNCTION_PLAN_B.md (深入理解)
   ↓
5. HYPERPARAMETER_TUNING_GUIDE.md (调整优化)
```

---

## ✅ 检查清单

开始训练前:
- [ ] 已安装依赖: `pip install torch matplotlib numpy pandas`
- [ ] 已创建目录: `mkdir -p results/plots results/models results/logs`
- [ ] 已备份配置: `cp config.py config.py.backup`
- [ ] 已阅读本文档

训练中:
- [ ] 每隔 5 个 Episode 运行 `python plot_training_curves.py`
- [ ] 记录关键指标变化
- [ ] 如有异常立即参考"常见调整"

训练后:
- [ ] 所有指标达到目标值
- [ ] 曲线呈现理想特征
- [ ] 保存最终 checkpoint

---

## 🚨 常见错误

### ❌ 错误 1: 同时调整多个参数
```python
# 错误
W_MATCH = 1.5
W_COMPLETION = 2.5
W_WAIT = 2.5
```
**后果**: 无法判断哪个参数有效

### ❌ 错误 2: 调整后立即评估
```python
# 错误
W_MATCH = 1.5
python plot_training_curves.py  # 运行 1 个 Episode 后
```
**后果**: 数据不足，无法得出结论

### ❌ 错误 3: 权重调整过大
```python
# 错误
W_MATCH = 1.2 → 3.0  # 增加 2.5 倍
```
**后果**: 奖励信号过强，模型不稳定

### ❌ 错误 4: 忽视曲线异常
```python
# 错误
Loss 始终很高 → 继续训练而不调整
```
**后果**: 浪费计算资源，无法改进

---

## 📞 快速问题解答

**Q: 应该训练多少个 Episode?**
A: 建议至少 50 个 Episode，前 10 个用于初期调整，20-40 个用于优化，40+ 个用于稳定。

**Q: 每次调整后要训练多久?**
A: 建议至少 5-10 个 Episode 再评估效果，前期可能有波动。

**Q: 如何选择合适的权重?**
A: 从默认值开始，根据曲线特征逐步调整。参考 HYPERPARAMETER_TUNING_GUIDE.md 中的场景。

**Q: 曲线波动大是正常的吗?**
A: 初期波动是正常的，但应该逐渐平稳。如果后期仍波动大，增加 EPSILON_START 以保持探索。

**Q: 如何知道何时停止调整?**
A: 当所有指标都达到目标值且保持稳定 5-10 个 Episode 后，可以停止调整。

---

## 🎯 成功标志

当你看到以下特征时，说明训练成功 ✅:

```
✓ Completion Rate > 90% 并保持稳定
✓ Avg Waiting Time < 150s 并继续下降
✓ Match Rate > 92% 并保持稳定
✓ Cancel Rate < 8% 并保持稳定
✓ Loss < 0.01 并趋于稳定
✓ Reward 保持在高水平
✓ 所有曲线呈现理想特征
```

---

## 📚 附录

### 权重快速参考

| 权重 | 默认值 | 增加效果 | 减少效果 |
|------|--------|---------|---------|
| W_MATCH | 1.2 | 更快匹配 | 匹配率下降 |
| W_COMPLETION | 2.0 | 更多完成 | 完成率下降 |
| W_WAIT_SCORE | 0.4 | 奖励短等待 | 忽视等待 |
| W_WAIT | 1.8 | 惩罚长等待 | 忽视长等待 |
| W_CANCEL | 1.0 | 严厉惩罚 | 取消率上升 |
| W_MATCH_SPEED | 0.5 | 强调速度 | 忽视速度 |

### 时间参数快速参考

| 参数 | 默认值 | 增加效果 | 减少效果 |
|------|--------|---------|---------|
| T_CHARACTERISTIC | 240s | 对长等待宽容 | 对长等待严格 |
| MAX_WAITING_TIME | 300s | 更宽松的超时 | 更严格的超时 |

---

**祝你训练顺利！** 🎉

如有问题，请参考详细文档或联系技术支持。

