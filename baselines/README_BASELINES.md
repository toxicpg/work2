# Baseline方法说明

本目录包含5个对比实验的baseline方法实现。

## 📊 所有Baseline方法

| 序号 | 方法 | 文件 | 描述 |
|------|------|------|------|
| 1 | **Random Walk** | `random_walk.py` | 随机游走：空闲车辆随机移动到相邻网格 |
| 2 | **Random Dispatch** | `random_dispatch.py` | 随机派遣：空闲车辆随机派遣到全局热点网格 |
| 3 | **SARSA-SAA** | `sarsa_saa.py` | SARSA(Δ) + 样本平均近似 |
| 4 | **H-MARL** | `train_hmarl.py` | 层级多智能体强化学习（MFuN） |
| 5 | **CNN-DDQN** | `cnn_ddqn.py` | 卷积神经网络 + Dueling DQN |

---

## 🚀 快速开始

### 方法1：运行单个baseline

```bash
# 从项目根目录运行

# Random Walk
python baselines/random_walk.py

# Random Dispatch
python baselines/random_dispatch.py

# SARSA-SAA
python baselines/sarsa_saa.py

# H-MARL
python baselines/train_hmarl.py

# CNN-DDQN
python baselines/cnn_ddqn.py
```

### 方法2：批量运行所有baseline

```bash
# 使用统一脚本
python run_all_baselines.py
```

运行后会提示选择：
- 运行所有baseline
- 选择特定baseline运行
- 退出

---

## 📁 输出结果

所有baseline的结果会保存在：
```
results/vehicles_{车辆数}/baselines/
├── random_walk_results_*.json
├── random_dispatch_results_*.json
├── sarsa_saa_results_*.json
├── hmarl_results_*.json
└── cnn_ddqn_results_*.json
```

每个结果文件包含：
- `completion_rate`: 完成率
- `cancel_rate`: 取消率
- `avg_waiting_time`: 平均等待时间
- `vehicle_utilization`: 车辆利用率
- `avg_total_revenue`: 平均总收入
- `daily_results`: 每天的详细结果

---

## 🔍 各方法详细说明

### 1. Random Walk（随机游走）

**策略**：当车辆空闲时间超过阈值，随机选择移动方向（上/下/左/右/停留）

**特点**：
- 最简单的baseline
- 无需训练
- 作为性能下界

**运行时间**：约5-10分钟

---

### 2. Random Dispatch（随机派遣）

**策略**：当车辆空闲时间超过阈值，随机派遣到热点网格

**特点**：
- 简单但比Random Walk更智能
- 考虑了热点区域
- 无需训练

**运行时间**：约5-10分钟

---

### 3. SARSA-SAA

**策略**：SARSA(Δ)算法 + Sample Average Approximation

**特点**：
- 基于Q-learning的方法
- 使用历史数据进行需求预测
- 需要简单训练

**运行时间**：约20-30分钟

**参考文献**：Yan et al. (2023) EJOR

---

### 4. H-MARL (MFuN)

**策略**：层级多智能体强化学习

**特点**：
- Manager-Worker架构
- Mean-Field近似
- 需要训练

**运行时间**：约30-60分钟

---

### 5. CNN-DDQN ⭐ 新增

**策略**：卷积神经网络 + Dueling Double DQN

**架构**：
```
Input (400, 5)
  ↓ Reshape to (5, 20, 20)
  ↓ CNN Layers (Conv2D + ReLU + BatchNorm)
  ↓ Flatten
  ↓ Feature Fusion (with position + day embeddings)
  ↓ Dueling DQN Head (Value + Advantage)
  ↓ Q-values (179,)
```

**特点**：
- 使用CNN提取空间特征
- 不使用图结构
- 经典深度强化学习baseline

**运行时间**：约30-60分钟

**模型参数**：约3.5M参数

---

## 📊 对比分析

预期性能排序（从低到高）：

```
Random Walk < Random Dispatch < SARSA-SAA ≈ H-MARL ≈ CNN-DDQN < Ours (MGCN-DDQN)
```

**关键对比点**：

1. **Random Walk vs Random Dispatch**：
   - 验证考虑热点的重要性

2. **SARSA-SAA vs H-MARL vs CNN-DDQN**：
   - 对比不同强化学习方法

3. **CNN-DDQN vs Ours**：
   - 验证图卷积网络（MGCN）的优势
   - CNN只能捕捉局部空间关系
   - MGCN可以捕捉全局拓扑结构

---

## 🔧 故障排查

### 问题1：模块导入错误

```bash
ImportError: No module named 'config'
```

**解决**：确保从项目根目录运行，或检查Python路径

---

### 问题2：CUDA内存不足

```bash
RuntimeError: CUDA out of memory
```

**解决**：
1. 减小batch_size（在config.py中）
2. 使用CPU模式：`DEVICE = 'cpu'`

---

### 问题3：数据文件未找到

```bash
FileNotFoundError: data/processed/...
```

**解决**：确保已运行数据预处理脚本

---

## 📈 结果可视化

运行完所有baseline后，使用以下脚本生成对比图：

```bash
python utils/shiyantu.py
```

会生成：
- 匹配率对比图
- 等待时间对比图
- 平均性能柱状图

---

## 🎯 下一步

1. 运行所有baseline获取结果
2. 运行你的完整模型（MGCN-DDQN）
3. 使用`utils/shiyantu.py`生成对比图表
4. 分析结果，撰写论文

---

## 📝 引用

如果你使用了这些baseline方法，请引用相关文献：

- **SARSA-SAA**: Yan et al. (2023) "Deep Reinforcement Learning for Ride-hailing Platform Order Dispatching", EJOR
- **H-MARL**: Yang et al. (2020) "Mean Field Multi-Agent Reinforcement Learning", ICML
- **CNN-DDQN**: Based on Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning", ICML

---

## ⚠️ 注意事项

1. 所有baseline使用相同的数据集和评估指标，确保公平对比
2. CNN-DDQN使用与主模型相同的训练参数（学习率、batch_size等）
3. 建议在GPU上运行深度学习方法（CNN-DDQN, H-MARL）
4. 结果可能因随机种子而略有不同

---

## 📧 联系

如有问题，请查看主README或提Issue。

