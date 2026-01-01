# 🧠 Models 模块完全指南

> 如果你刚刚回到这个项目，从这里开始！

## 📚 三份完整文档

我为你生成了三份详细的学习文档：

### 1. **MODELS_UNDERSTANDING.md** - 完整理论讲解
- 📊 整体架构图
- 🔧 分层讲解（从下往上）
- 🔄 数据流动示例
- 📐 维度变化总结
- 🎯 关键参数解释

**适合**：想要深入理解每一层的作用和原理

### 2. **MODELS_CODE_WALKTHROUGH.md** - 代码逐行讲解
- 🚀 GCN.py 详细代码注释
- 🚀 MGCN_Separate.py 详细代码注释
- 🚀 dispatcher.py 详细代码注释
- 🚀 Dueling 架构讲解
- 📊 完整数据流示例

**适合**：想要理解具体代码如何实现

### 3. **MODELS_QUICK_REFERENCE.md** - 快速参考卡片
- 🎯 核心问题与答案
- 🏗️ 架构速查表
- 🔢 关键参数
- 📊 维度变化速查
- 🎮 常见操作代码

**适合**：快速查询，或者已经理解基础想要快速回顾

---

## 🎯 学习路径建议

### 第一次学习（完整理解）
```
1. 读 MODELS_UNDERSTANDING.md
   ↓
2. 对照代码读 MODELS_CODE_WALKTHROUGH.md
   ↓
3. 自己运行代码，看维度变化
   ↓
4. 用 MODELS_QUICK_REFERENCE.md 检查理解
```

### 快速回顾（已学过）
```
1. 打开 MODELS_QUICK_REFERENCE.md
   ↓
2. 看关键问题与答案
   ↓
3. 需要细节时查 MODELS_CODE_WALKTHROUGH.md
```

### 调试代码
```
1. 查 MODELS_QUICK_REFERENCE.md 的"常见问题"
   ↓
2. 参考维度变化速查
   ↓
3. 对照代码和讲解文档
```

---

## 🔑 核心概念速记

### 三层网络

```
第1层：GCN (图卷积)
  作用：在单个图上做卷积
  输入：特征 + 邻接矩阵
  输出：卷积后的特征

第2层：MGCN_Separate (多图融合)
  作用：同时处理两个图，然后融合
  输入：特征 + 两个邻接矩阵
  输出：融合后的特征

第3层：MGCNVehicleDispatcher (决策网络)
  作用：整合所有信息，输出调度决策
  输入：城市状态 + 车辆位置 + 时间
  输出：Q值（每个热点一个）
```

### 三个关键设计

```
1. 分离邻接图和POI图
   为什么？捕捉不同类型的关系

2. 使用Dueling架构
   为什么？分离价值和优势，学习更稳定

3. 全局池化 + 嵌入 + 融合网络
   为什么？整合地理、时间、功能等多个信息源
```

### 三个关键维度

```
400 个网格
  ↓
5 维输入特征（订单、空闲车、繁忙车、sin(t)、cos(t)）
  ↓
32 维隐藏特征（经过 MGCN）
  ↓
64 维融合特征（加上位置和时间嵌入）
  ↓
179 维输出（每个热点一个 Q 值）
```

---

## 💻 快速上手

### 创建模型
```python
from models.dispatcher import create_dispatcher
from config import Config

config = Config()
model = create_dispatcher(config, neighbor_adj, poi_adj, 'dueling')
model = model.to(config.DEVICE)
```

### 前向传播
```python
node_features = torch.randn(32, 400, 5)      # 32个样本
vehicle_locs = torch.randint(0, 400, (32,))  # 32个车辆位置
day_of_week = torch.randint(0, 7, (32,))     # 32个星期几

q_values = model(node_features, vehicle_locs, day_of_week)
# 输出: (32, 179) - 每个样本179个热点的Q值
```

### 选择动作
```python
with torch.no_grad():
    action, q_values = model.select_action(
        node_features[0],
        vehicle_location=100,
        day_of_week=2,
        epsilon=0.1
    )
    # action: 0-178 的整数
```

---

## 🔍 理解数据流

### 输入数据是什么？

**node_features (B, 400, 5)**
```
每个网格的特征：
  特征0: 订单数量 (0-100)
  特征1: 空闲车辆数 (0-500)
  特征2: 繁忙车辆数 (0-500)
  特征3: sin(时间) (-1 到 1)
  特征4: cos(时间) (-1 到 1)
```

**vehicle_locations (B,)**
```
车辆当前所在的网格编号
0-399 中的任意一个
```

**day_of_week (B,)**
```
星期几
0=星期一, 1=星期二, ..., 6=星期日
```

### 输出数据是什么？

**q_values (B, 179)**
```
每个热点的 Q 值（价值估计）
数值越高 = 把车派到这个热点越好

例如：[2.1, 1.5, 3.2, 2.8, ...]
  热点0: Q=2.1
  热点1: Q=1.5
  热点2: Q=3.2 ← 最好的选择
  ...
```

---

## 🎓 进阶理解

### 为什么用图卷积？

```
城市是一个图：
  - 节点：400个网格
  - 边：相邻关系或POI相似性

GCN的优势：
  - 自动学习图的结构
  - 每个节点聚合邻接节点的信息
  - 比普通MLP更适合有结构的数据
```

### 为什么要融合两个图？

```
邻接图（地理关系）：
  网格100 ← 连接到 99, 101, 80, 120
  学到：地理位置很重要

POI图（功能相似性）：
  网格100（商业区）← 连接到其他商业区
  学到：功能类型很重要

融合后：
  既知道地理位置，也知道功能类型
  决策更准确
```

### 为什么要用Dueling架构？

```
传统DQN：
  Q(s,a) = 直接预测
  问题：当多个动作都好时，学习不稳定

Dueling DQN：
  V(s) = 这个状态本身有多好
  A(s,a) = 这个动作相对于其他动作有多好
  Q(s,a) = V(s) + A(s,a) - mean(A)

  优势：
  - V(s)学习稳定（与动作无关）
  - A(s,a)学习动作差异
  - 整体收敛更快
```

---

## 🐛 常见问题解决

### Q: 维度不匹配错误
**检查清单：**
- [ ] node_features 是 (B, 400, 5) 吗？
- [ ] vehicle_locations 是 (B,) 吗？
- [ ] day_of_week 是 (B,) 吗？
- [ ] 所有张量都在同一设备上吗？

### Q: 模型输出全是 NaN
**检查清单：**
- [ ] 输入数据有 NaN 吗？
- [ ] 学习率太高了吗？
- [ ] 梯度爆炸了吗？（看 grad_norm）
- [ ] 邻接矩阵有问题吗？

### Q: 模型性能不好
**检查清单：**
- [ ] 输入特征归一化了吗？
- [ ] 超参数调整过吗？
- [ ] 训练数据充分吗？
- [ ] 奖励函数合理吗？

---

## 📈 性能优化建议

### 内存优化
```python
# 预分配缓冲区而不是每次新建
self._idle_dist_buffer = np.zeros(self.config.NUM_GRIDS)

# 复用而不是新建
self._idle_dist_buffer.fill(0)
```

### 速度优化
```python
# 使用 einsum 替代循环
# 使用批量操作而不是逐个处理
# 缓存 K-D Tree 而不是每次重建
```

### 显存优化
```python
# 邻接矩阵存在 CPU，需要时移到 GPU
self.register_buffer('A_neighbor', kernels)

# 前向传播时：
A_on_device = self.A_neighbor.to(x.device)
```

---

## 🔗 与其他模块的关系

```
models/
├── GCN.py ──────────┐
├── MGCN_Separate.py ├──→ dispatcher.py
└── dispatcher.py ───┤
                     ├──→ trainer.py (训练)
                     ├──→ environment.py (推理)
                     └──→ evaluate.py (评估)
```

**调用关系：**
```
train.py
  └─ trainer.train_episode()
       ├─ 调用 dispatcher.forward() 计算 Q 值
       ├─ 调用 dispatcher.select_action() 选择动作
       └─ 计算损失并反向传播

environment.py
  └─ env._execute_proactive_dispatch()
       ├─ 调用 dispatcher.select_action() 做决策
       └─ 获取调度目标网格
```

---

## 📚 完整学习清单

- [ ] 理解 GCN 的基本原理
- [ ] 理解邻接矩阵的预处理
- [ ] 理解 MGCN_Separate 的两个分支
- [ ] 理解 Attention Pooling 的作用
- [ ] 理解 Embedding 层的作用
- [ ] 理解融合网络的设计
- [ ] 理解 Dueling 架构的优势
- [ ] 能够手工计算维度变化
- [ ] 能够修改超参数
- [ ] 能够调试维度错误

---

## 🎯 下一步

### 想理解训练过程？
→ 看 `models/trainer.py` 和 `TRAINING_GUIDE.md`

### 想理解环境交互？
→ 看 `environment.py` 和 `ENVIRONMENT_GUIDE.md`

### 想理解数据处理？
→ 看 `utils/data_process.py` 和 `DATA_GUIDE.md`

### 想理解完整流程？
→ 看 `train.py` 的主函数

---

## 💡 最后的话

这个 models 模块的设计很精妙：
- **分层清晰**：GCN → MGCN → Dispatcher
- **信息融合**：地理 + 功能 + 位置 + 时间
- **算法先进**：Dueling + PER + Double DQN
- **代码优化**：einsum + 缓冲区 + 批处理

现在你已经完全理解了！🎉

**有任何问题，查看相应的文档或代码注释。**

---

**最后更新：2025年12月31日**
**文档作者：CatPaw AI**

