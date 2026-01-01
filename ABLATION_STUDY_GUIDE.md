# 消融实验指南

## 概述

消融实验（Ablation Study）是一种系统的方法，通过逐个移除或修改模型的关键组件，来评估每个组件对最终性能的贡献度。

本项目提供了完整的消融实验框架，支持以下组件的消融：

1. **MGCN 图卷积网络** - 替换为简化的 MLP
2. **Dueling DQN** - 替换为标准 DQN
3. **优先级经验回放 (PER)** - 替换为统一采样
4. **多阶段奖励函数** - 替换为简化的单阶段奖励
5. **注意力融合** - 替换为简单拼接
6. **组合消融** - 多个组件的组合

## 快速开始

### 1. 运行所有消融实验

```bash
python run_ablation_simple.py --episodes 5
```

这将运行所有 7 种消融类型，每种类型 5 个 episode。

### 2. 运行特定的消融类型

```bash
# 运行完整模型（基准）
python run_ablation_simple.py --ablation full_model --episodes 10

# 运行无 MGCN 的版本
python run_ablation_simple.py --ablation no_mgcn --episodes 10

# 运行无 Dueling DQN 的版本
python run_ablation_simple.py --ablation no_dueling --episodes 10

# 运行无 PER 的版本
python run_ablation_simple.py --ablation no_per --episodes 10

# 运行无多阶段奖励的版本
python run_ablation_simple.py --ablation no_multi_stage_reward --episodes 10

# 运行无注意力融合的版本
python run_ablation_simple.py --ablation no_attention_fusion --episodes 10

# 运行最小化模型
python run_ablation_simple.py --ablation minimal --episodes 10
```

## 消融类型详解

### full_model（完整模型 - 基准）

包含所有优化组件：
- ✓ MGCN 图卷积网络
- ✓ Dueling DQN 架构
- ✓ 优先级经验回放 (PER)
- ✓ 多阶段奖励函数
- ✓ 注意力融合

这是性能的上界。

### no_mgcn（无 MGCN）

使用简化的 MLP 替代 MGCN：
- ✗ MGCN 被替换为 MLP
- ✓ 其他组件保持不变

**作用**: 评估 MGCN 对空间特征提取的贡献

### no_dueling（无 Dueling DQN）

使用标准 DQN 替代 Dueling DQN：
- ✓ MGCN 保持不变
- ✗ Dueling DQN 被替换为标准 DQN
- ✓ 其他组件保持不变

**作用**: 评估 Dueling 架构（值流和优势流分离）的贡献

### no_per（无 PER）

使用统一采样替代优先级经验回放：
- ✓ MGCN 保持不变
- ✓ Dueling DQN 保持不变
- ✗ PER 被替换为统一采样
- ✓ 其他组件保持不变

**作用**: 评估优先级采样对训练效率的贡献

### no_multi_stage_reward（无多阶段奖励）

使用简化的单阶段奖励替代多阶段奖励：
- ✓ 其他组件保持不变
- ✗ 多阶段奖励被替换为简化奖励

简化奖励配置：
```python
REWARD_WEIGHTS = {
    'W_MATCH': 1.0,      # 仅保留匹配奖励
    'W_WAIT': 0.0,       # 禁用等待时间惩罚
    'W_CANCEL': 1.0,     # 仅保留取消惩罚
    'W_WAIT_SCORE': 0.0, # 禁用等待评分
    'W_COMPLETION': 1.0, # 仅保留完成奖励
    'W_MATCH_SPEED': 0.0 # 禁用快速匹配奖励
}
```

**作用**: 评估复杂奖励函数设计的贡献

### no_attention_fusion（无注意力融合）

使用简单拼接替代注意力融合：
- ✓ 其他组件保持不变
- ✗ 注意力融合被替换为简单拼接

**作用**: 评估注意力机制对特征融合的贡献

### minimal（最小化模型）

保留最基础的组件：
- ✓ MGCN 保持不变
- ✗ Dueling DQN 被替换为标准 DQN
- ✗ PER 被替换为统一采样
- ✗ 多阶段奖励被替换为简化奖励
- ✗ 注意力融合被替换为简单拼接

**作用**: 评估所有优化组件的累积贡献

## 输出和结果分析

### 日志文件

每个消融实验会生成日志文件：
```
results/logs/ablation_<ablation_type>_<timestamp>.txt
```

### 结果文件

所有消融实验的结果保存在：
```
results/ablation_studies/ablation_results_<timestamp>.json
```

### 对比报告

运行完成后会打印对比报告，包括：

1. **性能对比表格**
   - 平均奖励 (Avg Reward)
   - 标准差 (Std Reward)
   - 平均损失 (Avg Loss)
   - 最终 Epsilon 值

2. **性能差异分析**
   - 相对于完整模型的性能差异
   - 百分比变化

示例输出：
```
==================================================
性能差异分析 (相对于完整模型)
==================================================

no_mgcn
  平均奖励: 1234.56 (vs 1456.78, 差异: -222.22, -15.3%)
  平均损失: 0.0234

no_dueling
  平均奖励: 1398.90 (vs 1456.78, 差异: -57.88, -4.0%)
  平均损失: 0.0198
```

## 解释消融实验结果

### 关键指标

1. **平均奖励 (Avg Reward)**
   - 越高越好
   - 反映模型的总体性能

2. **平均损失 (Avg Loss)**
   - 越低越好
   - 反映训练的稳定性

3. **标准差 (Std Reward)**
   - 越低越好
   - 反映性能的稳定性

### 如何判断组件的重要性

假设完整模型的平均奖励为 1456.78：

- **性能下降 > 20%**: 该组件非常重要，对模型性能有显著影响
- **性能下降 10-20%**: 该组件较为重要，有一定的贡献
- **性能下降 5-10%**: 该组件有适度的贡献
- **性能下降 < 5%**: 该组件的贡献较小

### 示例分析

假设你得到以下结果：

```
完整模型: 平均奖励 = 1456.78

no_mgcn: 平均奖励 = 1234.56 (-15.3%)  ← MGCN 很重要
no_dueling: 平均奖励 = 1398.90 (-4.0%)  ← Dueling DQN 不太重要
no_per: 平均奖励 = 1420.34 (-2.5%)  ← PER 影响较小
```

**结论**:
1. MGCN 是最关键的组件（下降 15.3%）
2. Dueling DQN 有一定作用（下降 4.0%）
3. PER 的作用较小（下降 2.5%）

## 高级用法

### 1. 自定义消融类型

编辑 `models/ablation_dispatcher.py` 中的 `AblationDispatcher` 类，添加新的消融类型。

### 2. 修改消融参数

编辑 `ablation_study.py` 中的 `AblationConfig.ABLATION_TYPES` 字典。

### 3. 批量运行实验

```bash
# 运行多个 episode 来获得更稳定的结果
for i in {1..3}; do
    python run_ablation_simple.py --episodes 20
done
```

### 4. 分析结果

```python
import json
import pandas as pd

# 加载结果
with open('results/ablation_studies/ablation_results_<timestamp>.json', 'r') as f:
    results = json.load(f)

# 转换为 DataFrame
data = []
for ablation_type, result in results.items():
    data.append({
        'Type': ablation_type,
        'Avg Reward': result['avg_reward'],
        'Avg Loss': result['avg_loss']
    })

df = pd.DataFrame(data)
print(df.sort_values('Avg Reward', ascending=False))
```

## 最佳实践

1. **运行足够的 episode**
   - 至少 5-10 个 episode 来获得稳定的结果
   - 对于关键的消融类型，运行 20+ 个 episode

2. **固定随机种子**
   - 确保可重现性
   - 使用相同的数据集

3. **记录实验条件**
   - 记录运行时间、硬件配置等
   - 便于后续对比

4. **逐步分析**
   - 先运行完整模型和最小化模型
   - 再逐个移除单个组件
   - 最后分析组件之间的相互作用

5. **考虑统计显著性**
   - 如果性能差异很小（< 2%），可能不具有统计显著性
   - 运行更多 episode 来确认

## 常见问题

### Q: 为什么某个消融实验的性能比完整模型还好？

A: 这可能是由于：
1. 随机波动（运行更多 episode 来确认）
2. 该组件在某些情况下可能会引入噪声
3. 数据集的特殊性质

### Q: 消融实验需要多长时间？

A: 取决于：
- Episode 数量
- 数据集大小
- 硬件配置

一般来说：
- 5 个 episode：约 30-60 分钟
- 10 个 episode：约 1-2 小时
- 20 个 episode：约 2-4 小时

### Q: 如何比较两个不同的消融实验？

A: 使用结果 JSON 文件，对比以下指标：
- 平均奖励
- 平均损失
- 标准差
- 收敛速度

## 参考资源

- [Ablation Study - Wikipedia](https://en.wikipedia.org/wiki/Ablation_(machine_learning))
- [How to Design and Interpret a Robustness Study](https://openreview.net/forum?id=Uytzf1I8M3)
- [A Primer on Neural Network Architectures for Natural Language Processing](https://arxiv.org/abs/1807.02291)

