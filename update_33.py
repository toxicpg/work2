#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Read the file
with open('METHODOLOGY_CHAPTER.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line numbers for 3.3.2 and 3.3.3
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if '#### 动机' in line and i > 990:  # Around line 999
        start_idx = i
    if '### 3.3.3 目标网络与双 DQN' in line:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print(f"Could not find sections: start={start_idx}, end={end_idx}")
    exit(1)

print(f"Found 3.3.2 section from line {start_idx} to {end_idx}")

# New content for 3.3.2
new_content = """#### 3.3.2.1 标准经验回放的局限性

虽然标准的经验回放已经大大改善了 DQN 的训练稳定性，但它仍然存在一个关键问题：**所有经验被等同对待**。

具体来说，在标准 DQN 中，缓冲区中的每条经验被采样的概率相等。然而，从学习的角度看，不同的经验对模型的贡献是不同的：

- **"惊喜"的经验**（即 TD 误差大的经验）表明模型的预测与实际结果差异大，这些经验对学习最有价值
- **"预期内"的经验**（即 TD 误差小的经验）表明模型预测准确，对学习的贡献较小

标准均匀采样的问题：

1. **样本效率低下**：浪费计算资源在已经学好的样本上
2. **学习缓慢**：模型需要更多步骤才能学到困难的任务
3. **收敛不稳定**：某些关键但稀少的经验可能被忽视

#### 3.3.2.2 优先级经验回放的核心思想

**优先级经验回放 (Prioritized Experience Replay, PER)** 通过根据经验的**学习价值**（用 TD 误差衡量）来调整采样概率，从而优先学习"惊喜"的经验。

关键直觉：如果一条经验的 TD 误差大，说明模型对这条经验的预测偏离实际很远，这条经验能够提供更多的学习信号。因此，我们应该更频繁地采样这样的经验。

#### 3.3.2.3 优先级的定义与计算

对于缓冲区中的第 $i$ 条经验，其**优先级** (priority) 定义为：

$$p_i = (|TD\\_error_i| + \\epsilon)^\\alpha$$

其中：

- $|TD\\_error_i| = |r_i + \\gamma \\max_{a'} Q(s_i', a'; \\theta^-) - Q(s_i, a_i; \\theta)|$：第 $i$ 条经验的 TD 误差绝对值
- $\\epsilon = 1 \\times 10^{-6}$：一个小的常数，确保即使 TD 误差为 0 的经验也有被采样的概率（避免某些经验永远不被采样）
- $\\alpha \\in [0, 1]$：优先级指数，控制优先级的强度

采样概率为：

$$P(i) = \\frac{p_i}{\\sum_j p_j}$$

**$\\alpha$ 参数的含义**：

- $\\alpha = 0$：优先级对采样概率无影响，退化为标准均匀采样
- $\\alpha = 1$：采样概率完全由 TD 误差决定，最激进的优先级采样
- $0 < \\alpha < 1$：介于两者之间，在我们的工作中取 $\\alpha = 0.4$，提供中等强度的优先级

在我们的设置中，$\\alpha = 0.4$ 的选择基于以下考虑：

1. **避免过度优先化**：如果 $\\alpha$ 过大，某些高 TD 误差的样本会被反复采样，可能导致过拟合
2. **保持多样性**：中等强度的 $\\alpha$ 保证了即使 TD 误差较小的经验也有合理的采样概率
3. **稳定训练**：经验表明 $\\alpha = 0.4$ 能够在多种任务上提供稳定的性能改进

#### 3.3.2.4 重要性采样权重的补偿机制

优先级采样改变了数据的分布。具体来说，高 TD 误差的样本被过度采样，而低 TD 误差的样本被欠采样。这会导致**分布偏差** (distribution bias)。

为了补偿这种偏差，我们使用**重要性采样权重** (Importance Sampling Weights)。对于第 $i$ 条经验，其重要性采样权重定义为：

$$w_i = \\left(\\frac{1}{N \\cdot P(i)}\\right)^\\beta$$

其中：

- $N = 50,000$：缓冲区大小
- $P(i)$：第 $i$ 条经验的采样概率
- $\\beta \\in [0, 1]$：重要性采样指数，控制补偿强度

**$\\beta$ 参数的含义**：

- $\\beta = 0$：无补偿，采样权重全为 1（可能导致有偏估计）
- $\\beta = 1$：完全补偿，使得加权后的梯度期望是无偏的
- $0 < \\beta < 1$：部分补偿

在实际训练中，$\\beta$ 从初始值 $\\beta_0 = 0.4$ 逐步增加到 1，在 100,000 个训练步骤后达到 1。这种**渐进的补偿策略**的好处：

1. **早期训练的稳定性**：早期训练时，优先级采样的效果尚未充分体现，较小的 $\\beta$ 避免过度补偿
2. **逐步适应**：模型逐步适应优先级采样带来的偏差
3. **最终无偏**：最终 $\\beta = 1$ 确保梯度期望无偏

$\\beta$ 的调度函数为：

$$\\beta_t = \\min\\left(1, \\beta_0 + (1 - \\beta_0) \\cdot \\frac{t}{T_\\beta}\\right)$$

其中 $t$ 是当前训练步数，$T_\\beta = 100,000$ 是达到 $\\beta = 1$ 所需的步数。

#### 3.3.2.5 修正的损失函数

使用 PER 后，DQN 的损失函数修改为：

$$\\mathcal{L}(\\theta) = \\mathbb{E}_{i \\sim P(\\alpha)} \\left[w_i \\left(r_i + \\gamma \\max_{a'} Q(s_i', a'; \\theta^-) - Q(s_i, a_i; \\theta)\\right)^2\\right]$$

其中期望是关于优先级分布 $P(\\alpha)$ 的。在实现中，这对应于：

$$\\mathcal{L}(\\theta) = \\frac{1}{B} \\sum_{i \\in \\text{batch}} w_i \\cdot \\text{TD\\_error}_i^2$$

其中 $B$ 是 batch 大小（128 在我们的设置中）。

#### 3.3.2.6 优先级更新机制

在每次训练迭代后，我们需要更新缓冲区中采样经验的优先级。更新规则为：

$$p_i^{\\text{new}} = (|TD\\_error_i^{\\text{new}}| + \\epsilon)^\\alpha$$

其中 $TD\\_error_i^{\\text{new}}$ 是用更新后的网络参数 $\\theta$ 计算的新 TD 误差。

这种在线更新优先级的方式确保了：

1. **动态适应**：随着网络参数的变化，优先级动态调整
2. **持续学习**：之前的"惊喜"样本可能随着学习逐步变成"预期内"，优先级会相应降低

#### 3.3.2.7 我们的 PER 配置

在本工作中，我们使用以下 PER 参数：

| 参数 | 值 | 说明 | 范围 |
|------|-----|------|------|
| $\\alpha$ | 0.4 | 优先级强度 | [0, 1] |
| $\\beta_{\\text{start}}$ | 0.4 | 初始重要性采样强度 | [0, 1] |
| $\\beta_{\\text{end}}$ | 1.0 | 最终重要性采样强度 | [0, 1] |
| $\\beta_{\\text{frames}}$ | 100,000 | $\\beta$ 从 start 增加到 end 需要的步数 | - |
| 缓冲区大小 | 50,000 | 经验回放缓冲区的最大容量 | - |
| 优先级剪裁 | $[0.01, 1.0]$ | 优先级的最小和最大值 | - |

**参数选择的理由**：

1. **$\\alpha = 0.4$**：
   - 避免过度优先级化（$\\alpha = 1$ 可能导致模型过度关注异常样本）
   - 保持一定程度的随机性（$\\alpha = 0$ 会退化为标准经验回放）
   - 在我们的实验中，0.4 提供了最好的性能

2. **$\\beta$ 的渐进增加**：
   - 在训练早期（$\\beta = 0.4$），允许更多的采样偏差，使模型能够更自由地探索
   - 在训练后期（$\\beta = 1.0$），完全补偿采样偏差，确保收敛的正确性

3. **缓冲区大小 50,000**：
   - 足够大以保持样本多样性
   - 足够小以保持计算效率
   - 对于我们的问题规模是一个很好的折衷

4. **优先级剪裁**：
   - 防止极端的优先级值（如 0 或非常大的值）导致数值不稳定
   - 确保每个样本都有最小的被采样概率

"""

# Replace the section
new_lines = lines[:start_idx] + [new_content + '\n'] + lines[end_idx:]

# Write back
with open('METHODOLOGY_CHAPTER.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Successfully updated 3.3.2 section (replaced {end_idx - start_idx} lines with new content)")

