# Midjourney 提示词 - 直接复制粘贴版本

> 复制下面的提示词，粘贴到 Midjourney 网页上即可生成图片

---

## 图表 1：完整模型架构图

**复制以下内容粘贴到 Midjourney：**

```
Generate a professional deep learning model architecture diagram showing complete data flow from input to output. Top: Input layer labeled "Input State (B, 400, 5)" with 5 input features: order count, idle vehicles, busy vehicles, time_sin, time_cos. Upper-middle: MGCN layer (Multi-Graph Convolutional Network) showing two parallel branches (Adjacency Graph Branch, POI Graph Branch), labeled "(B, 400, 64)". Middle: Feature Fusion Layer showing global pooling, position embedding, day embedding fusion, labeled "(B, 64)". Lower-middle: Dueling DQN Head showing separated value stream and advantage stream, labeled "(B, 1)" and "(B, 179)". Bottom: Output Q-values "(B, 179)". Use clear colors (blue, green, purple) to distinguish different modules. Rounded rectangles for each module. Clear arrows connecting modules. Dimension labels beside each module. Professional academic style, similar to papers. White background, high resolution. MGCN section highlighted with dashed box. Fusion operations marked with special symbols (⊕ for addition). Attention fusion marked with star symbol. --ar 16:9 --q 2
```

---

## 图表 2：MGCN 详细设计图

**复制以下内容粘贴到 Midjourney：**

```
Generate a detailed architecture diagram of Multi-Graph Convolutional Network (MGCN) showing parallel processing of two graph types. Left side: Adjacency Matrix (400×400) in blue. Center: Input Features H^(0) (B, 400, 5). Right side: POI Matrix (400×400) in green. Show two parallel GCN branches processing separately. First layer output: (B, 400, 64) for both branches. Attention fusion mechanism in the middle combining both branches with weights α_n and α_p. Second layer: Both branches process again. Final output: (B, 400, 32) after fusion. Adjacency branch: Blue color scheme. POI branch: Green color scheme. Fusion operations: Purple color. Attention weights: Orange highlights. Label each layer with dimensions. Show mathematical formula for attention fusion: H_fused = α_n⊙H_neighbor + α_p⊙H_poi. Include parameter counts (4736 + 4736 + 196 total). Professional academic diagram style. --ar 16:9 --q 2
```

---

## 图表 3：Dueling DQN 架构图

**复制以下内容粘贴到 Midjourney：**

```
Generate a professional diagram of Dueling DQN architecture showing separated value and advantage streams. Top: Fused Feature input (B, 64). Left branch (Red): Value Stream with Linear layer 64→128, ReLU activation, Linear layer 128→1, Output V(s) (B, 1). Right branch (Blue): Advantage Stream with Linear layer 64→128, ReLU activation, Linear layer 128→179, Output A(s,a) (B, 179). Bottom: Q-value aggregation showing formula Q(s,a) = V(s) + [A(s,a) - mean(A)]. Symmetric layout with value stream on left (red) and advantage stream on right (blue). Aggregation section at bottom in purple. Clear dimension labels on each layer. Mathematical formula prominently displayed. Parameter counts: Value stream 8,449 | Advantage stream 31,411. Professional academic style. --ar 16:9 --q 2
```

---

## 图表 4：训练循环流程图

**复制以下内容粘贴到 Midjourney：**

```
Generate a comprehensive flowchart of the complete training loop for reinforcement learning agent. Show six main steps in sequence: Step 1 (Blue) Action Selection - If random() < ε: random action, Else: argmax Q(state). Step 2 (Blue) Execute Action - next_state, reward, done = env.step(action), episode_reward += reward. Step 3 (Green) Store Experience - replay_buffer.push(state, action, reward, next_state, done). Step 4 (Red) Train Model (every 30 ticks) - Sample batch from PER buffer (B=128), Compute TD target: y = r + γ×max Q(s',a';θ⁻), Compute loss: L = mean(weights × (y - Q)²), Backpropagation and gradient clipping, Parameter update: θ ← θ - α∇L, Update PER priorities. Step 5 (Purple) Update Target Network (every 1000 steps) - θ⁻ ← θ (hard update). Step 6 (Orange) Episode End - Decay epsilon: ε ← max(ε_min, ε × 0.95). Use nested loops to show episode loop and tick loop. Color-code each step type. Show decision diamonds for conditional statements. Include frequency annotations (every 30 ticks, every 1000 steps, etc.). Professional flowchart style with clear connections. Total 500 episodes, 2880 ticks per episode. --ar 16:9 --q 2
```

---

## 图表 5：Epsilon 衰减曲线

**复制以下内容粘贴到 Midjourney：**

```
Generate a professional line chart showing epsilon decay schedule for exploration probability. X-axis: Episode (0 to 500). Y-axis: Epsilon value (0% to 60%). Curve: Exponential decay following formula ε_t = max(0.05, 0.6 × 0.95^t). Line color: Blue, line width 2-3 pixels. Key points marked with red dots: Episode 0: ε = 60%, Episode 50: ε ≈ 5%, Episode 100+: ε = 5% (flat line). Label three phases with different background colors: Exploration Phase (0-50): Light green background, Learning Phase (50-400): Light yellow background, Convergence Phase (400-500): Light blue background. Add mathematical formula: ε_t = max(0.05, 0.6 × 0.95^t). Include grid lines for easier reading. Professional academic chart style. High resolution. --ar 16:9 --q 2
```

---

## 图表 6：优先级经验回放 (PER) 机制图

**复制以下内容粘贴到 Midjourney：**

```
Generate a detailed flowchart showing the Prioritized Experience Replay (PER) mechanism with 5 steps. Step 1 (Blue box): Replay Buffer with samples showing 5-6 sample boxes with different TD_error values. Sample 1: TD_error = 0.05 (low) - blue color. Sample 2: TD_error = 2.50 (high) - orange color. Sample 4: TD_error = 5.00 (highest) - red color. Indicate "50K total samples". Step 2 (Green box): Priority Distribution showing bar chart with sampling probabilities. Sample 4: 40% (tallest bar, red). Sample 2: 35% (orange). Sample 1: 8% (blue). Formula annotation: p_i = |TD_error_i|^α / Σ|TD_error|^α (α=0.4). Step 3 (Purple box): Mini-batch with Importance Sampling Weights showing 4-5 selected samples with corresponding weight values. Formula: w_i = (1/(N×p_i))^β. Annotation: β increases from 0.4 to 1.0. Step 4 (Red box): Weighted Loss Calculation showing formula: L = mean(w_i × (y_i - Q_i)²). Step 5 (Orange box): Update Priorities showing priority update after training. Flow from top to bottom. Each step in a colored box. Clear arrows between steps. Professional academic style. --ar 16:9 --q 2
```

---

## 图表 7：目标网络的作用对比图

**复制以下内容粘贴到 Midjourney：**

```
Generate a side-by-side comparison diagram showing the difference between training with and without target network. Left side (Red - Without Target Network): Time t: Main network θ = [0.5, 0.3, 0.2]. Compute TD target: y = r + γ max Q(s', a; θ). Calculate loss: L = (y - Q(s,a;θ))². Update: θ ← θ - α∇L. Result: θ changes to [0.48, 0.32, 0.19]. Problem: Target y also changes (moving target). Outcome: Unstable training ✗. Right side (Green - With Target Network): Time t: Main network θ = [0.5, 0.3, 0.2]. Target network θ⁻ = [0.4, 0.25, 0.18] (old parameters). Compute TD target: y = r + γ max Q(s', a; θ⁻). Calculate loss: L = (y - Q(s,a;θ))². Update: θ ← θ - α∇L. Result: θ changes but θ⁻ stays fixed. Time t+1000: Update target network θ⁻ ← θ. Outcome: Stable training ✓. Vertical split layout. Red color scheme for left (unstable). Green color scheme for right (stable). Use ✗ and ✓ symbols for outcomes. Highlight the key difference: target network parameters stay fixed. Show parameter values clearly. Include arrows showing parameter changes. Professional comparison diagram style. --ar 16:9 --q 2
```

---

## 图表 8：训练三阶段分析图

**复制以下内容粘贴到 Midjourney：**

```
Generate a multi-panel figure showing three phases of training with multiple metrics. Create 3 subplots stacked vertically. Panel 1 - Training Loss: X-axis: Episode (0-500). Y-axis: Loss value. Curve: High fluctuation early (0-100), gradual decrease (100-400), flat (400-500). Divide into three phases with vertical dashed lines at episode 100 and 400. Phase labels: Exploration | Learning | Convergence. Panel 2 - Match Rate: X-axis: Episode (0-500). Y-axis: Match Rate (%). Curve: Low and unstable (0-100), gradual increase (100-400), high and stable (400-500). Same phase divisions. Target line at 90% marked with horizontal dashed line. Panel 3 - Average Waiting Time: X-axis: Episode (0-500). Y-axis: Waiting Time (seconds). Curve: High and unstable (0-100), decreasing (100-400), stable and low (400-500). Same phase divisions. Target line at 300 seconds marked with horizontal dashed line. Summary Table at bottom: Three columns for three phases. Rows: ε value, Loss trend, Match rate, Wait time. Color-coded cells matching phase colors. Green background for Exploration Phase (0-100). Yellow background for Learning Phase (100-400). Blue background for Convergence Phase (400-500). Professional academic figure style. High resolution. --ar 16:9 --q 2
```

---

## 图表 9：奖励函数曲线

**复制以下内容粘贴到 Midjourney：**

```
Generate a professional line chart showing the reward decay function. X-axis: Waiting Time w_t (0 to 600 seconds). Y-axis: Completion Reward R_completion (0 to 2.0). Curve: Exponential decay following formula R = 2.0 × exp(-w_t / 240). Line color: Blue, line width 2-3 pixels. Key points marked with red dots and annotations: w_t = 0s: R = 2.0 (maximum reward). w_t = 240s: R ≈ 0.74 (characteristic time). w_t = 600s: R ≈ 0.05 (minimal reward). Mathematical formula prominently displayed: R_completion = W_completion × exp(-w_t / T_0). Parameter values: W_completion = 2.0, T_0 = 240 seconds. Add shaded regions: Green region (0-240s): High reward zone. Yellow region (240-400s): Medium reward zone. Red region (400-600s): Low reward zone. Include grid lines. Label axes clearly. Professional academic chart style. Clear exponential decay curve. High resolution. White background. --ar 16:9 --q 2
```

---

## 图表 10：网格化地理表示图

**复制以下内容粘贴到 Midjourney：**

```
Generate a professional map visualization showing city grid division. Show a 20×20 grid representing Beijing city center. Grid size: 400 squares (20×20). Color scheme: Light blue: Regular grids. Dark blue/highlighted: Hot spot grids (179 hotspots). Red dots: Specific hotspot locations. Add coordinate labels: X-axis: 0-19. Y-axis: 0-19. Title: "City Grid Division for Ride-hailing Dispatch". Legend showing: Blue square: Regular grid. Dark blue/red dot: Hotspot grid (179 total). Add scale: "Each grid ~2.5 km × 2.5 km". Show some order/vehicle density as heatmap overlay (optional). Professional map style. Clear grid lines. Color gradient for density visualization. High resolution. Satellite map background (optional, can be simple). --ar 16:9 --q 2
```

---

## 图表 11：模拟器架构图

**复制以下内容粘贴到 Midjourney：**

```
Generate a system architecture diagram for the ride-hailing simulation environment. Show 6 main components connected in a system. Central: Event Queue - Priority queue showing events sorted by time. Different event types: Order events, Vehicle events, Dispatch events. Surrounding components (connected to Event Queue): 1. Order Manager (top-left) - Order generation from dataset, Order matching logic, Timeout handling. 2. Vehicle Manager (top-right) - Vehicle state management (idle, dispatching, waiting, executing), Position updates, Status transitions. 3. Dispatch Executor (middle-left) - Decision validation, Dispatch operations, Action execution. 4. Reward Calculator (middle-right) - Multi-stage reward computation, Business metrics calculation, Training signal generation. 5. State Observer (bottom-center) - Grid aggregation, Time encoding, State generation for RL model. 6. RL Model (bottom) - Decision making, Action selection, Feedback loop. Central event queue as hub. Surrounding components arranged in circle. Clear arrows showing data flow. Color-code by function (management, execution, observation, decision). Include data flow annotations. Professional system architecture style. --ar 16:9 --q 2
```

---

## 图表 12：订单生命周期状态转换图

**复制以下内容粘贴到 Midjourney：**

```
Generate a state transition diagram showing the complete lifecycle of a ride-hailing order. Show 6 states with transitions: States (boxes): 1. Created (green) - Order generated. 2. Waiting (yellow) - Waiting for driver. 3. Matched (orange) - Driver accepted. 4. Executing (blue) - Passenger in vehicle. 5. Completed (green) - Order finished. 6. Cancelled (red) - Order timed out. Transitions (arrows): Created → Waiting (order enters system). Waiting → Matched (driver accepts). Waiting → Cancelled (timeout). Matched → Executing (driver picks up passenger). Executing → Completed (reached destination). Timeline example on the right: T=0s: Created. T=50s: Waiting. T=120s: Matched. T=180s: Executing. T=450s: Completed. Total waiting time = 450s. State boxes with different colors. Clear directional arrows. Label each transition with condition. Include timeline visualization. Show time examples. Professional state diagram style. Include legend explaining each state. --ar 16:9 --q 2
```

---

## 使用说明

1. **复制提示词**：选择上面的任何一个提示词（在 ``` 之间的内容）
2. **打开 Midjourney**：访问 https://www.midjourney.com/ 或在 Discord 中使用 Midjourney Bot
3. **粘贴提示词**：将复制的内容粘贴到输入框中
4. **发送**：按 Enter 键发送，等待 Midjourney 生成图片
5. **优化**：如果效果不理想，可以：
   - 在提示词末尾添加 `--niji` 使用日本风格（有时效果更好）
   - 改变宽高比：`--ar 4:3` 或 `--ar 1:1`
   - 提高质量：`--q 2` 或 `--hd`
   - 重新生成：点击图片下方的刷新按钮

---

## 快速参考

| 图表 | 用途 |
|------|------|
| 1 | 整个模型的架构概览 |
| 2 | MGCN 多图卷积网络的详细设计 |
| 3 | Dueling DQN 的双流架构 |
| 4 | 强化学习训练的完整流程 |
| 5 | 探索率随时间的衰减 |
| 6 | 优先级经验回放机制 |
| 7 | 目标网络稳定性对比 |
| 8 | 训练过程中的三个阶段 |
| 9 | 奖励函数的指数衰减 |
| 10 | 城市网格划分和热点分布 |
| 11 | 模拟器系统架构 |
| 12 | 订单的完整生命周期 |

---

## 提示

- 英文提示词通常效果更好
- 所有提示词都已包含 `--ar 16:9 --q 2` 参数，适合论文插图
- 如果需要调整，可以修改宽高比或质量参数
- 生成通常需要 1-2 分钟，耐心等待
- 可以多次生成以获得最好的效果，然后选择最满意的版本

