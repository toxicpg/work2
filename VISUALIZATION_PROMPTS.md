# 论文图表生成 - AI Prompt 提示词

这些 Prompt 可以用于 Midjourney、DALL-E、Stable Diffusion 等 AI 图像生成模型。

---

## 图表 1：完整模型架构图

### Prompt（中文版本）
```
请生成一张专业的深度学习模型架构图，展示从输入到输出的完整数据流。

内容要求：
- 顶部：输入层，标注"输入状态 (B, 400, 5)"，显示5个输入特征：订单数、空闲车、繁忙车、time_sin、time_cos
- 中上：MGCN 层（多图卷积网络），展示两个分支（邻接图分支、POI图分支）并行处理，标注"(B, 400, 64)"
- 中间：特征融合层，显示全局池化、位置编码、日期编码的融合，标注"(B, 64)"
- 中下：Dueling DQN 头，展示值流和优势流分离，标注"(B, 1)" 和 "(B, 179)"
- 底部：输出 Q 值 "(B, 179)"

设计风格：
- 使用蓝色、绿色、紫色等清晰的颜色区分不同模块
- 每个模块用圆角矩形表示
- 箭头清晰连接各个模块
- 在每个模块旁边标注维度信息
- 专业学术风格，类似于论文中的架构图
- 背景白色，高清晰度

技术细节：
- MGCN 部分用虚线框突出
- 融合操作用特殊符号表示（如⊕表示加法）
- 注意力融合用星形符号标注
```

### Prompt（英文版本，质量通常更好）
```
Generate a professional deep learning model architecture diagram showing complete data flow from input to output.

Requirements:
- Top: Input layer labeled "Input State (B, 400, 5)" with 5 input features: order count, idle vehicles, busy vehicles, time_sin, time_cos
- Upper-middle: MGCN layer (Multi-Graph Convolutional Network) showing two parallel branches (Adjacency Graph Branch, POI Graph Branch), labeled "(B, 400, 64)"
- Middle: Feature Fusion Layer showing global pooling, position embedding, day embedding fusion, labeled "(B, 64)"
- Lower-middle: Dueling DQN Head showing separated value stream and advantage stream, labeled "(B, 1)" and "(B, 179)"
- Bottom: Output Q-values "(B, 179)"

Design Style:
- Use clear colors (blue, green, purple) to distinguish different modules
- Rounded rectangles for each module
- Clear arrows connecting modules
- Dimension labels beside each module
- Professional academic style, similar to papers
- White background, high resolution

Technical Details:
- MGCN section highlighted with dashed box
- Fusion operations marked with special symbols (⊕ for addition)
- Attention fusion marked with star symbol
- Include parameter count annotations
```

---

## 图表 2：MGCN 详细设计图

### Prompt（英文版本）
```
Generate a detailed architecture diagram of Multi-Graph Convolutional Network (MGCN) showing parallel processing of two graph types.

Requirements:
- Left side: Adjacency Matrix (400×400) in blue
- Center: Input Features H^(0) (B, 400, 5)
- Right side: POI Matrix (400×400) in green
- Show two parallel GCN branches processing separately
- First layer output: (B, 400, 64) for both branches
- Attention fusion mechanism in the middle combining both branches with weights α_n and α_p
- Second layer: Both branches process again
- Final output: (B, 400, 32) after fusion

Color scheme:
- Adjacency branch: Blue color scheme
- POI branch: Green color scheme
- Fusion operations: Purple color
- Attention weights: Orange highlights

Annotations:
- Label each layer with dimensions
- Show mathematical formula for attention fusion: H_fused = α_n⊙H_neighbor + α_p⊙H_poi
- Include parameter counts (4736 + 4736 + 196 total)
- Professional academic diagram style
```

---

## 图表 3：Dueling DQN 架构图

### Prompt（英文版本）
```
Generate a professional diagram of Dueling DQN architecture showing separated value and advantage streams.

Requirements:
- Top: Fused Feature input (B, 64)
- Left branch (Red): Value Stream
  - Linear layer: 64 → 128
  - ReLU activation
  - Linear layer: 128 → 1
  - Output: V(s) (B, 1)
- Right branch (Blue): Advantage Stream
  - Linear layer: 64 → 128
  - ReLU activation
  - Linear layer: 128 → 179
  - Output: A(s,a) (B, 179)
- Bottom: Q-value aggregation showing formula Q(s,a) = V(s) + [A(s,a) - mean(A)]

Design:
- Symmetric layout with value stream on left (red) and advantage stream on right (blue)
- Aggregation section at bottom in purple
- Clear dimension labels on each layer
- Mathematical formula prominently displayed
- Parameter counts: Value stream 8,449 | Advantage stream 31,411
- Professional academic style

Additional annotations:
- Explain why subtracting mean(A) ensures identifiability
- Show that value stream is shared across all actions
- Highlight that advantage stream outputs action-specific advantages
```

---

## 图表 4：训练循环流程图

### Prompt（英文版本）
```
Generate a comprehensive flowchart of the complete training loop for reinforcement learning agent.

Requirements:
Show six main steps in sequence:

Step 1 (Blue): Action Selection
- If random() < ε: random action
- Else: argmax Q(state)

Step 2 (Blue): Execute Action
- next_state, reward, done = env.step(action)
- episode_reward += reward

Step 3 (Green): Store Experience
- replay_buffer.push(state, action, reward, next_state, done)

Step 4 (Red): Train Model (every 30 ticks)
- Sample batch from PER buffer (B=128)
- Compute TD target: y = r + γ×max Q(s',a';θ⁻)
- Compute loss: L = mean(weights × (y - Q)²)
- Backpropagation and gradient clipping
- Parameter update: θ ← θ - α∇L
- Update PER priorities

Step 5 (Purple): Update Target Network (every 1000 steps)
- θ⁻ ← θ (hard update)

Step 6 (Orange): Episode End
- Decay epsilon: ε ← max(ε_min, ε × 0.95)

Design:
- Use nested loops to show episode loop and tick loop
- Color-code each step type
- Show decision diamonds for conditional statements
- Include frequency annotations (every 30 ticks, every 1000 steps, etc.)
- Professional flowchart style with clear connections
- Total 500 episodes, 2880 ticks per episode
```

---

## 图表 5：Epsilon 衰减曲线

### Prompt（英文版本）
```
Generate a professional line chart showing epsilon decay schedule for exploration probability.

Requirements:
- X-axis: Episode (0 to 500)
- Y-axis: Epsilon value (0% to 60%)
- Curve: Exponential decay following formula ε_t = max(0.05, 0.6 × 0.95^t)
- Line color: Blue, line width 2-3 pixels
- Key points marked with red dots:
  - Episode 0: ε = 60%
  - Episode 50: ε ≈ 5%
  - Episode 100+: ε = 5% (flat line)

Annotations:
- Label three phases with different background colors:
  - Exploration Phase (0-50): Light green background
  - Learning Phase (50-400): Light yellow background
  - Convergence Phase (400-500): Light blue background
- Add mathematical formula: ε_t = max(0.05, 0.6 × 0.95^t)
- Include grid lines for easier reading
- Professional academic chart style
- High resolution

Optional:
- Show comparison with other decay strategies (linear, exponential) as dashed lines
- Annotate what exploration probability means at each phase
```

---

## 图表 6：优先级经验回放 (PER) 机制图

### Prompt（英文版本）
```
Generate a detailed flowchart showing the Prioritized Experience Replay (PER) mechanism with 5 steps.

Requirements:

Step 1 (Blue box): Replay Buffer with samples
- Show 5-6 sample boxes with different TD_error values
- Sample 1: TD_error = 0.05 (low) - blue color
- Sample 2: TD_error = 2.50 (high) - orange color
- Sample 4: TD_error = 5.00 (highest) - red color
- Indicate "50K total samples"

Step 2 (Green box): Priority Distribution
- Show bar chart with sampling probabilities
- Sample 4: 40% (tallest bar, red)
- Sample 2: 35% (orange)
- Sample 1: 8% (blue)
- Others: percentages
- Formula annotation: p_i = |TD_error_i|^α / Σ|TD_error|^α (α=0.4)

Step 3 (Purple box): Mini-batch with Importance Sampling Weights
- Show 4-5 selected samples
- Each with corresponding weight value
- Formula: w_i = (1/(N×p_i))^β
- Annotation: β increases from 0.4 to 1.0

Step 4 (Red box): Weighted Loss Calculation
- Show formula: L = mean(w_i × (y_i - Q_i)²)
- Contrast with standard DQN: L = mean((y - Q)²)

Step 5 (Orange box): Update Priorities
- Show priority update after training
- Calculate new TD_error
- Update buffer priorities

Design:
- Flow from top to bottom
- Each step in a colored box
- Clear arrows between steps
- Color scheme: Blue (buffer) → Green (probability) → Purple (weights) → Red (loss) → Orange (update)
- Include mathematical formulas
- Professional academic style
```

---

## 图表 7：目标网络的作用对比图

### Prompt（英文版本）
```
Generate a side-by-side comparison diagram showing the difference between training with and without target network.

Left side (Red - Without Target Network):
- Time t: Main network θ = [0.5, 0.3, 0.2]
- Compute TD target: y = r + γ max Q(s', a; θ)
- Calculate loss: L = (y - Q(s,a;θ))²
- Update: θ ← θ - α∇L
- Result: θ changes to [0.48, 0.32, 0.19]
- Problem: Target y also changes (moving target)
- Outcome: Unstable training ✗

Right side (Green - With Target Network):
- Time t: Main network θ = [0.5, 0.3, 0.2]
- Target network θ⁻ = [0.4, 0.25, 0.18] (old parameters)
- Compute TD target: y = r + γ max Q(s', a; θ⁻)
- Calculate loss: L = (y - Q(s,a;θ))²
- Update: θ ← θ - α∇L
- Result: θ changes but θ⁻ stays fixed
- Time t+1000: Update target network θ⁻ ← θ
- Outcome: Stable training ✓

Design:
- Vertical split layout
- Red color scheme for left (unstable)
- Green color scheme for right (stable)
- Use ✗ and ✓ symbols for outcomes
- Highlight the key difference: target network parameters stay fixed
- Show parameter values clearly
- Include arrows showing parameter changes
- Professional comparison diagram style
```

---

## 图表 8：训练三阶段分析图

### Prompt（英文版本）
```
Generate a multi-panel figure showing three phases of training with multiple metrics.

Create 3 subplots stacked vertically:

Panel 1 - Training Loss:
- X-axis: Episode (0-500)
- Y-axis: Loss value
- Curve: High fluctuation early (0-100), gradual decrease (100-400), flat (400-500)
- Divide into three phases with vertical dashed lines at episode 100 and 400
- Phase labels: Exploration | Learning | Convergence

Panel 2 - Match Rate:
- X-axis: Episode (0-500)
- Y-axis: Match Rate (%)
- Curve: Low and unstable (0-100), gradual increase (100-400), high and stable (400-500)
- Same phase divisions
- Target line at 90% marked with horizontal dashed line

Panel 3 - Average Waiting Time:
- X-axis: Episode (0-500)
- Y-axis: Waiting Time (seconds)
- Curve: High and unstable (0-100), decreasing (100-400), stable and low (400-500)
- Same phase divisions
- Target line at 300 seconds marked with horizontal dashed line

Summary Table at bottom:
- Three columns for three phases
- Rows: ε value, Loss trend, Match rate, Wait time
- Color-coded cells matching phase colors

Design:
- Green background for Exploration Phase (0-100)
- Yellow background for Learning Phase (100-400)
- Blue background for Convergence Phase (400-500)
- Professional academic figure style
- High resolution
```

---

## 图表 9：奖励函数曲线

### Prompt（英文版本）
```
Generate a professional line chart showing the reward decay function.

Requirements:
- X-axis: Waiting Time w_t (0 to 600 seconds)
- Y-axis: Completion Reward R_completion (0 to 2.0)
- Curve: Exponential decay following formula R = 2.0 × exp(-w_t / 240)
- Line color: Blue, line width 2-3 pixels

Key points marked with red dots and annotations:
- w_t = 0s: R = 2.0 (maximum reward)
- w_t = 240s: R ≈ 0.74 (characteristic time)
- w_t = 600s: R ≈ 0.05 (minimal reward)

Annotations:
- Mathematical formula prominently displayed: R_completion = W_completion × exp(-w_t / T_0)
- Parameter values: W_completion = 2.0, T_0 = 240 seconds
- Add shaded regions:
  - Green region (0-240s): High reward zone
  - Yellow region (240-400s): Medium reward zone
  - Red region (400-600s): Low reward zone
- Include grid lines
- Label axes clearly

Design:
- Professional academic chart style
- Clear exponential decay curve
- High resolution
- White background
```

---

## 图表 10：网格化地理表示图

### Prompt（英文版本）
```
Generate a professional map visualization showing city grid division.

Requirements:
- Show a 20×20 grid representing Beijing city center
- Grid size: 400 squares (20×20)
- Color scheme:
  - Light blue: Regular grids
  - Dark blue/highlighted: Hot spot grids (179 hotspots)
  - Red dots: Specific hotspot locations
- Add coordinate labels:
  - X-axis: 0-19
  - Y-axis: 0-19

Annotations:
- Title: "City Grid Division for Ride-hailing Dispatch"
- Legend showing:
  - Blue square: Regular grid
  - Dark blue/red dot: Hotspot grid (179 total)
- Add scale: "Each grid ~2.5 km × 2.5 km"
- Show some order/vehicle density as heatmap overlay (optional)

Design:
- Professional map style
- Clear grid lines
- Color gradient for density visualization
- High resolution
- Satellite map background (optional, can be simple)
```

---

## 图表 11：模拟器架构图

### Prompt（英文版本）
```
Generate a system architecture diagram for the ride-hailing simulation environment.

Requirements:
Show 6 main components connected in a system:

Central: Event Queue
- Priority queue showing events sorted by time
- Different event types: Order events, Vehicle events, Dispatch events

Surrounding components (connected to Event Queue):

1. Order Manager (top-left)
- Order generation from dataset
- Order matching logic
- Timeout handling

2. Vehicle Manager (top-right)
- Vehicle state management (idle, dispatching, waiting, executing)
- Position updates
- Status transitions

3. Dispatch Executor (middle-left)
- Decision validation
- Dispatch operations
- Action execution

4. Reward Calculator (middle-right)
- Multi-stage reward computation
- Business metrics calculation
- Training signal generation

5. State Observer (bottom-center)
- Grid aggregation
- Time encoding
- State generation for RL model

6. RL Model (bottom)
- Decision making
- Action selection
- Feedback loop

Design:
- Central event queue as hub
- Surrounding components arranged in circle
- Clear arrows showing data flow
- Color-code by function (management, execution, observation, decision)
- Include data flow annotations
- Professional system architecture style
```

---

## 图表 12：订单生命周期状态转换图

### Prompt（英文版本）
```
Generate a state transition diagram showing the complete lifecycle of a ride-hailing order.

Requirements:
Show 6 states with transitions:

States (boxes):
1. Created (green) - Order generated
2. Waiting (yellow) - Waiting for driver
3. Matched (orange) - Driver accepted
4. Executing (blue) - Passenger in vehicle
5. Completed (green) - Order finished
6. Cancelled (red) - Order timed out

Transitions (arrows):
- Created → Waiting (order enters system)
- Waiting → Matched (driver accepts)
- Waiting → Cancelled (timeout)
- Matched → Executing (driver picks up passenger)
- Executing → Completed (reached destination)

Timeline example on the right:
- T=0s: Created
- T=50s: Waiting
- T=120s: Matched
- T=180s: Executing
- T=450s: Completed
- Total waiting time = 450s

Design:
- State boxes with different colors
- Clear directional arrows
- Label each transition with condition
- Include timeline visualization
- Show time examples
- Professional state diagram style
- Include legend explaining each state
```

---

## 通用提示词调整建议

### 如果要求高分辨率
在任何 Prompt 的末尾添加：
```
High resolution, 4K quality, professional publication-ready figure, 300 DPI
```

### 如果要求特定风格
```
Academic paper style, similar to IEEE/ACM publications
Scientific illustration, professional diagram style
Minimalist design with clear hierarchy
```

### 如果要求特定颜色
```
Use professional color scheme: Blue (#0066CC), Green (#00AA33), Purple (#9933FF), Orange (#FF9900), Red (#CC0000)
```

### 如果要求添加说明文字
```
Include clear labels and annotations
Add mathematical formulas where relevant
Include parameter values and dimensions
```

---

## 使用建议

### 对于 Midjourney：
- 使用英文版本的 Prompt
- 在末尾添加 `--ar 16:9` 或 `--ar 4:3` 指定宽高比
- 添加 `--q 2` 要求高质量渲染

### 对于 DALL-E 3：
- 英文 Prompt 效果最好
- 字数不超过 1000 字
- 具体描述颜色和布局

### 对于 Stable Diffusion：
- 可以使用中英文混合
- 添加 `(high quality, detailed, professional)` 提升质量
- 使用 `negative prompt` 排除不想要的元素

### 对于本地工具（如 draw.io）：
- 使用中文版本的 Prompt 更容易理解
- 逐步构建，先画框架再加细节

---

## 后处理建议

生成图片后，你可能需要在 Photoshop、Figma 或 draw.io 中进行微调：
1. 调整字体大小和位置
2. 添加论文中需要的特定标注
3. 统一颜色和风格
4. 调整分辨率和尺寸
5. 导出为 PDF 或 PNG 格式

