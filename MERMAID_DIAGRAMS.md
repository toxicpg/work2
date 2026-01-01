# Mermaid 图表代码 - 论文可视化

> 复制下面的代码到 https://mermaid.live 在线生成图表

---

## 图表 1：完整模型架构图

```mermaid
graph TD
    A["Input State<br/>(B, 400, 5)<br/>Orders|Idle Veh|Busy Veh|time_sin|time_cos"]

    B["Adjacency Graph<br/>GCN Branch<br/>(B, 400, 64)"]
    C["POI Graph<br/>GCN Branch<br/>(B, 400, 64)"]

    D["Feature Fusion Layer<br/>Global Pooling + Embeddings<br/>(B, 64)"]

    E["Value Stream<br/>64→128→ReLU→128→1<br/>V(s) (B, 1)"]
    F["Advantage Stream<br/>64→128→ReLU→128→179<br/>A(s,a) (B, 179)"]

    G["Q-value Aggregation<br/>Q(s,a) = V(s) + [A(s,a) - mean(A)]<br/>(B, 179)"]

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    D --> F
    E --> G
    F --> G

    style A fill:#E8F4F8,stroke:#0277BD,stroke-width:2px
    style B fill:#B3E5FC,stroke:#01579B,stroke-width:2px
    style C fill:#B3E5FC,stroke:#01579B,stroke-width:2px
    style D fill:#81D4FA,stroke:#00695C,stroke-width:2px
    style E fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style F fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style G fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
```

---

## 图表 2：MGCN 详细设计图

```mermaid
graph TD
    A["Adjacency Matrix<br/>(400×400)<br/>A_n"]
    B["Input Features<br/>H^(0)<br/>(B, 400, 5)"]
    C["POI Matrix<br/>(400×400)<br/>A_p"]

    D["GCN Layer 1<br/>H_n^(1)<br/>(B, 400, 64)"]
    E["GCN Layer 1<br/>H_p^(1)<br/>(B, 400, 64)"]

    F["Attention Fusion<br/>H_fused = α_n⊙H_n + α_p⊙H_p<br/>(B, 400, 64)"]

    G["Final Output<br/>H_out<br/>(B, 400, 32)"]

    A --> D
    B --> D
    C --> E
    B --> E
    D --> F
    E --> F
    F --> G

    style A fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style B fill:#DCEDC8,stroke:#558B2F,stroke-width:2px
    style C fill:#C8E6C9,stroke:#00695C,stroke-width:2px
    style D fill:#E1F5FE,stroke:#0277BD,stroke-width:2px
    style E fill:#E8F5E9,stroke:#00695C,stroke-width:2px
    style F fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
    style G fill:#FFF3E0,stroke:#F57F17,stroke-width:2px
```

---

## 图表 3：Dueling DQN 架构图

```mermaid
graph TD
    A["Fused Feature Input<br/>(B, 64)"]

    B["Value Stream"]
    C["Linear: 64 → 128"]
    D["ReLU"]
    E["Linear: 128 → 1"]
    F["V(s) (B, 1)"]

    G["Advantage Stream"]
    H["Linear: 64 → 128"]
    I["ReLU"]
    J["Linear: 128 → 179"]
    K["A(s,a) (B, 179)"]

    L["Q-value Aggregation<br/>Q(s,a) = V(s) + [A(s,a) - mean(A)]"]
    M["Output Q-values<br/>(B, 179)"]

    A --> C
    A --> H
    C --> D
    H --> I
    D --> E
    I --> J
    E --> F
    J --> K
    F --> L
    K --> L
    L --> M

    style A fill:#DCEDC8,stroke:#558B2F,stroke-width:2px
    style B fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#C62828
    style C fill:#FFEBEE,stroke:#C62828,stroke-width:1px
    style D fill:#FFEBEE,stroke:#C62828,stroke-width:1px
    style E fill:#FFEBEE,stroke:#C62828,stroke-width:1px
    style F fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    style G fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#1565C0
    style H fill:#E3F2FD,stroke:#1565C0,stroke-width:1px
    style I fill:#E3F2FD,stroke:#1565C0,stroke-width:1px
    style J fill:#E3F2FD,stroke:#1565C0,stroke-width:1px
    style K fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style L fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
    style M fill:#FFF3E0,stroke:#F57F17,stroke-width:2px
```

---

## 图表 4：训练循环流程图

```mermaid
graph TD
    Start["Start Training<br/>500 Episodes"]

    Loop1["For each Episode"]
    Reset["Reset Environment"]

    Loop2["For each Tick<br/>2880 ticks/episode"]

    Step1{"ε-Greedy<br/>Action Selection"}
    Random["Random Action"]
    Greedy["argmax Q(state)"]

    Step2["Execute Action<br/>env.step"]
    Step3["Store Experience<br/>Replay Buffer"]

    Step4{"Every 30 ticks?"}
    Train["Train Model<br/>Sample batch B=128<br/>Compute TD target<br/>Backward pass<br/>Update parameters"]

    Step5{"Every 1000 steps?"}
    UpdateTarget["Update Target Network<br/>θ⁻ ← θ"]

    CheckDone{"Episode Done?"}
    Decay["Decay Epsilon<br/>ε ← max(ε_min, ε × 0.95)"]

    End["End Training"]

    Start --> Loop1
    Loop1 --> Reset
    Reset --> Loop2
    Loop2 --> Step1
    Step1 -->|random < ε| Random
    Step1 -->|else| Greedy
    Random --> Step2
    Greedy --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 -->|Yes| Train
    Step4 -->|No| Step5
    Train --> Step5
    Step5 -->|Yes| UpdateTarget
    Step5 -->|No| CheckDone
    UpdateTarget --> CheckDone
    CheckDone -->|No| Loop2
    CheckDone -->|Yes| Decay
    Decay --> Loop1
    Loop1 -->|All episodes done| End

    style Start fill:#90CAF9,stroke:#1565C0,stroke-width:2px
    style Loop1 fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style Step1 fill:#FFE082,stroke:#F57F17,stroke-width:2px
    style Train fill:#EF9A9A,stroke:#C62828,stroke-width:2px
    style UpdateTarget fill:#CE93D8,stroke:#6A1B9A,stroke-width:2px
    style End fill:#90CAF9,stroke:#1565C0,stroke-width:2px
```

---

## 图表 5：Epsilon 衰减曲线

```mermaid
xychart-beta
    title "Epsilon Decay Schedule: ε(t) = max(0.05, 0.6 × 0.95^t)"
    x-axis [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    y-axis "Epsilon (Exploration Rate)" 0 --> 0.65
    line [0.6, 0.18, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
```

---

## 图表 6：奖励函数曲线

```mermaid
xychart-beta
    title "Reward Decay Function: R = 2.0 × exp(-w_t / 240)"
    x-axis [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    y-axis "Completion Reward R_completion" 0 --> 2.2
    line [2.0, 1.48, 1.09, 0.81, 0.60, 0.44, 0.33, 0.24, 0.18, 0.13, 0.10]
```

---

## 图表 7：PER 机制流程图

```mermaid
graph LR
    A["Step 1: Replay Buffer<br/>50K Samples<br/>TD_error values"]

    B["Step 2: Priority Distribution<br/>p_i = |TD_error_i|^α<br/>/ Σ|TD_error|^α"]

    C["Step 3: Importance Sampling<br/>w_i = 1/(N×p_i)^β<br/>β: 0.4 → 1.0"]

    D["Step 4: Weighted Loss<br/>L = mean(w_i × (y_i - Q_i)²)"]

    E["Step 5: Update Priorities<br/>p_i ← |TD_error_i|^α"]

    A --> B
    B --> C
    C --> D
    D --> E
    E -.-> A

    style A fill:#E1F5FE,stroke:#0277BD,stroke-width:2px
    style B fill:#E8F5E9,stroke:#558B2F,stroke-width:2px
    style C fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
    style D fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style E fill:#FFF3E0,stroke:#F57F17,stroke-width:2px
```

---

## 图表 8：目标网络对比

```mermaid
graph TD
    subgraph without["❌ Without Target Network"]
        A1["Main network θ = [0.5, 0.3, 0.2]"]
        A2["y = r + γ max Q(s', a; θ)"]
        A3["L = (y - Q(s,a;θ))²"]
        A4["Update: θ ← θ - α∇L"]
        A5["Result: θ = [0.48, 0.32, 0.19]"]
        A6["Problem: Moving target!"]
        A7["❌ Unstable Training"]

        A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
    end

    subgraph with["✓ With Target Network"]
        B1["Main θ = [0.5, 0.3, 0.2]<br/>Target θ⁻ = [0.4, 0.25, 0.18]"]
        B2["y = r + γ max Q(s', a; θ⁻)"]
        B3["L = (y - Q(s,a;θ))²"]
        B4["Update: θ ← θ - α∇L"]
        B5["Result: θ changes, θ⁻ fixed"]
        B6["Key: θ⁻ stays fixed!"]
        B7["✓ Stable Training"]
        B8["Update θ⁻ ← θ every 1000 steps"]

        B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8
    end

    style without fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style with fill:#E8F5E9,stroke:#00695C,stroke-width:2px
    style A7 fill:#EF5350,stroke:#C62828,stroke-width:2px,color:#fff
    style B7 fill:#66BB6A,stroke:#00695C,stroke-width:2px,color:#fff
```

---

## 图表 9：订单生命周期状态图

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Waiting: order enters system
    Waiting --> Matched: driver accepts
    Waiting --> Cancelled: timeout
    Matched --> Executing: driver picks up
    Executing --> Completed: reached destination
    Cancelled --> [*]
    Completed --> [*]

    note right of Created
        T=0s: Order generated
    end note

    note right of Waiting
        T=50s: Waiting for driver
        Max wait: 600s
    end note

    note right of Matched
        T=120s: Driver accepted
    end note

    note right of Executing
        T=180s: Passenger in vehicle
    end note

    note right of Completed
        T=450s: Trip completed
        Total wait: 450s
    end note
```

---

## 图表 10：模拟器架构图

```mermaid
graph TB
    subgraph core["Core Components"]
        EQ["Event Queue<br/>(Priority Queue)<br/>Orders|Vehicles|Dispatch"]
    end

    subgraph managers["Managers"]
        OM["Order Manager<br/>Generation|Matching|Timeout"]
        VM["Vehicle Manager<br/>State|Position|Transitions"]
    end

    subgraph execution["Execution"]
        DE["Dispatch Executor<br/>Validation|Operations|Execution"]
        RC["Reward Calculator<br/>Multi-stage rewards<br/>Business metrics"]
    end

    subgraph observation["Observation & Decision"]
        SO["State Observer<br/>Grid aggregation<br/>Time encoding|State generation"]
        RL["RL Model<br/>Decision making<br/>Action selection"]
    end

    EQ --> OM
    EQ --> VM
    EQ --> DE
    EQ --> RC
    EQ --> SO

    OM --> EQ
    VM --> EQ
    DE --> EQ
    RC --> EQ
    SO --> RL
    RL --> DE

    style EQ fill:#FF6B6B,stroke:#C62828,stroke-width:3px,color:#fff
    style OM fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style VM fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style DE fill:#DCEDC8,stroke:#558B2F,stroke-width:2px
    style RC fill:#DCEDC8,stroke:#558B2F,stroke-width:2px
    style SO fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
    style RL fill:#FFF3E0,stroke:#F57F17,stroke-width:2px
```

---

## 图表 11：MGCN 多图融合

```mermaid
graph LR
    A["Input Features<br/>H^0 (B, 400, 5)"]

    B["Adjacency Graph<br/>A_n (400×400)"]
    C["POI Graph<br/>A_p (400×400)"]

    D["GCN Branch 1<br/>H_n^1 (B, 400, 64)"]
    E["GCN Branch 2<br/>H_p^1 (B, 400, 64)"]

    F["Attention Weights<br/>α_n, α_p"]

    G["Fusion<br/>H_fused = α_n⊙H_n + α_p⊙H_p"]

    H["Output<br/>H_out (B, 400, 32)"]

    A --> D
    B --> D
    A --> E
    C --> E
    D --> F
    E --> F
    F --> G
    D --> G
    E --> G
    G --> H

    style A fill:#DCEDC8,stroke:#558B2F,stroke-width:2px
    style B fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style C fill:#C8E6C9,stroke:#00695C,stroke-width:2px
    style D fill:#E1F5FE,stroke:#0277BD,stroke-width:2px
    style E fill:#E8F5E9,stroke:#00695C,stroke-width:2px
    style F fill:#FFE082,stroke:#F57F17,stroke-width:2px
    style G fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px
    style H fill:#FFF3E0,stroke:#F57F17,stroke-width:2px
```

---

## 图表 12：训练指标监控

```mermaid
graph TD
    A["Training Loop"]

    B["Collect Metrics"]
    C["Loss"]
    D["Match Rate"]
    E["Waiting Time"]

    F["Exploration Phase<br/>Episode 0-100<br/>ε: 60% → 5%"]
    G["Learning Phase<br/>Episode 100-400<br/>ε: 5%"]
    H["Convergence Phase<br/>Episode 400-500<br/>ε: 5%"]

    I["Loss: High → Low"]
    J["Match Rate: Low → High"]
    K["Wait Time: High → Low"]

    A --> B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> F
    E --> F
    F --> G
    G --> H

    F --> I
    G --> J
    H --> K

    style F fill:#C8E6C9,stroke:#558B2F,stroke-width:2px
    style G fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style H fill:#B3E5FC,stroke:#0277BD,stroke-width:2px
    style I fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style J fill:#E8F5E9,stroke:#558B2F,stroke-width:2px
    style K fill:#E8F5E9,stroke:#558B2F,stroke-width:2px
```

---

## 使用说明

### 在线生成方法：

1. **打开 Mermaid 在线编辑器**
   - 访问 https://mermaid.live

2. **复制上面的代码**
   - 选择你想要的图表代码（从 ``` 到 ```）

3. **粘贴到编辑器**
   - 在左侧代码区粘贴代码

4. **自动生成图表**
   - 右侧会实时显示生成的图表

5. **导出图片**
   - 点击右上角的下载按钮
   - 选择 PNG 或 SVG 格式
   - 保存到本地

### 本地使用方法（如果安装了 Node.js）：

```bash
# 安装 mermaid-cli
npm install -g mermaid-cli

# 生成 PNG 图片
mmdc -i diagram.mmd -o diagram.png

# 生成 SVG 图片
mmdc -i diagram.mmd -o diagram.svg -t dark
```

### 在 Markdown 中使用：

```markdown
\`\`\`mermaid
graph TD
    A --> B
\`\`\`
```

---

## 图表说明

| 图表 | 类型 | 用途 |
|------|------|------|
| 1 | 数据流图 | 展示完整模型架构 |
| 2 | 数据流图 | MGCN 多图卷积网络 |
| 3 | 数据流图 | Dueling DQN 双流架构 |
| 4 | 流程图 | 训练循环完整流程 |
| 5 | 曲线图 | Epsilon 衰减曲线 |
| 6 | 曲线图 | 奖励函数曲线 |
| 7 | 流程图 | PER 优先级回放机制 |
| 8 | 对比图 | 目标网络稳定性 |
| 9 | 状态图 | 订单生命周期 |
| 10 | 架构图 | 模拟器系统架构 |
| 11 | 数据流图 | MGCN 多图融合 |
| 12 | 流程图 | 训练指标监控 |

---

## 自定义建议

### 修改颜色：
```mermaid
style NodeName fill:#新颜色代码,stroke:#边框颜色,stroke-width:2px
```

### 修改文本：
直接编辑引号内的内容

### 修改连接：
```mermaid
A --> B    # 普通箭头
A -.-> B   # 虚线箭头
A ==> B    # 加粗箭头
```

---

## 快速链接

- **Mermaid 官网**：https://mermaid.js.org/
- **在线编辑器**：https://mermaid.live/
- **文档**：https://mermaid.js.org/intro/
- **示例**：https://mermaid.js.org/ecosystem/integrations.html

