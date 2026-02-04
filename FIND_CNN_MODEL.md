# 🔍 如何找到 CNN 消融实验的模型

## 📂 模型保存位置（另一台设备上）

根据你的配置，CNN 消融实验的模型应该保存在：

### 1️⃣ **JSON 结果文件** (肯定存在)
```bash
results/vehicles_1800/ablation/ablation_results_YYYYMMDD_HHMMSS.json
```
或者旧路径：
```bash
results/ablation_studies/ablation_results_YYYYMMDD_HHMMSS.json
```

### 2️⃣ **模型检查点** (如果有保存的话)
根据代码分析，`run_ablation_simple.py` **没有自动保存模型**！

但如果你手动保存了，可能在：
```bash
results/vehicles_1800/models/cnn_ablation_episode_*.pt
```
或
```bash
results/vehicles_1800/ablation/cnn_ablation_episode_*.pt
```

### 3️⃣ **训练日志**
```bash
results/vehicles_1800/logs/ablation_cnn_YYYYMMDD_HHMMSS.txt
```

---

## 🔧 在另一台设备上运行这些命令查找：

### 查找所有消融实验相关文件：
```bash
cd /path/to/work2
find results -name "*ablation*" -o -name "*cnn*" | head -20
```

### 查找所有 .pt 模型文件：
```bash
find results -name "*.pt" | grep -i -E "(ablation|cnn)"
```

### 查找所有 JSON 结果文件：
```bash
find results -name "ablation_results_*.json"
```

### 查看具体目录内容：
```bash
ls -lh results/vehicles_1800/ablation/
ls -lh results/vehicles_1800/models/
ls -lh results/ablation_studies/
```

---

## ⚠️ 重要发现：

### 问题：`run_ablation_simple.py` 没有保存模型检查点！

查看代码发现，`run_ablation_simple.py` 的 `save_results()` 函数只保存了：
- ✅ JSON 结果文件（包含测试指标）
- ❌ **没有保存模型权重文件 (.pt)**

### 解决方案：

#### 方案1: 添加模型保存功能（推荐）
修改 `run_ablation_simple.py`，在训练完成后保存模型。

#### 方案2: 重新训练并保存
重新运行训练，但这次在训练后手动保存模型。

#### 方案3: 使用现有模型（如果存在）
如果你在训练过程中手动保存了模型，那么它应该在：
```bash
results/vehicles_1800/models/
```
或
```bash
results/vehicles_1800/ablation/
```

---

## 📋 传输到当前设备的步骤

### 1. 在另一台设备上打包模型和结果：
```bash
cd /path/to/work2
tar -czf cnn_ablation_results.tar.gz \
    results/vehicles_1800/ablation/ \
    results/vehicles_1800/models/ \
    results/vehicles_1800/logs/
```

### 2. 传输到当前设备：
```bash
# 使用 scp (如果两台机器在网络中)
scp user@remote:/path/to/work2/cnn_ablation_results.tar.gz ~/Downloads/

# 或使用其他方式（U盘、云盘等）
```

### 3. 在当前设备解压：
```bash
cd /Users/qiukuipeng/PycharmProjects/work2
tar -xzf ~/Downloads/cnn_ablation_results.tar.gz
```

### 4. 验证文件：
```bash
ls -lh results/vehicles_1800/ablation/
ls -lh results/vehicles_1800/models/
```

---

## 🚀 如果模型不存在，需要修改代码保存模型

我可以帮你修改 `run_ablation_simple.py` 添加模型保存功能，或者添加一个独立的保存函数到 `ablation_trainer.py`。

需要我帮你做吗？

