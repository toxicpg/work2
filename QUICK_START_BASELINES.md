# 🚀 Baseline实验快速启动

## 最简单的运行方式

### 在另一台机器上运行

```bash
# 1. 进入项目目录
cd /path/to/work2

# 2. 运行所有baseline实验（1800, 2000, 2200车辆）
python run_baselines_all_vehicles.py
```

**就这么简单！** 脚本会自动：
- ✅ 切换车辆数量配置
- ✅ 运行所有5个baseline
- ✅ 保存结果到对应文件夹
- ✅ 生成完整报告

---

## 预期时间

| 配置 | 时间 |
|------|------|
| 单个车辆配置（5个baseline） | ~2小时 |
| **全部3个车辆配置** | **~5-6小时** |

---

## 结果位置

```
results/
├── vehicles_1800/baselines/  ← 1800辆车的结果
├── vehicles_2000/baselines/  ← 2000辆车的结果
└── vehicles_2200/baselines/  ← 2200辆车的结果
```

每个文件夹包含5个方法的结果：
- `random_walk_results_*.json`
- `random_dispatch_results_*.json`
- `sarsa_saa_results_*.json`
- `hmarl_results_*.json`
- `cnn_ddqn_results_*.json`

---

## 后台运行（推荐）

如果需要长时间运行，建议后台执行：

### Linux/Mac:
```bash
nohup python run_baselines_all_vehicles.py > baseline.log 2>&1 &
```

查看日志：
```bash
tail -f baseline.log
```

### 使用screen:
```bash
screen -S baseline
python run_baselines_all_vehicles.py
# 按 Ctrl+A 然后 D 离开

# 重新连接
screen -r baseline
```

---

## 如果只想测试单个配置

```bash
# 1. 编辑 config.py
#    找到: TOTAL_VEHICLES = 2000
#    改为: TOTAL_VEHICLES = 1800  (或你想要的数量)

# 2. 运行单个baseline
python baselines/cnn_ddqn.py
```

---

## 结果包含的指标

每个结果文件包含：

```json
{
  "avg_results": {
    "completion_rate": 0.XX,      // 完成率
    "cancel_rate": 0.XX,          // 取消率
    "avg_waiting_time": XXX,      // 平均等待时间(秒)
    "vehicle_utilization": 0.XX,  // 车辆利用率
    "avg_total_revenue": XXXX     // 平均总收入
  }
}
```

---

## 常见问题

### Q: 运行中断了怎么办？
A: 重新运行脚本，它会跳过已完成的实验

### Q: CUDA内存不足？
A: 编辑 `config.py`，设置 `DEVICE = 'cpu'`

### Q: 某个baseline失败？
A: 查看错误日志，可以单独重跑该baseline

---

## 下一步

实验完成后：

1. **检查结果**：
   ```bash
   ls results/vehicles_*/baselines/
   ```

2. **生成对比图**：
   ```bash
   python utils/shiyantu.py
   ```

3. **分析数据**，撰写论文 📝

---

## 完整文档

详细说明请查看：
- `BASELINE_EXPERIMENTS_GUIDE.txt` - 完整实验指南
- `baselines/README_BASELINES.md` - 方法详细说明

---

## 快速命令汇总

```bash
# 运行所有配置
python run_baselines_all_vehicles.py

# 后台运行
nohup python run_baselines_all_vehicles.py > baseline.log 2>&1 &

# 查看日志
tail -f baseline.log

# 单独运行CNN-DDQN
python baselines/cnn_ddqn.py

# 测试CNN模型结构
python baselines/cnn_ddqn_model.py
```

---

**祝实验顺利！** 🎉

