## 问题判断
- 当前训练奖励在订单完成事件中计算为 wait_score=exp(-等待时间/T0)（c:\Users\admin\work2\environment.py:1007–1049）。这只间接优化等待时间，对“匹配率”没有显式正向激励。
- 绘图与评估主要看“总收入/完成率/等待时间”（c:\Users\admin\work2\train.py:41–51, 299–308；c:\Users\admin\work2\evaluate.py:183–197），与训练优化目标不完全一致。

## 奖励函数调整（符合“最小等待 + 最大匹配率”）
- 在订单完成事件，将奖励改为多目标组合：
  - R_complete = +w_match
  - R_wait = −w_wait * (total_wait_time_sec / T0)
  - R_cancel = −w_cancel（若订单被取消时记负奖励）
- 推荐初始权重：w_match=1.0，w_wait=1.0，w_cancel=1.0；后续可网格搜索微调。
- 若希望强化“及时匹配”，可在订单“匹配事件”发生时也给予小额即时奖励 R_match_tick=+w_match_tick（例如 0.2），但这需要在匹配发生处额外 PUSH 经验（目前仅在完成时 PUSH）。

## DQN 架构说明
- 网络结构为 Dueling DQN：优势-价值分解见 c:\Users\admin\work2\models\dispatcher.py:138–173。
- 更新算法为 Double DQN：主网络选 a*（argmax），目标网络估值 Q_target(S',a*)，见 c:\Users\admin\work2\models\trainer.py:243–248、249–255。

## 配套训练策略微调
- episode 增加至 50–100（c:\Users\admin\work2\config.py:106）。
- epsilon：起点 0.6–0.7，按 episode 衰减到 0.1；或按 tick 平滑衰减（c:\Users\admin\work2\models\trainer.py:346–353）。
- Buffer：MIN_REPLAY_SIZE≈8000、BATCH_SIZE≈256、TARGET_UPDATE_FREQ≈1000（c:\Users\admin\work2\config.py:86, 92–95）。

## 实施步骤
1. 修改 environment._process_events 的奖励组合，叠加匹配率与等待时间的项，取消时给负奖励。
2. 可选：在匹配发生处（OrderMatcher 之后）新增即时奖励 PUSH（需要在匹配处保存 S、A 并 PUSH）。
3. 调整超参数并重新训练，观察“平均等待时间下降、匹配率上升”的趋势。
4. 在图表中同时绘制“训练奖励（新定义）”与“业务指标”以验证对齐效果。

## 预期
- 训练过程中匹配率逐步上升、平均等待时间下降；验证与测试曲线与训练奖励趋势一致，提高整体稳定性。