========================================
可视化文件夹使用说明
========================================

📁 文件结构
-----------
visualization/
├── data_config.py           # 数据配置文件（在这里修改你的数据）
├── plot_main_results.py     # 主实验结果绘图脚本
├── plot_ablation.py         # 消融实验绘图脚本（待创建）
├── plot_distribution.py     # 初始分布实验绘图脚本（待创建）
└── README_VISUALIZATION.txt # 本说明文件


🚀 快速开始
-----------
1. 修改数据：
   打开 data_config.py，将你的实验数据填入对应位置

2. 运行绘图脚本：
   cd /Users/qiukuipeng/PycharmProjects/work2
   python visualization/plot_main_results.py

3. 查看结果：
   图表会保存在 results/visualizations/ 文件夹


📊 数据格式说明
---------------
主实验数据（7天数据）：
  - completion_rate: [day1, day2, ..., day7]  # 匹配率（小数，如0.85表示85%）
  - waiting_time: [day1, day2, ..., day7]     # 等待时间（秒）

消融/分布实验数据（平均值）：
  - completion_rate: 0.XXX  # 7天的平均匹配率
  - waiting_time: XXX.XX    # 7天的平均等待时间


📈 生成的图表
-------------
主实验结果：
  - fig1_completion_rate_combined.png  # 匹配率（堆叠柱+折线，3×子图）
  - fig2_waiting_time.png              # 等待时间（折线图）
  - fig3_average_performance.png       # 平均性能对比（柱状图）

消融实验结果（待创建）：
  - fig4_ablation_comparison.png       # 消融实验对比

初始分布实验（待创建）：
  - fig5_distribution_comparison.png   # 初始分布鲁棒性


⚙️ 自定义设置
-------------
如需修改图表样式（颜色、字体、大小等），请编辑对应的 .py 文件：
  - 颜色：搜索 color='#XXXXXX'
  - 字体大小：搜索 fontsize=XX
  - 图表尺寸：搜索 figsize=(宽, 高)


❓ 常见问题
-----------
Q: 中文显示乱码？
A: 在 plot_xxx.py 开头已设置中文字体，如仍有问题，请安装中文字体

Q: 如何修改图表配色？
A: 在脚本中搜索颜色代码：
   #2ECC71 = 绿色（已匹配）
   #E74C3C = 红色（未匹配）
   #3498DB = 蓝色（折线）

Q: 如何添加更多车辆数配置？
A: 在 data_config.py 中添加 data_XXXX = {...}，
   然后在绘图脚本中添加对应的处理逻辑


========================================
更新日期: 2026-02-04
========================================

