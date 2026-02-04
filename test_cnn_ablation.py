"""快速测试 CNN 消融实验模块"""
import torch
import sys
sys.path.append('.')
from config import Config
from models.ablation_dispatcher import CNNFeatureExtractor, create_ablation_dispatcher
from utils.graph_builder import GraphBuilder

print('=' * 70)
print('测试 CNN 消融实验模块')
print('=' * 70)

config = Config()
print(f'✓ Config 加载成功')
print(f'  - NUM_GRIDS: {config.NUM_GRIDS}')
print(f'  - GRID_SIZE: {config.GRID_SIZE}')
print(f'  - INPUT_DIM: {config.INPUT_DIM}')
print(f'  - HIDDEN_DIMS: {config.HIDDEN_DIMS}')
print(f'  - NUM_ACTIONS: {config.NUM_ACTIONS}')

# 创建测试图结构
graph_builder = GraphBuilder(config)
neighbor_adj, poi_adj = graph_builder.build_graphs()
print(f'\n✓ 图结构创建成功')

# 测试 CNN 特征提取器
print(f'\n--- 测试 CNNFeatureExtractor ---')
cnn = CNNFeatureExtractor(config)
batch_size = 4
node_features = torch.randn(batch_size, 400, 5)
print(f'输入: {node_features.shape}')
output = cnn(node_features)
print(f'输出: {output.shape}')
print(f'✓ CNN 特征提取器测试通过')

# 测试完整的 CNN Dispatcher
print(f'\n--- 测试完整的 CNN Dispatcher ---')
dispatcher = create_ablation_dispatcher(config, neighbor_adj, poi_adj, 'cnn')
vehicle_locations = torch.randint(0, 400, (batch_size,))
day_of_week = torch.randint(0, 7, (batch_size,))
print(f'输入: node_features={node_features.shape}, vehicle_locations={vehicle_locations.shape}, day_of_week={day_of_week.shape}')
q_values = dispatcher(node_features, vehicle_locations, day_of_week)
print(f'输出 Q值: {q_values.shape}')
print(f'✓ CNN Dispatcher 测试通过')

print(f'\n' + '=' * 70)
print(f'✅ 所有测试通过！可以开始训练')
print(f'=' * 70)

