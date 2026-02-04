#!/usr/bin/env python3
"""
测试H-MARL修复后的Manager输出
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from config import Config
from baselines.hmarl_agent import MFuN_Agent, ManagerNetwork

def test_manager_output():
    """测试Manager是否能输出负值"""
    print("="*70)
    print("测试1：Manager输出范围")
    print("="*70)

    config = Config()
    manager = ManagerNetwork(config)

    # 创建随机输入
    batch_size = 1
    state_dim = config.NUM_GRIDS * 5  # 2000
    state = torch.randn(batch_size, state_dim)

    # 前向传播
    sub_goals, value, hidden = manager(state, hidden=None)

    # 统计
    sub_goals_np = sub_goals[0].detach().numpy()
    print(f"\nsub_goals shape: {sub_goals.shape}")
    print(f"sub_goals range: [{sub_goals_np.min():.2f}, {sub_goals_np.max():.2f}]")
    print(f"负值数量: {(sub_goals_np < 0).sum()}/400")
    print(f"< -0.5 的数量: {(sub_goals_np < -0.5).sum()}/400  （这些grid会触发调度）")
    print(f"> 0的数量: {(sub_goals_np > 0).sum()}/400")
    print(f"\n✅ 如果'< -0.5的数量'大于0，则Manager可以正常工作")

    return (sub_goals_np < -0.5).sum() > 0


def test_agent_initialization():
    """测试Agent初始化"""
    print("\n" + "="*70)
    print("测试2：Agent初始化")
    print("="*70)

    config = Config()
    agent = MFuN_Agent(config)

    print(f"\n✅ Agent初始化成功")
    print(f"  Manager参数量: {sum(p.numel() for p in agent.manager.parameters()):,}")
    print(f"  Worker参数量: {sum(p.numel() for p in agent.worker_shared.parameters()):,}")

    return True


if __name__ == "__main__":
    print("\n🔧 H-MARL修复验证测试\n")

    test1_pass = test_manager_output()
    test2_pass = test_agent_initialization()

    print("\n" + "="*70)
    print("测试结果总结")
    print("="*70)
    print(f"  测试1（Manager输出负值）: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"  测试2（Agent初始化）: {'✅ 通过' if test2_pass else '❌ 失败'}")

    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！Manager现在可以输出负值，Worker将能够执行调度。")
    else:
        print("\n⚠️  部分测试失败，请检查修改。")

    print("="*70)

