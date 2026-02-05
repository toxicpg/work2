"""
紧急保存脚本 - 如果训练完忘记保存模型，用这个脚本手动保存

使用场景：
1. 训练刚结束，Python 进程还在运行
2. 你在 Jupyter notebook 或 IPython 中训练的
3. trainer 对象还在内存中

使用方法：
在训练完成的 Python 环境中执行：

```python
# 假设 trainer 是你的 AblationMGCNTrainer 对象
import torch
import os

# 保存模型
save_dir = 'results/vehicles_1800/ablation/'
os.makedirs(save_dir, exist_ok=True)

checkpoint = {
    'episode': 10,  # 替换为你训练的实际 episode 数
    'ablation_type': 'cnn',  # 你的消融类型
    'model_state_dict': trainer.main_net.state_dict(),
    'target_model_state_dict': trainer.target_net.state_dict(),
    'optimizer_state_dict': trainer.optimizer.state_dict(),
    'epsilon': trainer.epsilon,
    'train_step_count': trainer.train_step_count,
    'total_rewards': trainer.total_rewards[-100:],
    'losses': trainer.losses[-100:],
    'epsilon_history': trainer.epsilon_history[-100:],
}

checkpoint_path = os.path.join(save_dir, 'cnn_ablation_emergency_save.pt')
torch.save(checkpoint, checkpoint_path)
print(f"✓ 模型已保存到: {checkpoint_path}")
```

或者直接运行这个脚本（需要修改参数）：
    python emergency_save_model.py --trainer-var trainer --episode 10
"""

import argparse


def emergency_save_from_namespace(namespace_dict, ablation_type, episode, save_dir):
    """
    从命名空间字典中提取 trainer 并保存

    Args:
        namespace_dict: 包含 trainer 的字典（如 globals()）
        ablation_type: 消融类型
        episode: episode 数
        save_dir: 保存目录
    """
    import torch
    import os

    # 查找 trainer 对象
    trainer = None
    for key, value in namespace_dict.items():
        if 'trainer' in key.lower() or 'Trainer' in str(type(value)):
            trainer = value
            print(f"找到 trainer 对象: {key}")
            break

    if trainer is None:
        print("❌ 未找到 trainer 对象！")
        print("提示: 请确保 trainer 对象还在内存中")
        return False

    # 保存检查点
    try:
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'episode': episode,
            'ablation_type': ablation_type,
            'model_state_dict': trainer.main_net.state_dict(),
            'target_model_state_dict': trainer.target_net.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'epsilon': trainer.epsilon,
            'train_step_count': trainer.train_step_count,
            'total_rewards': getattr(trainer, 'total_rewards', [])[-100:],
            'losses': getattr(trainer, 'losses', [])[-100:],
            'epsilon_history': getattr(trainer, 'epsilon_history', [])[-100:],
        }

        checkpoint_path = os.path.join(save_dir, f'{ablation_type}_ablation_emergency_save_ep{episode}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ 模型已保存到: {checkpoint_path}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_manual_save_code(ablation_type='cnn', episode=10, save_dir='results/vehicles_1800/ablation/'):
    """打印手动保存代码"""
    code = f"""
# ========================================
# 手动保存模型代码（复制粘贴到你的 Python 环境）
# ========================================

import torch
import os

# 1. 确认 trainer 对象存在
print("Trainer 对象:", trainer)
print("Ablation 类型:", trainer.ablation_type)

# 2. 保存模型
save_dir = '{save_dir}'
os.makedirs(save_dir, exist_ok=True)

checkpoint = {{
    'episode': {episode},
    'ablation_type': '{ablation_type}',
    'model_state_dict': trainer.main_net.state_dict(),
    'target_model_state_dict': trainer.target_net.state_dict(),
    'optimizer_state_dict': trainer.optimizer.state_dict(),
    'epsilon': trainer.epsilon,
    'train_step_count': trainer.train_step_count,
    'total_rewards': trainer.total_rewards[-100:] if hasattr(trainer, 'total_rewards') else [],
    'losses': trainer.losses[-100:] if hasattr(trainer, 'losses') else [],
    'epsilon_history': trainer.epsilon_history[-100:] if hasattr(trainer, 'epsilon_history') else [],
}}

checkpoint_path = os.path.join(save_dir, '{ablation_type}_ablation_emergency_save_ep{episode}.pt')
torch.save(checkpoint, checkpoint_path)
print(f"✓ 模型已保存到: {{checkpoint_path}}")

# 3. 验证保存成功
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
    print(f"✓ 文件大小: {{size_mb:.2f}} MB")
else:
    print("❌ 文件未找到！")
"""
    print(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='紧急保存消融实验模型')
    parser.add_argument('--ablation', type=str, default='cnn',
                       help='消融类型 (默认: cnn)')
    parser.add_argument('--episode', type=int, default=10,
                       help='训练的 episode 数 (默认: 10)')
    parser.add_argument('--save-dir', type=str, default='results/vehicles_1800/ablation/',
                       help='保存目录')
    parser.add_argument('--print-code', action='store_true',
                       help='打印手动保存代码')

    args = parser.parse_args()

    if args.print_code:
        print("\n" + "="*80)
        print("紧急保存代码 - 复制粘贴到你训练的 Python 环境中执行")
        print("="*80)
        print_manual_save_code(args.ablation, args.episode, args.save_dir)
    else:
        print("\n" + "="*80)
        print("⚠️  这个脚本需要在训练环境中运行，且 trainer 对象必须在内存中")
        print("="*80)
        print("\n建议使用方式:")
        print("1. 运行 'python emergency_save_model.py --print-code' 获取保存代码")
        print("2. 复制生成的代码到你的训练环境（如 Jupyter notebook）")
        print("3. 执行代码保存模型")
        print("\n或者直接在训练环境中执行以下代码:\n")
        print_manual_save_code(args.ablation, args.episode, args.save_dir)

