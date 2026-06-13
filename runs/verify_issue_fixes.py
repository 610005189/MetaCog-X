# -*- coding: utf-8 -*-
"""验证三个Issue的修复"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from models.cognitive_particle import CognitiveParticle, DynamicScalingController

print("=" * 60)
print("Issue1: nn.Module子模块注册验证")
print("=" * 60)

# 创建两个实例
cp_enabled = CognitiveParticle(128, 16, 8, enable_orthogonality=True, enable_dynamic_scaling=True)
cp_disabled = CognitiveParticle(128, 16, 8, enable_orthogonality=False, enable_dynamic_scaling=False)

print(f"启用时 ortho_loss 类型: {type(cp_enabled.ortho_loss)}")
print(f"禁用时 ortho_loss: {cp_disabled.ortho_loss}")
print(f"启用时 scaling_controller 类型: {type(cp_enabled.scaling_controller)}")
print(f"禁用时 scaling_controller: {cp_disabled.scaling_controller}")

# 检查子模块注册
submodules = [n for n, p in cp_enabled.named_modules() if "ortho" in n or "scaling" in n]
print(f"注册的子模块: {submodules}")

# 验证 to(device) 是否正常工作
print("\n验证 model.to(device):")
cp_enabled.to("cpu")
print("  成功移动到CPU")

# 验证参数是否被正确追踪
params = [n for n, p in cp_enabled.named_parameters() if "ortho" in n or "scaling" in n]
print(f"  相关参数: {len(params)} 个")

print("\n" + "=" * 60)
print("Issue2: 梯度流验证")
print("=" * 60)

# 创建控制器
controller = DynamicScalingController(base_dims={"content": 128, "meta": 16, "awareness": 8})

# 创建需要梯度的输入 - 使用正确的task_id范围和context_features维度
task_id = torch.randint(0, controller.task_scales.num_embeddings, (1,), requires_grad=False)
context_features = torch.randn(1, 64, requires_grad=True)  # context_encoder期望[B, 64]

# 获取缩放因子
scales = controller(task_id=task_id, context_features=context_features)

# 检查 scales 是否是张量（有梯度）
print(f"content scale 类型: {type(scales['content'])}")
print(f"meta scale 类型: {type(scales['meta'])}")
print(f"awareness scale 类型: {type(scales['awareness'])}")

# 验证梯度是否可以传播
if isinstance(scales['content'], torch.Tensor):
    print("  scales 是张量，梯度可以传播")
    # 尝试反向传播
    loss = scales['content'] + scales['meta'] + scales['awareness']
    loss.backward()
    print(f"  context_features.grad 是否存在: {context_features.grad is not None}")
else:
    print("  WARNING: scales 不是张量，梯度无法传播")

print("\n" + "=" * 60)
print("Issue3: 损失计算验证")
print("=" * 60)

from training.enlightenment_finetune import ImitationLearning
from config import MetaCogXConfig
from models import MetaCogXModel, SparseMetaController, EnlightenmentTrigger

# 创建模型
config = MetaCogXConfig(d_model=128, d_meta=16, d_aware=8, num_layers=4, num_heads=4, max_seq_len=64, vocab_size=260)
model = MetaCogXModel(config, enable_metacog=True)
controller = SparseMetaController(d_meta=16, d_aware=8, hidden_dim=32)
trigger = EnlightenmentTrigger(entropy_thresh=2.0, repeat_thresh=3, entropy_patience=3)

# 创建模仿学习器
imitation = ImitationLearning(model, controller, trigger, lr=1e-4)

# 测试损失计算
loss1 = imitation.compute_behavioral_cloning_loss(
    problem="test problem",
    expert_response="test response",
    enlightenment_triggered=True,
    action_taken="enlightenment"
)
loss2 = imitation.compute_behavioral_cloning_loss(
    problem="test problem",
    expert_response="test response",
    enlightenment_triggered=False,
    action_taken="normal"
)

print(f"enlightenment_triggered=True 时损失: {loss1.item():.4f}")
print(f"enlightenment_triggered=False 时损失: {loss2.item():.4f}")

# 验证损失是否不同
if loss1.item() != loss2.item():
    print("  PASS: 损失值不同，模型可以学习区分正确/错误预测")
else:
    print("  FAIL: 损失值相同，模型无法学习")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)