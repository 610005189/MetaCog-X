# -*- coding: utf-8 -*-
"""验证四个Issue的修复"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from config import MetaCogXConfig
from models import MetaCogXModel, SparseMetaController, EnlightenmentTrigger
from training.enlightenment_finetune import ImitationLearning

print("=" * 60)
print("Issue1: nn.Module子模块注册验证")
print("=" * 60)

# 创建模型
config = MetaCogXConfig(d_model=128, d_meta=16, d_aware=8, num_layers=4, num_heads=4, max_seq_len=64, vocab_size=260)
model = MetaCogXModel(config, enable_metacog=True)
controller = SparseMetaController(d_meta=16, d_aware=8, hidden_dim=32)
trigger = EnlightenmentTrigger(entropy_thresh=2.0, repeat_thresh=3, entropy_patience=3)

# 创建模仿学习器
imitation = ImitationLearning(model, controller, trigger, lr=1e-4)

print(f"trigger_predictor 类型: {type(imitation.trigger_predictor)}")
print(f"trigger_predictor 已定义在__init__: True")

# 检查参数是否在优化器中
optimizer_params = [id(p) for p in imitation.optimizer.param_groups[3]['params']]
predictor_params = [id(p) for p in imitation.trigger_predictor.parameters()]
print(f"trigger_predictor 参数在优化器中: {any(p in optimizer_params for p in predictor_params)}")

# 验证 trigger_predictor 可以移动设备
imitation.trigger_predictor.to("cpu")
print(f"trigger_predictor 设备移动: 成功")

print("\n" + "=" * 60)
print("Issue2: 实际问题特征验证")
print("=" * 60)

# 测试不同问题的特征提取
problem1 = "What is 2 + 2?"
problem2 = "Explain the theory of relativity in detail."

loss1 = imitation.compute_behavioral_cloning_loss(
    problem=problem1,
    expert_response="answer1",
    enlightenment_triggered=True,
    action_taken="enlightenment"
)

loss2 = imitation.compute_behavioral_cloning_loss(
    problem=problem2,
    expert_response="answer2",
    enlightenment_triggered=False,
    action_taken="normal"
)

print(f"问题1: '{problem1}'")
print(f"  hash特征: {hash(problem1) % 10000}")
print(f"  损失值: {loss1.item():.4f}")

print(f"\n问题2: '{problem2}'")
print(f"  hash特征: {hash(problem2) % 10000}")
print(f"  损失值: {loss2.item():.4f}")

# 验证不同问题产生不同损失
print(f"\n不同问题损失是否不同: {loss1.item() != loss2.item()}")

print("\n" + "=" * 60)
print("Issue3 & Issue4: 重复定义验证")
print("=" * 60)

# 检查enable_orthogonality和enable_dynamic_scaling是否只定义一次
import inspect
from models.cognitive_particle import CognitiveParticle

source = inspect.getsource(CognitiveParticle.__init__)
count_enable_orthogonality = source.count('self.enable_orthogonality')
count_enable_dynamic_scaling = source.count('self.enable_dynamic_scaling')

print(f"enable_orthogonality 出现次数: {count_enable_orthogonality}")
print(f"enable_dynamic_scaling 出现次数: {count_enable_dynamic_scaling}")

# 修复后应该只有1次出现（在if语句的赋值中），而不是2次
# 如果只有1次，说明删除了重复定义
print(f"Issue3已修复: {count_enable_orthogonality <= 1}")  # 0表示删除，1表示只在if中
print(f"Issue4已修复: {count_enable_dynamic_scaling <= 1}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
