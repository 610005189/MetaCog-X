"""验证文档示例是否可运行"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import MetaCogXConfig
from models.metacogx_model import MetaCogXModel

print("=== 测试 API 文档示例 ===")

# 1. 配置测试
config = MetaCogXConfig.tiny()
print(f"1. 配置创建成功: {config}")

# 2. 模型创建测试
model = MetaCogXModel(config, enable_metacog=True)
print(f"2. 模型创建成功, 参数量: {model.get_num_params()}")

# 3. 前向传播测试
input_ids = torch.randint(0, config.vocab_size, (2, 32))
outputs = model(input_ids)
print(f"3. 前向传播成功, logits shape: {outputs['logits'].shape}")
print(f"   Mode: {outputs['mode']}, Switches: {outputs['switch_stats']['switches']}")

# 4. 生成测试
model.eval()
with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=10, verbose=False)
print(f"4. 生成测试成功, generated shape: {generated.shape}")

# 5. 带标签的前向传播测试
labels = input_ids.clone()
outputs_with_loss = model(input_ids, labels=labels)
print(f"5. 带损失的前向传播成功, loss: {outputs_with_loss['loss'].item():.4f}")

# 6. 获取内部状态测试
outputs_meta = model(input_ids, return_meta=True)
print(f"6. 获取内部状态成功:")
print(f"   Meta shape: {outputs_meta['meta'].shape}")
print(f"   Awareness shape: {outputs_meta['awareness'].shape}")

print("=== 所有示例测试通过 ===")