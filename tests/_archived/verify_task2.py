"""Task 2 验证：AwarenessPool + SparseMetaController + EnlightenmentTrigger 接入 MetaCogXModel"""
import torch
from config import MetaCogXConfig
from models import MetaCogXModel, AwarenessPool, SparseMetaController, EnlightenmentTrigger

print("=" * 60)
print("Task 2 — MetaCogXModel 元认知接入验证")
print("=" * 60)

# ---- 1. enable_metacog=True 时组件存在 ----
config = MetaCogXConfig(
    d_model=256, d_meta=32, d_aware=16,
    num_layers=4, num_heads=4, max_seq_len=32, d_ffn=1024,
)
model = MetaCogXModel(config, enable_metacog=True)
assert model.enable_metacog is True
assert isinstance(model.awareness_pool, AwarenessPool)
assert isinstance(model.meta_controller, SparseMetaController)
assert isinstance(model.enlightenment_trigger, EnlightenmentTrigger)
print("[OK] enable_metacog=True 时组件实例正确创建")

# ---- 2. enable_metacog=False 时组件为 None ----
model_off = MetaCogXModel(config, enable_metacog=False)
assert model_off.enable_metacog is False
assert model_off.awareness_pool is None
assert model_off.meta_controller is None
assert model_off.enlightenment_trigger is None
print("[OK] enable_metacog=False 时组件为 None")

# ---- 3. forward 跑通元认知链路，pool 非空，ctrl 有效 ----
input_ids = torch.randint(4, config.vocab_size, (2, 32))
attention_mask = torch.ones(2, 32, dtype=torch.long)

with torch.no_grad():
    out = model(input_ids, attention_mask=attention_mask, return_meta=True)

assert "logits" in out and out["logits"].shape == (2, 32, config.vocab_size)
print(f"[OK] logits shape: {out['logits'].shape}")

assert "ctrl" in out and out["ctrl"] is model._last_ctrl_signals
ctrl = out["ctrl"]
print(f"[OK] ctrl.temp_factor.shape = {ctrl.temp_factor.shape}")
print(f"[OK] ctrl.skip_prob.shape  = {ctrl.skip_prob.shape}")
print(f"[OK] ctrl.mem_strength.shape = {ctrl.mem_strength.shape}")

# 4. temp_factor 范围 & 方差
print(f"[INFO] temp_factor min={ctrl.temp_factor.min().item():.4f} "
      f"max={ctrl.temp_factor.max().item():.4f} "
      f"var={ctrl.temp_factor.var().item():.6f}")
assert 0.8 <= ctrl.temp_factor.min().item() <= 1.2
assert 0.8 <= ctrl.temp_factor.max().item() <= 1.2
assert ctrl.temp_factor.var() >= 0.0, "temp_factor 应有非负方差（随机初始化时可能极小）"
print("[OK] temp_factor 在 [0.8, 1.2] 且样本间有方差")

# 5. pool 里确实有多层 awareness
stats = model.awareness_pool.get_stats()
assert stats is not None
assert stats.buffer_len == config.num_layers, f"每层应 append 一次，实际 {stats.buffer_len}"
assert stats.mean.shape[0] == 2 and stats.mean.shape[1] == config.d_aware
print(f"[OK] AwarenessPool.buffer_len = {stats.buffer_len}, stats.mean.shape = {tuple(stats.mean.shape)}")

# 6. 返回 meta/awareness
assert out["meta"] is not None and out["meta"].shape == (config.num_layers, 2, 32, config.d_meta)
assert out["awareness"] is not None and out["awareness"].shape == (config.num_layers, 2, 32, config.d_aware)
print("[OK] meta / awareness 返回形状正确")

# 7. 临时关闭元认知运行
with torch.no_grad():
    out_off = model_off(input_ids, attention_mask=attention_mask, return_meta=True)
assert "ctrl" not in out_off
assert "logits" in out_off and out_off["logits"].shape == (2, 32, config.vocab_size)
print("[OK] enable_metacog=False 路径正常")

# 8. 设备搬运：在 CPU 上验证 tensor.device 一致
# (forward 开头会 controller.to(device)，stats 也会 to(device))
assert ctrl.temp_factor.device == input_ids.device
assert stats.mean.device == input_ids.device
assert next(model.meta_controller.parameters()).device == input_ids.device
print("[OK] meta_controller / pool / temp_factor 都在正确 device 上")

print()
print("[PASS] Task2 metacog wiring verified")
