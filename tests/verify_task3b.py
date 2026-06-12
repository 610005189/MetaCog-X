import sys, os, torch, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MetaCogXConfig
from models import MetaCogXModel
from training.losses import TotalLoss
import torch.nn.functional as F

print("=" * 70)
print("Task 1 (3b): Controller anti-collapse entropy + layer diversity")
print("=" * 70)

# -------- 1. SparseMetaController return_logits=True --------
print("\n[1] SparseMetaController return_logits")
cfg = MetaCogXConfig(
    d_model=128, d_meta=16, d_aware=8,
    num_layers=2, num_heads=4, max_seq_len=32, d_ffn=256, vocab_size=100,
)
m = MetaCogXModel(cfg, enable_metacog=True)
ids = torch.randint(0, 100, (2, 16))
msk = torch.ones(2, 16)

out = m(ids, attention_mask=msk, return_meta=True, enable_metacog=True)
assert m._last_ctrl_logits is not None and m._last_ctrl_logits.shape == (2, 3), \
    f"model._last_ctrl_logits shape = {m._last_ctrl_logits.shape} but expected (2, 3)"
print(f"[PASS] model._last_ctrl_logits shape = {m._last_ctrl_logits.shape}")

ctrl, ctrl_logits = m.meta_controller(
    out["meta"].mean(dim=(0, 2)),  # 先对 layer、seq 平均 -> [B, d_meta]
    m.awareness_pool.get_stats() if m.awareness_pool else None,
    return_logits=True,
)
assert ctrl_logits is not None and ctrl_logits.shape == (2, 3), \
    f"ctrl_logits shape = {ctrl_logits.shape} but expected (2, 3)"
print(f"[PASS] SparseMetaController return_logits OK, shape={ctrl_logits.shape}")

# 默认 return_logits=False 应保持原签名（ControlSignals 不是 tuple）
ctrl_only = m.meta_controller(
    out["meta"].mean(dim=(0, 2)),
    m.awareness_pool.get_stats() if m.awareness_pool else None,
    return_logits=False,
)
assert hasattr(ctrl_only, "temp_factor"), "ctrl_only should be ControlSignals dataclass"
print("[PASS] return_logits=False returns ControlSignals (backward compat)")

# -------- 2. TotalLoss 6 分量 (ce/meta/aware/entropy_bonus/layer_div + loss_total) --------
print("\n[2] TotalLoss components")
loss_fn = TotalLoss(alpha=0.01, beta=0.005, gamma=0.02, delta=0.005, ignore_index=0)
loss, comp = loss_fn(
    out["logits"], ids, out["meta"], out["awareness"],
    aware_pool_buffer=None, content_per_layer=None,
    ctrl_logits=m._last_ctrl_logits,
)
print(f"  loss_components keys = {sorted(comp.keys())}")
print(f"  comp = { {k: float(v.detach()) if hasattr(v, 'detach') else v for k, v in comp.items()} }")
assert comp["loss_ce"] > 0, f"ce should be > 0, got {comp['loss_ce']}"
assert comp["loss_meta"] > 0, f"meta should be > 0, got {comp['loss_meta']}"
assert comp["loss_aware"] > 0, f"aware should be > 0, got {comp['loss_aware']}"
assert "entropy_bonus" in comp, "missing entropy_bonus"
assert "layer_div" in comp, "missing layer_div"
print("[PASS] TotalLoss 6-component OK")

# -------- 3. delta=0 vs delta>0: delta>0 应让总 loss 更大 --------
print("\n[3] delta=0 vs delta=0.005")
loss_nodiv_fn = TotalLoss(alpha=0.01, beta=0.005, gamma=0.02, delta=0.0, ignore_index=0)
loss_div_fn = TotalLoss(alpha=0.01, beta=0.005, gamma=0.02, delta=0.005, ignore_index=0)

loss_nodiv, comp_nodiv = loss_nodiv_fn(
    out["logits"], ids, out["meta"], out["awareness"], None, None, m._last_ctrl_logits)
loss_div, comp_div = loss_div_fn(
    out["logits"], ids, out["meta"], out["awareness"], None, None, m._last_ctrl_logits)

loss_nodiv_f = float(loss_nodiv.detach()) if hasattr(loss_nodiv, "detach") else float(loss_nodiv)
loss_div_f = float(loss_div.detach()) if hasattr(loss_div, "detach") else float(loss_div)
layer_div_f = float(comp_div["layer_div"].detach()) if hasattr(comp_div["layer_div"], "detach") else float(comp_div["layer_div"])
layer_nodiv_f = float(comp_nodiv["layer_div"].detach()) if hasattr(comp_nodiv["layer_div"], "detach") else float(comp_nodiv["layer_div"])
print(f"  loss (delta=0.005)  = {loss_div_f:.6f}  layer_div = {layer_div_f:.6f}")
print(f"  loss (delta=0.000) = {loss_nodiv_f:.6f}  layer_div = {layer_nodiv_f:.6f}")
assert loss_div_f > loss_nodiv_f, \
    f"delta>0 should increase loss: {loss_div_f} vs {loss_nodiv_f}"
assert layer_div_f > 0, "layer_div should be > 0"
print("[PASS] delta>0 increases loss as expected")

# -------- 4. gamma>0 vs gamma=0 --------
print("\n[4] gamma=0 vs gamma=0.02")
loss_nogamma_fn = TotalLoss(alpha=0.01, beta=0.005, gamma=0.00, delta=0.005, ignore_index=0)
loss_gamma_fn = TotalLoss(alpha=0.01, beta=0.005, gamma=0.02, delta=0.005, ignore_index=0)
_, c0 = loss_nogamma_fn(out["logits"], ids, out["meta"], out["awareness"], None, None, m._last_ctrl_logits)
_, c1 = loss_gamma_fn(out["logits"], ids, out["meta"], out["awareness"], None, None, m._last_ctrl_logits)
eb0 = float(c0["entropy_bonus"].detach()) if hasattr(c0["entropy_bonus"], "detach") else float(c0["entropy_bonus"])
eb1 = float(c1["entropy_bonus"].detach()) if hasattr(c1["entropy_bonus"], "detach") else float(c1["entropy_bonus"])
print(f"  gamma=0.00 entropy_bonus = {eb0:.6f}")
print(f"  gamma=0.02 entropy_bonus = {eb1:.6f}")
assert abs(eb0) < 1e-9, f"entropy_bonus should be 0 when gamma=0, got {eb0}"
# entropy_bonus = gamma * KL(p||uniform). KL >= 0 always (only = 0 when p uniform). so entropy_bonus >= 0 when gamma>0.
assert eb1 > 0, f"entropy_bonus should be > 0 when gamma>0, got {eb1}"
print("[PASS] gamma>0 activates entropy bonus regularization")

# -------- 5. ab_trainer 10 steps --------
print("\n[5] ab_trainer 10 steps (metacog, gamma=0.02 delta=0.005)")
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
r = subprocess.run(
    [
        sys.executable,
        os.path.join(repo_root, "training", "ab_trainer.py"),
        "--variant", "metacog",
        "--steps", "10",
        "--d_model", "128",
        "--d_meta", "16",
        "--d_aware", "8",
        "--num_layers", "2",
        "--num_heads", "4",
        "--batch_size", "2",
        "--max_seq_len", "32",
        "--gamma", "0.02",
        "--delta", "0.005",
        "--max_train_samples", "50",
        "--print_every", "5",
    ],
    capture_output=True, text=True, cwd=repo_root,
)
print(r.stdout[-1500:])
if r.returncode != 0:
    print("--- STDERR ---")
    print(r.stderr)
assert r.returncode == 0, f"ab_trainer failed: {r.stderr[-500:]}"
out_tail = r.stdout[-800:]
assert "entropy=" in out_tail or "layer_div=" in out_tail, \
    f"trainer output missing entropy/layer_div tail: {out_tail}"
print("[PASS] ab_trainer 10 steps OK (with entropy + layer_div columns)")

# -------- 6. full_test --------
print("\n[6] full_test")
r2 = subprocess.run(
    [
        sys.executable, "run.py", "--mode", "full_test",
        "--d_model", "128", "--d_meta", "16", "--d_aware", "8",
        "--num_layers", "2", "--num_heads", "4", "--max_seq_len", "32",
    ],
    capture_output=True, text=True, cwd=repo_root,
)
print(r2.stdout[-1500:])
if r2.returncode != 0:
    print("--- STDERR ---")
    print(r2.stderr[-1000:])
assert r2.returncode == 0, f"full_test failed: {r2.stderr[-500:]}"
print("[PASS] full_test OK")

# -------- 7. run_tests --------
print("\n[7] run_tests")
r3 = subprocess.run(
    [sys.executable, os.path.join(repo_root, "tests", "run_tests.py")],
    capture_output=True, text=True, cwd=repo_root,
)
print(r3.stdout[-1500:])
if r3.returncode != 0:
    print("--- STDERR ---")
    print(r3.stderr[-1000:])
assert r3.returncode == 0, f"run_tests failed: {r3.stderr[-500:]}"
print("[PASS] run_tests OK")

print("\n" + "=" * 70)
print("[ALL PASS] Task 1 (3b): Controller anti-collapse + layer diversity OK")
print("=" * 70)
