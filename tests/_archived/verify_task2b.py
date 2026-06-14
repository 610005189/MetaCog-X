import sys, os, subprocess, traceback, time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def run(args):
    r = subprocess.run(
        [sys.executable, os.path.join(_PROJECT_ROOT, "training", "ab_trainer.py")] + args,
        cwd=_PROJECT_ROOT,
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": _PROJECT_ROOT + ";" + os.environ.get("PYTHONPATH", "")},
    )
    return r

sys.path.insert(0, _PROJECT_ROOT)
start = time.time()

print("[1/5] variant=gpt: enable_metacog=False, CE-only")
try:
    from config import MetaCogXConfig
    from models import MetaCogXModel
    from training.losses import TotalLoss
    import torch
    cfg = MetaCogXConfig(d_model=128, d_meta=16, d_aware=8, num_layers=2, num_heads=4, max_seq_len=32, d_ffn=256, vocab_size=100)
    m_gpt = MetaCogXModel(cfg, enable_metacog=False)
    ids = torch.randint(0, 100, (2, 16))
    msk = torch.ones(2, 16)
    out = m_gpt(ids, attention_mask=msk, labels=None, return_meta=False, enable_metacog=False)
    assert out["logits"].shape == (2, 16, 100)
    print("[PASS] variant=gpt forward OK")
except Exception as e:
    print(f"[FAIL] 1: {e}"); traceback.print_exc(); sys.exit(1)

print("[2/5] variant=metacog: enable_metacog=True, 三分支 + TotalLoss")
try:
    m_meta = MetaCogXModel(cfg, enable_metacog=True)
    out2 = m_meta(ids, attention_mask=msk, labels=None, return_meta=True, enable_metacog=True)
    assert out2["meta"] is not None and out2["awareness"] is not None
    loss, comp = TotalLoss()(out2["logits"], ids, out2["meta"], out2["awareness"], None)
    assert comp["loss_meta"].item() > 0 and comp["loss_aware"].item() > 0
    print(f"[PASS] variant=metacog forward + TotalLoss OK: meta={comp['loss_meta'].item():.4f} aware={comp['loss_aware'].item():.4f}")
except Exception as e:
    print(f"[FAIL] 2: {e}"); traceback.print_exc(); sys.exit(1)

print("[3/5] ab_trainer 跑 10 steps (gpt):")
try:
    r = run([
        "--variant", "gpt", "--steps", "10", "--d_model", "128", "--num_layers", "2",
        "--num_heads", "4", "--batch_size", "2", "--max_seq_len", "32",
        "--max_train_samples", "50",
    ])
    assert r.returncode == 0, f"rc={r.returncode} stderr={r.stderr}"
    print(f"[PASS] ab_trainer gpt 10 steps OK\n{r.stdout[-600:]}")
except Exception as e:
    print(f"[FAIL] 3: {e}"); traceback.print_exc(); sys.exit(1)

print("[4/5] ab_trainer 跑 10 steps (metacog):")
try:
    r2 = run([
        "--variant", "metacog", "--steps", "10", "--d_model", "128", "--d_meta", "16", "--d_aware", "8",
        "--num_layers", "2", "--num_heads", "4", "--batch_size", "2", "--max_seq_len", "32",
        "--max_train_samples", "50",
    ])
    assert r2.returncode == 0, f"rc={r2.returncode} stderr={r2.stderr}"
    print(f"[PASS] ab_trainer metacog 10 steps OK\n{r2.stdout[-600:]}")
except Exception as e:
    print(f"[FAIL] 4: {e}"); traceback.print_exc(); sys.exit(1)

print("[5/5] 参数量比较")
try:
    p_gpt = sum(p.numel() for p in m_gpt.parameters())
    p_meta = sum(p.numel() for p in m_meta.parameters())
    ratio = p_meta / p_gpt
    print(f"[INFO] params gpt={p_gpt:,} metacog={p_meta:,} ratio={ratio:.3f} (extra {(ratio-1)*100:.1f}%)")
    assert ratio < 1.10, "metacog extra params should be < 10%"
    print("[PASS] params ratio OK (< 10% extra)")
except Exception as e:
    print(f"[FAIL] 5: {e}"); traceback.print_exc(); sys.exit(1)

print(f"\n[PASS] Task2 A/B trainer verified OK in {time.time()-start:.1f}s")
