"""runs/run_ab_v2.py

3 组 A/B 比较 v2 (纯 AdamW+CE, 500 steps):
  1. gpt_plain        : enable_metacog=False
  2. alwayson_meta   : enable_metacog=True, 强制 meta head 始终激活 (mode_state='metacog' + 极低 gate 阈值)
  3. conditional_meta : enable_metacog=True, 正常 L1 gate + hysteresis

tiny byte-level 模型 (ByteTokenizer, d_model=128, d_meta=32, d_aware=16,
  num_layers=4, num_heads=4, d_ffn=512, max_seq_len=64, vocab_size=260)

--quick 把 steps 缩到 10, val 缩到 32, 30 秒内跑完
"""

import sys, os, math, random, time, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

try:
    import torch_directml
    DML_AVAIL = True
except Exception:
    DML_AVAIL = False

from config import MetaCogXConfig
from models import MetaCogXModel


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


PAD = 0
SPECIAL = 4
VOCAB = 256 + SPECIAL


class ByteTokenizer:
    def __init__(self, max_len=64):
        self.max_len = max_len
        self.vocab_size = VOCAB
        self.pad = PAD

    def encode(self, text):
        return [1] + [b + SPECIAL for b in text.encode("utf-8", "replace")]


class ByteDataset(Dataset):
    def __init__(self, texts, tok, max_len):
        self.items = []
        for text in texts:
            ids = tok.encode(text)
            if len(ids) < 16:
                ids = (ids * ((16 // max(1, len(ids))) + 1))[:16]
            for start in range(0, max(1, len(ids) - max_len + 1), max_len):
                chunk = ids[start : start + max_len]
                if len(chunk) < 16:
                    continue
                if len(chunk) < max_len:
                    chunk = chunk + [PAD] * (max_len - len(chunk))
                self.items.append(chunk)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids = torch.tensor(self.items[i], dtype=torch.long)
        return ids, (ids != PAD).long()


def collate(batch):
    return {
        "input_ids": torch.stack([x[0] for x in batch]),
        "attention_mask": torch.stack([x[1] for x in batch]),
    }


def build_texts(n=1200, seed=42):
    topics = [
        "Attention Is All You Need. Transformers stack self attention in blocks of residual layers. Scaled dot product attention relates tokens via query key value triplets.",
        "Adam optimizer combines momentum with adaptive learning rates. It estimates first and second moments of gradients per parameter.",
        "Residual connections ease optimization of deep stacks. Each layer learns delta on top of identity mapping so deeper nets train reliably.",
        "Dropout prevents co adaptation of hidden units. Each forward pass drops random units forcing robustness across pathways.",
        "Layer normalization stabilizes hidden activations. It computes mean and variance across feature dimensions and applies affine transform.",
        "Feedforward layers expand then compress. They apply pointwise nonlinearity in four times the model dimension then project back.",
        "Token embeddings map discrete indices to dense vectors. Position embeddings add sequence ordering information to each token.",
        "Gradient clipping rescales gradients whose norm exceeds a threshold. This avoids exploding gradients in long backprop chains.",
    ]
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(topics)
        out.append(
            t + " Sample index {}. More text here for diversity in byte patterns.".format(i)
        )
    return out


def validate_ppl(model, dl, device, enable_metacog, mode="normal", collect_stats=False):
    """返回 (avg_ce_loss, ppl, switches, plain_pct, score_mean, ctrl_std)"""
    model.eval()
    model._switch_stats = {k: 0 for k in model._switch_stats} if hasattr(model, "_switch_stats") else {
        "switches": 0, "total_forward": 0, "plain_steps": 0, "meta_steps": 0,
    }
    total_loss = 0.0
    total_count = 0
    switches = 0
    plain_steps = 0
    meta_steps = 0
    dilemmas = []
    abs_ctrls = []
    n_batches = 0
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(device)
            msk = b["attention_mask"].to(device)
            if mode == "alwayson" and enable_metacog and hasattr(model, "l1_gate"):
                model.l1_gate.enter_thresh = -1.0
                model.l1_gate.exit_thresh = -1.0
                model._mode_state = "metacog"
                model._plain_countdown = 0
                model._meta_countdown = 0
                if hasattr(model.l1_gate, "net") and hasattr(model.l1_gate.net[3], "bias"):
                    nn.init.constant_(model.l1_gate.net[3].bias, 5.0)
            elif mode == "conditional" and enable_metacog and hasattr(model, "l1_gate"):
                pass
            out = model(ids, attention_mask=msk, enable_metacog=enable_metacog)
            lg = out["logits"][:, :-1, :]
            tgt = ids[:, 1:]
            pm = msk[:, 1:].float()
            ce = F.cross_entropy(
                lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                ignore_index=PAD, reduction="none",
            ).reshape(ids.size(0), -1)
            loss = (ce * pm).sum()
            cnt = pm.sum()
            total_loss += float(loss.item())
            total_count += float(cnt.item())
            ss = out.get("switch_stats", {})
            switches += int(ss.get("switches", 0))
            plain_steps += int(ss.get("plain_steps", 0))
            meta_steps += int(ss.get("meta_steps", 0))
            ds = out.get("last_dilemma_score", None)
            if ds is not None:
                dilemmas.append(float(ds))
            tf_raw = out.get("ctrl_tf_raw_logit", None)
            if tf_raw is not None and isinstance(tf_raw, torch.Tensor):
                abs_ctrls.extend(tf_raw.detach().cpu().abs().reshape(-1).tolist())
            n_batches += 1
    avg_loss = total_loss / max(1e-9, total_count)
    ppl = math.exp(min(20, avg_loss))
    total_forward = plain_steps + meta_steps
    plain_pct = (plain_steps / total_forward) if total_forward > 0 else 1.0
    score_mean = float("nan")
    if dilemmas:
        score_mean = float(sum(dilemmas) / len(dilemmas))
    ctrl_std = float("nan")
    if abs_ctrls:
        # "mean absolute deviation" of controller logits  — 用平均绝对偏差
        if len(abs_ctrls) >= 2:
            mean_a = sum(abs_ctrls) / len(abs_ctrls)
            ctrl_std = float(sum(abs(v - mean_a) for v in abs_ctrls) / len(abs_ctrls))
        else:
            ctrl_std = float(abs_ctrls[0])
    return avg_loss, ppl, switches, plain_pct, score_mean, ctrl_std


def run_variant(name, variant_mode, cfg_kwargs, tr_dl, va_dl, device, steps=500, lr=2e-3):
    """variant_mode in {'plain', 'alwayson', 'conditional'}"""
    random.seed(0); torch.manual_seed(0)

    use_metacog = variant_mode != "plain"

    cfg = MetaCogXConfig(
        vocab_size=260,
        d_model=cfg_kwargs["d_model"],
        d_meta=cfg_kwargs["d_meta"],
        d_aware=cfg_kwargs["d_aware"],
        num_layers=cfg_kwargs["layers"],
        num_heads=cfg_kwargs["heads"],
        d_ffn=cfg_kwargs["d_ffn"],
        max_seq_len=cfg_kwargs["seq"],
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
    )

    # disable_tri_attn / use_dmn — 兼容未来 ablation
    model = MetaCogXModel(cfg, enable_metacog=use_metacog)

    # alwayson: 强制 meta head 永远激活 (极低 threshold + 初始 bias 大)
    if variant_mode == "alwayson":
        model._mode_state = "metacog"
        model._plain_countdown = 0
        model._meta_countdown = 0
        if hasattr(model, "l1_gate") and hasattr(model.l1_gate, "enter_thresh"):
            model.l1_gate.enter_thresh = -1.0
            model.l1_gate.exit_thresh = -1.0
            last = model.l1_gate.net[3]
            if hasattr(last, "bias") and last.bias is not None:
                nn.init.constant_(last.bias, 5.0)
        for sdict_k in getattr(model, "_switch_stats", {}):
            model._switch_stats[sdict_k] = 0
    
    # conditional: 设置合适的 gate 阈值，让它能在 plain 和 metacog 之间切换
    elif variant_mode == "conditional":
        if hasattr(model, "enter_thresh"):
            model.enter_thresh = 0.45  # 设置在 score=0.5 以下，确保能触发切换
            model.exit_thresh = 0.35   # 退出阈值，保持 hysteresis
            model.enter_patience = 1   # 降低耐心要求
            model.exit_patience = 1

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    print("\n" + "=" * 92, flush=True)
    print(" variant : {}  enable_metacog={}  mode={}  params={:,}".format(
        name, use_metacog, variant_mode, total_params
    ), flush=True)
    print("=" * 92, flush=True)
    print(" {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "step", "loss", "val_ppl", "val_loss", "mode", "switches", "plain%", "score", "ctrlStd"
    ), flush=True)
    print("-" * 92, flush=True)

    model.train()
    it = iter(tr_dl)
    for s in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(tr_dl)
            b = next(it)
        ids = b["input_ids"].to(device)
        msk = b["attention_mask"].to(device)

        if variant_mode == "alwayson" and hasattr(model, "l1_gate") and hasattr(model.l1_gate, "enter_thresh"):
            model.l1_gate.enter_thresh = -1.0
            model.l1_gate.exit_thresh = -1.0
            model._mode_state = "metacog"
            model._plain_countdown = 0
            model._meta_countdown = 0

        opt.zero_grad()
        out = model(ids, attention_mask=msk, enable_metacog=use_metacog)
        lg = out["logits"][:, :-1, :]
        tgt = ids[:, 1:]
        pm = msk[:, 1:].float()
        ce = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
            ignore_index=PAD, reduction="none",
        ).reshape(ids.size(0), -1)
        loss = (ce * pm).sum() / pm.sum().clamp(min=1.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if steps > 1:
            sched.step()

        if (s + 1) % max(1, (1 if steps <= 10 else (20 if steps <= 50 else 100))) == 0 or (s + 1) == steps:
            vloss, vppl, sw, pp, score_m, ctrl_std = validate_ppl(
                model, va_dl, device, use_metacog, mode=variant_mode, collect_stats=True
            )
            cur_mode = out.get("mode", "plain")
            print(" {:6d} {:10.4f} {:10.2f} {:10.4f} {:>10} {:10d} {:9.1f}% {:10.4f} {:10.4f}".format(
                s + 1, float(loss.item()), vppl, vloss, cur_mode, sw, pp * 100.0, score_m, ctrl_std
            ), flush=True)

    final_vloss, final_vppl, final_sw, final_pp, final_score_mean, final_ctrl_std = validate_ppl(
        model, va_dl, device, use_metacog, mode=variant_mode, collect_stats=True
    )

    print("-" * 92, flush=True)
    print(" FINAL ppl={:.4f}  loss={:.4f}  switches={}  plain={:.1f}%  score={:.3f}  ctrl_std={:.4f}".format(
        final_vppl, final_vloss, final_sw, final_pp * 100.0, final_score_mean, final_ctrl_std
    ), flush=True)

    return {
        "name": name,
        "enable_metacog": use_metacog,
        "mode": variant_mode,
        "final_ppl": float(final_vppl),
        "final_loss": float(final_vloss),
        "switches": int(final_sw),
        "plain_pct": float(final_pp),
        "score_mean": float(final_score_mean if not math.isnan(final_score_mean) else 0.0),
        "ctrl_std": float(final_ctrl_std if not math.isnan(final_ctrl_std) else 0.0),
    }


def parse_args():
    p = argparse.ArgumentParser("run_ab_v2.py  A/B evaluation")
    p.add_argument("--quick", action="store_true", help="steps=10 val=32 (~30s)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--val", type=int, default=300)
    p.add_argument("--out", type=str, default=str(ROOT / "runs" / "ab_results_v3.json"))
    p.add_argument("--dmodel", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    steps = 10 if args.quick else args.steps
    val_n = 32 if args.quick else args.val
    seq = args.seq
    d_model = args.dmodel
    layers = args.layers
    heads = args.heads
    d_ffn = 512
    d_meta = 32
    d_aware = 16
    batch = 32

    device = pick_device()
    dev_repr = str(device)

    tok = ByteTokenizer(max_len=seq)

    pin = False
    num_work = 0

    all_texts = build_texts(n=1200, seed=42)
    tr_ds = ByteDataset(all_texts[:900], tok, seq)
    va_ds = ByteDataset(all_texts[900 : 900 + val_n], tok, seq)
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, collate_fn=collate,
                       num_workers=num_work, pin_memory=pin, persistent_workers=False)
    va_dl = DataLoader(va_ds, batch_size=batch, shuffle=False, collate_fn=collate,
                       num_workers=num_work, pin_memory=pin, persistent_workers=False)

    print("=" * 92, flush=True)
    print(" A/B EVALUATION v3  (3 variants)".center(92), flush=True)
    print(" device={}  d_model={}  layers={}  heads={}  seq={}  d_ffn={}".format(
        dev_repr, d_model, layers, heads, seq, d_ffn
    ), flush=True)
    print(" d_meta={}  d_aware={}  batch={}  steps={}  val_batches={}".format(
        d_meta, d_aware, batch, steps, len(va_ds)
    ), flush=True)
    print(" OMP={}  MKL={}  DML_AVAIL={}  QUICK={}".format(
        os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"), DML_AVAIL, args.quick
    ), flush=True)
    print("=" * 92, flush=True)

    cfg_kwargs = dict(
        d_model=d_model, d_meta=d_meta, d_aware=d_aware,
        layers=layers, heads=heads, d_ffn=d_ffn, seq=seq,
    )

    t0 = time.time()
    variants = [
        ("gpt_plain",        "plain"),
        ("alwayson_meta",     "alwayson"),
        ("conditional_meta", "conditional"),
    ]
    results = []
    for name, mode in variants:
        r = run_variant(name, mode, cfg_kwargs, tr_dl, va_dl, device, steps=steps)
        results.append(r)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    wall = time.time() - t0

    print("\n" + "=" * 92, flush=True)
    print(" FINAL SUMMARY".center(92), flush=True)
    print("=" * 92, flush=True)
    print(" {:<22} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "variant", "mode", "val_ppl", "val_loss", "switches", "plain%", "score", "ctrlStd"
    ), flush=True)
    print("-" * 92, flush=True)
    for r in results:
        print(" {:<22} {:>8} {:10.2f} {:10.4f} {:10d} {:9.1f}% {:10.4f} {:10.4f}".format(
            r["name"], r["mode"], r["final_ppl"], r["final_loss"], r["switches"],
            r["plain_pct"] * 100.0, r["score_mean"], r["ctrl_std"]
        ), flush=True)
    print("=" * 92, flush=True)
    print(" wall_seconds={:.1f}".format(wall), flush=True)

    out_obj = {
        "wall_seconds": float(wall),
        "device_picked": dev_repr,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "dml_available": DML_AVAIL,
        "d_model": d_model, "layers": layers, "heads": heads, "seq": seq,
        "steps": steps, "val_samples": len(va_ds),
        "batch": batch, "lr": 2e-3,
        "variants": results,
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(" RESULTS JSON -> {}".format(out_path), flush=True)


if __name__ == "__main__":
    main()
