"""Ablation triple attention — 每种 fusion 子进程独立运行"""
import sys, math, random, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INNER = r'''
import sys, math, random
from pathlib import Path
ROOT_INNER = Path(r"{ROOT}")
sys.path.insert(0, str(ROOT_INNER))
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import MetaCogXConfig
from models import MetaCogXModel, TripleAttention

PAD = 0
SPECIAL = 4
VOCAB = 256 + SPECIAL
FUSION = "{FUSION}"

class ByteTokenizer:
    def __init__(self, max_len=64):
        self.max_len = max_len
        self.vocab_size = VOCAB
        self.pad = PAD
    def encode(self, text):
        return [1] + [b + SPECIAL for b in text.encode('utf-8', 'replace')]

class ByteDataset(Dataset):
    def __init__(self, texts, tok, max_len):
        self.items = []
        for text in texts:
            ids = tok.encode(text)
            if len(ids) < 16:
                ids = (ids * ((16 // max(1, len(ids))) + 1))[:16]
            for start in range(0, max(1, len(ids) - max_len + 1), max_len):
                chunk = ids[start : start + max_len]
                if len(chunk) < 16: continue
                if len(chunk) < max_len: chunk = chunk + [PAD] * (max_len - len(chunk))
                self.items.append(chunk)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ids = torch.tensor(self.items[i], dtype=torch.long)
        return ids, (ids != PAD).long()

def collate(batch):
    return {"input_ids": torch.stack([x[0] for x in batch]), "attention_mask": torch.stack([x[1] for x in batch])}

def build_texts(n=1200, seed=42):
    topics = [
        "Attention Is All You Need. Transformers stack self attention.",
        "Adam optimizer combines momentum with adaptive learning rates.",
        "Residual connections ease optimization of deep stacks.",
        "Dropout prevents co adaptation of hidden units.",
        "Layer normalization stabilizes hidden activations.",
        "Feedforward layers expand then compress.",
        "Token embeddings map discrete indices to dense vectors.",
        "Gradient clipping rescales gradients whose norm exceeds a threshold.",
    ]
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(topics)
        out.append(t + " Sample index {}. More text here.".format(i))
    return out

def build_model(fusion):
    cfg = MetaCogXConfig(
        d_model=128, d_meta=32, d_aware=16, num_layers=4, num_heads=4,
        d_ffn=512, max_seq_len=64, vocab_size=260,
        attn_dropout=0.0, resid_dropout=0.0, ffn_dropout=0.0,
    )
    model = MetaCogXModel(cfg, enable_metacog=True)
    for layer in model.layers:
        d_model = layer.triple_attn.d_model
        d_meta = layer.triple_attn.d_meta
        d_aware = layer.triple_attn.d_aware
        nh = layer.triple_attn.num_heads
        layer.triple_attn = TripleAttention(
            d_model=d_model, d_meta=d_meta, d_aware=d_aware,
            num_heads=nh, dropout=0.0, fusion=fusion,
        )
    return model

def evaluate(model, dl):
    model.eval()
    tl, tc = 0.0, 0.0
    with torch.no_grad():
        for b in dl:
            ids, msk = b["input_ids"], b["attention_mask"]
            o = model(ids, attention_mask=msk, enable_metacog=True)
            lg = o["logits"][:, :-1, :]; tgt = ids[:, 1:]; pm = msk[:, 1:].float()
            ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                 ignore_index=PAD, reduction="none").reshape(ids.size(0), -1)
            tl += float((ce * pm).sum().item())
            tc += float(pm.sum().item())
    avg = tl / max(1e-9, tc)
    return avg, math.exp(min(20, avg))

random.seed(0); torch.manual_seed(0)
tok = ByteTokenizer(64)
all_t = build_texts(1200, 42)
tr_ds = ByteDataset(all_t[:900], tok, 64)
va_ds = ByteDataset(all_t[900:], tok, 64)
tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate)
va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate)

model = build_model(FUSION)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

hist_ctrl, hist_modes = [], []
it = iter(tr_dl)
last_o = None
for s in range(200):
    try: b = next(it)
    except StopIteration: it = iter(tr_dl); b = next(it)
    ids, msk = b["input_ids"], b["attention_mask"]
    model.train(); opt.zero_grad()
    o = model(ids, attention_mask=msk, enable_metacog=True)
    last_o = o
    lg = o["logits"][:, :-1, :]; tgt = ids[:, 1:]; pm = msk[:, 1:].float()
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                         ignore_index=PAD, reduction="none").reshape(ids.size(0), -1)
    loss = (ce * pm).sum() / pm.sum().clamp(min=1.0)
    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    hist_modes.append(o.get("mode", "plain"))
    if "ctrl" in o and o["ctrl"] is not None and hasattr(o["ctrl"], "temp_factor"):
        hist_ctrl.append(float(o["ctrl"].temp_factor.detach().cpu().mean().item()))

vloss, vppl = evaluate(model, va_dl)
ctrl_std = float("nan")
if len(hist_ctrl) > 1:
    c = torch.tensor(hist_ctrl[-100:]); ctrl_std = float(c.std(unbiased=False).item())
ss = last_o.get("switch_stats", {}) if last_o else {}
sw = int(ss.get("switches", 0)) if ss else 0
total = (ss.get("plain_steps", 0) + ss.get("meta_steps", 0)) if ss else 0
plain_pct = 100.0 * ss.get("plain_steps", 0) / max(1, total) if ss else 100.0
mode = last_o.get("mode", "?") if last_o else "?"
print("RESULT fusion={} ppl={:.2f} loss={:.4f} ctrl_std={:.4f} sw={} plain={:.1f}% mode={}".format(
    FUSION, vppl, vloss, ctrl_std, sw, plain_pct, mode))
'''

fusions = ("additive_bias", "multiplicative_gate", "concat_proj")
results = []
for fusion in fusions:
    script = INNER.strip().replace("{ROOT}", str(ROOT)).replace("{FUSION}", fusion)
    out = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT), capture_output=True, text=True)
    print("=== {} ===".format(fusion), flush=True)
    if out.returncode != 0:
        print("ERROR:", out.stderr[-1500:], flush=True)
    else:
        for line in out.stdout.strip().splitlines():
            if "RESULT" in line or "FUSION" in line or " ppl=" in line or " ppl " in line or "loss=" in line or " sw=" in line:
                print(line, flush=True)
                results.append(line)
    print(flush=True)

print("\nSUMMARY TABLE")
print("fusion                    val_ppl  val_loss  ctrl_std  switches  plain%   mode")
print("-" * 72)
for line in results:
    parts = {}
    for key in ("fusion", "ppl", "loss", "ctrl_std", "sw", "plain", "mode"):
        pass
    import re
    m = re.search(r"fusion=(\S+).*?ppl=([\d.]+).*?loss=([\d.]+).*?ctrl_std=([\d.nan]+).*?sw=(\d+).*?plain=([\d.]+)%.*?mode=(\S+)", line)
    if m:
        print("{:<24} {:>8} {:>9} {:>9} {:>8} {:>8}% {:>10}".format(
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)))
print("=" * 72)
