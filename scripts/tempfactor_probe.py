"""Train tiny byte-level model (~256 vocab) for 800 CPU steps, then probe temp_factor.

Byte-level tokenizer is trivial (bytes 0..255 + 3 specials).
Model: d_model=64, d_meta=16, d_aware=8, 2 layers, 2 heads, d_ffn=256, seq=64.
~600K params. 800 CPU steps → ppl should drop to <20.
"""
import os, sys, math, time, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MetaCogXConfig
from models import MetaCogXModel
from training.losses import TotalLoss

PAD = 0; SOS = 1; EOS = 2; UNK = 3; SPECIAL = 4
VOCAB = 256 + SPECIAL

class ByteTokenizer:
    def __init__(self, max_len=64):
        self.max_len = max_len
        self.vocab_size = VOCAB
        self.pad_id = PAD
    def encode(self, text):
        ids = [SOS] + [b + SPECIAL for b in text.encode("utf-8", errors="replace")]
        if len(ids) > self.max_len: ids = ids[:self.max_len]
        return ids
    def __call__(self, text, max_length=None, padding='max_length', truncation=True, return_tensors='pt'):
        ml = max_length or self.max_len
        ids = self.encode(text) if truncation else self.encode(text)[:ml]
        if len(ids) < ml: ids = ids + [PAD] * (ml - len(ids))
        import torch as _t
        return {"input_ids": _t.tensor(ids, dtype=_t.long).unsqueeze(0),
                "attention_mask": _t.tensor([1]*sum(1 for x in ids if x != PAD) + [0]*(ml - sum(1 for x in ids if x != PAD)), dtype=_t.long).unsqueeze(0)}
    def __len__(self): return self.vocab_size

class ByteTextSet(Dataset):
    def __init__(self, texts, tok, max_len):
        self.tok = tok; self.max_len = max_len
        self.items = []
        for t in texts:
            ids = tok.encode(t)
            for i in range(0, max(1, len(ids) - max_len + 1), max_len // 2):
                chunk = ids[i:i+max_len]
                if len(chunk) < 8: continue
                if len(chunk) < max_len: chunk = chunk + [PAD] * (max_len - len(chunk))
                self.items.append(chunk)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ids = torch.tensor(self.items[idx], dtype=torch.long)
        msk = (ids != PAD).long()
        return ids, msk

def mk_texts(n=1000):
    topics = [
        "Attention Is All You Need. Transformers stack self-attention over token embeddings. Positional encodings break permutation invariance.",
        "Adam optimizer combines momentum with adaptive learning rates per parameter. Cross-entropy loss is the standard next-token objective.",
        "Residual connections let gradients flow through deep stacks. Layer norm stabilizes per-token magnitudes before each sublayer.",
        "Dropout randomly zeroes hidden units to prevent co-adaptation. BERT applies bidirectional attention; GPT applies causal masks.",
    ]
    out = []
    for i in range(n):
        out.append(topics[i % len(topics)] + f" The training sample id is {i}. This sentence adds diversity.")
    return out

def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    msk = torch.stack([b[1] for b in batch])
    return {"input_ids": ids, "attention_mask": msk}

def train(steps=800, bs=16, seq=64, lr=2e-3):
    device = torch.device("cpu")
    tok = ByteTokenizer(max_len=seq)
    all_texts = mk_texts(1000)
    train_ds = ByteTextSet(all_texts[:800], tok, seq)
    valid_ds = ByteTextSet(all_texts[800:], tok, seq)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=collate, drop_last=False)

    cfg = MetaCogXConfig(d_model=64, d_meta=16, d_aware=8, num_layers=2, num_heads=2,
                         max_seq_len=seq, d_ffn=256, vocab_size=tok.vocab_size)
    model = MetaCogXModel(cfg, enable_metacog=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[BYTE-TRAIN] params={n_params:,} vocab={tok.vocab_size} train_samples={len(train_ds)} valid_samples={len(valid_ds)}")

    loss_fn = TotalLoss(alpha=0.01, beta=0.005, gamma=0.0, delta=0.0, ignore_index=PAD)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    it = iter(train_dl)
    t0 = time.time()
    best_ppl = float("inf")
    model.train()
    for s in range(steps):
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        ids = b["input_ids"]; msk = b["attention_mask"]
        opt.zero_grad()
        out = model(ids, attention_mask=msk, return_meta=True, enable_metacog=True)
        L, _ = loss_fn(out["logits"], ids, out.get("meta"), out.get("awareness"))
        L.backward(); opt.step()
        if s % 200 == 0 or s == steps - 1:
            elapsed = time.time() - t0
            valid_ppl = evaluate(model, valid_dl, PAD)
            train_ppl = math.exp(max(float(L), 0))
            flag = " <<< BEST" if valid_ppl < best_ppl else ""
            best_ppl = min(best_ppl, valid_ppl)
            print(f"  step {s:4d} train_ppl={train_ppl:6.1f}  valid_ppl={valid_ppl:6.1f}  ({elapsed:.0f}s){flag}")

    print(f"[BYTE-TRAIN] final valid_ppl={valid_ppl:.2f} elapsed={time.time()-t0:.0f}s")
    return model, tok, valid_dl

def evaluate(model, dl, pad_id):
    model.eval()
    nll = 0.0; tok_count = 0
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"]; msk = b["attention_mask"]
            B, L = ids.shape
            out = model(ids, attention_mask=msk, return_meta=True, enable_metacog=True)
            lg = out["logits"][:, :-1, :]; tgt = ids[:, 1:]; pm = msk[:, 1:].float()
            ce = nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                ignore_index=pad_id, reduction="none").reshape(B, L-1)
            nll += float((ce * pm).sum().item())
            tok_count += int(pm.sum().item())
    avg = nll / max(tok_count, 1)
    return math.exp(max(avg, 0))

def patch_attn(model, val):
    from models.triple_attention import TripleAttention
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "TripleAttention":
            if not hasattr(mod, "_tf_orig"):
                mod._tf_orig = mod.forward
            orig = mod._tf_orig; tfv = float(val)
            def mk_patch(o, v):
                def patched(content, meta, awareness, mask=None, temp_factor=None):
                    tf = torch.full((content.size(0), 1), v, device=content.device)
                    return o(content, meta, awareness, mask=mask, temp_factor=tf)
                return patched
            mod.forward = mk_patch(orig, tfv); n += 1
    return n

def probe(model, tok, valid_dl, tf_steps=11, tf_min=0.3, tf_max=3.0):
    temps = [tf_min + (tf_max - tf_min) * i / (tf_steps - 1) for i in range(tf_steps)]
    temps[0] = tf_min; temps[-1] = tf_max
    results = []
    for tf in temps:
        patch_attn(model, tf)
        ppl = evaluate(model, valid_dl, tok.pad_id)
        results.append((tf, ppl))
        print(f"tf={tf:5.2f}  ppl={ppl:7.2f}")
    base = [p for tf,p in results if abs(tf-1.0)<0.06][0]
    mn = min(r[1] for r in results); mx = max(r[1] for r in results)
    span = (mx - mn) / base * 100 if base > 0 else 0
    best = min(results, key=lambda r: r[1])
    print(f"\n=== TEMP_FACTOR PROBE (byte-level, vocab={tok.vocab_size}) ===")
    print(f"BASELINE (tf≈1.0) ppl = {base:.2f}")
    print(f"BEST tf={best[0]:.2f} ppl={best[1]:.2f}  delta={(best[1]-base)/base*100:+.2f}%")
    print(f"PPL_SPAN across tf ∈ [{tf_min}, {tf_max}] = {span:.1f}%")
    verdict = "ACTION_SPACE_WEAK" if span < 3 else "ACTION_SPACE_MODERATE" if span < 10 else "ACTION_SPACE_STRONG"
    print(f"VERDICT: {verdict}")
    print("\nINTERPRETATION:")
    if span < 3:
        print("  Attention-temperature scaling barely affects ppl on a trained model.")
        print("  The controller's proposed action space (temp_factor ∈ [0.8, 1.2]) is WEAK.")
        print("  Recommendation: redesign controller — e.g. FFN residual gate, skip-layer decision.")
    elif span < 10:
        print("  Action space has mild leverage. PPO may help marginally.")
    else:
        print("  Action space is strong. PPO will have a useful gradient signal.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--tf_steps", type=int, default=11)
    ap.add_argument("--tf_min", type=float, default=0.3)
    ap.add_argument("--tf_max", type=float, default=3.0)
    args = ap.parse_args()
    model, tok, valid_dl = train(steps=args.steps)
    probe(model, tok, valid_dl, tf_steps=args.tf_steps, tf_min=args.tf_min, tf_max=args.tf_max)

if __name__ == "__main__":
    main()
