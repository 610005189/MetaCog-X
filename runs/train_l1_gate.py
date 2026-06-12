"""runs/train_l1_gate.py

自动标注"困境步骤"并训练 L1 DilemmaGate 做二分类。
数据: byte-level (BT) 8 主题 × 1200 句子 -> train/val DataLoader
模型: tiny MetaCogXModel (d_model=128, d_meta=32, d_aware=16,
       num_layers=4, num_heads=4, d_ffn=512, vocab_size=260)
特征: 每层 content attention 的 entropy (mean + std) × 4 层
       + logits max_prob, logits entropy, token_rep
       => input_dim = 2*4 + 3 = 11
标注: 先跑若干 batch 统计 entropy_mean 分布 -> 70 百分位阈值
      dilemma = (entropy_mean > th_entropy) OR (logits_maxprob < th_lmp)
              OR (token_rep >= 2)
训练: BCEWithLogitsLoss, lr=1e-3, 5 epochs, batch_size=32
输出: train acc / val acc / val F1 / pos_score_mean / neg_score_mean
"""

import sys, os, math, random, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from config import MetaCogXConfig
from models import MetaCogXModel
from models.dilemma_gate import (
    attention_entropy,
    logits_stats,
    token_repetition,
)

# =========================================================
# Byte-level 数据集 (内联)
# =========================================================
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


# =========================================================
# 特征采集
# =========================================================
def monkey_patch_model_for_zero_meta(model):
    """让 CognitiveParticle 输出零 meta 和零 awareness。
    TripleAttention 照常跑 content attention（最标准的 self-attention），
    meta/awareness 分支是零向量的注意力（不影响我们提取 content 分支的 attn）。"""
    from models import cognitive_particle

    orig = cognitive_particle.CognitiveParticle.forward

    def patched(self, x):
        content, _meta, _aware = orig(self, x)
        zero_meta = torch.zeros(
            content.size(0), content.size(1), self.d_meta, device=content.device
        )
        zero_aware = torch.zeros(
            content.size(0), content.size(1), self.d_aware, device=content.device
        )
        return content, zero_meta, zero_aware

    cognitive_particle.CognitiveParticle.forward = patched
    model._patched_cp = True
    return model


def collect_layer_attn(model):
    """从每个 layer 的 triple_attn 取出 content 分支的 attn_weights。
    返回 list[Tensor[B, H, L, L]]"""
    outs = []
    for layer in model.layers:
        w = getattr(layer.triple_attn, "_last_attn_c", None)
        if w is None:
            w = getattr(layer.triple_attn, "_last_attn_m", None)
        if w is None:
            w = getattr(layer.triple_attn, "_last_attn_a", None)
        outs.append(w.detach() if w is not None else None)
    return outs


def extract_gate_features(attn_list, logits, token_ids):
    """attn_list: list[Tensor[B, H, L, L]]
    返回: [B, F] F = 2*num_layers + 3
    """
    num_layers = len(attn_list)
    feats = []
    for w in attn_list:
        if w is None:
            raise RuntimeError("one layer returned None attention weights")
        ent = attention_entropy(w)  # [B, H, L]
        feats.append(ent.mean(dim=(1, 2)))
        feats.append(ent.std(dim=(1, 2), unbiased=False))

    st = logits_stats(logits[:, -1, :] if logits.dim() == 3 else logits)
    feats.append(st["max_prob"])
    feats.append(st["entropy"])

    rep = token_repetition(token_ids)
    feats.append(rep.mean(dim=-1))

    return torch.stack(feats, dim=-1)


# =========================================================
# 自动标注器
# =========================================================
def label_dilemma(feats, th_entropy, th_maxprob, rep_th=2):
    """feats: [B, F]
    F 布局: [ent_mean_l0, ent_std_l0, ent_mean_l1, ent_std_l1, ...,
             logits_maxprob, logits_entropy, token_rep_mean]
    """
    num_layers = (feats.size(1) - 3) // 2
    entropy_means = []
    for i in range(num_layers):
        entropy_means.append(feats[:, 2 * i])
    mean_entropy_across_layers = torch.stack(entropy_means, dim=0).mean(dim=0)  # [B]

    logits_maxprob = feats[:, -3]
    token_rep_mean = feats[:, -1]

    dilemma = (
        (mean_entropy_across_layers > th_entropy)
        | (logits_maxprob < th_maxprob)
        | (token_rep_mean >= rep_th)
    ).float()
    return dilemma, mean_entropy_across_layers


# =========================================================
# 主训练
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = ByteTokenizer(max_len=64)
    all_texts = build_texts(n=1200, seed=42)

    tr_ds = ByteDataset(all_texts[:900], tok, 64)
    va_ds = ByteDataset(all_texts[900:], tok, 64)
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)

    cfg = MetaCogXConfig(
        d_model=128,
        d_meta=32,
        d_aware=16,
        num_layers=4,
        num_heads=4,
        d_ffn=512,
        vocab_size=260,
        max_seq_len=64,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
    )
    model = MetaCogXModel(cfg, enable_metacog=False).to(args.device)

    num_layers = cfg.num_layers
    input_dim = 2 * num_layers + 3  # 11

    PRETRAIN_STEPS = 400
    print("=" * 70)
    print("L1 DilemmaGate Training")
    print(f"  device        : {args.device}")
    print(f"  backbone      : tiny MetaCogXModel {cfg.d_model}/{num_layers}/{cfg.num_heads}")
    print(f"  input_dim     : {input_dim}")
    print(f"  pretrain      : {PRETRAIN_STEPS} steps (byte-level CE, AdamW)")
    print(f"  train samples : {len(tr_ds)}")
    print(f"  val samples   : {len(va_ds)}")
    print("=" * 70)

    # ---------- Step 0: 预训练 backbone 让 attention/logits 真有分布差异 ----------
    print("\n[Step 0] Pretraining backbone ({} steps) ...".format(PRETRAIN_STEPS))
    model.train()
    opt_model = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    it = iter(tr_dl)
    for s in range(PRETRAIN_STEPS):
        try:
            b = next(it)
        except StopIteration:
            it = iter(tr_dl)
            b = next(it)
        ids = b["input_ids"].to(args.device)
        msk = b["attention_mask"].to(args.device)
        opt_model.zero_grad()
        o = model(ids, attention_mask=msk, enable_metacog=False)
        lg = o["logits"][:, :-1, :]
        tgt = ids[:, 1:]
        pm = msk[:, 1:].float()
        ce = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
            ignore_index=PAD, reduction="none",
        ).reshape(ids.size(0), -1)
        loss = (ce * pm).sum() / pm.sum().clamp(min=1.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_model.step()
        if (s + 1) % 200 == 0:
            ppl = math.exp(float(loss.item()))
            print("  step {:4d}  loss={:.4f}  ppl={:.2f}".format(s + 1, float(loss.item()), ppl))
    model.eval()

    # 预训练完成后再 monkey-patch（让 meta/awareness 置零）—— 我们其实不想让 meta/awareness 跑，
    # 预训练时 meta/awareness 由 CognitiveParticle 生成并随 model 更新，但这无关紧要。
    # 为了 TripleAttention 的 attention 纯粹基于 content，这里 patch CP 让 meta/awareness 置零。
    model = monkey_patch_model_for_zero_meta(model)

    # ---------- Step 1: 跑一遍 train 集 -> 统计特征分布 + 阈值 ----------
    print("\n[Step 1] Sampling train features ...")
    all_feats_tr = []
    all_entropy_agg = []
    all_logits_maxprob = []
    all_token_rep = []
    n_batches = 0
    with torch.no_grad():
        for b in tr_dl:
            ids = b["input_ids"].to(args.device)
            msk = b["attention_mask"].to(args.device)
            out = model(ids, attention_mask=msk, enable_metacog=False)
            attns = collect_layer_attn(model)
            feats = extract_gate_features(attns, out["logits"], ids)
            all_feats_tr.append(feats.cpu())

            num_layers_local = (feats.size(1) - 3) // 2
            ent_means = torch.stack(
                [feats[:, 2 * i] for i in range(num_layers_local)], dim=0
            ).mean(dim=0)
            all_entropy_agg.append(ent_means.cpu())
            all_logits_maxprob.append(feats[:, -3].cpu())
            all_token_rep.append(feats[:, -1].cpu())
            n_batches += 1

    all_feats_tr = torch.cat(all_feats_tr, dim=0)
    ent_all = torch.cat(all_entropy_agg, dim=0)
    lmp_all = torch.cat(all_logits_maxprob, dim=0)
    trp_all = torch.cat(all_token_rep, dim=0)

    # 70 百分位: entropy 越"高"越困境 -> 70% 分位当 th_entropy
    # maxprob 越"低"越困境 -> 下 30% 分位 (即 0.3 quantile) 当 th
    def q(vals, p):
        if vals.numel() == 0:
            return torch.tensor(0.0)
        return torch.quantile(vals.float(), p, interpolation="midpoint")

    q70_ent = q(ent_all, 0.70).item()
    q80_ent = q(ent_all, 0.80).item()
    q30_lmp = q(lmp_all, 0.30).item()
    q20_lmp = q(lmp_all, 0.20).item()

    # 先用相对宽松阈值，后面可微调
    th_entropy = q70_ent  # 约 top 30%
    th_maxprob = q30_lmp  # 约 bottom 30% max_prob
    rep_th = 2

    # 若正样本率 < 15% 太稀 -> 放宽；若 > 60% 太松 -> 收紧
    temp_labels, _ = label_dilemma(
        all_feats_tr, th_entropy, th_maxprob, rep_th
    )
    pos_rate = float(temp_labels.mean().item())
    print(
        "  entropy  q70={:.3f} q80={:.3f}  maxprob  q30={:.3f} q20={:.3f}".format(
            q70_ent, q80_ent, q30_lmp, q20_lmp
        )
    )
    print(
        "  initial pos_rate={:.3f}  (th_ent={:.3f}, th_lmp={:.3f}, rep>={})".format(
            pos_rate, th_entropy, th_maxprob, rep_th
        )
    )

    # 自适应阈值
    if pos_rate < 0.15:
        th_entropy = q(ent_all, 0.60).item()
        th_maxprob = q(lmp_all, 0.40).item()
    elif pos_rate > 0.60:
        th_entropy = q(ent_all, 0.85).item()
        th_maxprob = q(lmp_all, 0.15).item()

    temp_labels2, _ = label_dilemma(
        all_feats_tr, th_entropy, th_maxprob, rep_th
    )
    pos_rate2 = float(temp_labels2.mean().item())
    print(
        "  final   pos_rate={:.3f}  (th_ent={:.3f}, th_lmp={:.3f}, rep>={})".format(
            pos_rate2, th_entropy, th_maxprob, rep_th
        )
    )

    # ---------- Step 2: 收集完整 train / val 特征+标签 ----------
    def collect_all(dl):
        feats_list = []
        labels_list = []
        with torch.no_grad():
            for b in dl:
                ids = b["input_ids"].to(args.device)
                msk = b["attention_mask"].to(args.device)
                out = model(ids, attention_mask=msk, enable_metacog=False)
                attns = collect_layer_attn(model)
                feats = extract_gate_features(attns, out["logits"], ids)
                labs, _ = label_dilemma(feats, th_entropy, th_maxprob, rep_th)
                feats_list.append(feats.cpu())
                labels_list.append(labs.cpu())
        return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)

    print("\n[Step 2] Collecting train + val labeled features ...")
    feats_tr, labs_tr = collect_all(tr_dl)
    feats_va, labs_va = collect_all(va_dl)

    pos_tr = int(labs_tr.sum().item())
    pos_va = int(labs_va.sum().item())
    print(
        "  train pos={} neg={}  ({:.1f}%)".format(
            pos_tr, len(labs_tr) - pos_tr, 100 * pos_tr / max(1, len(labs_tr))
        )
    )
    print(
        "  val   pos={} neg={}  ({:.1f}%)".format(
            pos_va, len(labs_va) - pos_va, 100 * pos_va / max(1, len(labs_va))
        )
    )

    # ---------- Step 3: 训练 L1 DilemmaGate ----------
    # 先做特征标准化：用 train 统计量（均值 0 / 标准差 1），避免不同特征量纲差太大
    feats_mean = feats_tr.mean(dim=0)
    feats_std = feats_tr.std(dim=0, unbiased=False) + 1e-6
    feats_tr_n = (feats_tr - feats_mean) / feats_std
    feats_va_n = (feats_va - feats_mean) / feats_std

    print("\n  [feature stats per dim (train std)]")
    for i in range(input_dim):
        print("    f{:02d}: mean={:+.4f}  std={:.4f}  min={:.3f}  max={:.3f}".format(
            i, feats_tr[:, i].mean().item(), feats_tr[:, i].std(unbiased=False).item(),
            feats_tr[:, i].min().item(), feats_tr[:, i].max().item(),
        ))

    # 正负样本不平衡 -> BCEWithLogits + pos_weight
    pos_w_val = float(max(1.0, min(10.0,
        (labs_tr == 0).float().sum().item() / max(1, (labs_tr == 1).float().sum().item())
    )))
    print("  pos_weight for BCE: {:.3f}".format(pos_w_val))

    class GateLogits(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, hidden_dim2=32, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim2, 1),
            )
            # 最后一层偏置初始化为 -1 让初始 sigmoid≈0.25（偏向预测负样本）
            nn.init.constant_(self.net[-1].bias, -1.0)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    gate = GateLogits(input_dim=input_dim, hidden_dim=64, hidden_dim2=32, dropout=0.2).to(args.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w_val, device=args.device))
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)

    feats_tr_d = feats_tr_n.to(args.device)
    labs_tr_d = labs_tr.to(args.device)
    feats_va_d = feats_va_n.to(args.device)
    labs_va_d = labs_va.to(args.device)

    print("\n[Step 3] Training DilemmaGate ({} epochs, lr={}) ...".format(args.epochs, args.lr))
    n_total = feats_tr_d.size(0)
    bs = args.bs

    def evaluate(gate, X, y_true, th=0.5):
        gate.eval()
        with torch.no_grad():
            logits = gate(X)
        probs = torch.sigmoid(logits)
        preds = (probs >= th).float()
        acc = (preds == y_true).float().mean().item()
        try:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true.cpu().numpy(), preds.cpu().numpy(),
                average="binary", zero_division=0,
            )
        except Exception:
            p = r = f1 = 0.0
        pos_scores = probs[y_true == 1]
        neg_scores = probs[y_true == 0]
        pos_m = float(pos_scores.mean().item()) if pos_scores.numel() else 0.0
        neg_m = float(neg_scores.mean().item()) if neg_scores.numel() else 0.0
        return acc, p, r, f1, pos_m, neg_m, probs

    best_f1 = -1.0
    best_ckpt = None
    for ep in range(args.epochs):
        gate.train()
        idx = torch.randperm(n_total, device=args.device)
        total_loss = 0.0
        steps = 0
        for i in range(0, n_total, bs):
            sel = idx[i : i + bs]
            xb = feats_tr_d[sel]
            yb = labs_tr_d[sel]
            opt.zero_grad()
            logits = gate(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item())
            steps += 1

        va_acc, va_p, va_r, va_f1, pos_m, neg_m, va_probs = evaluate(gate, feats_va_d, labs_va_d, 0.5)
        tr_acc, tr_p, tr_r, tr_f1, _, _, _ = evaluate(gate, feats_tr_d, labs_tr_d, 0.5)

        # 阈值扫描（0.1~0.95）取 F1 最优值用于最终
        local_best_th = 0.5
        local_best_f1 = va_f1
        for tt in torch.linspace(0.1, 0.95, 86):
            preds = (va_probs >= tt.item()).float()
            try:
                _, _, ff1, _ = precision_recall_fscore_support(
                    labs_va_d.cpu().numpy(), preds.cpu().numpy(),
                    average="binary", zero_division=0,
                )
            except Exception:
                ff1 = 0.0
            if ff1 > local_best_f1:
                local_best_f1 = ff1
                local_best_th = tt.item()

        if local_best_f1 > best_f1:
            best_f1 = local_best_f1
            best_ckpt = {k: v.cpu().clone() for k, v in gate.state_dict().items()}

        print(
            "  epoch {:2d} loss={:.4f}  "
            "tr[acc={:.3f} P={:.3f} R={:.3f} F1={:.3f}]  "
            "va[acc={:.3f} P={:.3f} R={:.3f} F1={:.3f}]  "
            "pos_mu={:.3f} neg_mu={:.3f}  best@th={:.2f}/F1={:.3f}".format(
                ep + 1, total_loss / max(1, steps),
                tr_acc, tr_p, tr_r, tr_f1,
                va_acc, va_p, va_r, va_f1,
                pos_m, neg_m, local_best_th, local_best_f1,
            )
        )

    if best_ckpt is not None:
        gate.load_state_dict(best_ckpt)
        print("\n  (loaded best checkpoint, F1={:.3f})".format(best_f1))

    # ---------- Step 4: 保存训练好的门控参数 ----------
    print("\n[Step 4] Saving trained gate parameters ...")
    gate.eval()
    
    # 保存完整检查点：门控权重 + 特征标准化参数 + 最优阈值
    checkpoint = {
        'state_dict': best_ckpt if best_ckpt else gate.state_dict(),
        'feats_mean': feats_mean.cpu(),
        'feats_std': feats_std.cpu(),
        'input_dim': input_dim,
        'pos_rate': pos_rate2,
        'thresholds': {
            'th_entropy': th_entropy,
            'th_maxprob': th_maxprob,
            'rep_th': rep_th,
        },
        'suggested_enter_thresh': va_res["pos_mean"] * 0.7 + va_res["neg_mean"] * 0.3,
        'suggested_exit_thresh': va_res["pos_mean"] * 0.3 + va_res["neg_mean"] * 0.7,
        'metrics': {
            'val_f1': va_res["f1_best"],
            'val_pos_mean': va_res["pos_mean"],
            'val_neg_mean': va_res["neg_mean"],
            'train_f1': tr_res["f1_best"],
        }
    }
    
    # 保存到文件
    ckpt_path = ROOT / 'checkpoints' / 'l1_gate_best.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)
    print(f"  saved to: {ckpt_path}")
    
    # ---------- 最终汇总 ----------
    def final_eval(X, y_true):
        with torch.no_grad():
            logits = gate(X)
        probs = torch.sigmoid(logits)
        th_best = 0.5; f1_best = -1.0
        for tt in torch.linspace(0.1, 0.95, 86):
            preds = (probs >= tt.item()).float()
            try:
                _, _, ff1, _ = precision_recall_fscore_support(
                    y_true.cpu().numpy(), preds.cpu().numpy(),
                    average="binary", zero_division=0,
                )
            except Exception:
                ff1 = 0.0
            if ff1 > f1_best:
                f1_best = ff1; th_best = tt.item()
        preds_at_best = (probs >= th_best).float()
        try:
            acc_b = float((preds_at_best == y_true).float().mean().item())
            p_b, r_b, f1_b, _ = precision_recall_fscore_support(
                y_true.cpu().numpy(), preds_at_best.cpu().numpy(),
                average="binary", zero_division=0,
            )
        except Exception:
            acc_b = 0.0; p_b = r_b = f1_b = 0.0
        pos_s = probs[y_true == 1]
        neg_s = probs[y_true == 0]
        pm = float(pos_s.mean().item()) if pos_s.numel() else 0.0
        nm = float(neg_s.mean().item()) if neg_s.numel() else 0.0
        return {
            "th_best": th_best, "f1_best": f1_best,
            "acc_best": acc_b, "p_best": p_b, "r_best": r_b,
            "pos_mean": pm, "neg_mean": nm, "probs": probs,
        }

    tr_res = final_eval(feats_tr_d, labs_tr_d)
    va_res = final_eval(feats_va_d, labs_va_d)
    
    # 计算建议的自适应阈值
    suggested_enter = checkpoint['suggested_enter_thresh']
    suggested_exit = checkpoint['suggested_exit_thresh']
    score_gap = va_res["pos_mean"] - va_res["neg_mean"]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("-" * 70)
    print("  backbone           : tiny MetaCogXModel ({}L, d={}, h={})".format(num_layers, cfg.d_model, cfg.num_heads))
    print("  input_dim          : {}".format(input_dim))
    print("  label rule         : entropy_mean>{:.3f} OR logits_maxprob<{:.3f} OR token_rep>={}".format(th_entropy, th_maxprob, rep_th))
    print("  train pos%         : {:.1f}%".format(100 * pos_tr / max(1, len(labs_tr))))
    print("  val   pos%         : {:.1f}%".format(100 * pos_va / max(1, len(labs_va))))
    print("-" * 70)
    print("  -- train (best threshold) --")
    print("  thr={:.3f}  acc={:.4f}  P={:.4f}  R={:.4f}  F1={:.4f}  pos_mu={:.4f}  neg_mu={:.4f}".format(
        tr_res["th_best"], tr_res["acc_best"], tr_res["p_best"], tr_res["r_best"],
        tr_res["f1_best"], tr_res["pos_mean"], tr_res["neg_mean"],
    ))
    print("  -- val   (best threshold) --")
    print("  thr={:.3f}  acc={:.4f}  P={:.4f}  R={:.4f}  F1={:.4f}  pos_mu={:.4f}  neg_mu={:.4f}".format(
        va_res["th_best"], va_res["acc_best"], va_res["p_best"], va_res["r_best"],
        va_res["f1_best"], va_res["pos_mean"], va_res["neg_mean"],
    ))
    print("-" * 70)
    print("  score gap val      : {:.4f}".format(score_gap))
    print("-" * 70)
    print("  SUGGESTED THRESHOLDS for L1 Gate:")
    print("    enter_thresh      : {:.3f} (weighted between pos/neg means)".format(suggested_enter))
    print("    exit_thresh       : {:.3f} (weighted between neg/pos means)".format(suggested_exit))
    print("    enter_patience    : 2 (recommended)")
    print("    exit_patience     : 3 (recommended)")
    print("=" * 70)

    vf = va_res["f1_best"]
    vpos = va_res["pos_mean"]
    vneg = va_res["neg_mean"]
    if vf >= 0.6 and vpos > 0.7 and vneg < 0.4:
        verdict = ">> PASS  (val F1>=0.6, pos>0.7, neg<0.4)"
    elif vf >= 0.4:
        verdict = ">> OK (F1>0.4)  consider tuning thresholds"
    else:
        verdict = ">> NEEDS TUNING  (F1<0.4; adjust th_entropy/th_maxprob)"
    print(verdict)
    
    print("\n  To use this gate in MetaCogXModel:")
    print("  1. Load checkpoint: torch.load('checkpoints/l1_gate_best.pt')")
    print("  2. Set model.l1_gate.load_state_dict(checkpoint['state_dict'])")
    print("  3. Set model.enter_thresh = checkpoint['suggested_enter_thresh']")
    print("  4. Set model.exit_thresh = checkpoint['suggested_exit_thresh']")


if __name__ == "__main__":
    main()
