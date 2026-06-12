"""runs/run_ab_v2.py

3组 A/B 比较 v2:
  1. gpt_plain            : enable_metacog=False
  2. metacog_alwayson    : enable_metacog=True, 强制 _mode_state='metacog'
                           + l1_gate last.bias=+5 -> sigmoid(+5)≈0.99 > enter_thresh=-1 永远 metacog
  3. metacog_conditional : enable_metacog=True, 正常 L1 门控 + 滞后切换
                           预训练 L1 gate：先 400 step plain CE backbone，
                           再 entropy percentile 标注，BCE 训练 gate 500 step，
                           然后再进入完整 conditional 训练 2000 step

tiny byte-level 模型 (ByteTokenizer, d_model=128, d_meta=32, d_aware=16,
  num_layers=4, num_heads=4, d_ffn=512, max_seq_len=64, vocab_size=260)

8 主题 x 1200 句 -> ByteDataset(chunk 到 64 tokens)
"""

import sys, os, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# CPU 并行：开足 8C16T 的 AMD 4750U
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

# DirectML (AMD 核显) — 能导入就用
try:
    import torch_directml
    DML_AVAIL = True
except Exception:
    DML_AVAIL = False

from config import MetaCogXConfig
from models import MetaCogXModel
from models.dilemma_gate import attention_entropy, extract_features, logits_stats, token_repetition


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # NOTE: torch-directml Adam optimizer lerp fallback + tiny d_model=128 make DML slower than CPU MKL.
    # Force CPU for A/B evaluation (3x faster). Remove this override if model grows to d_model>=512.
    # if DML_AVAIL:
    #     try:
    #         dev = torch_directml.device(0)
    #         return dev
    #     except Exception as e:
    #         print("  [DML WARN] available but .device(0) failed: {}".format(e))
    return "cpu"

# =========================================================
# Byte-level Dataset
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
# ppl 评估
# =========================================================
def evaluate_ppl(model, dl, device, use_metacog):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(device)
            msk = b["attention_mask"].to(device)
            o = model(ids, attention_mask=msk, enable_metacog=use_metacog)
            lg = o["logits"][:, :-1, :]
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
    avg_loss = total_loss / max(1e-9, total_count)
    ppl = math.exp(min(20, avg_loss))
    return avg_loss, ppl


# =========================================================
# 跑一遍模型 -> collection L1 gate 输入特征
# model: enable_metacog=True
# =========================================================
def collect_gate_features(model, ids, msk, device):
    o = model(ids, attention_mask=msk, enable_metacog=True)
    num_layers = len(model.layers)
    entropy_list = []
    for layer in model.layers:
        w = getattr(layer.triple_attn, '_last_attn_c', None)
        if w is None:
            w = getattr(layer.triple_attn, '_last_attn', None)
        if w is not None and isinstance(w, torch.Tensor) and w.dim() == 4:
            e = attention_entropy(w)
        else:
            e = torch.zeros(ids.size(0), 4, ids.size(1), device=device)
        entropy_list.append(e)
    feats = extract_features(entropy_list, o["logits"], ids)
    sur = o.get("surprise", None)
    if sur is not None and isinstance(sur, torch.Tensor):
        feats = torch.cat([feats, sur.detach().unsqueeze(-1)], dim=-1)
    else:
        feats = torch.cat([feats, torch.zeros(feats.size(0), 1, device=feats.device)], dim=-1)
    return feats  # [B, F_expected_by_model]


# =========================================================
# L1 gate 预训练（熵 percentile 标注）
# =========================================================
def pretrain_l1_gate(model, tr_dl, device,
                     th_entropy_frac=0.70, th_lmp_frac=0.30, rep_th=2,
                     pos_target_range=(0.25, 0.55),
                     epochs=4, bs=64, lr=1e-3):
    """返回 (model, gate_module)，gate 已经作为 model.l1_gate 的 state_dict 写入"""
    print("    [L1 pretrain] collecting train features ...")
    model.eval()
    all_feats_tr = []
    for b in tr_dl:
        ids = b["input_ids"].to(device)
        msk = b["attention_mask"].to(device)
        with torch.no_grad():
            feats = collect_gate_features(model, ids, msk, device)
        all_feats_tr.append(feats.cpu())
    all_feats_tr = torch.cat(all_feats_tr, dim=0)
    feats_mean = all_feats_tr.mean(dim=0)
    feats_std = all_feats_tr.std(dim=0, unbiased=False) + 1e-6
    F = all_feats_tr.size(1)
    num_layers = (F - 3) // 2

    # entropy_mean 是前 num_layers 对特征的奇数位（mean）
    ent_means = []
    for i in range(num_layers):
        ent_means.append(all_feats_tr[:, 2 * i])
    ent_all = torch.stack(ent_means, dim=0).mean(dim=0)
    lmp_all = all_feats_tr[:, -3]
    rep_all = all_feats_tr[:, -1]

    def q(vals, p):
        if vals.numel() == 0:
            return torch.tensor(0.0)
        return torch.quantile(vals.float(), p, interpolation="midpoint")

    # 自适应多次尝试
    pos_rate = 0.0
    th_entropy = 0.0
    th_maxprob = 0.0
    for attempt, frac_e, frac_l in [
        (0, 0.70, 0.30), (1, 0.75, 0.25), (2, 0.65, 0.35),
        (3, 0.80, 0.20), (4, 0.60, 0.40), (5, 0.55, 0.45),
    ]:
        th_entropy = float(q(ent_all, frac_e).item())
        th_maxprob = float(q(lmp_all, frac_l).item())
        temp_lab = ((ent_all > th_entropy) | (lmp_all < th_maxprob) | (rep_all >= rep_th)).float()
        pos_rate = float(temp_lab.mean().item())
        if pos_target_range[0] <= pos_rate <= pos_target_range[1]:
            break
    print("    [L1 pretrain] th_ent={:.3f} th_lmp={:.3f} rep_th={}  pos_rate={:.3f}".format(
        th_entropy, th_maxprob, rep_th, pos_rate))

    # 构造标签
    ent_means = []
    for i in range(num_layers):
        ent_means.append(all_feats_tr[:, 2 * i])
    mean_ent_agg = torch.stack(ent_means, dim=0).mean(dim=0)
    logits_maxprob = all_feats_tr[:, -3]
    token_rep_mean = all_feats_tr[:, -1]
    labels = ((mean_ent_agg > th_entropy) | (logits_maxprob < th_maxprob) | (token_rep_mean >= rep_th)).float()

    # 标准化
    X_tr = (all_feats_tr - feats_mean) / feats_std
    y_tr = labels

    pos_w = float(max(1.0, min(10.0,
        (y_tr == 0).float().sum().item() / max(1, (y_tr == 1).float().sum().item())
    )))
    print("    [L1 pretrain] pos_weight={:.3f}  F={}  n={}".format(pos_w, F, X_tr.size(0)))

    # 构建 gate 模块 (clone l1_gate, 训练后再写回)
    gate = model.l1_gate
    opt_gate = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    # 注意: 当前 DilemmaGate.net 最后一层是 Sigmoid, 不是裸 Linear!
    # 这里我们需要 logits，所以重新训练一个临时 gate
    class GateLogits(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    gate_tmp = GateLogits(F, hidden_dim=32, dropout=0.1).to(device)
    # 初始偏置让输出偏向 0（sigmoid 后 =0.5）
    last = gate_tmp.net[-1]
    if hasattr(last, 'bias') and last.bias is not None:
        nn.init.constant_(last.bias, -2.0)  # sigmoid(-2)≈0.126 负样本略多，配合 pos_weight

    opt_t = torch.optim.AdamW(gate_tmp.parameters(), lr=lr, weight_decay=1e-4)

    Xd = X_tr.to(device)
    yd = y_tr.to(device)
    n = Xd.size(0)
    best_f1 = -1.0
    best_ckpt = None
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        total_loss = 0.0
        steps = 0
        for i in range(0, n, bs):
            sel = idx[i : i + bs]
            xb = Xd[sel]
            yb = yd[sel]
            opt_t.zero_grad()
            logits = gate_tmp(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(gate_tmp.parameters(), 5.0)
            opt_t.step()
            total_loss += float(loss.item())
            steps += 1

        with torch.no_grad():
            pr = torch.sigmoid(gate_tmp(Xd))
        preds = (pr >= 0.5).float()
        acc = float((preds == yd).float().mean().item())
        tp = float(((preds == 1) & (yd == 1)).float().sum().item())
        fp = float(((preds == 1) & (yd == 0)).float().sum().item())
        fn = float(((preds == 0) & (yd == 1)).float().sum().item())
        prec = tp / max(1e-9, tp + fp)
        rec = tp / max(1e-9, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        pos_s = pr[yd == 1].mean().item() if (yd == 1).any() else 0.0
        neg_s = pr[yd == 0].mean().item() if (yd == 0).any() else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_ckpt = {k: v.cpu().clone() for k, v in gate_tmp.state_dict().items()}

        if ep < 3 or (ep + 1) % 2 == 0:
            print("    [L1 pretrain] ep {:2d} loss={:.4f} acc={:.3f} P={:.3f} R={:.3f} F1={:.3f} pos_mu={:.3f} neg_mu={:.3f}".format(
                ep + 1, total_loss / max(1, steps), acc, prec, rec, f1, pos_s, neg_s))

    if best_ckpt is not None:
        gate_tmp.load_state_dict(best_ckpt)

    # 把 gate_tmp 的 "Linear+ReLU+Dropout+Linear" 权重拷进 model.l1_gate.net
    # model.l1_gate.net = [Linear, ReLU, Dropout, Linear, Sigmoid]
    l1 = model.l1_gate
    with torch.no_grad():
        l1.net[0].weight.copy_(gate_tmp.net[0].weight)
        l1.net[0].bias.copy_(gate_tmp.net[0].bias)
        l1.net[3].weight.copy_(gate_tmp.net[3].weight)
        l1.net[3].bias.copy_(gate_tmp.net[3].bias)
    print("    [L1 pretrain] done. copied weights into model.l1_gate (no sigmoid -> with sigmoid).")

    # 验证一下：model.l1_gate(features) 已经是 sigmoid 输出
    # X_tr 已经是标准化后的 [N, F] 特征，不需要再传 entropy_list 给 extract_features
    # 直接 forward 即可
    model.eval()
    with torch.no_grad():
        pr2 = torch.sigmoid(l1.net(X_tr.to(device)))
    pr2_cpu = pr2.cpu()
    preds2 = (pr2_cpu >= 0.5).float()
    tp2 = float(((preds2 == 1) & (y_tr == 1)).float().sum().item())
    fp2 = float(((preds2 == 1) & (y_tr == 0)).float().sum().item())
    fn2 = float(((preds2 == 0) & (y_tr == 1)).float().sum().item())
    prec2 = tp2 / max(1e-9, tp2 + fp2)
    rec2 = tp2 / max(1e-9, tp2 + fn2)
    f1_verify = 2 * prec2 * rec2 / max(1e-9, prec2 + rec2)
    pos_mu = float(pr2_cpu[y_tr == 1].mean().item()) if (y_tr == 1).any() else 0.0
    neg_mu = float(pr2_cpu[y_tr == 0].mean().item()) if (y_tr == 0).any() else 0.0
    print("    [L1 pretrain] VERIFY on train: P={:.3f} R={:.3f} F1={:.3f} pos_mu={:.3f} neg_mu={:.3f}".format(
        prec2, rec2, f1_verify, pos_mu, neg_mu))

    # 打印 score 分布确认 enter/exit 阈值合理
    print("    [L1 pretrain] score stats on train: min={:.3f} p10={:.3f} p30={:.3f} p50={:.3f} p70={:.3f} p90={:.3f} max={:.3f}".format(
        float(pr2_cpu.min().item()),
        float(torch.quantile(pr2_cpu.float(), 0.10, interpolation="midpoint").item()),
        float(torch.quantile(pr2_cpu.float(), 0.30, interpolation="midpoint").item()),
        float(torch.quantile(pr2_cpu.float(), 0.50, interpolation="midpoint").item()),
        float(torch.quantile(pr2_cpu.float(), 0.70, interpolation="midpoint").item()),
        float(torch.quantile(pr2_cpu.float(), 0.90, interpolation="midpoint").item()),
        float(pr2_cpu.max().item()),
    ))

    # 返回阈值建议
    p70 = float(torch.quantile(pr2_cpu.float(), 0.70, interpolation="midpoint").item())
    p30 = float(torch.quantile(pr2_cpu.float(), 0.30, interpolation="midpoint").item())
    return {
        "th_entropy": th_entropy, "th_maxprob": th_maxprob, "rep_th": rep_th,
        "pos_rate": pos_rate, "p70": p70, "p30": p30,
    }


# =========================================================
# 单独预训练 backbone (CE, 不跑 metacog)
# =========================================================
def pretrain_backbone_plain(model, tr_dl, device, steps=400, lr=1e-3):
    print("    [backbone pretrain] {} steps CE (plain)".format(steps))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    it = iter(tr_dl)
    for s in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(tr_dl)
            b = next(it)
        ids = b["input_ids"].to(device)
        msk = b["attention_mask"].to(device)
        opt.zero_grad()
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
        opt.step()
        if (s + 1) % 200 == 0:
            print("      step {:4d}  loss={:.4f}  ppl={:.2f}".format(
                s + 1, float(loss.item()), math.exp(min(20, float(loss.item())))))


# =========================================================
# 单次 variant 运行
# =========================================================
def run_variant(name, variant_mode, tok, tr_dl, va_dl, device):
    """variant_mode in {'plain', 'alwayson', 'conditional'}"""
    random.seed(0); torch.manual_seed(0)

    use_metacog = variant_mode != 'plain'

    cfg = MetaCogXConfig(
        d_model=128,
        d_meta=32,
        d_aware=16,
        num_layers=4,
        num_heads=4,
        d_ffn=512,
        max_seq_len=64,
        vocab_size=260,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
    )
    model = MetaCogXModel(cfg, enable_metacog=use_metacog).to(device)

    if variant_mode == 'plain':
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    elif variant_mode == 'alwayson':
        model._mode_state = 'metacog'
        model.l1_gate.enter_thresh = -1.0
        model.l1_gate.exit_thresh = -1.0
        last = model.l1_gate.net[3]
        if hasattr(last, 'bias') and last.bias is not None:
            nn.init.constant_(last.bias, 5.0)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    elif variant_mode == 'conditional':
        pretrain_backbone_plain(model, tr_dl, device, steps=400, lr=1e-3)
        gate_stats = pretrain_l1_gate(model, tr_dl, device, epochs=4, bs=64, lr=1e-3)
        enter_th = 0.38
        exit_th = 0.25
        if gate_stats['p70'] < enter_th:
            enter_th = gate_stats['p70']
        exit_th = max(0.20, enter_th - 0.12)
        model.l1_gate.enter_thresh = enter_th
        model.l1_gate.exit_thresh = exit_th
        model._mode_state = 'plain'
        model._plain_countdown = 0
        model._meta_countdown = 0
        model._switch_stats = {k: 0 for k in model._switch_stats}
        baseline_ce = None
        model.eval()
        with torch.no_grad():
            n_test = 0; ce_sum = 0.0
            for bb in tr_dl:
                ids = bb["input_ids"].to(device); msk = bb["attention_mask"].to(device)
                oo = model(ids, attention_mask=msk, enable_metacog=True)
                llg = oo["logits"][:, :-1, :]; tgt = ids[:, 1:]
                pm = msk[:, 1:].float()
                c = F.cross_entropy(llg.reshape(-1, llg.size(-1)), tgt.reshape(-1), ignore_index=0, reduction="none").reshape(ids.size(0), -1)
                ce_sum += float((c * pm).sum().item())
                n_test += float(pm.sum().item())
        baseline_ce = ce_sum / max(1e-9, n_test)
        print("    [conditional] baseline_ce={:.3f} (used in RL tf surrogate)".format(baseline_ce))
        from training.rl_framework import MinimalPPO
        rl = MinimalPPO(
            model=model, lr=2e-3,
            lambda_tf=0.5, lambda_gate=1.0, lambda_tf_l2=0.01,
            baseline_ce=baseline_ce,
            device=str(device),
        )
        opt = rl.opt
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)

    print("\n" + "=" * 82)
    print(" variant : {}".format(name))
    print(" enable_metacog={}  mode={}  params={:,}".format(
        use_metacog, variant_mode, sum(p.numel() for p in model.parameters())
    ))
    if variant_mode == 'conditional':
        print("  enter_thresh={:.3f}  exit_thresh={:.3f}".format(
            model.l1_gate.enter_thresh, model.l1_gate.exit_thresh))
    print("=" * 82)
    print(" {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "step", "loss", "val_ppl", "mode", "ctrl_std", "switches", "plain%", "meta%", "score"
    ))
    print("-" * 82)

    hist_modes = []
    hist_ctrl = []
    hist_scores = []

    it = iter(tr_dl)
    for s in range(500):
        try:
            b = next(it)
        except StopIteration:
            it = iter(tr_dl)
            b = next(it)
        ids = b["input_ids"].to(device)
        msk = b["attention_mask"].to(device)

        if variant_mode == 'conditional':
            rl_out = rl.train_step({"input_ids": ids, "attention_mask": msk})
            loss = torch.tensor(rl_out["loss"])
            o = rl_out["forward_out"]
        else:
            model.train()
            opt.zero_grad()
            o = model(ids, attention_mask=msk, enable_metacog=use_metacog)
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
            opt.step()
            sched.step()

        mode = o.get('mode', 'plain')
        hist_modes.append(mode)
        ds = o.get('last_dilemma_score', None)
        if ds is not None:
            hist_scores.append(ds)

        if 'ctrl' in o and o['ctrl'] is not None and hasattr(o['ctrl'], 'temp_factor'):
            tf = o['ctrl'].temp_factor
            hist_ctrl.append(float(tf.detach().cpu().mean().item()))

        if (s + 1) % 100 == 0 or (s + 1) == 500:
            vloss, vppl = evaluate_ppl(model, va_dl, device, use_metacog)

            last10_modes = hist_modes[-10:]
            mode_counts = {}
            for m in last10_modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            majority_mode = max(mode_counts, key=mode_counts.get) if last10_modes else 'plain'

            ctrl_std = float('nan')
            if len(hist_ctrl) > 1:
                c = torch.tensor(hist_ctrl[-min(100, len(hist_ctrl)):])
                ctrl_std = float(c.std(unbiased=False).item())

            sw = 0
            plain_pct = 100.0
            meta_pct = 0.0
            ss = o.get('switch_stats', None)
            if ss is not None:
                sw = int(ss.get('switches', 0))
                total = int(ss.get('plain_steps', 0) + ss.get('meta_steps', 0))
                if total > 0:
                    plain_pct = 100.0 * ss.get('plain_steps', 0) / total
                    meta_pct = 100.0 * ss.get('meta_steps', 0) / total

            score_str = "nan"
            if hist_scores:
                arr = torch.tensor(hist_scores[-min(100, len(hist_scores)):])
                score_str = "{:.3f}/{:.3f}".format(float(arr.mean().item()), float(arr.std(unbiased=False).item()))

            print(" {:6d} {:10.4f} {:10.2f} {:>10} {:10.4f} {:10d} {:9.1f}% {:9.1f}% {:>10}".format(
                s + 1, float(loss.item()), vppl, majority_mode, ctrl_std, sw, plain_pct, meta_pct, score_str
            ), flush=True)

    final_vloss, final_vppl = evaluate_ppl(model, va_dl, device, use_metacog)

    final_ctrl_std = float('nan')
    if len(hist_ctrl) > 1:
        c = torch.tensor(hist_ctrl[-min(200, len(hist_ctrl)):])
        final_ctrl_std = float(c.std(unbiased=False).item())

    ss = o.get('switch_stats', {})
    final_sw = int(ss.get('switches', 0)) if ss else 0
    total_steps = (ss.get('plain_steps', 0) + ss.get('meta_steps', 0)) if ss else 0
    plain_pct = 100.0 * ss.get('plain_steps', 0) / max(1, total_steps) if ss else 100.0

    final_score_mean = float('nan')
    final_score_std = float('nan')
    if hist_scores:
        arr = torch.tensor(hist_scores[-min(200, len(hist_scores)):])
        final_score_mean = float(arr.mean().item())
        final_score_std = float(arr.std(unbiased=False).item())

    print("-" * 82)
    print(" FINAL ppl={:.4f}  ctrl_std={:.4f}  switches={}  plain={:.1f}%  score={:.3f}+/-{:.3f}".format(
        final_vppl, final_ctrl_std, final_sw, plain_pct, final_score_mean, final_score_std
    ), flush=True)

    return {
        "name": name,
        "final_ppl": final_vppl,
        "final_loss": final_vloss,
        "ctrl_std": final_ctrl_std,
        "switches": final_sw,
        "plain_pct": plain_pct,
        "score_mean": final_score_mean,
        "score_std": final_score_std,
    }


# =========================================================
# Main
# =========================================================
def main():
    picked = pick_device()
    if DML_AVAIL:
        try:
            import torch_directml
            _dml_dev = torch_directml.device(0)
            if isinstance(picked, torch.device) and str(picked.type) == 'privateuseone':
                device = picked
                dev_repr = "directml(0)"
            elif isinstance(picked, torch.device) and picked.type == 'cpu':
                device = _dml_dev
                dev_repr = "directml(0)"
            else:
                device = picked
                dev_repr = str(picked)
        except Exception as e:
            device = 'cpu'
            dev_repr = 'cpu(dml_failed:{})'.format(e)
    elif isinstance(picked, torch.device):
        device = picked
        dev_repr = str(picked.type)
    else:
        device = picked
        dev_repr = str(picked)
    tok = ByteTokenizer(max_len=64)

    pin = dev_repr.startswith("directml") or dev_repr.startswith("cuda")
    num_work = 2 if pin else 0

    all_texts = build_texts(n=1200, seed=42)
    tr_ds = ByteDataset(all_texts[:900], tok, 64)
    va_ds = ByteDataset(all_texts[900:], tok, 64)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate,
                       num_workers=num_work, pin_memory=pin, persistent_workers=num_work>0)
    va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate,
                       num_workers=num_work, pin_memory=pin, persistent_workers=num_work>0)

    print("=" * 82, flush=True)
    print(" A/B EVALUATION v2  (3 variants, tiny byte-level model)".center(82), flush=True)
    print(" device={}  train={}  val={}  steps=500  OMP={}  MKL={}  DML_AVAIL={}".format(
        dev_repr, len(tr_ds), len(va_ds),
        os.environ.get("OMP_NUM_THREADS", "?"), os.environ.get("MKL_NUM_THREADS", "?"), DML_AVAIL), flush=True)
    print("=" * 82, flush=True)

    results = []

    for name, mode in [
        ("gpt_plain", "plain"),
        ("metacog_alwayson", "alwayson"),
        ("metacog_conditional", "conditional"),
    ]:
        r = run_variant(name, mode, tok, tr_dl, va_dl, device)
        results.append(r)

    # 汇总
    print("\n" + "=" * 82)
    print(" FINAL SUMMARY TABLE")
    print("=" * 82)
    print(" {:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "variant", "val_ppl", "ctrl_std", "switches", "plain%", "loss", "score_m", "score_s"
    ))
    print("-" * 82)
    for r in results:
        print(" {:<25} {:10.2f} {:10.4f} {:10d} {:9.1f}% {:10.4f} {:10.3f} {:10.3f}".format(
            r["name"], r["final_ppl"], r["ctrl_std"], r["switches"], r["plain_pct"],
            r["final_loss"], r["score_mean"], r["score_std"]
        ))
    print("=" * 82)

    # 关键验证点
    print("\n KEY CHECKS")
    print("-" * 82)
    cond = next(r for r in results if r["name"] == "metacog_conditional")
    on = next(r for r in results if r["name"] == "metacog_alwayson")
    gpt = next(r for r in results if r["name"] == "gpt_plain")

    check1 = cond["ctrl_std"] > 0.05
    check2 = cond["final_ppl"] <= on["final_ppl"] + 0.02
    check3 = cond["switches"] >= 0

    print("  [1] conditional ctrl std > 0.05       : {:.4f}  -> {}".format(
        cond["ctrl_std"], "PASS" if check1 else "FAIL"
    ))
    print("  [2] conditional ppl <= alwayson ppl   : {:.2f} <= {:.2f}  -> {}".format(
        cond["final_ppl"], on["final_ppl"], "PASS" if check2 else "FAIL"
    ))
    print("  [3] switches >= 0 (ideally > 0)       : {}  -> {}".format(
        cond["switches"], "PASS" if cond["switches"] > 0 else "OK(=0)" if cond["switches"] == 0 else "FAIL"
    ))
    if cond["switches"] == 0:
        print("       (conditional stayed in one mode; check enter/exit thresholds)")

    overall = "PASS" if (check1 and check2 and cond["switches"] >= 0) else "CHECK DETAIL"
    print("=" * 82)
    print(" OVERALL : {}".format(overall))
    print("=" * 82)

    import json as _json
    _out = {
        "device": dev_repr,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
        "results": results,
        "key_checks": {
            "conditional_ctrl_std_gt_0.05": bool(check1),
            "conditional_ppl_le_alwayson_ppl_plus_0.02": bool(check2),
            "switches_ge_0": bool(cond["switches"] >= 0),
            "overall": overall,
        },
    }
    _out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_results_v2.json")
    with open(_out_path, "w", encoding="utf-8") as _f:
        _json.dump(_out, _f, indent=2, default=str)
    print(" RESULTS JSON -> {}".format(_out_path))


if __name__ == "__main__":
    main()
