"""runs/eval.py - 统一评估脚本

支持加载预训练模型检查点并进行全面评估，输出：
- Perplexity (PPL)
- 门控触发率和模式切换统计
- Controller 信号统计
- 条件激活 PPL 差异分析
"""

import sys, os, math, argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import MetaCogXConfig
from models import MetaCogXModel


# =========================================================
# 数据集（复用 run_ab_v2.py 的逻辑）
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


class ByteDataset(torch.utils.data.Dataset):
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


def build_texts(n=500, seed=123):
    """生成评估用文本"""
    topics = [
        "Attention Is All You Need. Transformers stack self attention in blocks of residual layers. Scaled dot product attention relates tokens via query key value triplets.",
        "Adam optimizer combines momentum with adaptive learning rates. It estimates first and second moments of gradients per parameter.",
        "Residual connections ease optimization of deep stacks. Each layer learns delta on top of identity mapping so deeper nets train reliably.",
        "Dropout prevents co adaptation of hidden units. Each forward pass drops random units forcing robustness across pathways.",
        "Layer normalization stabilizes hidden activations. It computes mean and variance across feature dimensions and applies affine transform.",
        "Feedforward layers expand then compress. They apply pointwise nonlinearity in four times the model dimension then project back.",
        "Token embeddings map discrete indices to dense vectors. Position embeddings add sequence ordering information to each token.",
        "Gradient clipping rescales gradients whose norm exceeds a threshold. This avoids exploding gradients in long backprop chains.",
        "Learning rate scheduling adjusts step size dynamically. Cosine annealing cycles between high and low rates to escape local minima.",
        "Transformer decoders mask future positions during training. This autoregressive constraint ensures causality in generation.",
        "Multi-head attention computes multiple parallel attention heads. Each head learns distinct token relationships across the sequence.",
        "Positional encoding injects sequential information. Sinusoidal functions encode absolute position without additional parameters.",
        "Beam search explores multiple hypotheses during generation. It maintains top-k candidates to find optimal output sequences.",
        "Self-supervised learning trains on unlabeled data. Contrastive objectives learn useful representations without explicit labels.",
        "Knowledge distillation transfers expertise from large models. A smaller student model mimics a larger teacher's outputs.",
    ]
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(topics)
        out.append(t + " Evaluation sample {}. Additional text for diversity.".format(i))
    return out


# =========================================================
# 评估函数
# =========================================================
@torch.no_grad()
def evaluate_model(model, dl, device, enable_metacog=True, mode="conditional"):
    """
    评估模型并返回详细统计
    
    Args:
        model: MetaCogXModel
        dl: DataLoader
        device: str, 'cpu' or 'cuda'
        enable_metacog: bool, 是否启用元认知
        mode: str, 'plain', 'alwayson', or 'conditional'
    
    Returns:
        dict of metrics
    """
    model.eval()
    
    # 重置状态
    if hasattr(model, "_switch_stats"):
        model._switch_stats = {k: 0 for k in model._switch_stats}
    else:
        model._switch_stats = {
            "switches": 0, "total_forward": 0, "plain_steps": 0, "meta_steps": 0,
        }
    
    total_loss = 0.0
    total_count = 0
    switches = 0
    plain_steps = 0
    meta_steps = 0
    dilemma_scores = []
    temp_factors = []
    tf_raw_logits = []
    
    for b in dl:
        ids = b["input_ids"].to(device)
        msk = b["attention_mask"].to(device)
        
        # 根据模式设置门控
        if mode == "alwayson" and enable_metacog and hasattr(model, "l1_gate"):
            model.l1_gate.enter_thresh = -1.0
            model.l1_gate.exit_thresh = -1.0
            model._mode_state = "metacog"
            model._plain_countdown = 0
            model._meta_countdown = 0
        elif mode == "plain" and enable_metacog and hasattr(model, "l1_gate"):
            model.l1_gate.enter_thresh = 2.0  # 永远不进入
            model._mode_state = "plain"
        
        out = model(ids, attention_mask=msk, enable_metacog=enable_metacog, return_meta=True)
        
        # 计算损失
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
        
        # 收集统计
        ss = out.get("switch_stats", {})
        switches += int(ss.get("switches", 0))
        plain_steps += int(ss.get("plain_steps", 0))
        meta_steps += int(ss.get("meta_steps", 0))
        
        ds = out.get("last_dilemma_score", None)
        if ds is not None:
            dilemma_scores.append(float(ds))
        
        tf = out.get("ctrl_tf", None)
        if tf is not None and isinstance(tf, torch.Tensor):
            temp_factors.extend(tf.detach().cpu().reshape(-1).tolist())
        
        tf_raw = out.get("ctrl_tf_raw_logit", None)
        if tf_raw is not None and isinstance(tf_raw, torch.Tensor):
            tf_raw_logits.extend(tf_raw.detach().cpu().reshape(-1).tolist())
    
    # 计算指标
    avg_loss = total_loss / max(1e-9, total_count)
    ppl = math.exp(min(20, avg_loss))
    total_forward = plain_steps + meta_steps
    plain_pct = (plain_steps / total_forward) if total_forward > 0 else 1.0
    meta_pct = 1.0 - plain_pct
    
    # 困境分数统计
    score_mean = float(sum(dilemma_scores) / len(dilemma_scores)) if dilemma_scores else float("nan")
    score_std = float((sum((x - score_mean)**2 for x in dilemma_scores) / len(dilemma_scores))**0.5) if dilemma_scores else float("nan")
    
    # 温度因子统计
    tf_mean = float(sum(temp_factors) / len(temp_factors)) if temp_factors else float("nan")
    tf_std = float((sum((x - tf_mean)**2 for x in temp_factors) / len(temp_factors))**0.5) if temp_factors else float("nan")
    
    # Controller 原始 logit 统计
    ctrl_std = float(sum(tf_raw_logits) / len(tf_raw_logits)) if tf_raw_logits else float("nan")
    if len(tf_raw_logits) >= 2:
        ctrl_std = float(sum(abs(x - ctrl_std) for x in tf_raw_logits) / len(tf_raw_logits))
    
    return {
        "ppl": ppl,
        "ce_loss": avg_loss,
        "switches": switches,
        "plain_steps": plain_steps,
        "meta_steps": meta_steps,
        "total_forward": total_forward,
        "plain_pct": plain_pct,
        "meta_pct": meta_pct,
        "dilemma_score_mean": score_mean,
        "dilemma_score_std": score_std,
        "temp_factor_mean": tf_mean,
        "temp_factor_std": tf_std,
        "ctrl_std": ctrl_std,
        "mode": mode,
        "enable_metacog": enable_metacog,
    }


def evaluate_all_modes(model, dl, device):
    """评估所有模式并返回对比结果"""
    results = {}
    
    # Plain 模式（关闭元认知）
    print("  Evaluating: plain mode (no metacog)")
    results["plain"] = evaluate_model(model, dl, device, enable_metacog=False)
    
    # Always-on 模式（始终开启元认知）
    print("  Evaluating: alwayson mode")
    results["alwayson"] = evaluate_model(model, dl, device, enable_metacog=True, mode="alwayson")
    
    # Conditional 模式（条件激活）
    print("  Evaluating: conditional mode")
    results["conditional"] = evaluate_model(model, dl, device, enable_metacog=True, mode="conditional")
    
    return results


def print_results(results):
    """打印结构化评估报告"""
    print("\n" + "=" * 75)
    print("MODEL EVALUATION REPORT")
    print("=" * 75)
    
    # 表头
    print("\n  {:<15}  {:>8}  {:>8}  {:>6}  {:>8}  {:>8}  {:>10}".format(
        "Mode", "PPL", "CE Loss", "Switch", "Plain%", "Meta%", "Dilemma"
    ))
    print("  " + "-" * 70)
    
    for mode, metrics in results.items():
        print("  {:<15}  {:>8.4f}  {:>8.4f}  {:>6d}  {:>8.2f}%  {:>8.2f}%  {:>10.4f}".format(
            mode,
            metrics["ppl"],
            metrics["ce_loss"],
            metrics["switches"],
            metrics["plain_pct"] * 100,
            metrics["meta_pct"] * 100,
            metrics["dilemma_score_mean"],
        ))
    
    # 对比分析
    print("\n" + "-" * 75)
    print("  COMPARISON ANALYSIS")
    print("-" * 75)
    
    plain = results["plain"]
    alwayson = results["alwayson"]
    conditional = results["conditional"]
    
    print("  PPL Delta (alwayson vs plain): +{:.2f}%".format(
        (alwayson["ppl"] - plain["ppl"]) / plain["ppl"] * 100
    ))
    print("  PPL Delta (conditional vs plain): +{:.2f}%".format(
        (conditional["ppl"] - plain["ppl"]) / plain["ppl"] * 100
    ))
    print("  PPL Delta (conditional vs alwayson): {:.2f}%".format(
        (conditional["ppl"] - alwayson["ppl"]) / alwayson["ppl"] * 100
    ))
    
    print("\n  Controller Analysis:")
    print("    alwayson temp_factor: {:.4f} ± {:.4f}".format(
        alwayson["temp_factor_mean"], alwayson["temp_factor_std"]
    ))
    print("    conditional temp_factor: {:.4f} ± {:.4f}".format(
        conditional["temp_factor_mean"], conditional["temp_factor_std"]
    ))
    print("    conditional ctrl_std: {:.6f}".format(conditional["ctrl_std"]))
    
    print("\n  Mode Switching (conditional):")
    print("    Total switches: {}".format(conditional["switches"]))
    print("    Meta mode usage: {:.2f}%".format(conditional["meta_pct"] * 100))
    
    print("\n" + "=" * 75)


def load_model(checkpoint_path, device):
    """加载模型检查点"""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        cfg = checkpoint['config']
    elif 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
    else:
        # 默认配置
        cfg = MetaCogXConfig(
            d_model=128,
            d_meta=32,
            d_aware=16,
            num_layers=4,
            num_heads=4,
            d_ffn=512,
            vocab_size=260,
            max_seq_len=64,
        )
    
    model = MetaCogXModel(cfg, enable_metacog=True).to(device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("  Warning: No model state dict found in checkpoint")
    
    # 加载 L1 Gate 检查点（如果存在）
    if 'l1_gate_state_dict' in checkpoint:
        if hasattr(model, 'l1_gate'):
            model.l1_gate.load_state_dict(checkpoint['l1_gate_state_dict'])
            print("  Loaded L1 Gate state")
    
    # 设置阈值（如果提供）
    if 'enter_thresh' in checkpoint:
        model.enter_thresh = checkpoint['enter_thresh']
    if 'exit_thresh' in checkpoint:
        model.exit_thresh = checkpoint['exit_thresh']
    
    print(f"  Model loaded: d_model={cfg.d_model}, num_layers={cfg.num_layers}, num_heads={cfg.num_heads}")
    print(f"  Enable metacog: {model.enable_metacog}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaCog-X model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of evaluation samples")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()
    
    print("=" * 75)
    print("MetaCog-X Unified Evaluation Script")
    print("=" * 75)
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Evaluation samples: {args.num_samples}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    
    # 构建数据集
    tok = ByteTokenizer(max_len=64)
    texts = build_texts(n=args.num_samples, seed=123)
    ds = ByteDataset(texts, tok, 64)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    
    # 加载模型或创建新模型
    if args.checkpoint and os.path.exists(args.checkpoint):
        model = load_model(args.checkpoint, args.device)
    else:
        print("\nNo checkpoint provided, creating fresh model")
        cfg = MetaCogXConfig(
            d_model=128,
            d_meta=32,
            d_aware=16,
            num_layers=4,
            num_heads=4,
            d_ffn=512,
            vocab_size=260,
            max_seq_len=64,
        )
        model = MetaCogXModel(cfg, enable_metacog=True).to(args.device)
    
    # 执行评估
    print("\nRunning evaluation...")
    results = evaluate_all_modes(model, dl, args.device)
    
    # 打印报告
    print_results(results)
    
    # 保存结果（如果指定）
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()