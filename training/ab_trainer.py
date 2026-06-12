"""MetaCog-X A/B evaluation trainer.

4 种 variant（全部 enable_metacog=True，参数量完全一致；
 唯一差异是 TotalLoss 的辅助损失权重 alpha/beta/gamma/delta）：
  - gpt:                 alpha=beta=gamma=delta=0（退化为纯 CE，meta/aware 无梯度信号）
  - metacog_aware_only:  beta=args.beta,  其余 0（awareness 辅助损失单独开）
  - metacog_meta_only:   alpha=args.alpha,其余 0（meta 辅助损失单独开）
  - metacog_full:        alpha/beta/gamma/delta 全开

固定: d_model=256, num_layers=4, num_heads=4, bs=4, seq=128, lr=2e-4, AdamW.
仅 seed / steps / variant / data 路径可变.
"""

import argparse, os, random, sys, time, math, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MetaCogXConfig
from models import MetaCogXModel
from data.hf_dataset import load_wikitext_dataset
from training.losses import TotalLoss
from transformers import AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    ids = torch.stack([b[0] for b in batch])
    msk = torch.stack([b[1] for b in batch])
    return {"input_ids": ids, "attention_mask": msk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["gpt", "metacog_aware_only", "metacog_meta_only", "metacog_full"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--d_meta", type=int, default=32)
    ap.add_argument("--d_aware", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--d_ffn", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_csv", default=None)
    ap.add_argument("--cache_dir", default="data")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--save_ckpt", default=None)
    ap.add_argument("--print_every", type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ab_trainer] variant={args.variant} seed={args.seed} device={device}")

    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    train_ds = load_wikitext_dataset(
        "train", args.cache_dir, tok, args.max_seq_len, args.max_train_samples)
    valid_ds = load_wikitext_dataset("validation", args.cache_dir, tok, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=False)

    d_ffn = args.d_ffn or (args.d_model * 4)
    cfg = MetaCogXConfig(
        d_model=args.d_model,
        d_meta=args.d_meta,
        d_aware=args.d_aware,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        d_ffn=d_ffn,
        vocab_size=tok.vocab_size,
    )

    # ---- 关键：全部 4 组 enable_metacog=True，参数量一致 ----
    enable_metacog = True
    model = MetaCogXModel(cfg, enable_metacog=enable_metacog).to(device)

    if args.variant == "gpt":
        alpha, beta, gamma, delta = 0.0, 0.0, 0.0, 0.0
    elif args.variant == "metacog_aware_only":
        alpha, beta, gamma, delta = 0.0, args.beta, 0.0, 0.0
    elif args.variant == "metacog_meta_only":
        alpha, beta, gamma, delta = args.alpha, 0.0, 0.0, 0.0
    elif args.variant == "metacog_full":
        alpha, beta, gamma, delta = args.alpha, args.beta, args.gamma, args.delta
    else:
        raise ValueError(f"unknown variant {args.variant}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  variant={args.variant} enable_metacog={enable_metacog}")
    print(f"  loss weights: alpha={alpha} beta={beta} gamma={gamma} delta={delta}")
    print(f"  total_params={total_params:,} trainable={trainable:,}")

    loss_fn = TotalLoss(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        ignore_index=tok.pad_token_id
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    csv_rows = []
    step = 0
    best_valid_ppl = float("inf")
    t0 = time.time()
    print(f"  training for {args.steps} steps, evaluating every {args.eval_every} steps...")

    train_iter = iter(train_loader)

    model.train()
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        out = model(ids, attention_mask=msk, labels=None, return_meta=True, enable_metacog=True)
        logits = out["logits"]
        meta = out.get("meta")
        aware = out.get("awareness")
        loss, comp = loss_fn(
            logits, ids, meta, aware,
            aware_pool_buffer=None,
            ctrl_logits=model._last_ctrl_logits
        )
        ce_val = comp["loss_ce"].item()
        meta_val = comp["loss_meta"].item()
        aware_val = comp["loss_aware"].item()
        total_val = comp["loss_total"].item()
        entropy_val = comp["entropy_bonus"].item() if "entropy_bonus" in comp else 0.0
        layerdiv_val = comp["layer_div"].item() if "layer_div" in comp else 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % args.print_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  step {step:>5}/{args.steps} | total={total_val:.4f} "
                  f"ce={ce_val:.4f} meta={meta_val:.6f} "
                  f"aware={aware_val:.6f} entropy={entropy_val:.6f} "
                  f"layer_div={layerdiv_val:.6f} | {elapsed:>5.1f}s")

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            valid_ce_sum = 0.0
            valid_tokens = 0
            with torch.no_grad():
                for vb in valid_loader:
                    vids = vb["input_ids"].to(device)
                    vmsk = vb["attention_mask"].to(device)
                    vout = model(vids, attention_mask=vmsk, labels=None,
                                  return_meta=False, enable_metacog=True)
                    vlogits = vout["logits"]
                    sl = vlogits[..., :-1, :].contiguous()
                    la = vids[..., 1:].contiguous()
                    ma = vmsk[..., 1:].contiguous()
                    la_masked = la.masked_fill(ma == 0, tok.pad_token_id)
                    nll = nn.functional.cross_entropy(
                        sl.view(-1, sl.size(-1)),
                        la_masked.view(-1),
                        ignore_index=tok.pad_token_id,
                        reduction='sum'
                    )
                    valid_ce_sum += nll.item()
                    valid_tokens += ma.sum().item()
            valid_ppl = math.exp(valid_ce_sum / max(1, valid_tokens))
            row = {
                "variant": args.variant,
                "step": step,
                "train_total": total_val,
                "train_ce": ce_val,
                "train_meta": meta_val,
                "train_aware": aware_val,
                "valid_ppl": valid_ppl,
                "elapsed_sec": time.time() - t0,
                "params": total_params,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
            }
            csv_rows.append(row)
            best_valid_ppl = min(best_valid_ppl, valid_ppl)
            print(f"  [eval] step={step} valid_ppl={valid_ppl:.2f} best_ppl={best_valid_ppl:.2f}")
            model.train()

    if args.save_csv:
        out_dir = os.path.dirname(os.path.abspath(args.save_csv)) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"  saved CSV -> {args.save_csv}")

    if args.save_ckpt:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_ckpt)
        print(f"  saved ckpt -> {args.save_ckpt}")

    final_ce = csv_rows[-1]["train_ce"] if csv_rows else float("nan")
    elapsed = time.time() - t0
    print(f"\n=== DONE variant={args.variant} ===")
    print(f"  final_train_CE={final_ce:.4f}")
    print(f"  best_valid_ppl={best_valid_ppl:.2f}")
    print(f"  total_steps={args.steps} elapsed={elapsed:.1f}s device={device}")

    model.eval()
    def gen(prompt):
        ids = tok.encode(prompt)
        if len(ids) > args.max_seq_len - 1:
            ids = ids[-(args.max_seq_len - 1):]
        input_ids = torch.tensor([ids]).to(device)
        gen_kwargs = dict(max_new_tokens=min(10, args.max_seq_len - len(ids) - 1), temperature=0.7, top_k=30, verbose=False)
        if gen_kwargs["max_new_tokens"] < 1:
            gen_kwargs["max_new_tokens"] = 1
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        return tok.decode(out[0].tolist())

    try:
        p1 = gen("The meaning of life is")
        p2 = gen("Artificial intelligence")
        print(f"  gen1: {p1}")
        print(f"  gen2: {p2}")
    except Exception as e:
        print(f"  [gen skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
