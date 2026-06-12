import os, sys, csv, torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MetaCogXConfig
from models import MetaCogXModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data.hf_dataset import load_wikitext_dataset


def load_model(ckpt_path, d_model=256, d_meta=32, d_aware=16, num_layers=4, num_heads=4, d_ffn=1024,
               vocab_size=50257, max_seq_len=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MetaCogXConfig(d_model=d_model, d_meta=d_meta, d_aware=d_aware, num_layers=num_layers, num_heads=num_heads,
                         d_ffn=d_ffn, vocab_size=vocab_size, max_seq_len=max_seq_len)
    model = MetaCogXModel(cfg, enable_metacog=True).to(device)
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)["model"]
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model, device


def ctrl_entropy(ctrl_logits):
    p = torch.softmax(ctrl_logits, dim=-1)
    return -(p * torch.log2(p + 1e-9)).sum(dim=-1).mean().item()


def run_probe(model, device, tok, loader, real_prompts, gibber_prompts):
    metas = []
    ctrls_logits = []
    ctrls = []
    awares = []
    for i, b in enumerate(loader):
        if i >= 5:
            break
        ids = b["input_ids"].to(device)
        msk = b["attention_mask"].to(device)
        with torch.no_grad():
            out = model(ids, attention_mask=msk, return_meta=True, enable_metacog=True)
        metas.append(out["meta"].detach().cpu())
        if getattr(model, "_last_ctrl_logits", None) is not None:
            ctrls_logits.append(model._last_ctrl_logits.detach().cpu())
        ctrl = out.get("ctrl") or getattr(model, "_last_ctrl_signals", None)
        if ctrl is not None and hasattr(ctrl, "temp_factor"):
            ctrls.append(ctrl.temp_factor.detach().cpu())
        awares.append(out["awareness"].detach().cpu())

    meta_batched = torch.cat(metas, dim=1)
    num_layers = meta_batched.shape[0]
    cos_mat = torch.zeros(num_layers, num_layers)
    for i in range(num_layers):
        for j in range(num_layers):
            vi = meta_batched[i].mean(dim=(0, 1))
            vj = meta_batched[j].mean(dim=(0, 1))
            cos_mat[i, j] = torch.nn.functional.cosine_similarity(vi.unsqueeze(0), vj.unsqueeze(0)).item()
    off_diag = cos_mat[torch.triu(torch.ones_like(cos_mat), diagonal=1).bool()].mean().item()

    if ctrls_logits:
        cl = torch.cat(ctrls_logits)
        entro = ctrl_entropy(cl)
    else:
        entro = float("nan")
    if ctrls:
        tfs = torch.cat(ctrls)
        tf_std = tfs.std().item()
        tf_mean = tfs.mean().item()
    else:
        tf_std, tf_mean = float("nan"), float("nan")

    def get_aware_batch(prompts):
        ids_list = [tok.encode(p) for p in prompts]
        max_l = max(len(x) for x in ids_list)
        ids = torch.stack([torch.nn.functional.pad(torch.tensor(x), (0, max_l - len(x)),
                                                    value=tok.pad_token_id) for x in ids_list])
        msk = (ids != tok.pad_token_id).long()
        with torch.no_grad():
            out = model(ids.to(device), attention_mask=msk.to(device), return_meta=True, enable_metacog=True)
        return out["awareness"][-1].mean(dim=1).detach().cpu()

    aw_real = get_aware_batch(real_prompts)
    aw_gib = get_aware_batch(gibber_prompts)
    aw_dists = [(aw_real[i] - aw_gib[j]).norm().item()
                for i in range(aw_real.shape[0]) for j in range(aw_gib.shape[0])]
    intra_real = [(aw_real[i] - aw_real[j]).norm().item()
                  for i in range(aw_real.shape[0]) for j in range(i + 1, aw_real.shape[0])]
    aw_ratio = np.mean(aw_dists) / (np.mean(intra_real) + 1e-6)

    meta_centroids = [m.mean(dim=(1, 2)).mean(dim=0) for m in metas]
    inter_batch = []
    for i in range(len(meta_centroids) - 1):
        inter_batch.append((meta_centroids[i + 1] - meta_centroids[i]).pow(2).mean().item())

    return {
        "layer_meta_cos_offdiag": off_diag,
        "controller_entropy_bits": entro,
        "temp_factor_mean": tf_mean,
        "temp_factor_std": tf_std,
        "awareness_inter_dist": np.mean(aw_dists),
        "awareness_intra_dist": np.mean(intra_real),
        "awareness_ratio": aw_ratio,
        "inter_batch_meta_mse": np.mean(inter_batch) if inter_batch else float("nan"),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    real_prompts = [
        "Artificial intelligence is",
        "The Transformer architecture",
        "Natural language processing",
        "Deep learning models have",
    ]
    gibber_prompts = [
        "asdf jkl; foo bar zzz",
        "qqqq wwww eeee rrr",
        "!!!! ???? |||| ****",
        "mno pqr stu vwx yz",
    ]

    ckpt_dir = r"d:\Projects\MetaCog-X\runs"
    variants = [
        ("GPT (α=β=γ=δ=0)",               os.path.join(ckpt_dir, "gpt_300.pt")),
        ("Aware only (β=0.005)",          os.path.join(ckpt_dir, "metacog_aware_only_300.pt")),
        ("Meta only (α=0.01)",            os.path.join(ckpt_dir, "metacog_meta_only_300.pt")),
        ("Full (α+β+γ+δ)",                os.path.join(ckpt_dir, "metacog_full_300.pt")),
    ]

    valid_ds = load_wikitext_dataset("validation", tokenizer=tok, max_length=128)
    def collate(batch):
        return {"input_ids": torch.stack([b[0] for b in batch]),
                "attention_mask": torch.stack([b[1] for b in batch])}
    loader = DataLoader(valid_ds, batch_size=4, shuffle=True, collate_fn=collate)

    results = []
    for label, ckpt in variants:
        print(f"\n[{label}] loading {os.path.basename(ckpt)}...")
        model, _ = load_model(ckpt)
        r = run_probe(model, device, tok, loader, real_prompts, gibber_prompts)
        results.append((label, r))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'=' * 125}")
    print("PROBE SUMMARY (4 variants, 300 steps)")
    print(f"{'=' * 125}")
    header = (f"{'Variant':<25} {'LayerMetaOffDiag':>16} {'CtrlEntropy(b)':>15} "
              f"{'TempFactorStd':>15} {'AwInter':>10} {'AwRatio':>10} {'BatchMetaMSE':>14}")
    print(header)
    print("-" * 125)
    for label, r in results:
        print(f"{label:<25} {r['layer_meta_cos_offdiag']:>16.4f} {r['controller_entropy_bits']:>15.4f} "
              f"{r['temp_factor_std']:>15.6f} {r['awareness_inter_dist']:>10.3f} "
              f"{r['awareness_ratio']:>10.2f} {r['inter_batch_meta_mse']:>14.8f}")

    print(f"\n--- Interpretations ---")
    max_ent = max(r["controller_entropy_bits"] for _, r in results if not np.isnan(r["controller_entropy_bits"]))
    min_ent = min(r["controller_entropy_bits"] for _, r in results if not np.isnan(r["controller_entropy_bits"]))
    print(f"Controller entropy range: {min_ent:.3f} ~ {max_ent:.3f} (max H = log2(3) = 1.58 bits)")
    for label, r in results:
        std = r["temp_factor_std"]
        if np.isnan(std):
            tag = "N/A"
        elif std < 0.01:
            tag = "COLLAPSED (std<0.01)"
        elif std < 0.1:
            tag = "PARTIALLY_DECOUPLED"
        else:
            tag = "WELL_DECOUPLED"
        print(f"  {label:<25} tf_std={std:>10.6f} -> {tag}")
    for label, r in results:
        c = r["layer_meta_cos_offdiag"]
        if c < 0.95:
            tag = "GOOD DIVERSITY (cos<0.95)"
        elif c < 0.99:
            tag = "WEAK DIVERSITY"
        else:
            tag = "NO DIVERSITY (cos>=0.99)"
        print(f"  {label:<25} layer_cos={c:>10.4f} -> {tag}")
    for label, r in results:
        rat = r["awareness_ratio"]
        if rat > 10:
            tag = "EXCELLENT DISTINCTION (ratio>10)"
        elif rat > 2:
            tag = "GOOD DISTINCTION (ratio>2)"
        elif rat > 1:
            tag = "BASIC DISTINCTION"
        else:
            tag = "POOR DISTINCTION"
        print(f"  {label:<25} aw_ratio={rat:>10.2f} -> {tag}")

    out_csv = r"d:\Projects\MetaCog-X\runs\probe_summary_4variants.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "layer_meta_cos_offdiag", "controller_entropy_bits", "temp_factor_mean",
                    "temp_factor_std", "awareness_inter_dist", "awareness_intra_dist", "awareness_ratio",
                    "inter_batch_meta_mse"])
        for label, r in results:
            w.writerow([label, r["layer_meta_cos_offdiag"], r["controller_entropy_bits"],
                        r["temp_factor_mean"], r["temp_factor_std"],
                        r["awareness_inter_dist"], r["awareness_intra_dist"],
                        r["awareness_ratio"], r["inter_batch_meta_mse"]])
    print(f"\nCSV saved -> {out_csv}")


if __name__ == "__main__":
    main()
