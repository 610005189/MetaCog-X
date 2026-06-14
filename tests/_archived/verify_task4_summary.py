"""MetaCog-X Task 4 / Task 2 summary aggregator.

Reads 4 CSVs produced by ab_trainer.py and prints a clean 4× variant ppl table,
plus winner ranking and ablation analysis.
"""
import csv, os, math

VARIANTS = {
    "gpt":                 {"label": "GPT (baseline)",    "alpha": 0.0, "beta": 0.000, "gamma": 0.00, "delta": 0.000},
    "metacog_aware_only":  {"label": "Aware only",         "alpha": 0.0, "beta": 0.005, "gamma": 0.00, "delta": 0.000},
    "metacog_meta_only":   {"label": "Meta only",          "alpha": 0.01, "beta": 0.000, "gamma": 0.00, "delta": 0.000},
    "metacog_full":        {"label": "MetaCog-X (full)",  "alpha": 0.01, "beta": 0.005, "gamma": 0.02, "delta": 0.005},
}

FILES = {
    "gpt":                 "runs/gpt_300.csv",
    "metacog_aware_only":  "runs/metacog_aware_only_300.csv",
    "metacog_meta_only":   "runs/metacog_meta_only_300.csv",
    "metacog_full":        "runs/metacog_full_300.csv",
}


def main():
    print("=" * 80)
    print("MetaCog-X Task 2 — A/B Ablation Summary")
    print("Variant setup: all 4 groups use enable_metacog=True (identical param count).")
    print("               Control variable is only alpha/beta/gamma/delta (aux-loss weights).")
    print("=" * 80)

    rows_by_variant = {}
    for key, path in FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows_by_variant[key] = rows

    if not rows_by_variant:
        print("NO CSV FOUND")
        return

    params = list(rows_by_variant.values())[0][0]["params"]

    # --- per-variant best ppl + final ppl ---
    summary = {}
    for key, rows in rows_by_variant.items():
        best_ppl = min(float(r["valid_ppl"]) for r in rows)
        final_ppl = float(rows[-1]["valid_ppl"])
        final_ce = float(rows[-1]["train_ce"])
        final_step = int(rows[-1]["step"])
        summary[key] = {
            "best_ppl": best_ppl,
            "final_ppl": final_ppl,
            "final_ce": final_ce,
            "final_step": final_step,
            "label": VARIANTS[key]["label"],
            "alpha": VARIANTS[key]["alpha"],
            "beta":  VARIANTS[key]["beta"],
            "gamma": VARIANTS[key]["gamma"],
            "delta": VARIANTS[key]["delta"],
        }

    # --- print 4× table ---
    print()
    print(f"Param count (identical across 4 variants) : {params}")
    print(f"Variant setup (all enable_metacog=True)")
    print()
    header = f"{'Variant':<24} {'Step':>4} {'Train CE':>9} {'Val PPL':>9} {'Best PPL':>9} {'α':>5} {'β':>5} {'γ':>5} {'δ':>5}"
    print(header)
    print("-" * len(header))
    for key, s in summary.items():
        print(f"{s['label']:<24} {s['final_step']:>4} {s['final_ce']:>9.4f} {s['final_ppl']:>9.2f} {s['best_ppl']:>9.2f} "
              f"{s['alpha']:>5.2f} {s['beta']:>5.3f} {s['gamma']:>5.2f} {s['delta']:>5.3f}")

    # --- ranking ---
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["best_ppl"])
    print()
    gpt_best = summary["gpt"]["best_ppl"]
    print("=== Ranking by best validation perplexity (lower is better) ===")
    for i, (key, s) in enumerate(ranked, 1):
        delta_ppl = s["best_ppl"] - gpt_best
        pct = delta_ppl / gpt_best * 100.0 if gpt_best > 0 else 0.0
        print(f"  #{i}. {s['label']:<24}  best_ppl={s['best_ppl']:.2f}  "
              f"(vs GPT baseline: delta={delta_ppl:+.2f} ppl, rel={pct:+.1f}%)")

    winner = ranked[0]
    baseline = summary["gpt"]
    diff_vs_gpt = baseline["best_ppl"] - winner[1]["best_ppl"]
    pct_vs_gpt = diff_vs_gpt / baseline["best_ppl"] * 100

    print()
    print("=== Winner ===")
    print(f"  {winner[1]['label']} (best_ppl={winner[1]['best_ppl']:.2f}) wins by {diff_vs_gpt:.2f} ppl ({pct_vs_gpt:+.1f}% vs GPT baseline).")

    # --- ablation analysis ---
    print()
    print("=== Ablation Analysis ===")

    aware_only = summary["metacog_aware_only"]
    meta_only  = summary["metacog_meta_only"]
    full       = summary["metacog_full"]
    gpt_s      = summary["gpt"]

    # Awareness contribution
    aware_delta = gpt_s["best_ppl"] - aware_only["best_ppl"]
    print(f"  Awareness-only (β alone) vs GPT baseline improvement  : {aware_delta:+.2f} ppl = {aware_delta / gpt_s['best_ppl'] * 100:+.1f}%")

    # Meta contribution
    meta_delta = gpt_s["best_ppl"] - meta_only["best_ppl"]
    print(f"  Meta-only (α alone)     vs GPT baseline improvement  : {meta_delta:+.2f} ppl = {meta_delta / gpt_s['best_ppl'] * 100:+.1f}%")

    # Full vs best single
    best_single = min(aware_only["best_ppl"], meta_only["best_ppl"])
    full_delta = best_single - full["best_ppl"]
    print(f"  Full (α+β+γ+δ) vs best single auxiliary gain         : {full_delta:+.2f} ppl = {full_delta / best_single * 100:+.1f}%")

    # Full vs GPT
    full_vs_gpt = gpt_s["best_ppl"] - full["best_ppl"]
    print(f"  Full (meta+aware+controller regs) vs GPT baseline    : {full_vs_gpt:+.2f} ppl = {full_vs_gpt / gpt_s['best_ppl'] * 100:+.1f}%")

    print()
    print("=== Aux Loss Weight Interpretation ===")
    print("  α = 0.01  Meta temporal-consistency   loss (相邻层 meta 一致)")
    print("  β = 0.005 Awareness self-prediction   loss (layer i 预测 layer i+1 awareness)")
    print("  γ = 0.02  Controller entropy bonus    (push ctrl softmax away from collapse)")
    print("  δ = 0.005 Layer meta diversity         (pull inter-layer meta centroids apart)")
    print()
    print("NOTE on fairness:")
    print("  All 4 variants use ENABLE_METACOG=True → identical parameter count (17,020,086).")
    print("  Variant = which auxiliary losses have non-zero gradient signal.")
    print("  'gpt' variant runs TripleAttention but alpha=beta=gamma=delta=0 → meta/awareness/controller branches")
    print("  still forward-pass but produce zero loss-grad → effectively pure-content CE training.")

    print()
    print("=== Data ===")
    print("  WikiText-2 raw-v1 S3 URL returned 403 in this environment.")
    print("  datasets.load_dataset('wikitext','wikitext-2-raw-v1') timed out (corporate net).")
    print("  Fallback: bundled-fallback with ~820 AI/NLP/Transformer sentences.")
    print("  So ppl numbers are not comparable with published WikiText-2 baselines.")
    print("  Relative ranking between variants IS meaningful (same seed + same data + same params).")


if __name__ == "__main__":
    main()
