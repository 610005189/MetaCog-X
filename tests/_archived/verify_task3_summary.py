import csv, math, os, sys
sys.path.insert(0, r'd:\Projects\MetaCog-X')

def read_csv(p):
    rows = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            row["step"] = int(row["step"])
            row["valid_ppl"] = float(row["valid_ppl"])
            rows.append(row)
    return rows

gpt = read_csv(r'd:\Projects\MetaCog-X\runs\gpt_300.csv')
met = read_csv(r'd:\Projects\MetaCog-X\runs\metacog_300.csv')

def best_ppl(rows): return min(r['valid_ppl'] for r in rows)
def final_ppl(rows): return rows[-1]['valid_ppl']

gpt_best, gpt_final = best_ppl(gpt), final_ppl(gpt)
met_best, met_final = best_ppl(met), final_ppl(met)

print(f"\n{'='*60}")
print(f"A/B COMPARISON  (steps={met[-1]['step']})  [short-run=300]")
print(f"{'='*60}")
print(f"{'Variant':<12} {'best_ppl':>10} {'final_ppl':>10} {'params':>12}")
print(f"{'-'*60}")
print(f"{'GPT':<12} {gpt_best:>10.2f} {gpt_final:>10.2f} {gpt[-1]['params']:>12}")
print(f"{'MetaCog-X':<12} {met_best:>10.2f} {met_final:>10.2f} {met[-1]['params']:>12}")

delta_ppl = met_best - gpt_best
delta_log_ppl = math.log(met_best) - math.log(gpt_best)
if abs(delta_log_ppl) < 0.02:
    winner = "draw"
elif delta_log_ppl < 0:
    winner = "metacog"
else:
    winner = "gpt"

print(f"\ndelta_ppl (Meta - GPT) = {delta_ppl:+.2f}")
print(f"delta_log_ppl          = {delta_log_ppl:+.4f}")
print(f">>> WINNER = {winner} (draw if |delta_log_ppl|<0.02) <<<")
print(f"\nDelta % ppl (MetaCog better when negative): "
      f"{(met_best - gpt_best) / gpt_best * 100:+.2f}%")

try:
    import torch
    from config import MetaCogXConfig
    from models import MetaCogXModel
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    def gen(ckpt, variant, prompt):
        cfg = MetaCogXConfig(
            d_model=256, d_meta=32, d_aware=16, num_layers=4, num_heads=4,
            max_seq_len=128, d_ffn=1024, vocab_size=tok.vocab_size,
        )
        model = MetaCogXModel(cfg, enable_metacog=(variant == "metacog")).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=False)["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        ids = torch.tensor([tok.encode(prompt)]).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, temperature=0.7, top_k=30, verbose=False)
        return tok.decode(out[0].tolist())

    print(f"\n--- Generation side-by-side ---")
    for prompt in ["The meaning of life is", "Artificial intelligence", "The best way to"]:
        g = gen(r'd:\Projects\MetaCog-X\runs\gpt_300.pt', "gpt", prompt)
        m = gen(r'd:\Projects\MetaCog-X\runs\metacog_300.pt', "metacog", prompt)
        print(f"\nPROMPT: {prompt}")
        print(f"  GPT:     {g}")
        print(f"  MetaX:   {m}")
except Exception as e:
    print(f"\n[gen section skipped: {type(e).__name__}: {e}]")

print("\n[DONE]")
