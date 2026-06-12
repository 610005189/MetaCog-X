import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logf = open("tests/_diag_log.txt", "w", encoding="utf-8")
def log(s):
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()

try:
    log("Step 1: tokenizer")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    log(f"  vocab_size={tok.vocab_size}")

    log("Step 2: dataset")
    from data.hf_dataset import load_wikitext_dataset
    ds = load_wikitext_dataset("train", "data", tok, 32, 5)
    log(f"  len={len(ds)}")
    ids, msk = ds[0]
    log(f"  first ids shape={ids.shape}")

    log("Step 3: model gpt variant")
    from config import MetaCogXConfig
    from models import MetaCogXModel
    import torch, torch.nn as nn
    cfg = MetaCogXConfig(d_model=128, d_meta=16, d_aware=8, num_layers=2, num_heads=4, max_seq_len=32, d_ffn=256, vocab_size=tok.vocab_size)
    model = MetaCogXModel(cfg, enable_metacog=False)
    log(f"  params={sum(p.numel() for p in model.parameters()):,}")

    log("Step 4: forward/back")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
        }
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for batch in dl:
        b_ids = batch["input_ids"]
        b_msk = batch["attention_mask"]
        opt.zero_grad()
        out = model(b_ids, attention_mask=b_msk, labels=None, return_meta=False, enable_metacog=False)
        lg = out["logits"]
        sl = lg[..., :-1, :].contiguous()
        la = b_ids[..., 1:].contiguous()
        mm = b_msk[..., 1:].contiguous()
        la = la.masked_fill(mm == 0, tok.pad_token_id)
        loss = nn.functional.cross_entropy(sl.view(-1, sl.size(-1)), la.view(-1), ignore_index=tok.pad_token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log(f"  step loss={loss.item():.4f}")

    log("Step 5: generate")
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tok.encode("The meaning of life is")])
        out = model.generate(input_ids, max_new_tokens=10, temperature=0.7, top_k=30, verbose=False)
    text = tok.decode(out[0].tolist())
    log(f"  gen OK: {text[:80]}")

    log("\n[PASS] all steps OK")
except Exception as e:
    log("CRASH: " + repr(e))
    traceback.print_exc(file=logf)
    logf.flush()

logf.close()
