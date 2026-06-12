import sys, os, traceback, torch

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    print("Step 1: importing config")
    from config import MetaCogXConfig
    print("Step 2: importing models")
    from models import MetaCogXModel
    print("Step 3: importing losses")
    from training.losses import TotalLoss
    print("Step 4: importing tokenizer")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    print("Step 5: importing dataset")
    from data.hf_dataset import load_wikitext_dataset
    print("Step 6: loading train dataset (max 20 samples)")
    ds = load_wikitext_dataset("train", "data", tok, 32, 20)
    print(f"  got {len(ds)} samples, first_ids shape: {ds[0][0].shape}")
    print("Step 7: forward/back with gpt variant")
    cfg = MetaCogXConfig(d_model=128, d_meta=16, d_aware=8, num_layers=2, num_heads=4, max_seq_len=32, d_ffn=256, vocab_size=tok.vocab_size)
    model = MetaCogXModel(cfg, enable_metacog=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    from torch.utils.data import DataLoader
    ids_s = torch.stack([b[0] for b in ds])
    msk_s = torch.stack([b[1] for b in ds])
    print(f"  dataset shape: {ids_s.shape}")
    # first 5 batches of size 2
    import torch.nn as nn
    for i in range(5):
        b_ids = ids_s[i*2:(i+1)*2]
        b_msk = msk_s[i*2:(i+1)*2]
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
        print(f"  step {i+1} loss={loss.item():.4f}")
    print("Step 8: generation")
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tok.encode("The meaning of life is")])
        out = model.generate(input_ids, max_new_tokens=10, temperature=0.7, top_k=30, verbose=False)
    text = tok.decode(out[0].tolist())
    print(f"  gen OK: {text[:80]}")
    print("\n[PASS] all mini steps OK")
except Exception as e:
    print("CRASH:", e)
    traceback.print_exc()
