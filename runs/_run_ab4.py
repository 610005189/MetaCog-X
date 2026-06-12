import torch, os, sys, time, math, csv, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from config import MetaCogXConfig
from models import MetaCogXModel
from data.hf_dataset import load_wikitext_dataset
from torch.utils.data import DataLoader
from training.losses import TotalLoss


def run(variant, alpha, beta, gamma, delta):
    tok = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    train_ds = load_wikitext_dataset('train', 'data', tok, 128)
    valid_ds = load_wikitext_dataset('validation', 'data', tok, 128)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        msk = torch.stack([b[1] for b in batch])
        return {'input_ids': ids, 'attention_mask': msk}

    bs = 4
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=collate, drop_last=False)

    cfg = MetaCogXConfig(d_model=256, d_meta=32, d_aware=16, num_layers=4,
                         num_heads=4, d_ffn=1024, max_seq_len=128, vocab_size=tok.vocab_size)
    model = MetaCogXModel(cfg, enable_metacog=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] enable_metacog=True params={params:,}")
    loss_fn = TotalLoss(alpha=alpha, beta=beta, gamma=gamma, delta=delta, ignore_index=tok.pad_token_id)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    csv_rows = []
    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()
    while step < 300:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        ids = batch['input_ids']; msk = batch['attention_mask']
        opt.zero_grad()
        out = model(ids, attention_mask=msk, labels=None, return_meta=True, enable_metacog=True)
        loss, comp = loss_fn(out['logits'], ids, out.get('meta'), out.get('awareness'),
                             ctrl_logits=model._last_ctrl_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            ce = comp["loss_ce"].item()
            if step % 50 == 0:
                model.eval(); vs = 0.0; vn = 0
                with torch.no_grad():
                    for vb in valid_loader:
                        vids = vb['input_ids']; vmsk = vb['attention_mask']
                        vo = model(vids, attention_mask=vmsk, labels=None,
                                   return_meta=False, enable_metacog=True)
                        sl = vo['logits'][..., :-1, :].contiguous()
                        la = vids[..., 1:].contiguous()
                        ma = vmsk[..., 1:].contiguous()
                        lm = la.masked_fill(ma == 0, tok.pad_token_id)
                        nll = torch.nn.functional.cross_entropy(
                            sl.view(-1, sl.size(-1)), lm.view(-1),
                            ignore_index=tok.pad_token_id, reduction='sum').item()
                        vs += nll; vn += ma.sum().item()
                ppl = math.exp(vs / max(1, vn))
                row = {"variant": variant, "step": step, "train_ce": ce,
                       "valid_ppl": ppl, "elapsed": elapsed, "params": params,
                       "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}
                csv_rows.append(row)
                print(f"[{variant}] step={step} ce={ce:.4f} valid_ppl={ppl:.2f} t={elapsed:.1f}s")
            else:
                print(f"[{variant}] step={step} ce={ce:.4f} t={elapsed:.1f}s")

    csv_path = os.path.join('runs', f'{variant}_300.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    torch.save({"model": model.state_dict()}, os.path.join('runs', f'{variant}_300.pt'))
    print(f"[{variant}] DONE -> {csv_path}")


if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    variants = [
        ("gpt",                 0.0, 0.0,   0.0,   0.0),
        ("metacog_aware_only",  0.0, 0.005, 0.0,   0.0),
        ("metacog_meta_only",   0.01, 0.0,   0.0,   0.0),
        ("metacog_full",        0.01, 0.005, 0.02,  0.005),
    ]
    for v in variants:
        print("\n" + "="*60)
        print(f"STARTING: {v[0]}  alpha={v[1]} beta={v[2]} gamma={v[3]} delta={v[4]}")
        print("="*60)
        run(*v)
