import sys, os, traceback

try:
    print("Step A: loading dataset only")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    print("Step B: importing hf_dataset")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from data import hf_dataset
    print("Step C: calling load_wikitext_dataset with max_train_samples=5")
    ds = hf_dataset.load_wikitext_dataset("train", "data", tok, 32, 5)
    print(f"  returned len={len(ds)}")
    print("Step D: accessing first sample")
    ids, msk = ds[0]
    print(f"  ids shape={ids.shape} dtype={ids.dtype}")
    print("Step E: accessing all 5 samples (full pre-tokenize)")
    for i in range(len(ds)):
        _ = ds[i]
        print(f"  sample {i} OK")
    print("Step F: dataloader batch of 2")
    from torch.utils.data import DataLoader
    import torch
    def collate_fn(batch):
        ids = torch.stack([b[0] for b in batch])
        msk = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "attention_mask": msk}
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for i, b in enumerate(dl):
        print(f"  batch {i}: ids={b['input_ids'].shape}")
    print("\n[PASS] dataset all OK")
except Exception as e:
    print("CRASH:", repr(e))
    traceback.print_exc()
