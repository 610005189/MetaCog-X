import sys
sys.path.insert(0, r'd:\Projects\MetaCog-X')
from transformers import AutoTokenizer
print("[prep] loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
print(f"[prep] tokenizer ok, vocab_size={tok.vocab_size}", flush=True)

from data.hf_dataset import load_wikitext_dataset
print("[prep] loading datasets...", flush=True)
tr = load_wikitext_dataset("train", "data", tok, 128)
va = load_wikitext_dataset("validation", "data", tok, 128)
print(f"[prep] train={len(tr)} valid={len(va)}", flush=True)

import importlib, models, config
print("[prep] importing models...", flush=True)
from config import MetaCogXConfig
from models import MetaCogXModel
cfg = MetaCogXConfig(d_model=256, d_meta=32, d_aware=16, num_layers=4, num_heads=4, max_seq_len=128, d_ffn=1024, vocab_size=tok.vocab_size)
print("[prep] constructing gpt model...", flush=True)
m_gpt = MetaCogXModel(cfg, enable_metacog=False)
print("[prep] constructing metacog model...", flush=True)
m_met = MetaCogXModel(cfg, enable_metacog=True)
pg = sum(p.numel() for p in m_gpt.parameters())
pm = sum(p.numel() for p in m_met.parameters())
print(f"[prep] gpt params={pg:,} metacog params={pm:,}", flush=True)
print("[prep] DONE", flush=True)
