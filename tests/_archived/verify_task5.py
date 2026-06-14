import sys
sys.path.insert(0, r'd:\Projects\MetaCog-X')
import torch
try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not installed, skip")
    sys.exit(0)

t = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
enc = t.encode("The quick brown fox jumps over the lazy dog")
dec = t.decode(enc)
assert "quick" in dec or "fox" in dec, f"decode failed: {dec}"
assert t.vocab_size == 50257
print(f"[PASS] tokenizer roundtrip: vocab={t.vocab_size}, decoded={dec}")

from config import MetaCogXConfig
from models import MetaCogXModel
from data.hf_dataset import HFDataset

texts = ["The meaning of life is", "Artificial intelligence is the future"]
tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
ds = HFDataset(tok, texts, max_length=32)
ids, mask = ds[0]
assert ids.shape[0] == 32
assert mask.shape[0] == 32
dec2 = tok.decode(ids.tolist())
print(f"[PASS] dataset: shape={ids.shape}, decoded={dec2[:40]}")

config = MetaCogXConfig(d_model=256, d_meta=32, d_aware=16, num_layers=4, num_heads=4, max_seq_len=32, d_ffn=1024, vocab_size=t.vocab_size)
model = MetaCogXModel(config, enable_metacog=True)
prompt = "The quick"
prompt_ids = torch.tensor([tok.encode(prompt)])
out = model.generate(prompt_ids, max_new_tokens=8, temperature=1.0, top_k=20)
gen_text = tok.decode(out[0].tolist())
print(f"[PASS] generate(8 tokens): {gen_text}")
