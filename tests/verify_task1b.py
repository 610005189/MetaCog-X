import sys
sys.path.insert(0, r'd:\Projects\MetaCog-X')
from data.hf_dataset import load_wikitext_dataset, HFDataset
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
if t.pad_token is None:
    t.pad_token = t.eos_token; t.pad_token_id = t.eos_token_id

train = load_wikitext_dataset("train", tokenizer=t, max_length=64, max_train_samples=100)
valid = load_wikitext_dataset("validation", tokenizer=t, max_length=64)
ids, mask = train[0]
assert ids.shape[0] == 64
dec = t.decode(ids.tolist())
print(f"[PASS] train len={len(train)}, valid len={len(valid)}, first_decoded={dec[:40]}")
