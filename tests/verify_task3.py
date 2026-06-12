import sys, torch
from config import MetaCogXConfig
from models import MetaCogXModel, EnlightenmentTrigger

config = MetaCogXConfig(d_model=256, d_meta=32, d_aware=16, num_layers=4, num_heads=4, max_seq_len=32, d_ffn=1024)
model = MetaCogXModel(config, enable_metacog=True)
model.enlightenment_trigger.repeat_thresh = 1
model.enlightenment_trigger.entropy_thresh = 5.0

input_ids = torch.tensor([[4, 5, 6, 5, 5, 5]])
model.eval()
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=10, temperature=1.0, top_k=10, verbose=False)

hit = {'count': 0}
orig_fwd = model.enlightenment_trigger.forward
def patched_fwd(*args, **kwargs):
    hit['count'] += 1
    return orig_fwd(*args, **kwargs)
model.enlightenment_trigger.forward = patched_fwd
out = model.generate(input_ids, max_new_tokens=10, verbose=False)
assert hit['count'] > 0, f"trigger never called during generate"
print(f"[PASS] Task3 trigger called {hit['count']} times during generate")
