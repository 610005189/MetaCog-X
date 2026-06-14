import sys, torch
from config import MetaCogXConfig
from models import MetaCogXModel
from train import Trainer
from data.dataset import SimpleTextDataset, DummyTokenizer, create_dataloader

config = MetaCogXConfig(d_model=256, d_meta=32, d_aware=16, num_layers=4, num_heads=4, max_seq_len=32, d_ffn=1024)
model = MetaCogXModel(config, enable_metacog=True)
tok = DummyTokenizer(config.vocab_size)
texts = ["The quick brown fox jumps over the lazy dog."] * 8
ds = SimpleTextDataset(texts, tok, max_length=32)
loader = create_dataloader(ds, batch_size=4, shuffle=False)
trainer = Trainer(model=model, config={"lr": 1e-4, "weight_decay": 0.01, "alpha_meta": 0.01, "beta_aware": 0.005})

first_batch = next(iter(loader))
m0 = trainer.train_step(first_batch)
first_loss = m0['loss']
print(f"step0 total={m0['loss']:.4f} ce={m0.get('loss_ce', float('nan')):.4f} meta={m0.get('loss_meta', float('nan')):.4f} aware={m0.get('loss_aware', float('nan')):.4f}")

last_loss = first_loss
for i in range(20):
    mb = next(iter(loader))
    m = trainer.train_step(mb)
    last_loss = m['loss']

print(f"step{21} total={last_loss:.4f}")
print(f"[INFO] Task4 aux losses drop: {first_loss:.4f} -> {last_loss:.4f}")
