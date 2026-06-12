import sys, os
sys.path.insert(0, r'd:\Projects\MetaCog-X')
os.environ['OMP_NUM_THREADS']='16'; os.environ['MKL_NUM_THREADS']='16'
import torch
from config import MetaCogXConfig
from models import MetaCogXModel
cfg = MetaCogXConfig(vocab_size=260, d_model=128, d_meta=32, d_aware=16, num_layers=4, num_heads=4, d_head=64, d_ffn=512, max_seq_len=64, use_flash_attn=False)
m = MetaCogXModel(cfg, enable_metacog=True)
print('has l1_gate attr?', hasattr(m, 'l1_gate'))
print('has layers[0].gate?', hasattr(m.layers[0], 'gate'))
print('has layers[0].metacog_head?', hasattr(m.layers[0], 'metacog_head'))
if hasattr(m.layers[0], 'gate'):
    g = m.layers[0].gate
    print('layers[0].gate type:', type(g))
    for attr in ['threshold', 'enter_thresh', 'exit_thresh', 'bias', 'enabled', 'mode_state', 'net']:
        print('  gate.%s: %s' % (attr, getattr(g, attr, 'N/A')))
ids = torch.randint(0, 256, (4, 64)); msk = torch.ones(4, 64)
o = m(ids, attention_mask=msk, enable_metacog=True)
print('output keys:', list(o.keys()))
print('mode_had_metacog shape:', o['mode_had_metacog'].shape if 'mode_had_metacog' in o else None)
print('mode_had_metacog sample:', o['mode_had_metacog'].flatten()[:10] if 'mode_had_metacog' in o else None)
print('last_dilemma_score shape:', o['last_dilemma_score'].shape if 'last_dilemma_score' in o else None)
print('last_dilemma_score sample:', o['last_dilemma_score'].flatten()[:10] if 'last_dilemma_score' in o else None)
print('switch_stats:', o.get('switch_stats'))
print('mode:', o.get('mode'))
