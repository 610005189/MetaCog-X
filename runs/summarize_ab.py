import json
with open(r'd:\Projects\MetaCog-X\runs\ab_results_v3.json') as f:
    d = json.load(f)
print('Wall time: %.1fs = %.1fmin' % (d['wall_seconds'], d['wall_seconds']/60))
base_ppl = d['variants'][0]['final_ppl']
for v in d['variants']:
    ppl = v['final_ppl']
    loss = v['final_loss']
    delta = (ppl - base_ppl) / base_ppl * 100
    print('%-18s ppl=%.4f loss=%.4f delta_vs_plain=%+.2f%% switches=%d plain=%.2f ctrl_std=%.6f' % (
        v['name'], ppl, loss, delta, v['switches'], v['plain_pct'], v['ctrl_std']))
