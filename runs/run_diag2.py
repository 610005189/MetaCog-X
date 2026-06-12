import sys, os, subprocess
r = subprocess.run(
    [sys.executable, 'training/ab_trainer.py', '--variant', 'gpt',
     '--steps', '10', '--d_model', '128', '--num_layers', '2', '--num_heads', '4',
     '--batch_size', '2', '--max_seq_len', '32', '--max_train_samples', '50', '--eval_every', '999999'],
    cwd=r'D:\Projects\MetaCog-X', capture_output=True, text=True, timeout=180
)
with open('diag_out.txt', 'w', encoding='utf-8') as f:
    f.write('RC=' + str(r.returncode) + '\n')
    f.write('STDOUT:\n' + r.stdout + '\n')
    f.write('STDERR:\n' + r.stderr)
print('done')
