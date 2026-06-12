import sys, os, subprocess
r = subprocess.run(
    [sys.executable, 'training/ab_trainer.py', '--variant', 'gpt',
     '--steps', '10', '--d_model', '128', '--num_layers', '2', '--num_heads', '4',
     '--batch_size', '2', '--max_seq_len', '32', '--max_train_samples', '50',
     '--eval_every', '999999'],
    cwd=r'D:\Projects\MetaCog-X', capture_output=True, text=True
)
print('RC=', r.returncode)
print('STDOUT:')
print(r.stdout)
print('STDERR:')
print(r.stderr)
