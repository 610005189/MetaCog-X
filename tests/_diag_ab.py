import sys, os, subprocess, traceback

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "ab_trainer.py")
script = os.path.abspath(script)
print("script=", script)

try:
    r = subprocess.run(
        [sys.executable, script,
         "--variant", "gpt", "--steps", "10", "--d_model", "128",
         "--num_layers", "2", "--num_heads", "4", "--batch_size", "2",
         "--max_seq_len", "32", "--max_train_samples", "50",
         "--eval_every", "999999"],
        cwd=os.path.dirname(script),
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    print("RC=", r.returncode)
    print("--- STDOUT ---")
    print(r.stdout[-2000:])
    print("--- STDERR ---")
    print(r.stderr[-2000:])
except Exception as e:
    print("OUTER ERR:", e)
    traceback.print_exc()
