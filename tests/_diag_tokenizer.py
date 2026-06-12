import sys, os, subprocess

for i in range(10):
    r = subprocess.run(
        [sys.executable, "-c",
         "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('gpt2', local_files_only=True); print('OK1')"],
        capture_output=True, text=True, timeout=60)
    print(f"i={i} rc={r.returncode} out={r.stdout.strip()[:60]} err={r.stderr.strip()[:80]}")
