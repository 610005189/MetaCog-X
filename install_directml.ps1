# MetaCog-X: Python 3.11 + torch 2.3.1 + torch-directml
# 用法：右键 → 以管理员身份运行 PowerShell → 执行：
#   powershell -ExecutionPolicy Bypass -File install_directml.ps1
# 或直接：复制整段内容粘贴到管理员 PowerShell

$py = "C:\Users\wiggin\AppData\Local\Programs\Python\Python311\python.exe"
if (!(Test-Path $py)) {
    Write-Host "[ERROR] Python 3.11 not found at $py" -ForegroundColor Red
    Write-Host "        py -0p 确认 3.11 的实际路径，改脚本里的 `$py 变量" -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Upgrade pip..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip --quiet --disable-pip-version-check

Write-Host "[2/5] Install torch 2.3.1 + CPU wheels (清华大学镜像 + PyTorch 官方)..." -ForegroundColor Cyan
& $py -m pip install --user --no-cache-dir --timeout 600 `
    intel-openmp==2021.4.0 tbb==2021.13.1 mkl==2021.4.0 numpy==1.26.4 pillow==10.3.0 `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple

& $py -m pip install --user --no-cache-dir --timeout 600 `
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 `
    --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FALLBACK] Try install torch from PyPI default index (might not find 2.3.1)..." -ForegroundColor Yellow
    & $py -m pip install --user torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --timeout 600
}

Write-Host "[3/5] Install torch-directml 0.2.5.dev240914..." -ForegroundColor Cyan
& $py -m pip install --user --no-cache-dir --timeout 600 `
    torch-directml==0.2.5.dev240914 `
    --extra-index-url https://pypi.org/simple `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FALLBACK] Manual download + install latest wheel..." -ForegroundColor Yellow
    curl.exe -L --connect-timeout 30 -o "$env:TEMP\torch_directml.whl" `
        "https://files.pythonhosted.org/packages/84/8b/00528e6c75e030cc5f1fc1d08c58c46ecdbec9cd406b1dfd03023e3af4aa/torch_directml-0.2.5.dev240914-cp311-cp311-win_amd64.whl"
    & $py -m pip install --user "$env:TEMP\torch_directml.whl" --no-deps
}

Write-Host "[4/5] Install project dependencies..." -ForegroundColor Cyan
& $py -m pip install --user matplotlib tqdm tiktoken datasets accelerate tensorboard --timeout 300 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "[5/5] Verify..." -ForegroundColor Cyan
& $py -c "
import torch, numpy
print(f'  Python: {__import__(\"sys\").version}')
print(f'  torch: {torch.__version__}  cuda: {torch.cuda.is_available()}  threads: {torch.get_num_threads()}')
try:
    import torch_directml
    dev = torch_directml.device(0)
    x = torch.randn(4096, 4096).to(dev)
    y = x @ x
    print(f'  torch_directml OK  device={dev}  matmul={y.shape}')
except Exception as e:
    print(f'  torch_directml FAIL: {e}')
print('  Done. Run: py -3.11 -m runs.run_ab_v2')
"

Write-Host "[DONE] All done. Now run:" -ForegroundColor Green
Write-Host "  cd d:\Projects\MetaCog-X" -ForegroundColor Green
Write-Host "  py -3.11 runs/run_ab_v2.py" -ForegroundColor Green
