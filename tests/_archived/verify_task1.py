"""Task 1 验证：TripleAttention 的 causal + padding mask。"""
import sys
sys.path.insert(0, r"d:\Projects\MetaCog-X")

import torch
from models.triple_attention import TripleAttention


def check_causal_only(attn, L, tol, name):
    """每个 query 行 i 对 key 位置 j>i 的权重和应为 0。

    attn: [B, H, L, L]
    """
    ok = True
    # 上三角 j>i
    mask_future = torch.triu(torch.ones(L, L, device=attn.device), diagonal=1).bool()  # [L, L]
    future_sum = attn[:, :, mask_future].sum(dim=-1)  # [B, H]
    m = future_sum.abs().max().item()
    print(f"  causal [{name}]: max |attn on j>i| = {m:.3e}")
    if m > tol:
        ok = False
    return ok


def check_padding_only(attn, padding_positions, tol, name):
    """padding 列上所有 query 的注意力和应为 0（排除 causal 自然为 0 的 i<j 情况）。

    attn: [B, H, L, L]
    padding_positions: 1D tensor，哪些 key 位置是 pad
    """
    L = attn.shape[-1]
    pad = torch.zeros(L, dtype=torch.bool, device=attn.device)
    pad[padding_positions] = True
    ok = True
    # 对每个 query i，遍历 j in padding_positions：若 j<=i 则不应有注意力
    # 构造 [L, L] mask：(j 是 pad) 且 (j <= i)
    row = torch.arange(L, device=attn.device).unsqueeze(1)     # [L,1]
    col = torch.arange(L, device=attn.device).unsqueeze(0)     # [1,L]
    key_is_pad = pad.unsqueeze(0)  # [1, L]
    # 对每个行 i，j 是 pad 且 j<=i（即 causal 允许的范围里但 j 是 pad）
    m_bool = key_is_pad & (col <= row)  # [L, L]
    if not m_bool.any():
        print(f"  padding [{name}]: no (j<=i & j=pad) entries, skip")
        return True
    pad_sum = attn[:, :, m_bool].sum(dim=-1)  # [B, H]
    v = pad_sum.abs().max().item()
    print(f"  padding [{name}]: max |attn on (j<=i & j=pad)| = {v:.3e}")
    if v > tol:
        ok = False
    return ok


def main():
    torch.manual_seed(0)

    ta = TripleAttention(d_model=256, d_meta=32, d_aware=16, num_heads=4)
    ta.eval()

    B, L = 2, 16
    content = torch.randn(B, L, 256)
    meta = torch.randn(B, L, 32)
    awareness = torch.randn(B, L, 16)

    tol = 1e-6
    all_ok = True

    # --- 子测试 1：causal mask 单独（mask 全 1） ---
    print("[1] causal-only (mask=全 1)")
    mask_no_pad = torch.ones(B, L)
    with torch.no_grad():
        ta(content, meta, awareness, mask=mask_no_pad)

    for name, attn in [("content", ta._last_attn_c),
                       ("meta", ta._last_attn_m),
                       ("awareness", ta._last_attn_a)]:
        all_ok &= check_causal_only(attn, L, tol, name)

    # --- 子测试 2：padding mask（最后两个位置 pad） ---
    print("\n[2] causal + padding (最后 2 位置 pad)")
    mask_pad = torch.ones(B, L)
    mask_pad[:, 14:] = 0
    with torch.no_grad():
        ta(content, meta, awareness, mask=mask_pad)

    for name, attn in [("content", ta._last_attn_c),
                       ("meta", ta._last_attn_m),
                       ("awareness", ta._last_attn_a)]:
        all_ok &= check_causal_only(attn, L, tol, name)
        all_ok &= check_padding_only(attn, torch.tensor([14, 15], device=attn.device), tol, name)

    # --- 子测试 3：mask=None ---
    print("\n[3] mask=None (纯 causal)")
    with torch.no_grad():
        ta(content, meta, awareness, mask=None)

    for name, attn in [("content", ta._last_attn_c),
                       ("meta", ta._last_attn_m),
                       ("awareness", ta._last_attn_a)]:
        all_ok &= check_causal_only(attn, L, tol, name)

    if all_ok:
        print("\n[PASS] Task1 causal + padding mask verified")
        return 0
    else:
        print("\n[FAIL] Task1 验证失败，请查看上面的 max 值")
        return 1


if __name__ == "__main__":
    sys.exit(main())
