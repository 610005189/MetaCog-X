"""三重注意力机制 (Triple Attention) — v2 带三种融合模式 + DMN h_self 注入

同时计算基于 content、meta、awareness 三个向量空间的注意力，
然后融合为统一的输出。

fusion 控制 meta/awareness 分支如何与 content 分支融合：
  - additive_bias       : 当前/原有实现；meta/aware 各自算 attn 再加到 Vc（等价于 concat-proj 的简化版）
  - multiplicative_gate : logits_c * (1 + sigmoid(gate_m)) * (1 + sigmoid(gate_a))，在 QKV=content 路径上的 logits 级乘法门
  - concat_proj         : 标准 attn_weights 后把 [attn_c, attn_m, attn_a] concat -> Linear 新的 attn_weights，再乘 Vc

注入 h_self_proj (来自 DMN)：对 Vc 做 h_self_proj 投影加到 V 上：Vc = Vc + h_self_proj
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TripleAttention(nn.Module):
    """三重注意力机制"""

    VALID_FUSION = ("additive_bias", "multiplicative_gate", "concat_proj")

    def __init__(
        self,
        d_model: int,
        d_meta: int,
        d_aware: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion: str = "additive_bias",
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        assert fusion in self.VALID_FUSION, (
            f"fusion must be one of {self.VALID_FUSION}, got '{fusion}'"
        )

        self.d_model = d_model
        self.d_meta = d_meta
        self.d_aware = d_aware
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)
        self.fusion = fusion

        self.q_proj_c = nn.Linear(d_model, d_model)
        self.k_proj_c = nn.Linear(d_model, d_model)
        self.v_proj_c = nn.Linear(d_model, d_model)

        self.q_proj_m = nn.Linear(d_meta, d_model)
        self.k_proj_m = nn.Linear(d_meta, d_model)

        self.q_proj_a = nn.Linear(d_aware, d_model)
        self.k_proj_a = nn.Linear(d_aware, d_model)

        # 融合输出投影（additive_bias 也用，保持原有的 concat 投影）
        self.out_proj = nn.Linear(d_model, d_model)
        self.fusion_linear = nn.Linear(d_model * 3, d_model)

        # multiplicative_gate 专属：meta / aware 各自一个 gate 投影到 d_head 再 sigmoid
        if fusion == "multiplicative_gate":
            self.meta_gate = nn.Linear(d_meta, d_model)
            self.aware_gate = nn.Linear(d_aware, d_model)

        # concat_proj 专属：attn_weights concat 后 Linear 融合（softmax 归一化）
        if fusion == "concat_proj":
            # attn weights shape [B,H,L,L]; 三分支 concat -> Linear 沿最后一维(L)不工作
            # 我们改为对 content logits 做 concat(meta_proj, aware_proj) -> Linear 融合 logits
            self.logits_fuse = nn.Linear(d_model * 3, d_model)

        # DMN 投影：h_self [B, 16] -> [B, H, L, d_head] 在 Vc 路径上加性注入
        self.h_self_proj = nn.Linear(16, d_model)

        self.dropout = nn.Dropout(dropout)

        self._last_attn_c: Optional[torch.Tensor] = None
        self._last_attn_m: Optional[torch.Tensor] = None
        self._last_attn_a: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def _build_attention_mask(
        self,
        mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        causal = self._causal_mask(seq_len, device)
        causal_add = (1.0 - causal) * (-1e9)

        if mask is None:
            return causal_add

        m = mask
        if m.dim() == 2:
            padding = m.float().unsqueeze(1).unsqueeze(2)
            padding_add = (1.0 - padding) * (-1e9)
            total = causal_add + padding_add
        elif m.dim() == 3:
            padding_add = (1.0 - m.float()).unsqueeze(2) * (-1e9)
            total = causal_add + padding_add
        elif m.dim() == 4:
            total = causal_add + m.float()
        else:
            total = causal_add

        return total

    def forward(
        self,
        content: torch.Tensor,
        meta: torch.Tensor,
        awareness: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temp_factor: Optional[torch.Tensor] = None,
        h_self: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            content: [B, L, d_model]
            meta: [B, L, d_meta]
            awareness: [B, L, d_aware]
            mask: [B, L] or 3D/4D
            temp_factor: [B, 1] optional temperature factor (scale)
            h_self: [B, 16] optional DMN hidden state injected into V

        Returns:
            output: [B, L, d_model]
        """
        B, L, _ = content.shape
        device = content.device

        attn_mask = self._build_attention_mask(mask, L, device)

        Qc = self._split_heads(self.q_proj_c(content))
        Kc = self._split_heads(self.k_proj_c(content))
        Vc = self._split_heads(self.v_proj_c(content))

        # --- DMN 注入 V 路径 ---
        if h_self is not None:
            hp = self.h_self_proj(h_self)  # [B, d_model]
            hp = hp.view(B, self.num_heads, 1, self.d_head)  # [B, H, 1, d_head]
            Vc = Vc + hp  # 广播到 [B, H, L, d_head]

        if temp_factor is not None:
            scale = self.scale * temp_factor.view(B, 1, 1, 1)
        else:
            scale = self.scale

        logits_c = torch.matmul(Qc, Kc.transpose(-2, -1)) / scale

        if self.fusion == "multiplicative_gate":
            g_m = torch.sigmoid(self.meta_gate(meta))      # [B, L, d_model]
            g_a = torch.sigmoid(self.aware_gate(awareness))
            g_m = self._split_heads(g_m)                  # [B, H, L, d_head]
            g_a = self._split_heads(g_a)
            # 在 QKV=content 上做 logits 级乘法门（沿 d_head 最后一维 broadcast 回 L'）
            g_m_mean = g_m.mean(dim=-1, keepdim=True)     # [B, H, L, 1]
            g_a_mean = g_a.mean(dim=-1, keepdim=True)
            g_m_logits = g_m_mean.transpose(-2, -1)       # [B, H, 1, L]
            g_a_logits = g_a_mean.transpose(-2, -1)
            logits_c = logits_c * (1.0 + g_m_logits) * (1.0 + g_a_logits)

        if self.fusion == "concat_proj":
            # content logits 作为 base，再叠 concat(meta_proj, aware_proj) 的 logits 修正
            meta_proj = self.q_proj_m(meta)                # [B, L, d_model] (当作 d_model 投影)
            aware_proj = self.q_proj_a(awareness)
            fused = self.logits_fuse(torch.cat([content, meta_proj, aware_proj], dim=-1))
            fused = self._split_heads(fused)              # [B, H, L, d_head]
            # 沿最后一维 mean -> [B,H,L,1] -> transpose [B,H,1,L] 加到 logits
            fused_logit = fused.mean(dim=-1, keepdim=True).transpose(-2, -1)
            logits_c = logits_c + fused_logit

        logits_c = logits_c + attn_mask
        attn_c = F.softmax(logits_c, dim=-1)
        attn_c = self.dropout(attn_c)
        out_c = torch.matmul(attn_c, Vc)
        out_c = out_c.transpose(1, 2).reshape(B, L, self.d_model)

        # meta / awareness 分支注意力（additive_bias 保留原分支独立 attn 再 concat 融合）
        Qm = self._split_heads(self.q_proj_m(meta))
        Km = self._split_heads(self.k_proj_m(meta))
        logits_m = torch.matmul(Qm, Km.transpose(-2, -1)) / self.scale + attn_mask
        attn_m = F.softmax(logits_m, dim=-1)
        attn_m = self.dropout(attn_m)
        out_m = torch.matmul(attn_m, Vc)
        out_m = out_m.transpose(1, 2).reshape(B, L, self.d_model)

        Qa = self._split_heads(self.q_proj_a(awareness))
        Ka = self._split_heads(self.k_proj_a(awareness))
        logits_a = torch.matmul(Qa, Ka.transpose(-2, -1)) / self.scale + attn_mask
        attn_a = F.softmax(logits_a, dim=-1)
        attn_a = self.dropout(attn_a)
        out_a = torch.matmul(attn_a, Vc)
        out_a = out_a.transpose(1, 2).reshape(B, L, self.d_model)

        # concat 融合
        fused = torch.cat([out_c, out_m, out_a], dim=-1)
        output = self.fusion_linear(fused)
        output = self.out_proj(output)
        output = self.dropout(output)

        self._last_attn_c = attn_c.detach()
        self._last_attn_m = attn_m.detach()
        self._last_attn_a = attn_a.detach()
        self._last_attn = attn_c.detach()

        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_meta={self.d_meta}, d_aware={self.d_aware}, "
            f"num_heads={self.num_heads}, fusion={self.fusion}"
        )
