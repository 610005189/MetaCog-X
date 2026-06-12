"""认知Transformer层 (Cognitive Transformer Layer)

每个MetaCogXLayer包含：
1. 三重注意力 (TripleAttention)
2. 前馈网络 (FFN)
3. 残差连接和层归一化
4. Meta/Awareness 更新
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .triple_attention import TripleAttention


class MetaCogXLayer(nn.Module):
    """认知Transformer层

    整合三重注意力、FFN、残差归一化，同时维护meta和awareness的更新。
    采用Pre-LN（预层归一化）结构以提高训练稳定性。
    """

    def __init__(
        self,
        d_model: int,
        d_meta: int,
        d_aware: int,
        num_heads: int = 8,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1
    ):
        """
        Args:
            d_model: 内容向量维度
            d_meta: 元认知状态维度
            d_aware: 觉知维度
            num_heads: 注意力头数
            d_ffn: 前馈网络隐藏层维度
            dropout: 主dropout概率
            attn_dropout: 注意力dropout
            ffn_dropout: FFN dropout
        """
        super().__init__()

        # Pre-LN 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_meta = nn.LayerNorm(d_meta)
        self.norm_aware = nn.LayerNorm(d_aware)

        # 三重注意力
        self.triple_attn = TripleAttention(
            d_model=d_model,
            d_meta=d_meta,
            d_aware=d_aware,
            num_heads=num_heads,
            dropout=attn_dropout
        )

        # FFN (前馈网络)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(ffn_dropout)
        )

        # Meta 更新MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(d_meta, d_meta * 2),
            nn.GELU(),
            nn.Linear(d_meta * 2, d_meta)
        )

        # Awareness 更新MLP
        self.aware_mlp = nn.Sequential(
            nn.Linear(d_aware, d_aware * 2),
            nn.GELU(),
            nn.Linear(d_aware * 2, d_aware)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        content: torch.Tensor,
        meta: torch.Tensor,
        awareness: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temp_factor: Optional[torch.Tensor] = None,
        h_self: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content_norm = self.norm1(content)
        meta_norm = self.norm_meta(meta)
        aware_norm = self.norm_aware(awareness)

        attn_output = self.triple_attn(
            content_norm, meta_norm, aware_norm, mask, temp_factor, h_self
        )
        content = content + self.dropout(attn_output)

        # FFN + 残差连接
        ffn_output = self.ffn(self.norm1(content))
        content = content + ffn_output

        # 更新 meta（通过残差连接）
        meta = meta + self.dropout(self.meta_mlp(meta_norm))

        # 更新 awareness（通过残差连接）
        awareness = awareness + self.dropout(self.aware_mlp(aware_norm))

        return content, meta, awareness


class MetaCogXBlock(nn.Module):
    """MetaCogXBlock - Multi-head Meta Cognitive Block

    这是MetaCogXLayer的别名，保持命名一致性。
    """

    def __init__(self, config):
        super().__init__()
        self.layer = MetaCogXLayer(
            d_model=config.d_model,
            d_meta=config.d_meta,
            d_aware=config.d_aware,
            num_heads=config.num_heads,
            d_ffn=config.d_ffn,
            dropout=config.resid_dropout,
            attn_dropout=config.attn_dropout,
            ffn_dropout=config.ffn_dropout
        )

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)
