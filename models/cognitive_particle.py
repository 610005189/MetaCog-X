"""认知粒子生成器 (Cognitive Particle Generator)

将输入embedding分解为三个正交部分：
- content: 内容语义信息
- meta: 元认知状态
- awareness: 全局觉知向量
"""
import torch
import torch.nn as nn
from typing import Tuple


class CognitiveParticle(nn.Module):
    """认知粒子生成器

    将输入向量 x_emb [B, L, d_model] 分解为三个并行的向量空间。
    """

    def __init__(
        self,
        d_model: int,
        d_meta: int,
        d_aware: int,
        init_method: str = "linear"
    ):
        """
        Args:
            d_model: 内容向量维度
            d_meta: 元认知状态维度
            d_aware: 觉知维度
            init_method: 初始化方式，"linear"表示线性投影（推荐）
        """
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.d_aware = d_aware
        self.total_dim = d_model + d_meta + d_aware
        self.init_method = init_method

        # 线性投影：将d_model投影到三个子空间
        self.proj = nn.Linear(d_model, self.total_dim)

        # 可学习的缩放因子，增加表达能力
        self.content_scale = nn.Parameter(torch.ones(1))
        self.meta_scale = nn.Parameter(torch.ones(1))
        self.aware_scale = nn.Parameter(torch.ones(1))

    def forward(self, x_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_emb: [B, L, d_model] 输入embedding

        Returns:
            content: [B, L, d_model] 内容向量
            meta: [B, L, d_meta] 元认知状态
            awareness: [B, L, d_aware] 觉知向量
        """
        # 线性投影到三个子空间
        z = self.proj(x_emb)
        splits = [self.d_model, self.d_meta, self.d_aware]

        content, meta, awareness = torch.split(z, splits, dim=-1)

        # 应用可学习的缩放
        content = content * self.content_scale
        meta = meta * self.meta_scale
        awareness = awareness * self.aware_scale

        return content, meta, awareness

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_meta={self.d_meta}, d_aware={self.d_aware}, "
            f"init_method={self.init_method}"
        )
