"""认知粒子生成器 (Cognitive Particle Generator) v2

将输入embedding分解为三个正交部分：
- content: 内容语义信息
- meta: 元认知状态
- awareness: 全局觉知向量

增强功能：
- 正交分解约束
- 动态缩放机制
- 自适应投影矩阵
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class OrthogonalityLoss(nn.Module):
    """正交性约束损失

    确保三个向量空间之间保持正交。
    """

    def __init__(self, lambda_ortho: float = 1.0, lambda_norm: float = 0.1):
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_norm = lambda_norm

    def forward(
        self,
        content: torch.Tensor,
        meta: torch.Tensor,
        awareness: torch.Tensor
    ) -> torch.Tensor:
        """
        计算正交性损失

        Args:
            content: [B, L, d_model]
            meta: [B, L, d_meta]
            awareness: [B, L, d_aware]

        Returns:
            正交性损失
        """
        B, L, _ = content.shape

        # 归一化向量
        content_norm = F.normalize(content, dim=-1)
        meta_norm = F.normalize(meta, dim=-1)
        aware_norm = F.normalize(awareness, dim=-1)

        # 计算两两之间的余弦相似度
        # content vs meta
        sim_cm = (content_norm * meta_norm).sum(dim=-1).abs().mean()
        # content vs awareness
        sim_ca = (content_norm * aware_norm).sum(dim=-1).abs().mean()
        # meta vs awareness
        sim_ma = (meta_norm * aware_norm).sum(dim=-1).abs().mean()

        # 正交性损失：希望相似度尽可能小
        ortho_loss = sim_cm + sim_ca + sim_ma

        # 归一化损失：希望向量有合理的范数
        content_var = content.norm(dim=-1).var()
        meta_var = meta.norm(dim=-1).var()
        aware_var = awareness.norm(dim=-1).var()
        norm_loss = content_var + meta_var + aware_var

        # 总损失
        total_loss = self.lambda_ortho * ortho_loss + self.lambda_norm * norm_loss

        return total_loss


class DynamicScalingController(nn.Module):
    """动态缩放控制器

    根据任务类型和上下文动态调整向量空间维度。
    """

    def __init__(
        self,
        base_dims: Dict[str, int],
        num_task_types: int = 4,
        max_scale: float = 2.0,
        min_scale: float = 0.5
    ):
        super().__init__()
        self.base_dims = base_dims
        self.num_task_types = num_task_types
        self.max_scale = max_scale
        self.min_scale = min_scale

        # 任务特定的维度调整
        self.task_scales = nn.Embedding(num_task_types, 3)  # content, meta, awareness
        nn.init.constant_(self.task_scales.weight, 0.0)

        # 上下文感知的维度调整
        self.context_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(
        self,
        task_id: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        计算动态缩放因子

        Args:
            task_id: 任务类型ID
            context_features: 上下文特征 [B, 64]

        Returns:
            缩放因子字典
        """
        scales = {
            "content": 1.0,
            "meta": 1.0,
            "awareness": 1.0
        }

        # 任务特定调整 - 保留张量以维持梯度流
        if task_id is not None:
            task_scale = self.task_scales(task_id)  # [B, 3]
            task_scale_adjusted = task_scale.sigmoid() * (self.max_scale - self.min_scale) + self.min_scale
            scales["content"] = scales["content"] * task_scale_adjusted[0, 0]
            scales["meta"] = scales["meta"] * task_scale_adjusted[0, 1]
            scales["awareness"] = scales["awareness"] * task_scale_adjusted[0, 2]

        # 上下文感知调整 - 保留张量以维持梯度流
        if context_features is not None:
            context_scale = self.context_encoder(context_features)  # [B, 3]
            context_scale_adjusted = context_scale.sigmoid() * (self.max_scale - self.min_scale) + self.min_scale
            scales["content"] = scales["content"] * context_scale_adjusted[0, 0]
            scales["meta"] = scales["meta"] * context_scale_adjusted[0, 1]
            scales["awareness"] = scales["awareness"] * context_scale_adjusted[0, 2]

        return scales


class CognitiveParticle(nn.Module):
    """认知粒子生成器

    将输入向量 x_emb [B, L, d_model] 分解为三个并行的向量空间。
    """

    def __init__(
        self,
        d_model: int,
        d_meta: int,
        d_aware: int,
        init_method: str = "linear",
        enable_orthogonality: bool = True,
        enable_dynamic_scaling: bool = True
    ):
        """
        Args:
            d_model: 内容向量维度
            d_meta: 元认知状态维度
            d_aware: 觉知维度
            init_method: 初始化方式，"linear"表示线性投影（推荐）
            enable_orthogonality: 是否启用正交性约束
            enable_dynamic_scaling: 是否启启动态缩放
        """
        super().__init__()
        self.d_model = d_model
        self.d_meta = d_meta
        self.d_aware = d_aware
        self.total_dim = d_model + d_meta + d_aware
        self.init_method = init_method
        self.enable_orthogonality = enable_orthogonality
        self.enable_dynamic_scaling = enable_dynamic_scaling

        # 线性投影：将d_model投影到三个子空间
        self.proj = nn.Linear(d_model, self.total_dim)

        # 可学习的缩放因子，增加表达能力
        self.content_scale = nn.Parameter(torch.ones(1))
        self.meta_scale = nn.Parameter(torch.ones(1))
        self.aware_scale = nn.Parameter(torch.ones(1))

        # 正交性损失模块 - 始终创建，未启用时设为None
        self.enable_orthogonality = enable_orthogonality
        if enable_orthogonality:
            self.ortho_loss = OrthogonalityLoss(lambda_ortho=1.0, lambda_norm=0.1)
        else:
            self.ortho_loss = None

        # 动态缩放控制器 - 始终创建，未启用时设为None
        self.enable_dynamic_scaling = enable_dynamic_scaling
        if enable_dynamic_scaling:
            self.scaling_controller = DynamicScalingController(
                base_dims={"content": d_model, "meta": d_meta, "awareness": d_aware}
            )
        else:
            self.scaling_controller = None

        # 自适应投影矩阵（用于混合模式）
        self.mixing_weights = nn.Parameter(torch.eye(3))  # [3, 3]

    def _gram_schmidt(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Gram-Schmidt正交化

        Args:
            vectors: [B, L, 3, D] 三个向量的堆叠

        Returns:
            正交化后的向量
        """
        B, L, _, D = vectors.shape
        orthogonal = torch.zeros_like(vectors)

        # 第一个向量保持不变
        orthogonal[:, :, 0] = vectors[:, :, 0]

        # 第二个向量减去在第一个向量上的投影
        proj_1 = (vectors[:, :, 1] * orthogonal[:, :, 0]).sum(dim=-1, keepdim=True) * orthogonal[:, :, 0]
        orthogonal[:, :, 1] = vectors[:, :, 1] - proj_1

        # 第三个向量减去在前两个向量上的投影
        proj_2a = (vectors[:, :, 2] * orthogonal[:, :, 0]).sum(dim=-1, keepdim=True) * orthogonal[:, :, 0]
        proj_2b = (vectors[:, :, 2] * orthogonal[:, :, 1]).sum(dim=-1, keepdim=True) * orthogonal[:, :, 1]
        orthogonal[:, :, 2] = vectors[:, :, 2] - proj_2a - proj_2b

        # 归一化
        orthogonal = F.normalize(orthogonal, dim=-1)

        return orthogonal

    def compute_orthogonality_loss(
        self,
        content: torch.Tensor,
        meta: torch.Tensor,
        awareness: torch.Tensor
    ) -> torch.Tensor:
        """
        计算正交性损失

        Args:
            content: [B, L, d_model]
            meta: [B, L, d_meta]
            awareness: [B, L, d_aware]

        Returns:
            正交性损失（如果未启用则返回0）
        """
        if not self.enable_orthogonality or self.ortho_loss is None:
            return torch.tensor(0.0, device=content.device)

        return self.ortho_loss(content, meta, awareness)

    def forward(
        self,
        x_emb: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_emb: [B, L, d_model] 输入embedding
            task_id: 任务类型ID（用于动态缩放）
            context_features: 上下文特征（用于动态缩放）
            return_loss: 是否返回正交性损失

        Returns:
            content: [B, L, d_model] 内容向量
            meta: [B, L, d_meta] 元认知状态
            awareness: [B, L, d_aware] 觉知向量
            (可选) ortho_loss: 正交性损失
        """
        B, L, _ = x_emb.shape

        # 线性投影到三个子空间
        z = self.proj(x_emb)
        splits = [self.d_model, self.d_meta, self.d_aware]
        content, meta, awareness = torch.split(z, splits, dim=-1)

        # 应用基础缩放
        content = content * self.content_scale
        meta = meta * self.meta_scale
        awareness = awareness * self.aware_scale

        # 动态缩放（如果启用）
        if self.enable_dynamic_scaling and self.scaling_controller is not None:
            scales = self.scaling_controller(task_id, context_features)
            content = content * scales["content"]
            meta = meta * scales["meta"]
            awareness = awareness * scales["awareness"]

        # 混合模式：通过可学习的混合权重调整缩放因子
        # 注意：由于维度不同，不能直接混合向量，而是混合缩放因子
        if self.training:
            mixing = F.softmax(self.mixing_weights, dim=0)
            # 只调整缩放因子，不直接混合向量
            scale_adjustment = mixing.sum(dim=1) / 3.0  # 平均混合权重
            content = content * scale_adjustment[0]
            meta = meta * scale_adjustment[1]
            awareness = awareness * scale_adjustment[2]

        result = (content, meta, awareness)

        # 计算正交性损失（如果需要）
        if return_loss:
            ortho_loss = self.compute_orthogonality_loss(content, meta, awareness)
            result = result + (ortho_loss,)

        return result

    def get_orthogonality_metrics(
        self,
        content: torch.Tensor,
        meta: torch.Tensor,
        awareness: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算正交性指标

        Args:
            content: [B, L, d_model]
            meta: [B, L, d_meta]
            awareness: [B, L, d_aware]

        Returns:
            正交性指标字典
        """
        content_norm = F.normalize(content, dim=-1)
        meta_norm = F.normalize(meta, dim=-1)
        aware_norm = F.normalize(awareness, dim=-1)

        return {
            "content_meta_similarity": float((content_norm * meta_norm).sum(dim=-1).abs().mean().item()),
            "content_aware_similarity": float((content_norm * aware_norm).sum(dim=-1).abs().mean().item()),
            "meta_aware_similarity": float((meta_norm * aware_norm).sum(dim=-1).abs().mean().item()),
            "content_norm_std": float(content.norm(dim=-1).std().item()),
            "meta_norm_std": float(meta.norm(dim=-1).std().item()),
            "aware_norm_std": float(awareness.norm(dim=-1).std().item()),
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_meta={self.d_meta}, d_aware={self.d_aware}, "
            f"init_method={self.init_method}, orthogonality={self.enable_orthogonality}, "
            f"dynamic_scaling={self.enable_dynamic_scaling}"
        )
