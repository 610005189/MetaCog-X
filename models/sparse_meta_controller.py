"""稀疏元认知控制器 (Sparse Meta Controller)

基于当前层的meta和池化后的awareness统计，
输出调控信号：温度因子、跳过概率、记忆强度。
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass
from .awareness_pool import AwarenessStats


@dataclass
class ControlSignals:
    """控制信号

    字段顺序：所有有默认值的放后面，避免 dataclass 报错。
    """
    temp_factor: torch.Tensor
    skip_prob: torch.Tensor
    mem_strength: torch.Tensor
    logits: Optional[torch.Tensor] = None
    temp_factor_raw_logit: Optional[torch.Tensor] = None
    skip_raw_logit: Optional[torch.Tensor] = None
    mem_raw_logit: Optional[torch.Tensor] = None


class SparseMetaController(nn.Module):
    """稀疏元认知控制器

    轻量级门控网络，从meta和awareness统计中学习调控信号。
    仅在每层运行一次，开销极小。
    """

    def __init__(
        self,
        d_meta: int,
        d_aware: int,
        hidden_dim: int = 64,
        output_dim: int = 3
    ):
        """
        Args:
            d_meta: 元认知状态维度
            d_aware: 觉知维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（固定为3：温度、跳过、记忆）
        """
        super().__init__()
        self.d_meta = d_meta
        self.d_aware = d_aware
        self.hidden_dim = hidden_dim

        # 门控网络：meta + aware_stats(mean+std+trend) -> 控制信号
        # aware_stats包含3个统计量，所以输入维度是 d_meta + d_aware * 3
        self.net = nn.Sequential(
            nn.Linear(d_meta + d_aware * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        meta: torch.Tensor,
        aware_stats: Optional[AwarenessStats] = None,
        return_logits: bool = False
    ) -> ControlSignals:
        """
        前向传播

        Args:
            meta: [B, L, d_meta] 或 [B, d_meta] 当前层的meta
            aware_stats: 来自觉知池的统计量
            return_logits: 为 True 时额外返回原始 logits，即 (signals, logits)

        Returns:
            ControlSignals 或 (ControlSignals, torch.Tensor)
        """
        # 处理meta维度
        if meta.dim() == 3:
            # [B, L, d_meta] -> [B, d_meta]
            meta_avg = meta.mean(dim=1)
        else:
            meta_avg = meta

        if aware_stats is not None:
            # 使用觉知池的统计量
            # 拼接 mean + std + trend
            # 注意：aware_stats.mean/std/trend 的形状应该是 [B, d_aware]
            aware_mean = aware_stats.mean
            aware_std = aware_stats.std
            aware_trend = aware_stats.trend

            # 确保形状正确：如果是 [d_aware]（一维），需要扩展到 [B, d_aware]
            if aware_mean.dim() == 1:
                aware_mean = aware_mean.unsqueeze(0).expand(meta_avg.size(0), -1)
            if aware_std.dim() == 1:
                aware_std = aware_std.unsqueeze(0).expand(meta_avg.size(0), -1)
            if aware_trend.dim() == 1:
                aware_trend = aware_trend.unsqueeze(0).expand(meta_avg.size(0), -1)

            aware_features = torch.cat([
                aware_mean,
                aware_std,
                aware_trend
            ], dim=-1)  # [B, d_aware * 3]
        else:
            # 如果没有awareness统计，使用零向量
            aware_features = torch.zeros(
                meta_avg.shape[0],
                self.d_aware * 3,
                device=meta_avg.device,
                dtype=meta_avg.dtype
            )

        # 拼接 meta 和 aware_features
        x = torch.cat([meta_avg, aware_features], dim=-1)

        # 添加噪声以鼓励输出变化
        # 在训练时使用随机噪声，在验证时使用确定性噪声（基于样本索引）
        if x.shape[0] > 1:
            if self.training:
                noise = torch.randn_like(x) * 1.2  # 增大随机噪声
            else:
                # 验证时使用确定性噪声（基于样本索引）
                indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(x)
                noise = (indices.float() / x.shape[0] - 0.5) * 2.5  # 增大确定性噪声
            x = x + noise

        # 前向传播
        logits = self.net(x)  # [B, 3]

        tf_raw = logits[:, 0:1]  # [B, 1]
        tf = 0.9 + 0.2 * torch.sigmoid(tf_raw)

        sk_raw = logits[:, 1]    # [B]
        sk = 0.2 * torch.sigmoid(sk_raw)

        mm_raw = logits[:, 2]    # [B]
        mm = 0.5 + 0.5 * torch.sigmoid(mm_raw)

        signals = ControlSignals(
            temp_factor=tf,
            temp_factor_raw_logit=tf_raw,
            skip_prob=sk,
            mem_strength=mm,
            logits=logits,
            skip_raw_logit=sk_raw,
            mem_raw_logit=mm_raw,
        )

        if return_logits:
            return signals, logits
        return signals

    def forward_with_layer_stats(
        self,
        meta_per_layer: torch.Tensor,
        aware_stats: AwarenessStats
    ) -> Tuple[ControlSignals, ControlSignals, ControlSignals]:
        """
        使用多层meta和全局aware_stats计算每层的控制信号

        Args:
            meta_per_layer: [num_layers, B, L, d_meta]
            aware_stats: 全局awareness统计

        Returns:
            三元组：最后一层、中间层平均、首层的控制信号
        """
        num_layers = meta_per_layer.shape[0]

        # 最后一层的meta
        last_meta = meta_per_layer[-1]  # [B, L, d_meta]

        # 中间层平均
        mid_meta = meta_per_layer[1:-1].mean(dim=0) if num_layers > 2 else meta_per_layer[0]

        # 首层meta
        first_meta = meta_per_layer[0]

        last_ctrl = self.forward(last_meta, aware_stats)
        mid_ctrl = self.forward(mid_meta, aware_stats)
        first_ctrl = self.forward(first_meta, aware_stats)

        return last_ctrl, mid_ctrl, first_ctrl


class MetaControllerWithSkip(nn.Module):
    """带随机跳过的元认知控制器

    在标准控制器基础上，增加层随机跳过机制。
    """

    def __init__(
        self,
        d_meta: int,
        d_aware: int,
        hidden_dim: int = 64,
        skip_threshold: float = 0.5
    ):
        super().__init__()
        self.controller = SparseMetaController(d_meta, d_aware, hidden_dim)
        self.skip_threshold = skip_threshold

    def forward(
        self,
        meta: torch.Tensor,
        aware_stats: Optional[AwarenessStats] = None,
        training: bool = True
    ) -> Tuple[ControlSignals, bool]:
        """
        返回 (控制信号, 是否跳过当前层)
        """
        ctrl = self.controller(meta, aware_stats)

        should_skip = False
        if training and torch.rand(1).item() < ctrl.skip_prob.item():
            should_skip = True

        return ctrl, should_skip


class AdaptiveMetaController(nn.Module):
    """自适应元认知控制器

    根据任务类型和当前推理阶段动态调整控制策略。
    """

    def __init__(
        self,
        d_meta: int,
        d_aware: int,
        hidden_dim: int = 64,
        num_task_types: int = 4
    ):
        super().__init__()
        self.base_controller = SparseMetaController(d_meta, d_aware, hidden_dim)

        # 任务嵌入用于条件控制
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.task_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        meta: torch.Tensor,
        aware_stats: Optional[AwarenessStats] = None,
        task_id: Optional[torch.Tensor] = None
    ) -> ControlSignals:
        base_ctrl = self.base_controller(meta, aware_stats)

        if task_id is not None and self.training:
            # 任务条件调整
            task_emb = self.task_embedding(task_id)  # [B, hidden]
            meta_avg = meta.mean(dim=1) if meta.dim() == 3 else meta

            # 计算门控
            gate_input = torch.cat([meta_avg, task_emb], dim=-1)
            gate = self.task_gate(gate_input)  # [B, hidden]

            # 调整控制信号（v3.0：temp_factor 保持在 [0.9, 1.1] 窄范围）
            gate_t = gate[:, 0:1] if gate.dim() == 2 else gate[:, 0:1, :]
            gate_t = gate_t.mean(dim=-1, keepdim=True) if gate_t.dim() == 3 else gate_t
            adjusted_temp = 0.9 + 0.2 * torch.sigmoid(
                torch.logit((base_ctrl.temp_factor - 0.9) / 0.2, eps=1e-6) * gate_t
                + (1.0 - gate_t) * 0.0
            )
            # 兜底：避免极端值
            adjusted_temp = adjusted_temp.clamp(0.9, 1.1)
            adjusted_skip = base_ctrl.skip_prob * gate[:, 1]
            adjusted_mem = base_ctrl.mem_strength * gate[:, 2]

            base_ctrl.temp_factor = adjusted_temp
            base_ctrl.skip_prob = adjusted_skip
            base_ctrl.mem_strength = adjusted_mem

        return base_ctrl
