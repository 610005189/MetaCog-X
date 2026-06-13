"""MetaCog-X 损失函数

包含：
1. 语言建模交叉熵损失
2. Meta时序一致性损失
3. Awareness自预测损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class MetaConsistencyLoss(nn.Module):
    """Meta时序一致性损失

    鼓励相邻步的meta相似，除非content发生突变。
    这有助于学习稳定的元认知表征。
    """

    def __init__(self, margin: float = 0.1):
        """
        Args:
            margin: 边界值，用于对比损失
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        meta_per_layer: torch.Tensor,
        content_per_layer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            meta_per_layer: [num_layers, B, L, d_meta]
            content_per_layer: [num_layers, B, L, d_model] (可选)

        Returns:
            一致性损失
        """
        num_layers = meta_per_layer.shape[0]

        if num_layers < 2:
            return torch.zeros(1, device=meta_per_layer.device).squeeze()

        # 计算相邻层meta的差异
        total_loss = torch.zeros(1, device=meta_per_layer.device, dtype=meta_per_layer.dtype).squeeze()
        count = 0

        for i in range(num_layers - 1):
            # meta差异
            meta_diff = F.mse_loss(meta_per_layer[i], meta_per_layer[i + 1])

            # 如果提供了content，检测是否发生突变
            if content_per_layer is not None:
                content_diff = F.mse_loss(content_per_layer[i], content_per_layer[i + 1])
                threshold = content_diff * 2
                if meta_diff > threshold:
                    total_loss = total_loss + (meta_diff - threshold)
            else:
                total_loss = total_loss + meta_diff

            count += 1

        return total_loss / max(count, 1)


class AwarenessPredictionLoss(nn.Module):
    """Awareness自预测损失

    用前一层的awareness预测后一层，提高表征质量。
    无需外部觉知池数据，完全基于当前batch的 layer-wise awareness。
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 延迟在 forward 中构建 MLP，因为 d_aware 依赖输入
        self._projector = None

    def forward(
        self,
        aware_per_layer: torch.Tensor,
        aware_pool_buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            aware_per_layer: [num_layers, B, L, d_aware]
            aware_pool_buffer: 预留参数，当前未使用

        Returns:
            自预测损失（标量tensor）
        """
        num_layers = aware_per_layer.shape[0]
        if num_layers < 2:
            return torch.zeros(1, device=aware_per_layer.device).squeeze()

        d_aware = aware_per_layer.shape[-1]

        # 延迟构建 MLP
        if self._projector is None or self._projector[0].in_features != d_aware:
            self._projector = nn.Sequential(
                nn.Linear(d_aware, max(d_aware * 2, self.hidden_dim)),
                nn.ReLU(),
                nn.Linear(max(d_aware * 2, self.hidden_dim), d_aware)
            ).to(aware_per_layer.device)

        # 预测：用 layer i 的 awareness 预测 layer i+1 的 awareness
        # 取序列和batch的平均，简化为 [num_layers, d_aware]
        layer_means = aware_per_layer.mean(dim=(1, 2))  # [num_layers, d_aware]

        total_loss = torch.zeros(1, device=aware_per_layer.device,
                                 dtype=aware_per_layer.dtype).squeeze()
        count = 0
        for i in range(num_layers - 1):
            pred = self._projector(layer_means[i].detach())  # 预测下一层
            target = layer_means[i + 1]
            total_loss = total_loss + F.mse_loss(pred, target)
            count += 1

        return total_loss / max(count, 1)


class TotalLoss(nn.Module):
    """总损失函数

    L_total = L_ce + α * L_meta_consistency + β * L_aware_pred
            + γ * KL(softmax(ctrl_logits) || uniform)   # entropy bonus (push controller away from collapse)
            + δ * layer_div_cosine                     # 惩罚层间 meta 相似性
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.005,
        gamma: float = 0.0,
        delta: float = 0.0,
        ignore_index: int = 0
    ):
        """
        Args:
            alpha: meta一致性损失权重
            beta: awareness自预测损失权重
            gamma: controller 熵正则权重（>0 鼓励 entropy 高 = KL 均匀分布小）
            delta: 层间 meta 分化惩罚权重（>0 拉开层间 meta 差异）
            ignore_index: 忽略的token id
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.meta_loss = MetaConsistencyLoss()
        self.aware_loss = AwarenessPredictionLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        meta_per_layer: Optional[torch.Tensor] = None,
        aware_per_layer: Optional[torch.Tensor] = None,
        aware_pool_buffer: Optional[torch.Tensor] = None,
        content_per_layer: Optional[torch.Tensor] = None,
        ctrl_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: [B, L, vocab]
            labels: [B, L]
            meta_per_layer: [num_layers, B, L, d_meta]
            aware_per_layer: [num_layers, B, L, d_aware]
            aware_pool_buffer: [T, d_aware]
            content_per_layer: [num_layers, B, L, d_model]
            ctrl_logits: [B, 3] controller 输出的原始 logits

        Returns:
            (总损失, 损失分量字典)
        """
        # 1. 交叉熵损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # 2. Meta一致性损失
        loss_meta = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if meta_per_layer is not None:
            loss_meta = self.meta_loss(meta_per_layer, content_per_layer)

        # 3. Awareness自预测损失
        loss_aware = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if aware_per_layer is not None:
            loss_aware = self.aware_loss(aware_per_layer, aware_pool_buffer)

        # 4. Controller 熵正则（防塌陷）
        # KL(softmax(logits) || Uniform) = sum q_i log(q_i / (1/K)) = sum q_i log q_i + log(K) = log(K) - H(q)
        # 当 softmax 接近均匀时 KL≈0；当 one-hot 时 KL≈log(K)。
        # 最小化 KL 等价于最大化 entropy；gamma > 0 时 push KL 小 → push entropy 大 → 防塌陷。
        loss_entropy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.gamma > 0 and ctrl_logits is not None:
            K = ctrl_logits.size(-1)
            Kf = float(K)
            logits_sf = torch.softmax(ctrl_logits, dim=-1)
            kl_uniform = (logits_sf * (torch.log(logits_sf + 1e-9) - math.log(Kf))).sum(dim=-1).mean()
            # kl_uniform 此时 = -H - log K，为负且绝对值越大越均匀。我们想要"越均匀 penalty 越小"，
            # 所以真正的 KL = -kl_uniform。
            loss_entropy = self.gamma * (-kl_uniform)

        # 5. Layer diversity 正则
        layer_div_val = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.delta > 0 and meta_per_layer is not None and meta_per_layer.dim() >= 4:
            meta_centroid = meta_per_layer.mean(dim=(1, 2))  # [num_layers, d_meta]
            nl = meta_centroid.shape[0]
            if nl >= 2:
                cos_sum = 0.0
                pairs = 0
                for i in range(nl):
                    for j in range(i + 1, nl):
                        ci = F.normalize(meta_centroid[i], dim=0)
                        cj = F.normalize(meta_centroid[j], dim=0)
                        cos_sum += (ci @ cj).abs()
                        pairs += 1
                if pairs > 0:
                    layer_div_val = cos_sum / pairs

        loss_total = loss_ce + self.alpha * loss_meta + self.beta * loss_aware \
            + loss_entropy + self.delta * layer_div_val

        loss_components = {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_meta": loss_meta,
            "loss_aware": loss_aware,
            "entropy_bonus": loss_entropy,
            "layer_div": layer_div_val,
            "ce": loss_ce,
            "meta": loss_meta,
            "aware": loss_aware,
        }

        return loss_total, loss_components


class AuxiliaryLossCalculator:
    """辅助损失计算器

    在训练过程中计算各种辅助指标用于监控。
    """

    def __init__(self, config):
        self.config = config
        self.total_loss_fn = TotalLoss(
            alpha=config.alpha_meta,
            beta=config.beta_aware
        )

    def compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        meta_per_layer: Optional[torch.Tensor] = None,
        aware_per_layer: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        计算各种评估指标

        Returns:
            指标字典
        """
        metrics = {}

        # Perplexity
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        metrics['perplexity'] = torch.exp(ce).item()
        metrics['cross_entropy'] = ce.item()

        # Meta变异度
        if meta_per_layer is not None:
            meta_std = meta_per_layer.std().item()
            meta_mean = meta_per_layer.mean().item()
            metrics['meta_std'] = meta_std
            metrics['meta_mean'] = meta_mean

        # Awareness变异度
        if aware_per_layer is not None:
            aware_std = aware_per_layer.std().item()
            aware_mean = aware_per_layer.mean().item()
            metrics['aware_std'] = aware_std
            metrics['aware_mean'] = aware_mean

        return metrics
