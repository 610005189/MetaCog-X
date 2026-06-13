"""觉知池 (Awareness Pool)

维护最近N步的全局awareness向量的滑动窗口，
实时计算统计量（均值、方差、趋势）用于检测异常。
"""
import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class AwarenessStats:
    """觉知统计量"""
    mean: torch.Tensor      # 指数滑动均值
    std: torch.Tensor        # 指数滑动标准差
    trend: torch.Tensor      # 趋势（最后一步减第一步）
    buffer_len: int          # 当前buffer长度


class AwarenessPool(nn.Module):
    """觉知池

    维护一个滑动窗口，记录最近N步的awareness向量，
    并计算统计量供元认知控制器使用。
    """

    def __init__(
        self,
        capacity: int = 64,
        feature_dim: int = 16,
        decay: float = 0.95,
        device: str = "cpu"
    ):
        """
        Args:
            capacity: 滑动窗口容量
            feature_dim: awareness向量维度
            decay: 指数滑动平均的衰减率
            device: 计算设备
        """
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.decay = decay
        self.device = device

        # 滑动窗口buffer
        self.buffer: List[torch.Tensor] = []

        # 指数滑动统计量
        self.exp_avg: Optional[torch.Tensor] = None
        self.exp_var: Optional[torch.Tensor] = None

        # 追踪是否已初始化
        self._initialized = False

    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """确保张量在正确的设备上"""
        return tensor.to(self.device)

    def update(self, aware_tensor: torch.Tensor) -> None:
        """
        更新觉知池

        Args:
            aware_tensor: [B, L, d_aware] 或 [B, d_aware] 当前层的awareness
        """
        # 处理不同输入形状
        if aware_tensor.dim() == 3:
            # [B, L, d_aware] -> 取序列均值 [B, d_aware]
            mean_aware = aware_tensor.mean(dim=1).detach()
        elif aware_tensor.dim() == 2:
            # [B, d_aware]
            mean_aware = aware_tensor.detach()
        else:
            raise ValueError(f"aware_tensor 维度应为2或3，实际为 {aware_tensor.dim()}")

        mean_aware = self._ensure_device(mean_aware)

        # batch size 改变时重置，避免 buffer 中混有不同 batch 维度的张量
        if self.buffer:
            existing_bs = self.buffer[-1].shape[0]
            if mean_aware.shape[0] != existing_bs:
                self.reset()

        # 添加到buffer
        self.buffer.append(mean_aware)

        # 维护固定容量
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

        # 更新指数滑动统计量
        if not self._initialized:
            self.exp_avg = mean_aware.mean(dim=0)
            self.exp_var = torch.zeros_like(self.exp_avg)
            self._initialized = True
        else:
            # 计算batch均值
            batch_mean = mean_aware.mean(dim=0)

            # 更新指数滑动平均
            self.exp_avg = self.decay * self.exp_avg + (1 - self.decay) * batch_mean

            # 使用Welford在线算法更新方差
            # var_new = decay * var_old + (1-decay) * (x - mean) * (x - mean_new)
            delta = batch_mean - self.exp_avg
            self.exp_var = self.decay * self.exp_var + (1 - self.decay) * delta * delta

    def get_stats(self) -> Optional[AwarenessStats]:
        """
        获取当前统计量

        Returns:
            AwarenessStats 或 None（如果buffer为空）
        """
        if len(self.buffer) == 0:
            return None

        # 计算buffer内均值和方差
        stacked = torch.stack(self.buffer, dim=0)  # [T, B, d_aware]
        mean = stacked.mean(dim=0)
        # 使用无偏方差，确保T=1时variance为0而非NaN
        if stacked.shape[0] > 1:
            std = torch.sqrt(stacked.var(dim=0, unbiased=False) + 1e-8)
        else:
            std = torch.zeros_like(mean)

        # 计算趋势（最后一步减第一步）
        if len(stacked) > 1:
            trend = stacked[-1] - stacked[0]
        else:
            trend = torch.zeros_like(mean)

        return AwarenessStats(
            mean=mean,
            std=std,
            trend=trend,
            buffer_len=len(self.buffer)
        )

    def get_recent_awareness(self, k: int = 5) -> Optional[torch.Tensor]:
        """
        获取最近k步的awareness（用于awareness自预测损失）

        Args:
            k: 返回最近k步

        Returns:
            [k, d_aware] 或 None
        """
        if len(self.buffer) < k:
            return None

        recent = torch.stack(self.buffer[-k:], dim=0)  # [k, d_aware]
        return recent

    def reset(self) -> None:
        """重置觉知池"""
        self.buffer.clear()
        self.exp_avg = None
        self.exp_var = None
        self._initialized = False

    def get_buffer_as_tensor(self) -> Optional[torch.Tensor]:
        """
        获取整个buffer作为张量

        Returns:
            [T, d_aware] 或 None
        """
        if len(self.buffer) == 0:
            return None
        return torch.stack(self.buffer, dim=0)


class MultiLayerAwarenessPool:
    """多层觉知池

    为每层维护独立的觉知池，或者维护一个全局池。
    """

    def __init__(
        self,
        num_layers: int = 12,
        capacity: int = 64,
        feature_dim: int = 16,
        decay: float = 0.95,
        global_pool: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            num_layers: Transformer层数
            capacity: 每层池的容量
            feature_dim: awareness维度
            decay: 衰减率
            global_pool: 是否维护全局池
            device: 设备
        """
        self.num_layers = num_layers
        self.global_pool = global_pool

        if global_pool:
            # 单一全局池
            self.global_aware_pool = AwarenessPool(capacity, feature_dim, decay, device)
            self.layer_pools = None
        else:
            # 每层独立池
            self.global_aware_pool = None
            self.layer_pools = nn.ModuleList([
                AwarenessPool(capacity, feature_dim, decay, device)
                for _ in range(num_layers)
            ])

    def update(self, awareness_per_layer: torch.Tensor) -> None:
        """
        更新所有层的觉知池

        Args:
            awareness_per_layer: [num_layers, B, L, d_aware] 或 [B, L, d_aware]
        """
        if self.global_pool:
            # 全局池：对所有层取平均
            if awareness_per_layer.dim() == 4:
                # [num_layers, B, L, d_aware] -> [B, L, d_aware]
                pooled = awareness_per_layer.mean(dim=0)
            else:
                pooled = awareness_per_layer
            self.global_aware_pool.update(pooled)
        else:
            # 每层独立更新
            for layer_idx, pool in enumerate(self.layer_pools):
                layer_aware = awareness_per_layer[layer_idx]
                pool.update(layer_aware)

    def get_stats(self) -> Optional[AwarenessStats]:
        """获取全局统计量"""
        if self.global_pool and self.global_aware_pool:
            return self.global_aware_pool.get_stats()
        elif not self.global_pool and self.layer_pools:
            # 聚合所有层的统计
            all_stats = [pool.get_stats() for pool in self.layer_pools]
            valid_stats = [s for s in all_stats if s is not None]
            if not valid_stats:
                return None

            # 取平均
            avg_mean = torch.stack([s.mean for s in valid_stats]).mean(dim=0)
            avg_std = torch.stack([s.std for s in valid_stats]).mean(dim=0)
            avg_trend = torch.stack([s.trend for s in valid_stats]).mean(dim=0)

            return AwarenessStats(
                mean=avg_mean,
                std=avg_std,
                trend=avg_trend,
                buffer_len=valid_stats[0].buffer_len
            )
        return None

    def reset(self) -> None:
        """重置所有池"""
        if self.global_aware_pool:
            self.global_aware_pool.reset()
        if self.layer_pools:
            for pool in self.layer_pools:
                pool.reset()
