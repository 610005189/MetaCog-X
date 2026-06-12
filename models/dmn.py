"""DMN — Default Mode Network 默认模式网络

单层 GRU（hidden_size=16, input_size=4），每步单独调用（非序列输入）。
输入 4 维 self_features，输出 (h_self[B,16], surprise[B]∈[0,1])。
surprise = Sigmoid(Linear(32→1)(ReLU(Linear(16+4→32)(concat(h_proj, features)))))
GRU 内部维护 h0，reset() 清空。
"""
import torch
import torch.nn as nn
from typing import Optional


class DMN(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.h_proj = nn.Linear(hidden_size, hidden_size)

        self.surprise_mlp = nn.Sequential(
            nn.Linear(hidden_size + input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._h: Optional[torch.Tensor] = None

    def reset(self):
        self._h = None

    def forward(self, self_features: torch.Tensor):
        """self_features: [B, 4] -> (h_self:[B,16], surprise:[B])

        DMN 必须永远在 CPU 上跑，因为 torch-directml 不支持 aten::_thnn_fused_gru_cell。
        子模块第一次 forward 时被主模型 .to(device) 移到了 DML，这里强制拉回 CPU（参数小，no-op 检测）。
        输入若在 DML/CUDA，先拷到 CPU；输出再拷回原设备。张量很小（B×4/16），拷贝开销可忽略。
        """
        if self_features.dim() != 2:
            raise ValueError(
                f"DMN expects self_features [B, {self.input_size}], got {tuple(self_features.shape)}"
            )

        B = self_features.size(0)
        orig_device = self_features.device
        cpu = torch.device("cpu")

        self.gru = self.gru.to(cpu)
        self.h_proj = self.h_proj.to(cpu)
        self.surprise_mlp = self.surprise_mlp.to(cpu)

        self_features_cpu = self_features.to(cpu, non_blocking=False)

        if self._h is None or self._h.size(1) != B:
            h0 = torch.zeros(1, B, self.hidden_size, device=cpu)
        else:
            h0 = self._h.to(cpu)

        x = self_features_cpu.unsqueeze(1)  # [B, 1, 4]
        _, h_new = self.gru(x, h0)  # [1, B, H]
        self._h = h_new

        h_self = h_new.squeeze(0)  # [B, H]
        h_proj = self.h_proj(h_self)
        fused = torch.cat([h_proj, self_features_cpu], dim=-1)
        surprise = self.surprise_mlp(fused).squeeze(-1)  # [B]

        h_self = h_self.to(orig_device, non_blocking=False)
        surprise = surprise.to(orig_device, non_blocking=False)
        return h_self, surprise

    def get_h(self):
        if self._h is None:
            return None
        return self._h.squeeze(0)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
