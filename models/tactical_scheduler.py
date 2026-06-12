"""TacticalScheduler — L2 战术调度器骨架

策略库方案：Python dict 常量 STRATEGY_LIBRARY，无梯度更新 / EWC / 参数微调。
forward(features_seq:[B,T,d_seq]) -> (strategy_id:int, confidence:float)
"""
import torch
import torch.nn as nn


STRATEGY_LIBRARY = {
    0: {
        "name": "conservative_plain",
        "description": "plain 模式主导，降低 surprise 注入",
        "apply": lambda model: setattr(model, "_dmn_gate_scale", 0.5),
    },
    1: {
        "name": "balanced_meta",
        "description": "平衡元认知与普通模式",
        "apply": lambda model: setattr(model, "_dmn_gate_scale", 1.0),
    },
    2: {
        "name": "aggressive_meta",
        "description": "放大元认知信号，增强 surprise 权重",
        "apply": lambda model: setattr(model, "_dmn_gate_scale", 2.0),
    },
    3: {
        "name": "stability_first",
        "description": "模式滞后加宽，减少 switch",
        "apply": lambda model: None,
    },
    4: {
        "name": "explore_surprise",
        "description": "优先探索 DMN surprise 信号",
        "apply": lambda model: setattr(model, "_dmn_explore", True),
    },
}


class TacticalScheduler(nn.Module):
    def __init__(
        self,
        d_seq: int,
        T: int = 10,
        hidden_size: int = 32,
        num_strategies: int = 5,
    ):
        super().__init__()
        self.d_seq = d_seq
        self.T = T
        self.hidden_size = hidden_size
        self.num_strategies = num_strategies

        self.encoder = nn.LSTM(
            input_size=d_seq,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_strategies),
            nn.Softmax(dim=-1),
        )

    def forward(self, features_seq: torch.Tensor):
        """features_seq: [B, T, d_seq] -> (strategy_id:int, confidence:float)"""
        if features_seq.dim() == 2:
            features_seq = features_seq.unsqueeze(0)  # [1, T, d_seq]
        if features_seq.size(1) != self.T:
            T_act = features_seq.size(1)
            if T_act < self.T:
                pad = torch.zeros(
                    features_seq.size(0), self.T - T_act, features_seq.size(-1),
                    device=features_seq.device, dtype=features_seq.dtype,
                )
                features_seq = torch.cat([features_seq, pad], dim=1)
            else:
                features_seq = features_seq[:, : self.T, :]

        _, (h, _) = self.encoder(features_seq)  # h: [1, B, H]
        last_hidden = h.squeeze(0)                # [B, H]
        probs = self.head(last_hidden)           # [B, num_strategies]

        best_idx = int(probs[0].argmax().item())
        confidence = float(probs[0, best_idx].item())
        return best_idx, float(confidence)

    def list_strategies(self):
        return {k: STRATEGY_LIBRARY[k] for k in range(self.num_strategies)}

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, d_seq={self.d_seq}, hidden={self.hidden_size}, "
            f"num_strategies={self.num_strategies}"
        )
