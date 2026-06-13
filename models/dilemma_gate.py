import torch
import torch.nn as nn
from typing import List, Optional


def attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算 attention softmax 分布的熵
    attn_weights: [B, H, L, L] 的 softmax 输出
    返回: [B, H, L] 每头每位置的熵 (nats)
    """
    return -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)


def logits_stats(logits: torch.Tensor, eps: float = 1e-8) -> dict:
    """从 next-token logits 提取统计
    返回 dict: {
      'max_prob': [B],
      'entropy': [B],  (nats)
    }
    """
    p = torch.softmax(logits, dim=-1)
    return {
        'max_prob': p.max(dim=-1).values,
        'entropy': -(p * torch.log(p + eps)).sum(dim=-1),
    }


def token_repetition(tokens: torch.Tensor, window: int = 5) -> torch.Tensor:
    """向量化版本：统计序列中最近 window 个 token 的重复数
    tokens: [B, L]
    返回: [B, L] 每个位置最近 window 步重复 token 的计数（最多减到 0）
    """
    B, L = tokens.shape
    w = min(window, L)
    out = torch.zeros(B, L, device=tokens.device, dtype=torch.float32)
    for shift in range(1, w):
        same = (tokens[:, shift:] == tokens[:, :-shift]).float()
        out[:, shift:] += same
    return out


def extract_features(
    entropy_list: List[torch.Tensor],
    logits: Optional[torch.Tensor] = None,
    token_ids: Optional[torch.Tensor] = None,
    surprise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """构造门控输入特征向量
    支持两种 entropy_list 元素：
      - 4D [B, H, L, L]：未算熵的 attention softmax，函数内部算 entropy
      - 3D [B, H, L]：已算好的 attention entropy
    返回: [B, F] 特征向量，F = 2*num_layers + 3 (+1 if surprise given)
    """
    B = entropy_list[0].size(0)
    feats = []
    for e in entropy_list:
        if e.dim() == 4:
            e = attention_entropy(e)           # -> [B, H, L]
        feats.append(e.mean(dim=(1, 2)) if e.dim() >= 2 else e.mean(dim=-1))  # -> [B]
        feats.append(e.std(dim=(1, 2), unbiased=False) if e.dim() >= 2 else torch.zeros(B, device=e.device))  # -> [B]
    if logits is not None:
        st = logits_stats(logits[:, -1, :] if logits.dim() == 3 else logits)
        feats.append(st['max_prob'])
        feats.append(st['entropy'])
    else:
        feats.append(torch.ones(B, device=entropy_list[0].device) * 0.5)
        feats.append(torch.ones(B, device=entropy_list[0].device) * 0.0)
    if token_ids is not None:
        rep = token_repetition(token_ids)
        feats.append(rep.mean(dim=-1))
    else:
        feats.append(torch.zeros(B, device=entropy_list[0].device))
    if surprise is not None:
        if surprise.dim() == 0:
            feats.append(surprise.unsqueeze(0).expand(B))
        else:
            feats.append(surprise.reshape(-1)[:B])
    return torch.stack(feats, dim=-1)


class DilemmaGate(nn.Module):
    """L1 困境门控
    输入：backbone 采集的特征（attention 熵 + logits 统计 + token 重复）
    输出：dilemma_score ∈ [0, 1]（Sigmoid）
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, F] -> [B]"""
        return self.net(features).squeeze(-1)

    def __call__(
        self,
        entropy_list: List[torch.Tensor],
        logits: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        feats = extract_features(entropy_list, logits, token_ids, surprise)
        return self.forward(feats)
