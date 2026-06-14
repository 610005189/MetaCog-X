"""工具函数"""
import time
from typing import List, Tuple, Union
import numpy as np


def compute_token_repetition(tokens: Union[List[int], np.ndarray], window: int = 5) -> float:
    """计算 token 重复率

    Args:
        tokens: token 序列
        window: 窗口大小

    Returns:
        重复率 (0.0 ~ 1.0)
    """
    if len(tokens) < 2:
        return 0.0

    tokens = np.array(tokens)
    w = min(window, len(tokens))
    repetition_count = 0
    total_comparisons = 0

    for shift in range(1, w):
        same = (tokens[shift:] == tokens[:-shift])
        repetition_count += int(same.sum())
        total_comparisons += len(same)

    return repetition_count / total_comparisons if total_comparisons > 0 else 0.0


def compute_logists_entropy(logits: np.ndarray) -> float:
    """计算 logits 分布的熵

    Args:
        logits: 模型输出的 logits

    Returns:
        熵值 (nats)
    """
    # softmax
    exp_logits = np.exp(logits - np.max(logits))
    p = exp_logits / exp_logits.sum()

    # 计算熵
    eps = 1e-8
    entropy = -np.sum(p * np.log(p + eps))
    return float(entropy)


def compute_ngram_repetition(tokens: Union[List[int], np.ndarray], n: int = 3) -> float:
    """计算 n-gram 重复率

    Args:
        tokens: token 序列
        n: n-gram 大小

    Returns:
        重复率 (0.0 ~ 1.0)
    """
    if len(tokens) < n * 2:
        return 0.0

    tokens = np.array(tokens)
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)

    return 1.0 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0


def detect_consecutive_same(tokens: Union[List[int], np.ndarray], max_consecutive: int = 3) -> Tuple[bool, int]:
    """检测连续相同 token

    Args:
        tokens: token 序列
        max_consecutive: 最大允许连续数

    Returns:
        (是否异常, 最大连续数)
    """
    if len(tokens) < 2:
        return False, 0

    tokens = np.array(tokens)
    max_consec = 1
    current_consec = 1

    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 1

    return max_consec >= max_consecutive, max_consec


def current_timestamp() -> float:
    """获取当前时间戳"""
    return time.time()


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax 归一化

    Args:
        logits: 输入 logits

    Returns:
        概率分布
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()
