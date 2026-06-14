"""阈值配置"""
from typing import Final

# 默认阈值配置
DEFAULT_THRESHOLDS: Final[dict] = {
    "token_repetition": 0.3,       # Token 重复率阈值
    "logits_entropy": 2.5,         # Logits 熵阈值 (nats)
    "ngram_repetition": 0.4,       # N-gram 重复率阈值
    "consecutive_same": 3,        # 连续相同 token 阈值
}

# 困境置信度阈值
CONFIDENCE_THRESHOLDS: Final[dict] = {
    "high": 0.8,       # 高置信度
    "medium": 0.5,     # 中置信度
    "low": 0.3,        # 低置信度
}


def get_threshold(name: str, custom: dict = None) -> float:
    """获取指定阈值

    Args:
        name: 阈值名称
        custom: 自定义阈值字典

    Returns:
        阈值值
    """
    thresholds = DEFAULT_THRESHOLDS.copy()
    if custom:
        thresholds.update(custom)
    return thresholds.get(name, 0.0)


def validate_thresholds(thresholds: dict) -> bool:
    """验证阈值配置是否合法

    Args:
        thresholds: 阈值字典

    Returns:
        是否合法
    """
    required_keys = {"token_repetition", "logits_entropy", "ngram_repetition", "consecutive_same"}
    if not required_keys.issubset(thresholds.keys()):
        return False

    for key in required_keys:
        val = thresholds[key]
        if not isinstance(val, (int, float)) or val < 0 or val > 1:
            return False
    return True
