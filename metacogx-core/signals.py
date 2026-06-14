"""信号类型定义"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DilemmaType(Enum):
    """困境类型枚举"""
    SEMANTIC_STUCK = "SEMANTIC_STUCK"       # 语义卡顿
    SYNTAX_ANOMALY = "SYNTAX_ANOMALY"       # 句法异常
    PATTERN_REPEAT = "PATTERN_REPEAT"       # 模式重复
    GENERATION_STALL = "GENERATION_STALL"   # 生成停滞


@dataclass
class DilemmaSignal:
    """困境信号数据结构

    Attributes:
        dilemma_type: 困境类型
        confidence: 置信度 0.0 ~ 1.0
        features: 原始特征字典
        timestamp: 时间戳
    """
    dilemma_type: str
    confidence: float
    features: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        """数据验证"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.dilemma_type not in [dt.value for dt in DilemmaType]:
            raise ValueError(f"Invalid dilemma_type: {self.dilemma_type}")

    @property
    def is_dilemma(self) -> bool:
        """是否检测到困境"""
        return self.confidence >= 0.5

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "dilemma_type": self.dilemma_type,
            "confidence": self.confidence,
            "features": self.features,
            "timestamp": self.timestamp,
            "is_dilemma": self.is_dilemma
        }
