"""策略基类和接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(Enum):
    """信号类型枚举"""
    SEMANTIC_MISMATCH = "semantic_mismatch"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    LOW_CONFIDENCE = "low_confidence"
    REPETITION = "repetition"
    HALLUCINATION_RISK = "hallucination_risk"


@dataclass
class DilemmaSignal:
    """困境信号 - 表示检测到的认知困境"""
    signal_type: SignalType
    confidence: float = 0.0
    context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.signal_type, SignalType):
            self.signal_type = SignalType(self.signal_type)


@dataclass
class InterventionResult:
    """干预结果"""
    success: bool
    new_tokens: list = field(default_factory=list)
    message: str = ""
    metadata: dict = field(default_factory=dict)


class InterventionStrategy(ABC):
    """干预策略基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    def trigger_condition(self) -> dict:
        """触发条件 - 子类可重写"""
        return {}

    @abstractmethod
    def should_trigger(self, signal: DilemmaSignal) -> bool:
        """是否应该触发"""
        pass

    @abstractmethod
    def execute(self, context: dict) -> InterventionResult:
        """执行干预"""
        pass

    def validate_context(self, context: dict, required_keys: list[str]) -> bool:
        """验证上下文是否包含必需的键"""
        return all(key in context for key in required_keys)
