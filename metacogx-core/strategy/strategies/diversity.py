"""DiversityStrategy - 多样性策略"""

from typing import Optional

from ..base import DilemmaSignal, InterventionResult, InterventionStrategy, SignalType


class DiversityStrategy(InterventionStrategy):
    """多样性策略 - 增加 temperature 以提高生成多样性

    当检测到重复或幻觉风险时，增加温度以鼓励更多样化的生成。
    """

    def __init__(
        self,
        temperature_increment: float = 0.2,
        base_temperature: float = 0.7,
        max_temperature: float = 1.5,
        min_confidence_threshold: float = 0.0,
        max_confidence_threshold: float = 0.6,
    ):
        self._temp_increment = temperature_increment
        self._base_temp = base_temperature
        self._max_temp = max_temperature
        self._min_confidence = min_confidence_threshold
        self._max_confidence = max_confidence_threshold

    @property
    def name(self) -> str:
        return "diversity"

    @property
    def trigger_condition(self) -> dict:
        return {
            "signal_types": [SignalType.REPETITION, SignalType.HALLUCINATION_RISK],
            "confidence_range": (self._min_confidence, self._max_confidence),
            "temperature_increment": self._temp_increment,
        }

    def should_trigger(self, signal: DilemmaSignal) -> bool:
        """检测到重复或幻觉风险时触发"""
        if signal.signal_type not in [SignalType.REPETITION, SignalType.HALLUCINATION_RISK]:
            return False
        return signal.confidence <= self._max_confidence

    def execute(self, context: dict) -> InterventionResult:
        """执行多样性干预"""
        current_temp = context.get("temperature", self._base_temp)
        original_temp = current_temp

        new_temp = min(current_temp + self._temp_increment, self._max_temp)

        return InterventionResult(
            success=True,
            new_tokens=[],
            message=f"Increased temperature from {original_temp:.2f} to {new_temp:.2f}",
            metadata={
                "original_temperature": original_temp,
                "new_temperature": new_temp,
                "temperature_increment": self._temp_increment,
                "strategy": "diversity",
            }
        )

    def get_new_temperature(self, current_temp: Optional[float] = None) -> float:
        """获取新的温度值"""
        temp = current_temp if current_temp is not None else self._base_temp
        return min(temp + self._temp_increment, self._max_temp)
