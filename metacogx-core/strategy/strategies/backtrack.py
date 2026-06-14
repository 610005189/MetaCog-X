"""BacktrackStrategy - 回退策略"""

from typing import Optional

from ..base import DilemmaSignal, InterventionResult, InterventionStrategy, SignalType


class BacktrackStrategy(InterventionStrategy):
    """回退策略 - 回退指定步数重新生成

    当检测到语义不匹配或逻辑不一致时，回退到某个决策点重新生成。
    """

    def __init__(
        self,
        backtrack_steps: int = 5,
        min_confidence_threshold: float = 0.3,
        max_confidence_threshold: float = 0.8,
    ):
        self._backtrack_steps = backtrack_steps
        self._min_confidence = min_confidence_threshold
        self._max_confidence = max_confidence_threshold

    @property
    def name(self) -> str:
        return "backtrack"

    @property
    def trigger_condition(self) -> dict:
        return {
            "signal_types": [SignalType.SEMANTIC_MISMATCH, SignalType.LOGICAL_INCONSISTENCY],
            "confidence_range": (self._min_confidence, self._max_confidence),
        }

    def should_trigger(self, signal: DilemmaSignal) -> bool:
        """在指定置信度范围内检测到语义不匹配或逻辑不一致时触发"""
        if signal.signal_type not in [
            SignalType.SEMANTIC_MISMATCH,
            SignalType.LOGICAL_INCONSISTENCY,
        ]:
            return False

        return self._min_confidence <= signal.confidence <= self._max_confidence

    def execute(self, context: dict) -> InterventionResult:
        """执行回退干预"""
        if not self.validate_context(context, ["tokens", "decision_points"]):
            return InterventionResult(
                success=False,
                message="Context missing required keys: 'tokens' and 'decision_points'"
            )

        tokens: list = context["tokens"]
        decision_points: list = context["decision_points"]

        if len(decision_points) < self._backtrack_steps:
            return InterventionResult(
                success=False,
                message=f"Not enough decision points to backtrack {self._backtrack_steps} steps"
            )

        backtrack_index = decision_points[-(self._backtrack_steps + 1)]
        new_tokens = tokens[:backtrack_index]

        return InterventionResult(
            success=True,
            new_tokens=new_tokens,
            message=f"Backtracked {self._backtrack_steps} steps to index {backtrack_index}",
            metadata={
                "original_length": len(tokens),
                "new_length": len(new_tokens),
                "backtrack_steps": self._backtrack_steps,
            }
        )
