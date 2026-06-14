"""RerankStrategy - 重排策略"""

from typing import Optional

from ..base import DilemmaSignal, InterventionResult, InterventionStrategy, SignalType


class RerankStrategy(InterventionStrategy):
    """重排策略 - 从 top-k 候选中重新选择

    当检测到低置信度时，从 top-k 候选中选择更好的替代。
    """

    def __init__(
        self,
        top_k: int = 5,
        min_confidence_threshold: float = 0.0,
        max_confidence_threshold: float = 0.5,
    ):
        self._top_k = top_k
        self._min_confidence = min_confidence_threshold
        self._max_confidence = max_confidence_threshold

    @property
    def name(self) -> str:
        return "rerank"

    @property
    def trigger_condition(self) -> dict:
        return {
            "signal_types": [SignalType.LOW_CONFIDENCE],
            "confidence_range": (self._min_confidence, self._max_confidence),
            "top_k": self._top_k,
        }

    def should_trigger(self, signal: DilemmaSignal) -> bool:
        """低置信度时触发"""
        return signal.signal_type == SignalType.LOW_CONFIDENCE and \
               signal.confidence <= self._max_confidence

    def execute(self, context: dict) -> InterventionResult:
        """执行重排干预"""
        if not self.validate_context(context, ["candidates", "scores"]):
            return InterventionResult(
                success=False,
                message="Context missing required keys: 'candidates' and 'scores'"
            )

        candidates: list = context["candidates"]
        scores: list = context["scores"]

        if len(candidates) < 2:
            return InterventionResult(
                success=False,
                message="Not enough candidates for reranking"
            )

        if len(candidates) != len(scores):
            return InterventionResult(
                success=False,
                message="Candidates and scores length mismatch"
            )

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_candidates = [c for c, _ in ranked[:self._top_k]]

        selected = top_candidates[0] if top_candidates else candidates[0]
        new_tokens = [selected] if isinstance(selected, str) else list(selected)

        return InterventionResult(
            success=True,
            new_tokens=new_tokens,
            message=f"Reranked and selected from top {self._top_k} candidates",
            metadata={
                "top_k": self._top_k,
                "original_score": scores[0] if candidates else 0.0,
                "new_score": scores[candidates.index(selected)] if selected in candidates else 0.0,
            }
        )
