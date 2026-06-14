"""核心检测器模块"""
import numpy as np
from typing import Optional, List, Dict

from .signals import DilemmaSignal, DilemmaType
from .thresholds import DEFAULT_THRESHOLDS
from .utils import (
    compute_token_repetition,
    compute_logits_entropy,
    compute_ngram_repetition,
    detect_consecutive_same,
    current_timestamp,
)


class DilemmaDetector:
    """困境检测器

    用于检测文本生成过程中的困境状态，包括：
    - SEMANTIC_STUCK: 语义卡顿
    - SYNTAX_ANOMALY: 句法异常
    - PATTERN_REPEAT: 模式重复
    - GENERATION_STALL: 生成停滞
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """初始化检测器

        Args:
            thresholds: 自定义阈值字典，如果为 None 则使用 DEFAULT_THRESHOLDS
        """
        self.thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS.copy()

    def detect(self, tokens: List[int], logits: Optional[np.ndarray] = None) -> DilemmaSignal:
        """检测困境信号

        Args:
            tokens: token 序列
            logits: 模型输出的 logits，可以为 None

        Returns:
            DilemmaSignal 对象
        """
        features = self._extract_features(tokens, logits)
        dilemma_type, confidence = self._classify_dilemma(features)

        return DilemmaSignal(
            dilemma_type=dilemma_type,
            confidence=confidence,
            features=features,
            timestamp=current_timestamp()
        )

    def _extract_features(self, tokens: List[int], logits: Optional[np.ndarray]) -> Dict[str, float]:
        """提取特征

        Args:
            tokens: token 序列
            logits: 模型输出的 logits

        Returns:
            特征字典
        """
        features = {}

        # 1. Token 重复率
        features["token_repetition"] = compute_token_repetition(tokens)

        # 2. Logits 熵
        if logits is not None:
            features["logits_entropy"] = compute_logits_entropy(logits)
        else:
            features["logits_entropy"] = 0.0

        # 3. N-gram 重复率
        features["ngram_repetition"] = compute_ngram_repetition(tokens)

        # 4. 连续相同 token 检测
        is_consecutive, consecutive_count = detect_consecutive_same(
            tokens, self.thresholds["consecutive_same"]
        )
        features["consecutive_same"] = consecutive_count
        features["consecutive_alert"] = 1.0 if is_consecutive else 0.0

        return features

    def _classify_dilemma(self, features: Dict[str, float]) -> tuple:
        """根据特征分类困境

        Args:
            features: 特征字典

        Returns:
            (困境类型, 置信度)
        """
        # 计算各项异常得分
        token_rep_score = features["token_repetition"] / self.thresholds["token_repetition"] \
            if self.thresholds["token_repetition"] > 0 else 0.0

        entropy_score = features["logits_entropy"] / self.thresholds["logits_entropy"] \
            if self.thresholds["logits_entropy"] > 0 else 0.0

        ngram_score = features["ngram_repetition"] / self.thresholds["ngram_repetition"] \
            if self.thresholds["ngram_repetition"] > 0 else 0.0

        consecutive_score = 1.0 if features["consecutive_alert"] > 0 else 0.0

        # 综合得分
        total_score = (
            token_rep_score * 0.3 +
            entropy_score * 0.25 +
            ngram_score * 0.3 +
            consecutive_score * 0.15
        )

        # 分类
        if features["consecutive_alert"] > 0:
            dilemma_type = DilemmaType.PATTERN_REPEAT.value
            confidence = min(consecutive_score * 0.8 + total_score * 0.2, 1.0)
        elif token_rep_score > 1.5:
            dilemma_type = DilemmaType.PATTERN_REPEAT.value
            confidence = min(token_rep_score * 0.5, 1.0)
        elif ngram_score > 1.5:
            dilemma_type = DilemmaType.SYNTAX_ANOMALY.value
            confidence = min(ngram_score * 0.5, 1.0)
        elif entropy_score > 1.5:
            dilemma_type = DilemmaType.SEMANTIC_STUCK.value
            confidence = min(entropy_score * 0.5, 1.0)
        else:
            # 无明显困境
            dilemma_type = DilemmaType.GENERATION_STALL.value
            confidence = min(total_score * 0.6, 1.0)

        return dilemma_type, confidence

    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """更新阈值配置

        Args:
            thresholds: 新的阈值字典
        """
        self.thresholds.update(thresholds)

    def reset_thresholds(self) -> None:
        """重置为默认阈值"""
        self.thresholds = DEFAULT_THRESHOLDS.copy()

    def __call__(self, tokens: List[int], logits: Optional[np.ndarray] = None) -> DilemmaSignal:
        """便捷调用接口

        Args:
            tokens: token 序列
            logits: 模型输出的 logits

        Returns:
            DilemmaSignal 对象
        """
        return self.detect(tokens, logits)
