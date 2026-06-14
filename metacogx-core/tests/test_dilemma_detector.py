"""Tests for DilemmaDetector"""
import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from metacogx_core import DilemmaDetector, DilemmaType

class TestDilemmaDetector:
    def test_normal_tokens(self):
        """正常token不应触发困境"""
        detector = DilemmaDetector()
        tokens = [1, 2, 3, 4, 5]
        logits = [[0.1, 0.2, 0.3, 0.4] for _ in range(5)]
        
        signal = detector.detect(tokens, logits)
        # 正常情况置信度应较低
        assert signal.confidence < 0.5
    
    def test_repetition_detection(self):
        """重复token应触发困境"""
        detector = DilemmaDetector()
        # 高重复率的token序列
        tokens = [1, 1, 1, 1, 1, 1, 1, 1]
        logits = [[0.1, 0.2, 0.3, 0.4] for _ in range(8)]
        
        signal = detector.detect(tokens, logits)
        assert signal.type == DilemmaType.GENERATION_STALL
        assert signal.confidence > 0.5
    
    def test_entropy_threshold(self):
        """高熵应触发困境"""
        detector = DilemmaDetector()
        tokens = [1, 2, 3, 4, 5]
        # 高熵的logits（均匀分布）
        logits = [[1.0, 1.0, 1.0, 1.0] for _ in range(5)]
        
        signal = detector.detect(tokens, logits)
        assert signal.features.get('logitsEntropy') is not None
    
    def test_custom_thresholds(self):
        """自定义阈值应生效"""
        thresholds = {
            "tokenRepetition": 0.1,  # 更严格
            "logitsEntropy": 1.0,
            "ngramRepetition": 0.2,
            "consecutiveSame": 2
        }
        detector = DilemmaDetector(thresholds)
        tokens = [1, 1, 1, 2]
        logits = [[0.1, 0.2, 0.3, 0.4] for _ in range(4)]
        
        signal = detector.detect(tokens, logits)
        assert signal.confidence > 0.5
