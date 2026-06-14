"""Tests for intervention strategies"""
import pytest
import sys
sys.path.insert(0, '.')

from metacogx_core import DilemmaSignal, DilemmaType
from metacogx_core.strategy import BacktrackStrategy

class TestBacktrackStrategy:
    def test_trigger_condition(self):
        """BacktrackStrategy 应在正确条件下触发"""
        strategy = BacktrackStrategy()
        
        signal = DilemmaSignal(
            type=DilemmaType.SEMANTIC_STUCK,
            confidence=0.5,
            features={},
            timestamp=0
        )
        
        assert strategy.shouldTrigger(signal) is True
    
    def test_execute(self):
        """执行应正确回退"""
        strategy = BacktrackStrategy()
        
        context = {
            "tokens": [1, 2, 3, 4, 5],
            "logits": [],
            "prompt": "test",
            "cursorPosition": 5,
            "language": "python"
        }
        
        result = strategy.execute(context)
        assert result.success is True
        assert len(result.newTokens) < len(context["tokens"])
