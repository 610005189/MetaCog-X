"""Tests for StrategyRouter"""
import pytest
import sys
sys.path.insert(0, '.')

from metacogx_core import StrategyRouter, DilemmaSignal, DilemmaType
from metacogx_core.strategy import BacktrackStrategy, RerankStrategy

class TestStrategyRouter:
    def test_register_strategy(self):
        """策略注册应成功"""
        router = StrategyRouter()
        router.register(BacktrackStrategy())
        assert len(router.listStrategies()) == 1
    
    def test_route_to_triggered_strategy(self):
        """应路由到触发的策略"""
        router = StrategyRouter()
        router.register(BacktrackStrategy())
        
        signal = DilemmaSignal(
            type=DilemmaType.SEMANTIC_STUCK,
            confidence=0.6,
            features={},
            timestamp=0
        )
        
        context = {
            "tokens": [1, 2, 3, 4, 5],
            "logits": [],
            "prompt": "test",
            "cursorPosition": 5,
            "language": "python"
        }
        
        result = router.route(signal, context)
        assert result.success is True
