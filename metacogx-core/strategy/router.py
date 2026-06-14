"""策略路由器"""

from typing import Optional

from .base import DilemmaSignal, InterventionResult, InterventionStrategy


class StrategyRouter:
    """策略路由器 - 根据信号选择并执行策略"""

    def __init__(self):
        self.strategies: list[InterventionStrategy] = []
        self._strategy_map: dict[str, InterventionStrategy] = {}

    def register(self, strategy: InterventionStrategy) -> None:
        """注册策略"""
        self.strategies.append(strategy)
        self._strategy_map[strategy.name] = strategy

    def unregister(self, strategy_name: str) -> bool:
        """注销策略"""
        if strategy_name in self._strategy_map:
            strategy = self._strategy_map.pop(strategy_name)
            self.strategies.remove(strategy)
            return True
        return False

    def get_strategy(self, name: str) -> Optional[InterventionStrategy]:
        """获取指定名称的策略"""
        return self._strategy_map.get(name)

    def route(self, signal: DilemmaSignal, context: dict) -> InterventionResult:
        """根据信号选择策略并执行"""
        for strategy in self.strategies:
            if strategy.should_trigger(signal):
                result = strategy.execute(context)
                if result.success:
                    return result
        return InterventionResult(
            success=False,
            message="No strategy triggered or all strategies failed"
        )

    def list_strategies(self) -> list[str]:
        """列出所有已注册策略的名称"""
        return [s.name for s in self.strategies]
