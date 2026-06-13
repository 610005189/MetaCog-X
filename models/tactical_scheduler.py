"""TacticalScheduler — L2 战术调度器 v2

策略库方案：支持策略选择、执行和学习。

增强功能：
- 策略执行引擎
- 策略效果评估
- 策略梯度学习
- 策略演化机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class StrategyInfo:
    """策略信息"""
    id: int
    name: str
    description: str
    confidence: float
    reward: float = 0.0
    usage_count: int = 0


@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    strategy_id: int
    reward: float
    metrics: Dict[str, float]
    timestamp: int = 0


class StrategyLibrary:
    """策略库管理"""

    def __init__(self):
        self.strategies = {
            0: {
                "name": "conservative_plain",
                "description": "plain 模式主导，降低 surprise 注入",
                "params": {"dmn_gate_scale": 0.5, "exploration_bonus": 0.0},
            },
            1: {
                "name": "balanced_meta",
                "description": "平衡元认知与普通模式",
                "params": {"dmn_gate_scale": 1.0, "exploration_bonus": 0.3},
            },
            2: {
                "name": "aggressive_meta",
                "description": "放大元认知信号，增强 surprise 权重",
                "params": {"dmn_gate_scale": 2.0, "exploration_bonus": 0.5},
            },
            3: {
                "name": "stability_first",
                "description": "模式滞后加宽，减少 switch",
                "params": {"dmn_gate_scale": 0.8, "exploration_bonus": 0.1, "hysteresis_scale": 1.5},
            },
            4: {
                "name": "explore_surprise",
                "description": "优先探索 DMN surprise 信号",
                "params": {"dmn_gate_scale": 1.5, "exploration_bonus": 0.8, "surprise_weight": 2.0},
            },
            5: {
                "name": "precision_mode",
                "description": "高精度模式，降低温度因子",
                "params": {"dmn_gate_scale": 0.3, "exploration_bonus": 0.0, "temp_factor_scale": 0.9},
            },
            6: {
                "name": "creative_mode",
                "description": "创造性模式，增强探索",
                "params": {"dmn_gate_scale": 1.8, "exploration_bonus": 1.0, "temp_factor_scale": 1.1},
            },
            7: {
                "name": "recovery_mode",
                "description": "恢复模式，重置部分状态",
                "params": {"dmn_gate_scale": 0.5, "exploration_bonus": 0.2, "reset_awareness": True},
            },
        }

    def get_strategy(self, strategy_id: int) -> Optional[Dict]:
        """获取策略信息"""
        return self.strategies.get(strategy_id)

    def list_strategies(self) -> Dict[int, Dict]:
        """列出所有策略"""
        return self.strategies

    def add_strategy(self, strategy_id: int, config: Dict) -> None:
        """添加新策略"""
        self.strategies[strategy_id] = config

    def remove_strategy(self, strategy_id: int) -> None:
        """移除策略"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]


class StrategyExecutor:
    """策略执行器"""

    def __init__(self, model):
        self.model = model
        self.execution_history: List[ExecutionResult] = []
        self.strategy_usage: Dict[int, int] = {}

    def apply_strategy(self, strategy_id: int, strategy_params: Dict) -> ExecutionResult:
        """
        应用策略到模型

        Args:
            strategy_id: 策略ID
            strategy_params: 策略参数

        Returns:
            ExecutionResult
        """
        try:
            # 应用 DMN gate scale
            if "dmn_gate_scale" in strategy_params:
                setattr(self.model, "_dmn_gate_scale", strategy_params["dmn_gate_scale"])

            # 应用探索奖励
            if "exploration_bonus" in strategy_params:
                setattr(self.model, "_exploration_bonus", strategy_params["exploration_bonus"])

            # 应用温度因子缩放
            if "temp_factor_scale" in strategy_params:
                setattr(self.model, "_temp_factor_scale", strategy_params["temp_factor_scale"])

            # 应用滞后缩放
            if "hysteresis_scale" in strategy_params:
                if hasattr(self.model, 'dilemma_gate'):
                    self.model.dilemma_gate._hysteresis_scale = strategy_params["hysteresis_scale"]

            # 应用 surprise 权重
            if "surprise_weight" in strategy_params:
                if hasattr(self.model, 'dmn'):
                    self.model.dmn.surprise_weight = strategy_params["surprise_weight"]

            # 重置 awareness
            if strategy_params.get("reset_awareness", False):
                if hasattr(self.model, 'awareness_pool'):
                    self.model.awareness_pool.reset()

            # 更新使用计数
            self.strategy_usage[strategy_id] = self.strategy_usage.get(strategy_id, 0) + 1

            return ExecutionResult(
                success=True,
                strategy_id=strategy_id,
                reward=0.0,
                metrics={"applied_params": len(strategy_params)}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                strategy_id=strategy_id,
                reward=0.0,
                metrics={"error": str(e)}
            )

    def record_result(self, result: ExecutionResult) -> None:
        """记录执行结果"""
        self.execution_history.append(result)

    def get_strategy_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取策略统计信息"""
        stats = {}
        for strategy_id, count in self.strategy_usage.items():
            results = [r for r in self.execution_history if r.strategy_id == strategy_id]
            success_rate = sum(1 for r in results if r.success) / max(1, len(results))
            avg_reward = sum(r.reward for r in results) / max(1, len(results))
            stats[strategy_id] = {
                "usage_count": count,
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "total_executions": len(results)
            }
        return stats


class TacticalScheduler(nn.Module):
    """战术调度器

    根据特征序列选择最优策略。
    """

    def __init__(
        self,
        d_seq: int,
        T: int = 10,
        hidden_size: int = 32,
        num_strategies: int = 8,
        enable_learning: bool = True,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.d_seq = d_seq
        self.T = T
        self.hidden_size = hidden_size
        self.num_strategies = num_strategies
        self.enable_learning = enable_learning

        # 策略库
        self.strategy_library = StrategyLibrary()

        # 序列编码器
        self.encoder = nn.LSTM(
            input_size=d_seq,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # 策略选择头
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_strategies),
        )

        # 策略价值网络（用于策略评估）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # 策略偏好（可学习的策略权重）
        self.strategy_preferences = nn.Parameter(torch.zeros(num_strategies))

        # 优化器（如果启用学习）
        if enable_learning:
            self.optimizer = torch.optim.Adam(
                list(self.parameters()),
                lr=learning_rate,
                weight_decay=1e-4
            )

        # 策略执行器
        self.executor = None

    def set_model(self, model) -> None:
        """设置要控制的模型"""
        self.executor = StrategyExecutor(model)

    def forward(
        self,
        features_seq: torch.Tensor,
        return_probs: bool = False,
        return_value: bool = False
    ) -> Tuple[int, float]:
        """
        选择策略

        Args:
            features_seq: [B, T, d_seq] 特征序列
            return_probs: 是否返回概率分布
            return_value: 是否返回价值估计

        Returns:
            (strategy_id, confidence) 或扩展结果
        """
        if features_seq.dim() == 2:
            features_seq = features_seq.unsqueeze(0)  # [1, T, d_seq]

        # 处理长度不匹配
        if features_seq.size(1) != self.T:
            T_act = features_seq.size(1)
            if T_act < self.T:
                pad = torch.zeros(
                    features_seq.size(0), self.T - T_act, features_seq.size(-1),
                    device=features_seq.device, dtype=features_seq.dtype,
                )
                features_seq = torch.cat([features_seq, pad], dim=1)
            else:
                features_seq = features_seq[:, : self.T, :]

        # 编码特征序列
        _, (h, _) = self.encoder(features_seq)  # h: [1, B, H]
        last_hidden = h.squeeze(0)                # [B, H]

        # 计算策略logits
        logits = self.head(last_hidden)           # [B, num_strategies]

        # 添加可学习的策略偏好
        logits = logits + self.strategy_preferences

        # 计算概率
        probs = F.softmax(logits, dim=-1)        # [B, num_strategies]

        # 选择最佳策略
        best_idx = int(probs[0].argmax().item())
        confidence = float(probs[0, best_idx].item())

        result = (best_idx, confidence)

        if return_probs:
            result = result + (probs,)

        if return_value:
            value = self.value_head(last_hidden).squeeze(-1)  # [B]
            result = result + (float(value[0].item()),)

        return result

    def execute_strategy(self, strategy_id: int) -> ExecutionResult:
        """执行策略"""
        if self.executor is None:
            raise RuntimeError("Model not set. Call set_model() first.")

        strategy = self.strategy_library.get_strategy(strategy_id)
        if strategy is None:
            return ExecutionResult(
                success=False,
                strategy_id=strategy_id,
                reward=0.0,
                metrics={"error": "Strategy not found"}
            )

        result = self.executor.apply_strategy(strategy_id, strategy["params"])
        self.executor.record_result(result)
        return result

    def learn_from_reward(
        self,
        features_seq: torch.Tensor,
        reward: float,
        strategy_id: int,
        baseline: float = 0.0
    ) -> Dict[str, float]:
        """
        从奖励中学习

        Args:
            features_seq: 特征序列
            reward: 获得的奖励
            strategy_id: 执行的策略ID
            baseline: 奖励基线

        Returns:
            损失信息
        """
        if not self.enable_learning:
            return {"error": "Learning is disabled"}

        self.optimizer.zero_grad()

        # 获取策略概率
        if features_seq.dim() == 2:
            features_seq = features_seq.unsqueeze(0)

        _, (h, _) = self.encoder(features_seq)
        last_hidden = h.squeeze(0)
        logits = self.head(last_hidden) + self.strategy_preferences
        probs = F.softmax(logits, dim=-1)

        # 策略梯度：REINFORCE 算法
        advantage = reward - baseline
        log_prob = torch.log(probs[0, strategy_id] + 1e-8)
        policy_loss = -advantage * log_prob

        # 价值损失
        value = self.value_head(last_hidden).squeeze(-1)
        value_loss = F.mse_loss(value, torch.tensor([reward], device=value.device))

        # 熵正则化（鼓励探索）
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -0.01 * entropy

        # 总损失
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy_loss": float(entropy_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "advantage": advantage,
            "entropy": float(entropy.detach().cpu()),
        }

    def evolve_strategy(self, strategy_id: int, mutation_rate: float = 0.1) -> None:
        """
        策略演化：随机变异策略参数

        Args:
            strategy_id: 策略ID
            mutation_rate: 变异率
        """
        strategy = self.strategy_library.get_strategy(strategy_id)
        if strategy is None:
            return

        params = strategy["params"].copy()
        for key, value in params.items():
            if isinstance(value, float):
                params[key] = value * (1.0 + (torch.rand(1).item() - 0.5) * 2 * mutation_rate)
                # 保持合理范围
                if key == "dmn_gate_scale":
                    params[key] = max(0.1, min(3.0, params[key]))
                elif key == "exploration_bonus":
                    params[key] = max(0.0, min(2.0, params[key]))
                elif key == "temp_factor_scale":
                    params[key] = max(0.5, min(1.5, params[key]))

        strategy["params"] = params

    def combine_strategies(
        self,
        strategy_id1: int,
        strategy_id2: int,
        new_id: Optional[int] = None
    ) -> int:
        """
        策略组合：融合两个策略的参数

        Args:
            strategy_id1: 第一个策略ID
            strategy_id2: 第二个策略ID
            new_id: 新策略ID（可选）

        Returns:
            新策略ID
        """
        strategy1 = self.strategy_library.get_strategy(strategy_id1)
        strategy2 = self.strategy_library.get_strategy(strategy_id2)

        if strategy1 is None or strategy2 is None:
            raise ValueError("One or both strategies not found")

        # 合并参数（取平均）
        combined_params = {}
        all_keys = set(strategy1["params"].keys()) | set(strategy2["params"].keys())

        for key in all_keys:
            val1 = strategy1["params"].get(key, 1.0)
            val2 = strategy2["params"].get(key, 1.0)
            combined_params[key] = (val1 + val2) / 2.0

        # 生成新策略ID
        if new_id is None:
            new_id = max(self.strategy_library.strategies.keys()) + 1

        new_name = f"{strategy1['name']}_{strategy2['name']}"
        new_description = f"Combination of {strategy1['name']} and {strategy2['name']}"

        self.strategy_library.add_strategy(new_id, {
            "name": new_name,
            "description": new_description,
            "params": combined_params,
        })

        return new_id

    def list_strategies(self) -> Dict[int, StrategyInfo]:
        """列出所有策略信息"""
        stats = self.executor.get_strategy_stats() if self.executor else {}

        result = {}
        for idx, strategy in self.strategy_library.strategies.items():
            info = stats.get(idx, {})
            result[idx] = StrategyInfo(
                id=idx,
                name=strategy["name"],
                description=strategy["description"],
                confidence=0.0,
                reward=info.get("avg_reward", 0.0),
                usage_count=info.get("usage_count", 0)
            )
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取调度器统计"""
        if self.executor:
            return self.executor.get_strategy_stats()
        return {}

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, d_seq={self.d_seq}, hidden={self.hidden_size}, "
            f"num_strategies={self.num_strategies}, learning={self.enable_learning}"
        )
