"""开悟触发器 (Enlightenment Trigger) v2

检测推理是否陷入无效循环或高熵不确定性，决定是否干预。

增强功能：
- 推理质量监控
- 自适应阈值调整
- 任务难度感知
- 上下文感知
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TriggerAction(Enum):
    """触发动作类型"""
    NONE = "none"
    RESET = "reset"      # 重置上下文
    TOOL = "tool"        # 调用外部工具
    MODIFY = "modify"    # 修改推理路径
    REPHRASE = "rephrase"  # 重新表述
    BACKTRACK = "backtrack" # 回溯


@dataclass
class TriggerResult:
    """触发结果"""
    triggered: bool
    action: TriggerAction
    confidence: float          # 触发置信度
    reason: str                # 触发原因描述
    repeat_count: int          # 当前连续重复数
    current_entropy: float     # 当前熵值
    quality_score: float       # 推理质量评分
    task_difficulty: float     # 任务难度估计


@dataclass
class QualityMetrics:
    """推理质量指标"""
    perplexity: float
    entropy: float
    repetition_rate: float
    coherence_score: float
    confidence_score: float
    novelty_score: float


class QualityMonitor(nn.Module):
    """推理质量监控器

    实时监控推理过程的质量指标。
    """

    def __init__(
        self,
        history_window: int = 10,
        quality_threshold: float = 0.6
    ):
        super().__init__()
        self.history_window = history_window
        self.quality_threshold = quality_threshold
        self._quality_history: List[float] = []
        self._perplexity_history: List[float] = []

    def compute_perplexity(self, logits: torch.Tensor, labels: torch.Tensor = None) -> float:
        """计算困惑度"""
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        if labels is not None:
            # 使用真实标签计算困惑度
            if labels.dim() == 2:
                labels = labels[:, -1]
            ce = -torch.log(probs.gather(1, labels.unsqueeze(1)) + 1e-8).mean().item()
            return float(torch.exp(torch.tensor(ce)))
        else:
            # 使用最大概率估计
            max_prob = probs.max(dim=-1).values.mean().item()
            return 1.0 / max_prob if max_prob > 0 else float("inf")

    def compute_coherence(self, tokens: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> float:
        """计算连贯性分数"""
        if tokens is None or tokens.dim() < 2:
            return 0.5

        # 简单的连贯性估计：检查token分布的多样性
        B, L = tokens.shape
        unique_ratio = tokens.unique(dim=-1).shape[-1] / L
        return float(min(1.0, unique_ratio * 2))

    def compute_novelty(self, tokens: torch.Tensor, vocab_size: int = 50000) -> float:
        """计算新颖性分数"""
        if tokens is None:
            return 0.5

        # 简单的新颖性估计：使用稀有token的比例
        # 假设低频token（> vocab_size * 0.9）更具新颖性
        rare_token_ratio = (tokens > int(vocab_size * 0.9)).float().mean().item()
        return float(min(1.0, rare_token_ratio * 10))

    def compute_confidence(self, logits: torch.Tensor) -> float:
        """计算置信度"""
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1).values.mean().item()
        return float(max_prob)

    def forward(
        self,
        logits: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None
    ) -> QualityMetrics:
        """计算所有质量指标"""
        perplexity = self.compute_perplexity(logits, labels)
        entropy = self.compute_entropy(logits)
        repetition_rate = self.compute_repetition_rate(tokens)
        coherence = self.compute_coherence(tokens, attention_weights)
        confidence = self.compute_confidence(logits)
        novelty = self.compute_novelty(tokens)

        # 更新历史
        self._perplexity_history.append(perplexity)
        if len(self._perplexity_history) > self.history_window:
            self._perplexity_history.pop(0)

        # 综合质量评分（归一化）
        quality_score = (
            min(1.0, 1.0 - perplexity / 100) * 0.3 +
            min(1.0, 1.0 - entropy / 10) * 0.2 +
            (1.0 - repetition_rate) * 0.2 +
            coherence * 0.15 +
            confidence * 0.15
        )

        self._quality_history.append(quality_score)
        if len(self._quality_history) > self.history_window:
            self._quality_history.pop(0)

        return QualityMetrics(
            perplexity=perplexity,
            entropy=entropy,
            repetition_rate=repetition_rate,
            coherence_score=coherence,
            confidence_score=confidence,
            novelty_score=novelty
        )

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """计算输出分布的熵"""
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
        return entropy

    def compute_repetition_rate(self, tokens: Optional[torch.Tensor]) -> float:
        """计算重复率"""
        if tokens is None:
            return 0.0

        if tokens.dim() == 2:
            tokens = tokens[0]

        tokens_list = tokens.tolist()
        if len(tokens_list) < 2:
            return 0.0

        repeat_count = 0
        for i in range(1, len(tokens_list)):
            if tokens_list[i] == tokens_list[i-1]:
                repeat_count += 1

        return repeat_count / (len(tokens_list) - 1)

    def get_average_quality(self) -> float:
        """获取平均质量评分"""
        if not self._quality_history:
            return 0.5
        return sum(self._quality_history) / len(self._quality_history)

    def is_degrading(self, threshold: float = 0.1) -> bool:
        """检测质量是否在下降"""
        if len(self._quality_history) < 3:
            return False

        recent = self._quality_history[-3:]
        earlier = self._quality_history[:-3] if len(self._quality_history) > 3 else recent

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        return (earlier_avg - recent_avg) > threshold

    def reset(self) -> None:
        """重置监控器状态"""
        self._quality_history.clear()
        self._perplexity_history.clear()


class AdaptiveThresholdController(nn.Module):
    """自适应阈值控制器

    根据任务难度和上下文动态调整触发阈值。
    """

    def __init__(
        self,
        base_entropy_thresh: float = 2.5,
        base_repeat_thresh: int = 3,
        base_patience: int = 5,
        learning_rate: float = 0.1
    ):
        super().__init__()
        self.base_entropy_thresh = base_entropy_thresh
        self.base_repeat_thresh = base_repeat_thresh
        self.base_patience = base_patience
        self.learning_rate = learning_rate

        # 自适应参数
        self._entropy_adjustment = 0.0
        self._repeat_adjustment = 0.0
        self._patience_adjustment = 0.0

        # 历史记录
        self._difficulty_history: List[float] = []

    def estimate_task_difficulty(
        self,
        quality_metrics: QualityMetrics,
        input_length: int = 0,
        output_length: int = 0
    ) -> float:
        """
        估计任务难度

        Args:
            quality_metrics: 质量指标
            input_length: 输入长度
            output_length: 输出长度

        Returns:
            难度分数 [0, 1]，越高越难
        """
        # 基于质量指标的难度估计
        quality_based = (1.0 - quality_metrics.confidence_score) * 0.4 + \
                        (quality_metrics.entropy / 10) * 0.3 + \
                        quality_metrics.repetition_rate * 0.3

        # 基于长度的难度估计
        length_based = min(1.0, (input_length + output_length) / 500) * 0.3

        difficulty = min(1.0, quality_based + length_based)
        return difficulty

    def update_thresholds(self, difficulty: float) -> None:
        """根据难度更新阈值"""
        # 存储历史难度
        self._difficulty_history.append(difficulty)
        if len(self._difficulty_history) > 20:
            self._difficulty_history.pop(0)

        avg_difficulty = sum(self._difficulty_history) / len(self._difficulty_history)

        # 自适应调整：难度越高，阈值越宽松（允许更多"挣扎"）
        # 难度越低，阈值越严格（更早触发干预）
        target_entropy_adjust = (avg_difficulty - 0.5) * 2.0  # [-1, 1]
        target_repeat_adjust = -(avg_difficulty - 0.5) * 1.0  # [-0.5, 0.5]
        target_patience_adjust = (avg_difficulty - 0.5) * 3.0  # [-1.5, 1.5]

        # 平滑更新
        self._entropy_adjustment += (target_entropy_adjust - self._entropy_adjustment) * self.learning_rate
        self._repeat_adjustment += (target_repeat_adjust - self._repeat_adjustment) * self.learning_rate
        self._patience_adjustment += (target_patience_adjust - self._patience_adjustment) * self.learning_rate

    def get_current_thresholds(self) -> Dict[str, float]:
        """获取当前自适应阈值"""
        return {
            "entropy_thresh": max(1.0, self.base_entropy_thresh + self._entropy_adjustment),
            "repeat_thresh": max(2, int(self.base_repeat_thresh + self._repeat_adjustment)),
            "patience": max(2, int(self.base_patience + self._patience_adjustment)),
        }

    def reset(self) -> None:
        """重置自适应状态"""
        self._entropy_adjustment = 0.0
        self._repeat_adjustment = 0.0
        self._patience_adjustment = 0.0
        self._difficulty_history.clear()


class EnlightenmentTrigger(nn.Module):
    """开悟触发器

    基于规则的触发器，检测：
    1. 重复token（连续重复）
    2. 高熵不确定性（输出分布过于均匀）
    3. 异常awareness模式
    4. 推理质量下降
    """

    def __init__(
        self,
        entropy_thresh: float = 2.5,
        repeat_thresh: int = 3,
        entropy_patience: int = 5,
        high_entropy_bonus: float = 1.5,
        awareness_thresh: float = 0.8,
        trend_thresh: float = 0.5,
        enable_adaptive: bool = True,
        quality_threshold: float = 0.4
    ):
        """
        Args:
            entropy_thresh: 熵阈值，超过则认为不确定性过高
            repeat_thresh: 连续重复阈值，超过则触发
            entropy_patience: 高熵持续次数阈值
            high_entropy_bonus: 高熵时的额外置信度加成
            awareness_thresh: awareness异常阈值
            trend_thresh: 趋势异常阈值
            enable_adaptive: 是否启用自适应阈值
            quality_threshold: 质量评分阈值
        """
        super().__init__()
        self._base_entropy_thresh = entropy_thresh
        self._base_repeat_thresh = repeat_thresh
        self._base_entropy_patience = entropy_patience
        self.high_entropy_bonus = high_entropy_bonus
        self.awareness_thresh = awareness_thresh
        self.trend_thresh = trend_thresh
        self.enable_adaptive = enable_adaptive
        self.quality_threshold = quality_threshold

        # 质量监控器
        self.quality_monitor = QualityMonitor()

        # 自适应阈值控制器
        if enable_adaptive:
            self.threshold_controller = AdaptiveThresholdController(
                base_entropy_thresh=entropy_thresh,
                base_repeat_thresh=repeat_thresh,
                base_patience=entropy_patience
            )

        # 状态追踪
        self._entropy_counter = 0
        self._last_tokens: List[int] = []
        self._repeat_count = 0
        self._step = 0

    @property
    def entropy_thresh(self) -> float:
        if self.enable_adaptive and hasattr(self, 'threshold_controller'):
            return self.threshold_controller.get_current_thresholds()["entropy_thresh"]
        return self._base_entropy_thresh

    @property
    def repeat_thresh(self) -> int:
        if self.enable_adaptive and hasattr(self, 'threshold_controller'):
            return self.threshold_controller.get_current_thresholds()["repeat_thresh"]
        return self._base_repeat_thresh

    @property
    def entropy_patience(self) -> int:
        if self.enable_adaptive and hasattr(self, 'threshold_controller'):
            return self.threshold_controller.get_current_thresholds()["patience"]
        return self._base_entropy_patience

    def reset(self) -> None:
        """重置触发器状态"""
        self._entropy_counter = 0
        self._last_tokens.clear()
        self._repeat_count = 0
        self._step = 0
        self.quality_monitor.reset()
        if self.enable_adaptive and hasattr(self, 'threshold_controller'):
            self.threshold_controller.reset()

    def detect_repeat(self, tokens: torch.Tensor) -> Tuple[int, int]:
        """
        检测重复token

        Args:
            tokens: [B, L] 或 [L]

        Returns:
            (连续重复数, 最大连续重复数)
        """
        if tokens.dim() == 2:
            tokens = tokens[0]  # 取第一个样本

        tokens_list = tokens.tolist()

        # 更新历史
        self._last_tokens.extend(tokens_list)

        # 只保留最近N个token用于检测
        max_history = 20
        if len(self._last_tokens) > max_history:
            self._last_tokens = self._last_tokens[-max_history:]

        # 检测连续重复
        max_repeat = 0
        current_repeat = 1
        for i in range(1, len(self._last_tokens)):
            if self._last_tokens[i] == self._last_tokens[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        self._repeat_count = max_repeat
        return max_repeat, max_repeat

    def detect_awareness_anomaly(
        self,
        aware_stats: Optional[Any],
        expected_mean: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float]:
        """
        检测awareness是否异常

        Args:
            aware_stats: AwarenessStats对象
            expected_mean: 期望的均值（可选）

        Returns:
            (是否异常, 异常分数)
        """
        if aware_stats is None:
            return False, 0.0

        # 计算与零点的距离（异常模式通常远离正常范围）
        mean_norm = torch.norm(aware_stats.mean).item()
        std_norm = torch.norm(aware_stats.std).item()
        trend_norm = torch.norm(aware_stats.trend).item()

        # 异常分数
        anomaly_score = (mean_norm + std_norm + trend_norm) / 3

        is_anomaly = (
            mean_norm > self.awareness_thresh or
            std_norm > self.awareness_thresh or
            trend_norm > self.trend_thresh
        )

        return is_anomaly, anomaly_score

    def forward(
        self,
        logits: torch.Tensor,
        aware_stats: Optional[Any] = None,
        tokens: Optional[torch.Tensor] = None,
        step: int = 0,
        labels: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None
    ) -> TriggerResult:
        """
        前向传播，检测是否需要触发

        Args:
            logits: [B, L, vocab] 当前输出
            aware_stats: AwarenessStats 对象
            tokens: [B, L] 生成的token序列
            step: 当前推理步数
            labels: 真实标签（用于计算质量指标）
            attention_weights: 注意力权重（用于计算连贯性）

        Returns:
            TriggerResult
        """
        self._step = step
        triggered = False
        action = TriggerAction.NONE
        confidence = 0.0
        reason = "normal"
        quality_score = 0.5
        task_difficulty = 0.5

        # 1. 计算质量指标
        quality_metrics = self.quality_monitor(
            logits, tokens, labels, attention_weights
        )
        quality_score = (
            min(1.0, 1.0 - quality_metrics.perplexity / 100) * 0.3 +
            min(1.0, 1.0 - quality_metrics.entropy / 10) * 0.2 +
            (1.0 - quality_metrics.repetition_rate) * 0.2 +
            quality_metrics.coherence_score * 0.15 +
            quality_metrics.confidence_score * 0.15
        )

        # 2. 自适应阈值更新
        if self.enable_adaptive and hasattr(self, 'threshold_controller'):
            input_len = tokens.shape[1] if tokens is not None else 0
            task_difficulty = self.threshold_controller.estimate_task_difficulty(
                quality_metrics, input_length=input_len, output_length=step
            )
            self.threshold_controller.update_thresholds(task_difficulty)

        # 3. 检测重复token
        repeat_count = 0
        if tokens is not None:
            repeat_count, _ = self.detect_repeat(tokens)
            if repeat_count >= self.repeat_thresh:
                triggered = True
                action = TriggerAction.RESET
                confidence = min(1.0, repeat_count / self.repeat_thresh)
                reason = f"repeat_detected: {repeat_count} consecutive same tokens"

        # 4. 检测高熵
        entropy = quality_metrics.entropy
        is_high_entropy = entropy > self.entropy_thresh

        if is_high_entropy:
            self._entropy_counter += 1
            if self._entropy_counter >= self.entropy_patience:
                triggered = True
                if action == TriggerAction.NONE:
                    action = TriggerAction.TOOL
                confidence = max(confidence, self.high_entropy_bonus * (self._entropy_counter / self.entropy_patience))
                reason = f"high_entropy: {entropy:.3f} > {self.entropy_thresh} for {self._entropy_counter} steps"
        else:
            self._entropy_counter = max(0, self._entropy_counter - 1)

        # 5. 检测质量下降
        if quality_score < self.quality_threshold:
            triggered = True
            if action == TriggerAction.NONE:
                action = TriggerAction.BACKTRACK
            confidence = max(confidence, min(1.0, (self.quality_threshold - quality_score) * 5))
            reason += f" | low_quality: {quality_score:.3f} < {self.quality_threshold}"

        # 6. 检测awareness异常
        if aware_stats is not None:
            is_anomaly, anomaly_score = self.detect_awareness_anomaly(aware_stats)
            if is_anomaly:
                triggered = True
                if action == TriggerAction.NONE:
                    action = TriggerAction.MODIFY
                confidence = max(confidence, min(1.0, anomaly_score / 2.0))
                reason += f" | awareness_anomaly: score={anomaly_score:.3f}"

        return TriggerResult(
            triggered=triggered,
            action=action,
            confidence=confidence,
            reason=reason,
            repeat_count=repeat_count,
            current_entropy=entropy,
            quality_score=quality_score,
            task_difficulty=task_difficulty
        )


class AdaptiveEnlightenmentTrigger(nn.Module):
    """自适应开悟触发器

    根据任务类型和上下文动态调整触发策略。
    """

    def __init__(
        self,
        base_thresholds: Dict[str, float],
        num_task_types: int = 4,
        enable_adaptive: bool = True
    ):
        super().__init__()
        self.base_trigger = EnlightenmentTrigger(
            entropy_thresh=base_thresholds.get("entropy", 2.5),
            repeat_thresh=int(base_thresholds.get("repeat", 3)),
            entropy_patience=int(base_thresholds.get("patience", 5)),
            enable_adaptive=enable_adaptive
        )

        # 任务特定阈值调整
        self.task_thresholds = nn.Embedding(num_task_types, 4)  # 4个阈值参数
        nn.init.constant_(self.task_thresholds.weight, 0.0)

    def forward(
        self,
        logits: torch.Tensor,
        aware_stats=None,
        tokens: Optional[torch.Tensor] = None,
        step: int = 0,
        task_id: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> TriggerResult:
        result = self.base_trigger(logits, aware_stats, tokens, step, labels)

        # 如果指定了任务类型，动态调整触发置信度
        if task_id is not None and self.training:
            task_emb = self.task_thresholds(task_id)  # [B, 4]
            # 任务特定的置信度调整
            confidence_bonus = task_emb[0, 0].sigmoid().item() - 0.5
            result.confidence = min(1.0, max(0.0, result.confidence + confidence_bonus))

        return result


class EnlightenmentExecutor:
    """开悟执行器

    负责执行触发后的干预动作。
    """

    def __init__(self, tool_registry: Optional[Dict[str, callable]] = None):
        """
        Args:
            tool_registry: 工具注册表 {tool_name: callable}
        """
        self.tool_registry = tool_registry or {}
        self.reset_history: List[Dict] = []
        self.tool_calls: List[Dict] = []
        self.modify_history: List[Dict] = []

    def execute_reset(
        self,
        state: Dict[str, Any],
        reset_type: str = "full"
    ) -> Dict[str, Any]:
        """
        执行重置动作

        Args:
            state: 当前状态字典
            reset_type: 重置类型 ("full", "partial", "soft", "backtrack")

        Returns:
            重置后的状态
        """
        new_state = state.copy()

        if reset_type == "full":
            # 完全重置
            new_state["awareness_pool"].reset()
            new_state["hidden_states"] = None
            new_state["repeat_count"] = 0
            new_state["generation_history"] = []
        elif reset_type == "partial":
            # 部分重置：只重置awareness
            if "awareness_pool" in new_state:
                new_state["awareness_pool"].reset()
        elif reset_type == "soft":
            # 软重置：衰减但不完全清空
            if "awareness_pool" in new_state:
                pool = new_state["awareness_pool"]
                pool.buffer = pool.buffer[-len(pool.buffer)//2:]
        elif reset_type == "backtrack":
            # 回溯：回退到上一个状态
            if "generation_history" in new_state and len(new_state["generation_history"]) > 0:
                prev_state = new_state["generation_history"].pop()
                new_state.update(prev_state)

        # 记录
        self.reset_history.append({
            "reset_type": reset_type,
            "state_keys": list(state.keys()),
            "step": state.get("step", 0)
        })

        return new_state

    def execute_tool(
        self,
        tool_name: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            context: 上下文信息

        Returns:
            工具执行结果
        """
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_func = self.tool_registry[tool_name]
        result = tool_func(context)

        self.tool_calls.append({
            "tool_name": tool_name,
            "context_keys": list(context.keys()),
            "success": result is not None,
            "timestamp": context.get("step", 0)
        })

        return result

    def execute_modify(
        self,
        state: Dict[str, Any],
        modification_type: str = "temperature",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行修改动作

        Args:
            state: 当前状态
            modification_type: 修改类型 ("temperature", "top_k", "top_p", "rephrase")
            kwargs: 额外参数

        Returns:
            修改后的状态
        """
        new_state = state.copy()

        if modification_type == "temperature":
            factor = kwargs.get("factor", 1.5)
            current_temp = state.get("temperature", 1.0)
            new_state["temperature"] = current_temp * factor
        elif modification_type == "top_k":
            new_state["top_k"] = kwargs.get("top_k", 50)
        elif modification_type == "top_p":
            new_state["top_p"] = kwargs.get("top_p", 0.9)
        elif modification_type == "rephrase":
            new_state["rephrase_prompt"] = kwargs.get("prompt", "Please rephrase this.")

        self.modify_history.append({
            "modification_type": modification_type,
            "kwargs": kwargs,
            "step": state.get("step", 0)
        })

        return new_state

    def inject_result(
        self,
        tool_result: Any,
        injection_type: str = "awareness"
    ) -> Dict[str, Any]:
        """
        将工具结果注入到上下文中

        Args:
            tool_result: 工具返回结果
            injection_type: 注入类型 ("awareness", "token", "hidden", "memory")

        Returns:
            注入数据
        """
        return {
            "type": injection_type,
            "data": tool_result,
            "timestamp": torch.tensor([[0]])  # 占位符
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            "total_resets": len(self.reset_history),
            "total_tool_calls": len(self.tool_calls),
            "total_modifications": len(self.modify_history),
            "reset_types": [r["reset_type"] for r in self.reset_history],
            "tool_names": [c["tool_name"] for c in self.tool_calls],
            "success_rate": sum(1 for c in self.tool_calls if c["success"]) / max(1, len(self.tool_calls))
        }
