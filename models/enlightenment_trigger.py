"""开悟触发器 (Enlightenment Trigger)

检测推理是否陷入无效循环或高熵不确定性，决定是否干预。
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


@dataclass
class TriggerResult:
    """触发结果"""
    triggered: bool
    action: TriggerAction
    confidence: float          # 触发置信度
    reason: str                # 触发原因描述
    repeat_count: int          # 当前连续重复数
    current_entropy: float     # 当前熵值


class EnlightenmentTrigger(nn.Module):
    """开悟触发器

    基于规则的触发器，检测：
    1. 重复token（连续重复）
    2. 高熵不确定性（输出分布过于均匀）
    3. 异常awareness模式
    """

    def __init__(
        self,
        entropy_thresh: float = 2.5,
        repeat_thresh: int = 3,
        entropy_patience: int = 5,
        high_entropy_bonus: float = 1.5,
        awareness_thresh: float = 0.8,
        trend_thresh: float = 0.5
    ):
        """
        Args:
            entropy_thresh: 熵阈值，超过则认为不确定性过高
            repeat_thresh: 连续重复阈值，超过则触发
            entropy_patience: 高熵持续次数阈值
            high_entropy_bonus: 高熵时的额外置信度加成
            awareness_thresh: awareness异常阈值
            trend_thresh: 趋势异常阈值
        """
        super().__init__()
        self.entropy_thresh = entropy_thresh
        self.repeat_thresh = repeat_thresh
        self.entropy_patience = entropy_patience
        self.high_entropy_bonus = high_entropy_bonus
        self.awareness_thresh = awareness_thresh
        self.trend_thresh = trend_thresh

        # 状态追踪
        self._entropy_counter = 0
        self._last_tokens: List[int] = []
        self._repeat_count = 0

    def reset(self):
        """重置触发器状态"""
        self._entropy_counter = 0
        self._last_tokens.clear()
        self._repeat_count = 0

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        计算输出分布的熵

        Args:
            logits: [B, L, vocab] 或 [vocab]

        Returns:
            平均熵值
        """
        if logits.dim() == 3:
            # 取最后一个位置的logits
            logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
        return entropy

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
        aware_stats,
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
        aware_stats=None,
        tokens: Optional[torch.Tensor] = None,
        step: int = 0
    ) -> TriggerResult:
        """
        前向传播，检测是否需要触发

        Args:
            logits: [B, L, vocab] 当前输出
            aware_stats: AwarenessStats 对象
            tokens: [B, L] 生成的token序列
            step: 当前推理步数

        Returns:
            TriggerResult
        """
        triggered = False
        action = TriggerAction.NONE
        confidence = 0.0
        reason = "normal"

        # 1. 检测重复token
        repeat_count = 0
        if tokens is not None:
            repeat_count, _ = self.detect_repeat(tokens)
            if repeat_count >= self.repeat_thresh:
                triggered = True
                action = TriggerAction.RESET
                confidence = min(1.0, repeat_count / self.repeat_thresh)
                reason = f"repeat_detected: {repeat_count} consecutive same tokens"

        # 2. 检测高熵
        entropy = self.compute_entropy(logits)
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

        # 3. 检测awareness异常
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
            current_entropy=entropy
        )


class AdaptiveEnlightenmentTrigger(nn.Module):
    """自适应开悟触发器

    根据任务类型和上下文动态调整触发策略。
    """

    def __init__(
        self,
        base_thresholds: Dict[str, float],
        num_task_types: int = 4
    ):
        super().__init__()
        self.base_trigger = EnlightenmentTrigger(
            entropy_thresh=base_thresholds.get("entropy", 2.5),
            repeat_thresh=int(base_thresholds.get("repeat", 3)),
            entropy_patience=int(base_thresholds.get("patience", 5))
        )

        # 任务特定阈值调整
        self.task_thresholds = nn.Embedding(num_task_types, 4)  # 4个阈值参数

    def forward(
        self,
        logits: torch.Tensor,
        aware_stats=None,
        tokens: Optional[torch.Tensor] = None,
        step: int = 0,
        task_id: Optional[torch.Tensor] = None
    ) -> TriggerResult:
        result = self.base_trigger(logits, aware_stats, tokens, step)

        # 如果指定了任务类型，可以动态调整
        if task_id is not None and self.training:
            # 任务自适应调整（训练时）
            pass

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

    def execute_reset(
        self,
        state: Dict[str, Any],
        reset_type: str = "full"
    ) -> Dict[str, Any]:
        """
        执行重置动作

        Args:
            state: 当前状态字典
            reset_type: 重置类型 ("full", "partial", "soft")

        Returns:
            重置后的状态
        """
        new_state = state.copy()

        if reset_type == "full":
            # 完全重置
            new_state["awareness_pool"].reset()
            new_state["hidden_states"] = None
            new_state["repeat_count"] = 0
        elif reset_type == "partial":
            # 部分重置：只重置awareness
            if "awareness_pool" in new_state:
                new_state["awareness_pool"].reset()
        elif reset_type == "soft":
            # 软重置：衰减但不完全清空
            if "awareness_pool" in new_state:
                pool = new_state["awareness_pool"]
                pool.buffer = pool.buffer[-len(pool.buffer)//2:]

        # 记录
        self.reset_history.append({
            "reset_type": reset_type,
            "state_keys": list(state.keys())
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
            "success": result is not None
        })

        return result

    def inject_result(
        self,
        tool_result: Any,
        injection_type: str = "awareness"
    ) -> Dict[str, Any]:
        """
        将工具结果注入到上下文中

        Args:
            tool_result: 工具返回结果
            injection_type: 注入类型 ("awareness", "token", "hidden")

        Returns:
            注入数据
        """
        return {
            "type": injection_type,
            "data": tool_result
        }
