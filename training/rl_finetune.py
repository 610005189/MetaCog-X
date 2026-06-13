"""强化学习微调模块

包含：
1. PPO/GRPO 算法实现
2. 元认知控制器训练
3. 开悟触发器训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class Trajectory:
    """一条轨迹"""
    states: List[torch.Tensor]      # 每个时间步的状态
    actions: List[int]              # 每个时间步的动作
    rewards: List[float]            # 每个时间步的奖励
    log_probs: List[torch.Tensor]    # 每个动作的log概率
    values: List[torch.Tensor]       # 每个状态的价值估计
    entropies: List[torch.Tensor]    # 每个时间步的熵


class MetaControllerPPO:
    """元认知控制器的PPO训练

    训练稀疏元认知控制器学会使用控制信号。
    """

    def __init__(
        self,
        controller: nn.Module,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0
    ):
        """
        Args:
            controller: SparseMetaController
            lr: 学习率
            clip_eps: PPO裁剪范围
            entropy_coef: 熵正则系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
        """
        self.controller = controller
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            controller.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        self.device = next(controller.parameters()).device

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE (Generalized Advantage Estimation)

        Args:
            rewards: 奖励列表
            values: 价值列表
            gamma: 折扣因子
            lam: GAE参数

        Returns:
            (advantages, returns)
        """
        advantages = []
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = torch.tensor(0.0, device=self.device)
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value.item() - values[t].item()
            advantages.insert(0, last_gae * gamma * lam + delta)
            last_gae = advantages[0]

        advantages = torch.tensor(advantages, device=self.device)
        returns = advantages + torch.stack(values)

        # 标准化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(
        self,
        trajectories: List[Trajectory],
        old_log_probs: torch.Tensor
    ) -> Dict[str, float]:
        """
        PPO更新

        Args:
            trajectories: 轨迹列表
            old_log_probs: 旧的对数概率

        Returns:
            训练指标
        """
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for traj in trajectories:
            # 计算GAE
            advantages, returns = self.compute_gae(traj.rewards, traj.values)

            # 重新计算log_probs和entropies
            new_log_probs = torch.stack(traj.log_probs)
            entropies = torch.stack(traj.entropies)

            # PPO策略损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            values_pred = torch.stack(traj.values)
            value_loss = F.mse_loss(values_pred, returns)

            # 熵损失（鼓励探索）
            entropy_loss = -entropies.mean()

            # 总损失
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()

        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.max_grad_norm)
        self.optimizer.step()

        num_trajs = len(trajectories)
        return {
            "loss": total_loss / num_trajs,
            "policy_loss": total_policy_loss / num_trajs,
            "value_loss": total_value_loss / num_trajs,
            "entropy": total_entropy / num_trajs
        }


class GRPO:
    """GRPO (Group Relative Policy Optimization) 算法

    一种简化的策略优化方法，比PPO更简单。
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        lr: float = 1e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        # 冻结参考策略
        for p in self.ref_policy.parameters():
            p.requires_grad = False

    def compute_reward(
        self,
        task_success: bool,
        energy_cost: float,
        redundant_steps: int,
        enlightenment_count: int
    ) -> float:
        """
        计算奖励

        R = task_success - λ1 * energy - λ2 * redundant - λ3 * enlightenment
        """
        lambda1, lambda2, lambda3 = 0.01, 0.1, 0.5

        reward = float(task_success)
        reward -= lambda1 * energy_cost
        reward -= lambda2 * redundant_steps
        reward -= lambda3 * enlightenment_count

        return reward

    def update(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        GRPO更新

        Args:
            samples: 样本列表，每个包含 {states, actions, rewards, log_probs}
        """
        policy_losses = []
        entropy_losses = []

        for sample in samples:
            rewards = sample["rewards"]
            old_log_probs = sample["log_probs"]

            # 归一化奖励
            rewards = torch.tensor(rewards, dtype=torch.float32)
            if len(rewards) > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # 计算策略损失
            new_log_probs = torch.stack(sample["log_probs"])
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 组内相对优势
            advantages = rewards

            # GRPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 熵正则
            entropy_loss = -new_log_probs.mean()

            total_loss = policy_loss + self.entropy_coef * entropy_loss

            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())

            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return {
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "entropy_loss": sum(entropy_losses) / len(entropy_losses)
        }


class RewardCalculator:
    """奖励计算器

    根据任务成功率和各种成本计算奖励。
    """

    def __init__(
        self,
        lambda_energy: float = 0.01,
        lambda_redundant: float = 0.1,
        lambda_enlightenment: float = 0.5,
        lambda_ctrl_complexity: float = 0.01
    ):
        self.lambda_energy = lambda_energy
        self.lambda_redundant = lambda_redundant
        self.lambda_enlightenment = lambda_enlightenment
        self.lambda_ctrl = lambda_ctrl_complexity

    def compute_reward(
        self,
        task_success: bool,
        energy_cost: float,
        num_steps: int,
        optimal_steps: int,
        enlightenment_count: int,
        control_complexity: float
    ) -> float:
        """
        计算最终奖励

        Returns:
            奖励值
        """
        # 任务成功奖励
        success_reward = 1.0 if task_success else 0.0

        # 效率惩罚（超过最优步数的部分）
        redundant = max(0, num_steps - optimal_steps)
        efficiency_penalty = self.lambda_redundant * redundant

        # 能耗惩罚
        energy_penalty = self.lambda_energy * energy_cost

        # 开悟次数惩罚（开悟应该有成本）
        enlightenment_penalty = self.lambda_enlightenment * enlightenment_count

        # 控制复杂度惩罚
        ctrl_penalty = self.lambda_ctrl * control_complexity

        total_reward = (
            success_reward
            - efficiency_penalty
            - energy_penalty
            - enlightenment_penalty
            - ctrl_penalty
        )

        return total_reward

    def compute_step_reward(
        self,
        step: int,
        entropy: float,
        repeat_count: int,
        enlightenment_active: bool
    ) -> float:
        """
        计算每步的中间奖励（用于GAE）

        Returns:
            即时奖励
        """
        reward = 0.0

        # 熵奖励（低熵更好）
        if entropy > 2.5:
            reward -= 0.1
        elif entropy < 1.0:
            reward += 0.05

        # 重复惩罚
        if repeat_count > 0:
            reward -= 0.1 * repeat_count

        # 开悟激活奖励（鼓励正确的干预）
        if enlightenment_active:
            reward += 0.2

        return reward
