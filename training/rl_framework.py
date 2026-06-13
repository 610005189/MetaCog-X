"""RL 训练框架 v3 - 完整 PPO 实现

包含完整的 Proximal Policy Optimization (PPO) 算法实现：
- 独立的策略网络和价值网络
- GAE (Generalized Advantage Estimation) 优势估计
- PPO 裁剪损失 (Clipped Surrogate Objective)
- 策略熵正则化
- 与 MetaCogX 模型集成支持
"""
import math
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAINABLE_KEYWORDS = ()


def perplexity_from_loss(x: float) -> float:
    try:
        return float(math.exp(min(max(float(x), -20.0), 20.0)))
    except OverflowError:
        return float("inf")


class PolicyNetwork(nn.Module):
    """独立策略网络"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,  # temp_factor, skip_prob, mem_strength
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出原始 logits"""
        return self.net(x)


class ValueNetwork(nn.Module):
    """独立价值网络"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出状态价值估计"""
        return self.net(x).squeeze(-1)


class TrajectoryBuffer:
    """轨迹缓冲区，用于存储 PPO 训练数据"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: List[Dict[str, torch.Tensor]] = []

    def add(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """添加轨迹数据"""
        self.buffer.append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "values": values,
            "dones": dones,
        })
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()

    def get_all(self) -> Dict[str, torch.Tensor]:
        """获取所有数据并拼接"""
        if not self.buffer:
            return {}

        keys = self.buffer[0].keys()
        result = {}
        for key in keys:
            result[key] = torch.cat([item[key] for item in self.buffer], dim=0)
        return result

    def __len__(self) -> int:
        return len(self.buffer)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GAE (Generalized Advantage Estimation)

    Args:
        rewards: [T, B] 奖励序列
        values: [T+1, B] 价值估计（包含最后一步的 bootstrap）
        dones: [T, B] 终止标志
        gamma: 折扣因子
        lambda_: GAE 参数

    Returns:
        advantages: [T, B] 优势估计
        returns: [T, B] 目标价值（带优势的回报）
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lambda_ * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]

    returns = values[:-1] + advantages
    return advantages, returns


class PPO(nn.Module):
    """完整的 Proximal Policy Optimization 实现"""

    def __init__(
        self,
        model,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # 提取策略输入维度（基于模型的 controller）
        self._setup_policy_networks()

        # 优化器
        params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        # 轨迹缓冲区
        self.trajectory_buffer = TrajectoryBuffer()

    def _setup_policy_networks(self) -> None:
        """设置策略网络和价值网络"""
        # 输入维度：meta维度 + awareness统计维度
        input_dim = 128  # 默认值，实际使用时根据模型调整
        self.policy_net = PolicyNetwork(input_dim).to(self.device)
        self.value_net = ValueNetwork(input_dim).to(self.device)

    def _extract_features(self, out: Dict[str, Any]) -> torch.Tensor:
        """从模型输出中提取特征用于策略网络"""
        # 提取 meta 和 awareness 特征
        meta = out.get("meta", None)
        aware_stats = out.get("aware_stats", None)

        if meta is not None:
            if meta.dim() == 3:
                meta = meta.mean(dim=1)  # [B, L, D] -> [B, D]
        else:
            meta = torch.zeros(out["logits"].shape[0], 64, device=self.device)

        if aware_stats is not None:
            # 拼接 awareness 统计
            stats = torch.cat([
                aware_stats.mean,
                aware_stats.std,
                aware_stats.trend,
            ], dim=-1)
        else:
            stats = torch.zeros(out["logits"].shape[0], 64, device=self.device)

        return torch.cat([meta, stats], dim=-1)

    def _compute_log_probs(self, actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        # actions: [B, 3] (temp_factor, skip_prob, mem_strength)
        # logits: [B, 3] 原始输出
        probs = torch.sigmoid(logits)
        # 简化的对数概率计算（假设独立高斯分布）
        return torch.log(probs + 1e-8).sum(dim=-1)

    def _get_actions(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 logits 生成动作和对数概率"""
        # temp_factor: [0.9, 1.1]
        tf = 0.9 + 0.2 * torch.sigmoid(logits[:, 0])
        # skip_prob: [0, 0.2]
        sk = 0.2 * torch.sigmoid(logits[:, 1])
        # mem_strength: [0.5, 1.0]
        mm = 0.5 + 0.5 * torch.sigmoid(logits[:, 2])

        actions = torch.stack([tf, sk, mm], dim=-1)
        log_probs = self._compute_log_probs(actions, logits)

        return actions, log_probs

    def collect_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """收集轨迹数据"""
        self.model.eval()

        if labels is None:
            labels = batch.get("labels", batch.get("input_ids"))

        with torch.no_grad():
            out = self.model(**batch, return_meta=True, enable_metacog=True)
            features = self._extract_features(out)

            # 获取策略输出
            policy_logits = self.policy_net(features)
            value = self.value_net(features)
            actions, log_probs = self._get_actions(policy_logits)

            # 计算奖励（基于 perplexity）
            ce = F.cross_entropy(
                out["logits"][..., :-1, :].contiguous().view(-1, out["logits"].size(-1)),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=0,
                reduction="none",
            ).view(out["logits"].shape[0], -1).mean(dim=-1)
            reward = -ce  # 负的交叉熵作为奖励

        # 判断是否结束（简化：单步轨迹）
        dones = torch.zeros_like(reward)

        # 添加到缓冲区
        self.trajectory_buffer.add(
            states=features,
            actions=actions,
            rewards=reward.unsqueeze(0),
            log_probs=log_probs.unsqueeze(0),
            values=value.unsqueeze(0),
            dones=dones.unsqueeze(0),
        )

        return {
            "reward": float(reward.mean().cpu()),
            "value": float(value.mean().cpu()),
            "log_prob": float(log_probs.mean().cpu()),
            "trajectory_size": len(self.trajectory_buffer),
        }

    def update_policy(self) -> Dict[str, float]:
        """执行 PPO 策略更新"""
        self.model.train()
        self.policy_net.train()
        self.value_net.train()

        if len(self.trajectory_buffer) == 0:
            return {"error": "Empty trajectory buffer"}

        # 获取所有轨迹数据
        data = self.trajectory_buffer.get_all()
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        old_log_probs = data["log_probs"]
        old_values = data["values"]
        dones = data["dones"]

        T, B = rewards.shape

        # 计算 GAE 优势估计
        # 添加 bootstrap value（最后一步的价值估计）
        with torch.no_grad():
            last_value = self.value_net(states[-1] if T > 1 else states[0]).unsqueeze(0)
            values = torch.cat([old_values, last_value], dim=0)

        advantages, returns = compute_gae(
            rewards, values, dones, self.gamma, self.lambda_gae
        )

        # 扁平化数据用于 mini-batch
        states_flat = states.reshape(-1, states.size(-1))
        actions_flat = actions.reshape(-1, actions.size(-1))
        old_log_probs_flat = old_log_probs.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # 标准化优势
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0

        # PPO 更新循环
        for _ in range(self.ppo_epochs):
            # 随机打乱
            indices = torch.randperm(len(states_flat))
            for i in range(0, len(states_flat), self.mini_batch_size):
                batch_idx = indices[i:i + self.mini_batch_size]
                batch_states = states_flat[batch_idx]
                batch_actions = actions_flat[batch_idx]
                batch_old_log_probs = old_log_probs_flat[batch_idx]
                batch_advantages = advantages_flat[batch_idx]
                batch_returns = returns_flat[batch_idx]

                self.opt.zero_grad()

                # 获取新的策略输出
                new_logits = self.policy_net(batch_states)
                new_value = self.value_net(batch_states)
                _, new_log_probs = self._get_actions(new_logits)

                # PPO 裁剪损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # 价值损失
                value_loss = F.mse_loss(new_value, batch_returns)

                # 熵正则化（鼓励策略探索）
                probs = torch.sigmoid(new_logits)
                entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).sum(dim=-1).mean()
                entropy_loss = -self.entropy_coef * entropy

                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                self.opt.step()

                total_policy_loss += float(policy_loss.detach().cpu())
                total_value_loss += float(value_loss.detach().cpu())
                total_entropy_loss += float(entropy_loss.detach().cpu())

        # 清空缓冲区
        self.trajectory_buffer.clear()

        num_updates = self.ppo_epochs * (len(states_flat) // self.mini_batch_size)
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "total_updates": num_updates,
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        update_interval: int = 10,
    ) -> Dict[str, Any]:
        """执行一步训练（收集轨迹 + 可能的策略更新）"""
        collect_result = self.collect_trajectory(batch, labels)

        # 每 update_interval 步更新一次策略
        if len(self.trajectory_buffer) >= update_interval:
            update_result = self.update_policy()
            collect_result.update(update_result)

        return collect_result

    @torch.no_grad()
    def validate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """验证模型性能"""
        self.model.eval()
        self.policy_net.eval()
        self.value_net.eval()

        labels = batch.get("labels", batch.get("input_ids"))
        out = self.model(**batch, return_meta=True, enable_metacog=True)

        ce = F.cross_entropy(
            out["logits"][..., :-1, :].contiguous().view(-1, out["logits"].size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=0,
            reduction="mean",
        )
        ppl = perplexity_from_loss(float(ce.cpu()))

        # 获取策略评估
        features = self._extract_features(out)
        value = self.value_net(features).mean()

        return {
            "val_loss": float(ce.cpu()),
            "val_ppl": ppl,
            "val_value": float(value.cpu()),
            "mode": out.get("mode"),
            "switch_stats": out.get("switch_stats", {}),
            "dilemma_score": out.get("last_dilemma_score"),
        }


class MinimalPPO:
    """简化版 PPO（保持向后兼容）"""

    def __init__(
        self,
        model,
        lr: float = 2e-3,
        lambda_tf: float = 0.5,
        lambda_gate: float = 1.0,
        lambda_tf_l2: float = 0.01,
        lambda_intervene: float = 0.0,
        baseline_ce: float = 2.0,
        device: str = "cpu",
    ):
        self.device = device
        self.model = model.to(device)
        self.lambda_tf = lambda_tf
        self.lambda_gate = lambda_gate
        self.lambda_tf_l2 = lambda_tf_l2
        self.lambda_intervene = lambda_intervene
        self.baseline_ce = baseline_ce

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(self.model.parameters())
            for p in trainable_params:
                p.requires_grad = True
        self.opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    def cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction="mean",
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self.model.train()
        self.opt.zero_grad()

        out = self.model(**batch, return_meta=True, enable_metacog=True)

        if labels is None:
            labels = batch.get("labels", None)
            if labels is None:
                labels = batch["input_ids"]

        self._last_forward_out = out

        ce = self.cross_entropy(out["logits"], labels)
        ppl = perplexity_from_loss(float(ce.detach().cpu()))

        switches = int(out.get("switch_stats", {}).get("switches", 0))
        total_fwd = int(out.get("switch_stats", {}).get("total_forward", 1))
        meta_steps = int(out.get("switch_stats", {}).get("meta_steps", 0))
        plain_steps = int(out.get("switch_stats", {}).get("plain_steps", 0))

        mode = out.get("mode", "plain")
        ds = out.get("last_dilemma_score", None)
        tf_raw = out.get("ctrl_tf_raw_logit", None)  # [B, 1]

        total_loss = ce
        parts = {"ce": float(ce.detach().cpu())}

        # --- controller TF surrogate ---
        if tf_raw is not None and isinstance(tf_raw, torch.Tensor):
            ce_diff = ce - self.baseline_ce
            sign_coef = torch.tanh(ce_diff * 10.0)
            tf_mean = tf_raw.mean()
            loss_tf = -sign_coef * tf_mean
            total_loss = total_loss + self.lambda_tf * loss_tf

            loss_tf_l2 = tf_raw.pow(2).mean()
            total_loss = total_loss + self.lambda_tf_l2 * loss_tf_l2

            parts["tf_surrogate"] = float(loss_tf.detach().cpu())
            parts["tf_l2"] = float(loss_tf_l2.detach().cpu())
            parts["tf_sign"] = float(sign_coef.detach().cpu())
            parts["tf_mean"] = float(tf_raw.mean().detach().cpu())
            parts["tf_raw_grad_fn"] = str(tf_raw.grad_fn) if tf_raw.grad_fn is not None else "NONE"
            parts["ce_diff"] = float(ce_diff.detach().cpu())

        # --- gate BCE ---
        if ds is not None:
            score_t = torch.tensor(float(ds), device=ce.device)
            if mode == "metacog":
                target = torch.ones_like(score_t)
            else:
                target = torch.zeros_like(score_t)
            gate_loss = F.binary_cross_entropy(score_t, target, reduction="mean")
            total_loss = total_loss + self.lambda_gate * gate_loss
            parts["gate_bce"] = float(gate_loss.detach().cpu())

        # --- intervene_rate 正则（可选）---
        if self.lambda_intervene > 0 and (meta_steps + plain_steps) > 0:
            rate = meta_steps / float(meta_steps + plain_steps)
            total_loss = total_loss + self.lambda_intervene * torch.tensor(rate, device=ce.device)
            parts["intervene_rate"] = rate

        total_loss.backward()

        clip_val = 1.0
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=clip_val,
        )
        self.opt.step()

        return {
            "loss": float(total_loss.detach().cpu()),
            "ce_loss": float(ce.detach().cpu()),
            "ppl": ppl,
            "mode": mode,
            "switches": switches,
            "meta_steps": meta_steps,
            "plain_steps": plain_steps,
            "total_forward": total_fwd,
            "dilemma_score": ds,
            "switch_stats": out.get("switch_stats", {}),
            "parts": parts,
            "forward_out": self._last_forward_out,
        }

    @torch.no_grad()
    def validate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.model.eval()
        out = self.model(**batch, return_meta=True, enable_metacog=True)
        labels = batch.get("labels", batch.get("input_ids"))
        ce = self.cross_entropy(out["logits"], labels)
        ppl = perplexity_from_loss(float(ce.detach().cpu()))
        st = out.get("switch_stats", {})
        return {
            "val_loss": float(ce.detach().cpu()),
            "val_ppl": ppl,
            "mode": out.get("mode"),
            "switch_stats": st,
            "dilemma_score": out.get("last_dilemma_score"),
        }


RLFramework = PPO
