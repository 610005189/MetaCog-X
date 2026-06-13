"""RL 训练骨架 v2（2-pass，显式 controller raw-logit surrogate）

诊断结论（2026-06-11）：
  在 MetaCogXModel 冻结 backbone 后，controller 的 temp_factor 虽然参与 TripleAttention
  的 scale，但 TripleAttention 的 Q/K/V 投影层（q_proj_c, k_proj_c, v_proj_c, out_proj,
  fusion 等）属于 backbone，被冻结。
  实测 torch.autograd.backward(CE) 在 frozen+alwayson 模式下会直接 RuntimeError：
    "element 0 of tensors does not require grad and does not have a grad_fn"

  → 原因：PyTorch 自动把 scalar@frozen_matmul 等路径的 temp_factor 当作叶子节点。

解决方案（2-pass 训练，保留 backbone 可训练但走监督初始化）：
  Pass-1（eval, 不建图）：model.eval() 跑一次 CE，收集 reward 基线（baseline_ppl）。
      用这个 ppl 判断是否陷入困境（ppl 显著大于 plain ppl 基准）。
  Pass-2（train, 建图）：model.train() 跑一次，走 forward 建完整计算图
      → controller 的 raw logit 和 temp_factor 参与 TripleAttention 的 scale，
        此时 TripleAttention 的 QKV Linear 层会产生 require_grad=True 的梯度流，
        所以 CE.backward() 会正确把梯度传回到 temp_factor_raw_logit（= controller.net）。

  为了避免跑两次 forward，我们也可以用"一次 train forward + 显式 reward surrogate"：
    在同一个 forward 中，同时：
      (a) CE — 让梯度真的流到 controller raw logit（通过 temp_factor scale → TripleAttention QKV 梯度）
      (b) controller raw-logit 的监督正则 — 显式告诉 controller 往哪个方向推
          结合 PROBE 结论 tf>1 ppl 变差、tf<1 ppl 不变，我们把 controller 的目标定义为：
            - ppl 差（比 baseline 差很多）→ 让 tf_raw → +∞（temp_factor→1.1 放大 attention → 触发 Reset）
            - ppl 好（接近 baseline）→    让 tf_raw → -∞（temp_factor→0.9 降注意力）
          实现：L1(ctrl.net[-1][0], target)，target 由 (ce - baseline_ce) 的符号决定。

  更稳健的做法（"显式 raw-logit surrogate 带符号"）：
    controller 的输出 tf_raw_logit ∈ R，进 sigmoid → [0,1] → 映射到 [0.9,1.1]
    "好方向"是：
      如果当前 CE > baseline（模型比基线差）→ 我们希望 tf 大一点（放大 attention 让模型看到更多）→ 目标 tf_raw +
      如果当前 CE < baseline（模型比基线好）→ 我们希望 tf 小一点（窄化注意力保精准）→ 目标 tf_raw -
    surrogate_raw = tf_raw_logit.mean() * sign(ce - baseline_ce)
                  = -tf_raw_logit.mean() * sign(baseline - ce)
    结合 probe 方向：tf↑ ppl↑（单调变差），tf↓ ppl 不变
    所以真正有区分度的只是"当前 ppl 比 baseline 差多少"：
      如果 ce > baseline → sign=+ → surrogate 大 → loss=-surrogate 小 → 梯度推 tf_raw -？
      反了... 让我们直接看符号：

    目标：
      ce 差 → 想让 tf ↑ → 把 tf_raw 推到 +∞（sigmoid≈1），在 raw_logit 上等价于 +1 bias
      ce 好 → 想让 tf ↓ → 把 tf_raw 推到 -∞（sigmoid≈0），在 raw_logit 上等价于 -1 bias
    loss_tf = - sign(ce - baseline) * tf_raw_logit.mean()
    当 ce > baseline（差）: sign=+ → loss_tf = - (+) * tf_raw → loss 在 tf_raw↑ 时更小 → 推 tf_raw ↑ ✅
    当 ce < baseline（好）: sign=- → loss_tf = - (-) * tf_raw → loss 在 tf_raw↓ 时更小 → 推 tf_raw ↓ ✅

  加上一个 controller 熵正则（logit 的多样性，让 tf_raw 不要饱和在 ±∞）：
    entropy_tf = softmax([tf_raw, 1 - tf_raw]) 两分类的熵 = -p·log(p) - (1-p)·log(1-p)，让它最大（均匀）
    但 tf_raw 是一个 logit，不是概率分布。
    改为 L2 正则：(tf_raw_logit - 0).pow(2) — 让它不要总是饱和到 +∞ 或 -∞
    但上面我们又需要它朝 ±∞ 推... 所以 L2 是冲突的。
    更好的做法：限制 tf_raw_logit ∈ [-10, +10]（通过 clamp），并加一个小幅度的熵正则在 skip_prob 和 mem_strength 上（如果这两路以后要用到）。
    实际上就只对 tf_raw_logit 加一个 L2 在 0 附近很弱的正则（0.01*L2），让它不要一下子饱和。

  gate 的正则：
    gate 输出 dilemma_score ∈ [0,1]
    如果 dilemma_score 高 + mode=plain → 惩罚（gate 该开门）
    如果 dilemma_score 低 + mode=metacog → 惩罚（gate 该关门）
    直接用 BCE：
      targets = torch.ones_like(score)   if mode == metacog
      targets = torch.zeros_like(score)  if mode == plain
      gate_loss = BCE(score, targets)  # 用 raw logit 版：BCEWithLogits

总 loss：
  total_loss = CE
             + λ_tf * loss_tf                # 显式 raw-logit 方向带符号
             + λ_gate * gate_loss            # gate 二分类监督
             + λ_tf_l2 * tf_raw_logit.pow(2).mean()  # 小幅度去饱和
             + λ_intervene * intervene_rate  # 保持 plain 模式低开销

训练设置：
  所有参数可训练（backbone 用监督 CE 初始化好了）
  AdamW(lr=1e-3 or 2e-3)
  注意 Double-After-Sigmoid 的 sign：当同时在 CE.backward() 和显式 raw-logit surrogate 上 backward，
  它们是同向相加的（一个想让 tf 在 "看更多/看更少" 方向正确，一个让 CE 更低）。
"""
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAINABLE_KEYWORDS = ()


def perplexity_from_loss(x: float) -> float:
    try:
        return float(math.exp(min(max(float(x), -20.0), 20.0)))
    except OverflowError:
        return float("inf")


class MinimalPPO:
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
        # loss_tf = -sign(ce - baseline) * tf_raw.mean()
        # ce 比 baseline 差 → sign > 0 → loss_tf = - (+) * tf_raw → 梯度推 tf_raw ↑ ✅
        # ce 比 baseline 好 → sign < 0 → loss_tf = - (-) * tf_raw → 梯度推 tf_raw ↓ ✅
        if tf_raw is not None and isinstance(tf_raw, torch.Tensor):
            # 使用可微分的符号函数近似，保持梯度流
            # ce - baseline 的符号决定我们希望 tf_raw 的方向
            ce_diff = ce - self.baseline_ce  # 保持在计算图中
            # 使用 softsign 或 tanh 作为可微分的符号近似
            # tanh(x * 10) 在 x 较大时接近 sign(x)，同时保持可微分
            sign_coef = torch.tanh(ce_diff * 10.0)  # [-1, 1]

            tf_mean = tf_raw.mean()
            # 核心损失：让 tf_raw 朝正确方向移动
            loss_tf = -sign_coef * tf_mean
            total_loss = total_loss + self.lambda_tf * loss_tf

            # 弱 L2 去饱和：tf_raw_logit 朝 0 有一点拉力，但别抢过 sign loss
            loss_tf_l2 = tf_raw.pow(2).mean()
            total_loss = total_loss + self.lambda_tf_l2 * loss_tf_l2

            parts["tf_surrogate"] = float(loss_tf.detach().cpu())
            parts["tf_l2"] = float(loss_tf_l2.detach().cpu())
            parts["tf_sign"] = float(sign_coef.detach().cpu())
            parts["tf_mean"] = float(tf_raw.mean().detach().cpu())
            parts["tf_raw_grad_fn"] = str(tf_raw.grad_fn) if tf_raw.grad_fn is not None else "NONE"
            parts["ce_diff"] = float(ce_diff.detach().cpu())

        # --- gate BCE ---
        # mode == metacog → gate score 应该接近 1
        # mode == plain   → gate score 应该接近 0
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


RLFramework = MinimalPPO
