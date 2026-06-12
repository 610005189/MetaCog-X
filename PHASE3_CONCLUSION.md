# MetaCog-X Phase 3 Conclusion — A/B + Probe

日期: 2026-06-10 / 11
运行步数: 300 steps (all variants)
参数量: 17,020,086 (identical across the 4 variants)
控制变量: enable_metacog=True 对全部 4 组；差异仅在辅助损失权重 α/β/γ/δ。

---

## 1. Variant PPL 对比表（来自 `tests/verify_task4_summary.py`）

| Variant                 | α    | β     | γ    | δ     | Best PPL | Δ vs GPT | % vs GPT |
|-------------------------|------|-------|------|-------|---------:|---------:|---------:|
| GPT (baseline)          | 0.00 | 0.000 | 0.00 | 0.000 |     9.20 |     0.00 |     0.0% |
| Aware only (β alone)    | 0.00 | 0.005 | 0.00 | 0.000 |     8.79 |    -0.41 |    -4.5% |
| Meta only (α alone)     | 0.01 | 0.000 | 0.00 | 0.000 |     8.87 |    -0.34 |    -3.6% |
| MetaCog-X (full)        | 0.01 | 0.005 | 0.02 | 0.005 |     9.24 |    +0.04 |    +0.5% |

**WINNER: Aware only (β=0.005)** — perplexity 8.79，比 GPT 基线低 0.41（-4.5%）。
其次是 Meta only (α=0.01)，低 0.34（-3.6%）。
**Full variant 表现最差**，比 GPT 基线还高 +0.04。

> 解释：
> - GPT 变体 enable_metacog=True 但 α=β=γ=δ=0 → meta/awareness/controller 仍 forward 但零辅助梯度 → 实际上是纯 content CE 训练。
> - Aware-only (β) 通过 awareness self-prediction 损失拉低 ppl 0.41，是本阶段最有效的单一辅助项。
> - Meta-only (α) 通过 meta temporal-consistency 拉低 ppl 0.34。
> - Full (α+β+γ+δ) 出现**过正则化**：γ（controller entropy bonus）把 3 路 softmax 从高熵推向 collapse，δ（meta diversity）把 inter-layer meta centroid 硬拉开，两者叠加让 representation 变得不稳定，最终 ppl 比纯 GPT 还差。

---

## 2. Representation Probe 结果（来自 `scripts/representation_probe.py`）

### 2.1 6 指标总览表

| Variant                 | LayerMetaOffDiag | CtrlEntropy(b) | TempFactorStd  | AwInter | AwRatio | BatchMetaMSE |
|-------------------------|-----------------:|---------------:|---------------:|--------:|--------:|-------------:|
| GPT (α=β=γ=δ=0)         |           0.9957 |         1.5848 |      0.000000  |   0.433 |  13.53  |  3.0e-07     |
| Aware only (β=0.005)    |           0.9934 |         1.5847 |      0.000002  |   0.357 |  13.48  |  2.5e-07     |
| Meta only (α=0.01)      |           0.9955 |         1.5846 |      0.000002  |   0.392 |  13.00  |  2.0e-07     |
| Full (α+β+γ+δ)          |           0.0577 |         0.0249 |      0.000001  |   0.317 |  14.67  |  4.0e-08     |

### 2.2 指标解读

**(a) Layer-wise Meta Cosine Off-Diag（层间 meta 去相关度）**

- GPT / Aware-only / Meta-only：0.993~0.996 → **NO DIVERSITY**（层间 meta 基本一样）
- Full（仅这一组）：0.058 → **GOOD DIVERSITY**（4 层 meta 完全解耦）

→ δ=0.005 确实起作用，但 Full 中 0.058 意味着层间 meta 几乎正交 → **过强**，层间传递链条被打断，导致 content 分支也受影响。

**(b) Controller Entropy（3 路 softmax bits/token）**

理论最大值 log₂(3) ≈ 1.585 bits。

- GPT / Aware-only / Meta-only：1.5846~1.5848 → 几乎最大熵，三路均匀随机
- Full：0.025 → **几乎完全坍缩**

→ γ=0.02 熵奖励的方向写反了？代码里 γ 惩罚低熵 → 但 softmax 最大熵是均匀分布（loss 最大），训练时模型反而被推到 collapse？需要在 Phase 4 重新确认损失符号。

**(c) TempFactor Std**

所有 4 组 std ≈ 0 → **COLLAPSED**。

→ temp_factor 只取 0.8 + 0.4×sigmoid(logit₀)，controller collapse 时 sigmoid(logit₀) 饱和在 0 或 1 → temp_factor 固定在 0.8 或 1.2，所有 batch 一样。这意味着 TripleAttention content 分支实际上**没有被动态温度调节**。

**(d) Awareness Ratio（real-vs-gibberish 区分度）**

四组 aw_ratio ∈ [13.0, 14.7] → **EXCELLENT DISTINCTION**（ratio>10）。

→ awareness 自预测损失（β）虽然对 ppl 最有帮助，但即使 GPT 组（β=0）awareness 也能很好区分 real/gibberish → 区分能力主要来自 content 路径通过认知粒子生成器的共享表示。

**(e) Inter-batch Meta MSE**

Full 组最小（4e-08）→ 但 LayerMetaOffDiag 又显示它层间最解耦 → 两者组合说明 Full 的 meta 向量各层差异极大但每层内部跨 batch 又极稳定 → 是被 δ 推到各自方向的"饱和簇"，稳定但不再携带有用的跨层转移信号。

---

## 3. Run 与 Tests 验证

- `python run.py --mode full_test` — 7/7 PASS（参数量 17,020,086）
- `python tests/run_tests.py`
  - 单元测试 13/13 PASS
  - 集成测试 5/5 PASS
  - 评估指标 5/5 PASS（推理开销 +3.43% ≤10%，自我干预 71%≥60%，开悟解脱 70.5%≥50%，觉知召回 100%≥70%）

---

## 4. Phase 4 下一步（PPO 强化控制器）

Phase 3 结论：**辅助损失中 β（awareness self-prediction）和 α（meta temporal consistency）是有效的**，但 γ（controller entropy bonus）方向/强度需重调，δ（meta diversity）过强导致层间断裂，temp_factor 在所有组都 collapse → controller 没发挥作用。

Phase 4 切换到 PPO 直接优化 controller，计划：

1. **Controller 参数冻结，content/meta/aware backbone 用 Aware-only 权重初始化**（Phase 3 最优点 ppl=8.79）。
2. **PPO 只优化 `meta_controller.net[0]`（第一层线性层）+ 偏置**，避免过参数；每步以 `temp_factor` 对 TripleAttention 的 ppl 改善作为 reward。
3. **Reward = Δ_ppl**（连续）+ **entropy bonus**（防止 collapse）+ **temp_factor std ≥ 0.1 的正则项**。
4. **Gym**：10 个 bundle 样本算一次 PPO rollout，actor 每 5 步更新一次 critic 基线。
5. **Compare**：PPO-aware vs Aware-only（Phase 3 winner）vs GPT baseline，跑 300 steps + ppl + probe。

风险控制：
- PPO 探索率从 ε=0.1 开始线性衰减到 0.01。
- 每 50 steps 跑一次 probe（LayerMetaOffDiag / CtrlEntropy / TempFactorStd 三维监控 collapse）。
- 如果 TempFactorStd 前 20 steps 仍 < 0.01 → 扩大 action space 到直接输出 temp_factor 残差 [0.95, 1.05]。

---

## 5. Artifact 清单

- `runs/gpt_300.pt` / `runs/gpt_300.csv`
- `runs/metacog_aware_only_300.pt` / `runs/metacog_aware_only_300.csv`
- `runs/metacog_meta_only_300.pt` / `runs/metacog_meta_only_300.csv`
- `runs/metacog_full_300.pt` / `runs/metacog_full_300.csv`
- `runs/probe_summary_4variants.csv`
- `scripts/representation_probe.py`（Phase 3 升级版）
- `PHASE3_CONCLUSION.md`（本文档）
