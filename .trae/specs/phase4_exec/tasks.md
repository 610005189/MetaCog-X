# Phase IV 实证验证 — 任务分解

## [x] Task 0：Code Review / Baseline commit
- **Priority**: P0
- **Depends On**: None
- **Description**: 把 Phase 0–3 的 baseline 代码提交到 git（.gitignore 已配好），作为 Phase IV 的起点
- **Test Requirements**:
  - `git log --oneline -1` 返回 1 个 commit
  - `git status` 显示 clean working tree
- **Status**: ✅ Done (commit `342d97d`)

---

## [ ] Task 1：修复 run_ab_v2.py — 补全 alwayson + ppl/loss 写回
- **Priority**: P0
- **Depends On**: Task 0
- **Description**:
  1. 在 main() 的 variants 循环里加入 `alwayson_meta` variant（`enable_metacog=True`，并且在 forward 时强制 `gate.threshold = float('inf')` 或在 layer 里强制 `metacog head 永远执行`）
  2. 每跑完一个 variant 的 train 循环后，调用 validate_ppl(val_dl, model, enable_metacog, mode) 计算 final_ppl 和 final_loss
  3. conditional variant 额外做 switches 统计：在验证集 forward 时计数 metacog head 被激活的次数
  4. JSON 写出到 `runs/ab_results_v3.json`，所有 variant 的 ppl/loss/switches/plain_pct/score_mean 都非 null
  5. 打印格式改成 flush=True（避免输出被截断）
- **Acceptance Criteria Addressed**: AC-1（可对比的 JSON）, AC-2（三 variant）
- **Test Requirements**:
  - `programmatic` TR-1.1: 用 `--quick` 参数（steps=10, val_samples=32）跑完 → 3 个 variant 都在 JSON 里且 ppl/loss 非 null
  - `programmatic` TR-1.2: alwayson variant ppl ≥ plain（方向对）
  - `human-judgement` TR-1.3: 代码风格与现有 run_ab_v2.py 保持一致，没有新引入的无用 import
- **Notes**: forward 里 mode="alwayson" 的含义是 L1 Gate 永远不阻止 metacog head；实现时可以简单地在 run 脚本里在 forward 前设置 `model.gate.l1_threshold = float('inf')` 并在每个 layer 的 metacog head 里强制 enable

## [ ] Task 2：完整 A/B 实证验证（d_model=128）
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 用默认参数（steps=500, train_samples=1200, val_samples=300, d_model=128）跑完整 A/B
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3
- **Test Requirements**:
  - `programmatic` TR-2.1: 运行时间 ≤ 15 分钟（wall clock）
  - `programmatic` TR-2.2: conditional controller std 在 50–500 steps 之间 > 0.05
  - `programmatic` TR-2.3: conditional switches > 0
  - `programmatic` TR-2.4: plain ppl 和 alwayson ppl 差异在 ±20% 以内（训练都收敛）
  - `human-judgement` TR-2.5: console 输出格式清晰，JSON 结构与 spec 一致
- **Notes**: 如果 TR-2.2 失败（ctrl_std ≈ 0）→ conditional 模式在退化；可能是 controller 的输出被 Softmax 饱和，需要调 controller 的 init 或 lr

## [ ] Task 3：Triple Attention 消融
- **Priority**: P1
- **Depends On**: Task 2
- **Description**: 新增一个 variant `no_tri_attn`（d_model=128, plain backbone，但在 MetaCogXLayer 里 `tri_attn=None` 或 `disable_tri_attn=True`），跑 500 steps + ppl 验证
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `programmatic` TR-3.1: ppl(no_tri_attn) ≥ ppl(plain)（消融指标 ≥ baseline）
  - `programmatic` TR-3.2: 代码改动能通过一个 `disable_tri_attn=True` kwarg 关掉 Triple Attention
- **Notes**: 如果 ppl(no_tri_attn) 反而 < ppl(plain)，说明 Triple Attention 在当前 tiny 模型里是负贡献（可能是参数太多导致过拟合）

## [ ] Task 4：DMN 消融
- **Priority**: P1
- **Depends On**: Task 2
- **Description**: conditional variant 去掉 DMN（`use_dmn=False` 或在 layer 里跳过 surprise 分支），跑 500 steps + ppl 验证 + switches 统计
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-4.1: conditional(no_dmn) ppl 和 conditional(with_dmn) ppl 都在合理范围（≤ 15）
  - `programmatic` TR-4.2: no_dmn 的 switches 统计仍可计算（forward 不会因为 dmn=None 崩溃）
- **Notes**: DMN 是 surprise 驱动；没有 DMN 时 L1 Gate 仍然可以用其他 self_features 维度（ctrl_logits, mode_prob 等）工作

## [ ] Task 5（可选）：加宽到 d_model=256 再跑一轮
- **Priority**: P2
- **Depends On**: Task 2
- **Description**: 如果 Task 2 的 ppl 都非常接近（差异 < 2%），说明 tiny 模型 capacity 不够 → 把 d_model 改成 256 重跑 Task 2
- **Test Requirements**: 同 Task 2
- **Notes**: 预计耗时 ≈ 20 分钟

## [ ] Task 6：写总结 + Phase IV 结论
- **Priority**: P1
- **Depends On**: Task 2 + Task 3 + Task 4
- **Description**: 根据所有实验的 JSON 结果写一份 Phase IV 结论，包含：
  - 各 variant ppl/CE loss 对比表
  - conditional vs plain 的相对提升/下降百分比
  - Triple Attention 贡献度
  - DMN 贡献度
  - 是否值得进入 Phase V（投稿级训练）
- **Notes**: 只有当 conditional 有明确优势（ppl 至少不比 plain 差 2% 以上）时，才能认真考虑 Phase V

---

> 执行策略：按 Task 1 → 2 → 3 → 4 顺序串行，每步完成后立即验证 TR，发现 bug 就就地修，不进入下一个任务。
