# MetaCog-X 第三阶段 — 实施计划

## [ ] Task 1: Controller 防塌陷正则（entropy bonus + layer diversity）
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - SparseMetaController.forward 新增 `return_logits=False` 参数，True 时同时返回 `ctrl, logits`（或在 ControlSignals 上加一个 .logits 属性）。
  - 在 TotalLoss 里新增两个可开关正则项：
    ```
    total_loss = ce + α·meta + β·aware + γ·entropy_bonus + δ·layer_diversity
    entropy_bonus = -mean(H(controller_softmax))   # 让分布保持高熵
    layer_diversity = mean_over_pairs( cosine(meta_layer_i, meta_layer_j) )  # 越小越分化好
    ```
  - 在 MetaCogXModel 里保留 controller.forward 的 logits 输出（存 self._last_ctrl_logits），以便 TotalLoss 算 entropy。
  - ab_trainer.py 新增 `--gamma`（默认 0.02）、`--delta`（默认 0.005）。
  - 新增 ablation 开关：variant=gpt / aware_only / meta_only / full 各自 enable / disable 这些权重。
  - 默认值：alpha=0.01, beta=0.005, gamma=0.02, delta=0.005。
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-5
- **Test Requirements**:
  - `programmatic` TR-1.1: 训练 300 步后 temp_factor std ≥ 0.02。
  - `programmatic` TR-1.2: layer-wise meta cosine off-diag ≤ 0.95。
  - `programmatic` TR-1.3: python run.py --mode full_test 7/7 PASS（回归）。
  - `programmatic` TR-1.4: python tests/run_tests.py 23/23 PASS。
- **Notes**: entropy_bonus γ 太大可能让 controller 输出混乱；建议 γ 从 0.01 开始往上试。layer_diversity 的 δ 如果 push 太多可能让 meta 完全随机，建议 δ 从 0.001 起。

## [x] Task 2: 真数据 WikiText-2 加载 + 4 组 variant 300 步 A/B（bundle-fallback，full 退化）
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 联网环境运行 ab_trainer.py（不再用 fallback），或手工下载 wikitext-2-raw-v1.zip 解压到 data/wikitext-2/。
  - 跑 4 组：gpt / aware_only / meta_only / full 各 1000 步（CPU 上每 variant ~1.5h，合计 ~6h）。
  - 每 100 步 eval 一次。
  - 所有 4 组：同 config，同 seed，同 train/valid split，同 batch，同 optim。
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-2.1: 打印 train N 条、valid N 条、source=wikitext-online 或 wikitext-local-file（不是 bundled-fallback）。
  - `programmatic` TR-2.2: 4 组 best_valid_ppl 全部可比较。
  - `programmatic` TR-2.3: full ppl < gpt ppl（delta_log_ppl < 0）确认 10.1% 优势。
- **Notes**: 如果 CPU 太慢可以 --steps 500 先看趋势。

## [ ] Task 3: 表征探针升级（4 组 × 正则前后）
- **Priority**: P1
- **Depends On**: Task 1, Task 2
- **Description**:
  - scripts/representation_probe.py 接受 --variant 选项，加载 4 组 checkpoint 各跑一次。
  - 对每组输出：layer-wise cosine / temp_factor hist / awareness 区分度 / inter-batch meta MSE / controller entropy。
  - 新增 markdown summary：4 组 × 6 个指标 = 24 格子表。
- **Acceptance Criteria Addressed**: AC-1, AC-2
- **Test Requirements**:
  - `programmatic` TR-3.1: 4 组 probe 输出完整。
  - `human-judgement` TR-3.2: 人眼看 markdown 表就能判断谁好谁坏。

## [ ] Task 4: 完整回归 + 下一阶段 PPO 计划
- **Priority**: P1
- **Depends On**: Task 1..3
- **Description**:
  - python run.py --mode full_test 7/7 PASS。
  - python tests/run_tests.py 23/23 PASS。
  - 写 PHASE3_CONCLUSION.md：数据来源、4 组 ppl 表、controller entropy 对比、下一阶段 PPO 候选方案（哪个环境、哪个奖励、多少预算）。
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-4.1: 两条回归命令 exit_code=0。
  - `human-judgement` TR-4.2: PHASE3_CONCLUSION.md 里 WINNER + 下一步计划清晰。

