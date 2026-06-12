# MetaCog-X 阶段 IV — 实施计划（按优先级排序）

## [x] Task 0: 基线 A/B 验证 — run_ab_v2.py 跑完 + 结果整理
- **Priority**: P0
- **Depends On**: None（依赖已就绪：py3.11 + torch 2.4.1 + MKL）
- **Description**:
  - 直接在 py3.11 + MKL 上跑 `runs/run_ab_v2.py`（`$env:OMP_NUM_THREADS=16`）
  - 跑完后把 stdout 里 plain / alwayson / conditional 的 ppl、best_ce、switch_stats 提取到 `runs/ab_results_v2.json`
  - 预估：3.5 min
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-0.1: run_ab_v2.py 退出码为 0
  - `programmatic` TR-0.2: ab_results_v2.json 存在且含 3 个 variant key
  - `human-judgement` TR-0.3: 对比 conditional vs plain ppl 方向
- **Notes**: 如果当前正在跑，等就行。若崩了，按报错修

## [ ] Task 1: Triple Attention 消融脚本
- **Priority**: P0
- **Depends On**: Task 0（A/B 基线）
- **Description**:
  - 新建 `runs/run_ablation.py`
  - 参数：`--ablation {triple_content_only,l1_skipgate,dmn_surprise_off}`
  - triple_content_only：在 TripleAttention forward 里把 meta_vec / aware_vec 的 projection 置 0 或走 identity
  - l1_skipgate：conditional variant 里直接 enable_metacog=True 常开，不走 gate
  - dmn_surprise_off：L1 gate feature 收集去掉 surprise 那一维（features 从 F=12 变 F=11）
  - 每一个 ablation 都训练完整流程（backbone pretrain + L1/controller train + ppl eval）
  - 结果存入 `runs/ablation_triple.json` 等
- **Acceptance Criteria Addressed**: AC-2, AC-4
- **Test Requirements**:
  - `programmatic` TR-1.1: triple_content_only 跑完 ppl 出现在 json
  - `programmatic` TR-1.2: l1_skipgate 和 dmn_surprise_off 同理
  - `human-judgement` TR-1.3: 写一个 3 句中文分析在 stdout
- **Notes**: TripleAttention 里 meta/aware 分支是 proj_m / proj_a，置零即可

## [ ] Task 2: Checkpoint 模块 + run_ab_v2.py 自动 save
- **Priority**: P1
- **Depends On**: Task 0
- **Description**:
  - 新建 `training/checkpoint.py`（save / load / resume）
  - run_ab_v2.py 在每个 variant 跑完后 save：model_state_dict + opt_state_dict + variant_stats（ppl/ce/switch）
  - 文件名：`runs/ckpt_{variant}_v2.pt`
  - 新增 `--resume <path>` 参数（可选，用于从某 variant ckpt 继续 train）
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-2.1: run_ab_v2.py 跑完后 3 个 pt 文件存在
  - `programmatic` TR-2.2: 小脚本 load 一个 ckpt → model 前向 loss 连续
  - `human-judgement` TR-2.3: 不破坏现有流程

## [ ] Task 3: 快速复现脚本
- **Priority**: P2
- **Depends On**: Task 2
- **Description**:
  - 新建 `scripts/quick_repro.py`（Windows 专用，py3.11）
  - 步骤：(1) 检查 py3.11 (2) pip install -r requirements（torch torch-directml matplotlib）(3) 设置 OMP_NUM_THREADS (4) 跑 run_ab_v2.py (5) 打印结果
  - py 3.11 不存在时自动用 `winget install Python.Python.3.11`
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `human-judgement` TR-3.1: 在干净 Windows 机器手动跑通
  - `programmatic` TR-3.2: dry-run 开关可用

## [ ] Task 4: 默认 config 升级探索（可选）
- **Priority**: P2
- **Depends On**: Task 0 + Task 1
- **Description**:
  - 若 A/B 结果 ppl 差异太小（<1%），试把 d_model 128→256、d_ffn 512→1024、num_layers 2→4
  - 预估训练时长从 3.5min 升到 ~30min
  - 只在 Task 0 确认差异不显著时执行
- **Acceptance Criteria Addressed**: AC-2, AC-4（补充）
- **Test Requirements**:
  - `programmatic` TR-4.1: 新 config 的 variant ppl 差异更显著
- **Notes**: 标记为可选，取决于 Task 0 结果
