# MetaCog-X 阶段 IV — 实证验证 + 消融 + 小规模预训练 Product Requirement Document

## Overview
- **Summary**: MetaCog-X v3.0 三大架构组件（L1 Dilemma Gate、Triple Attention + Sparse Meta Controller、DMN）都已跑通；RL 训练框架（MinimalPPO with 2-pass REINFORCE surrogate）已验证有梯度。目前最大空白是 **A/B 实证对比、Triple Attention 消融、以及小规模数据预训练出可复现模型**。
- **Purpose**: 把"所有模块都能单独训练"升级为"端到端验证元认知模式确实比 plain/always-on 更好"，再把训练出的模型导出，作为后续完整预训练的起点。
- **Target Users**: 本项目开发者；后续可能扩展到论文实验可复现性。

## Goals
1. **A/B 实证对比**（run_ab_v2.py 的 plain / alwayson / conditional 三变体）跑完 + 记录 ppl / CE / switch_stats
2. **Triple Attention 消融**：只保留 content attention 关 meta/aware，对比 ppl 变化
3. **L1 Gate 消融**：conditional vs 直接 controller 常开（无 gate）
4. **DMN 消融**：surprise feature 关掉 / DMN 换成 2 层 MLP，重新跑 A/B
5. **小规模预训练 checkpoint**：run_ab_v2.py 里最优变体训练 checkpoint 化，能 `python train.py --resume`

## Non-Goals
- 大语料（Wiki/TinyStories）上的几千 step 预训练；只在 built_texts 生成的 toy 语料（2334 条 × 64 tokens）上做 A/B
- 多卡 / 云 GPU / CUDA / AMD ROCm 支持（已验证 DirectML 对 d_model=128 反慢，放弃）
- 任何新架构组件（tactical_scheduler / enlightenment_trigger 保持 stub 状态）
- 网页 demo / 可视化面板

## Background & Context
- 当前最可信的设备：Python 3.11 + torch 2.4.1 + MKL（16 线程）= 65ms/step → 完整 A/B 约 3.5 min
- DirectML 已跑通，但 216ms/step，d_model=128 下 kernel launch + Adam lerp fallback 拖后腿，放弃
- py3.8 旧环境 = 275ms/step，py3.11 快 4×
- 模型 74M 参数（默认 config vocab=256, d_model=128, d_ffn=512, num_layers=2, heads=2）
- 训练策略：backbone 400 step plain CE + L1 gate 4 epoch BCE + controller 500 step RL（tf surrogate + gate BCE + L2）

## Functional Requirements
- **FR-1**: run_ab_v2.py 跑完 → 生成 `runs/ab_results_v2.json`（含每 variant 的 ppl、best_ce、switch_rate）
- **FR-2**: 新增 `run_ablation.py`（独立脚本）→ 支持 `--ablation {triple_content_only,l1_skipgate,dmn_surprise_off}`
- **FR-3**: 新增 `training/checkpoint.py`（独立模块）→ save/load model + optimizer + variant_stats
- **FR-4**: run_ab_v2.py 训练结束后自动 save checkpoint + 打印 best variant
- **FR-5**: 新增 `scripts/quick_repro.py` → 一键 `pip install` + run_ab_v2.py 跑在干净环境

## Non-Functional Requirements
- **NFR-1**: 完整 A/B + 3 次消融总时长 ≤ 30 min（MKL CPU 16 线程）
- **NFR-2**: 所有产物（json / ckpt / log）命名一致，放在 `runs/YYYYMMDD-HHMM/` 下
- **NFR-3**: 每一步有 `--dry-run` 开关，不写任何文件

## Constraints
- **Technical**: 只在本机 CPU（AMD 5900HS，16C/32T）上跑，不引入新依赖
- **Business**: 本轮不超过 2 sessions
- **Dependencies**: torch 2.4.1 py3.11, 无外部数据

## Assumptions
- 2334 条 toy 语料足以看到方向差异（gate 开 vs 不开 ppl 降 1-3%）
- Controller RL 训练已经验证 TF raw logit 有梯度（= 训练是有效的）
- DML 无需再试（已经 benchmark 过慢，d_model=128 太小）

## Acceptance Criteria

### AC-1: A/B 三变体 ppl 出结果
- **Given**: run_ab_v2.py 在 py3.11 + MKL 上跑通
- **When**: main() 跑完
- **Then**: 输出 plain / alwayson / conditional 各自的 tr_val_ppl + best_ce + switch_stats 到 json
- **Verification**: `programmatic` — 文件存在且 3 个 variant 都有 ppl

### AC-2: Triple Attention 消融
- **Given**: run_ablation.py --ablation triple_content_only
- **When**: 跑完
- **Then**: ppl 比 full triple 显著上升（>3%）或持平；确认 meta/aware 是否真贡献
- **Verification**: `programmatic` — ppl 数值；`human-judgement` — 解读

### AC-3: Checkpoint save/load
- **Given**: run_ab_v2.py 跑完
- **When**: 读回 checkpoint
- **Then**: model.load_state_dict 无 missing/unexpected；optimizer 状态也可恢复
- **Verification**: `programmatic`

### AC-4: L1 Gate + DMN 消融
- **Given**: run_ablation.py 各 flag
- **When**: 跑完 l1_skipgate 和 dmn_surprise_off
- **Then**: 能判断 L1 gate 和 surprise 各有没有实作用
- **Verification**: `programmatic` + `human-judgement`

### AC-5: 可复现性脚本
- **Given**: 一台干净的 Windows 机器
- **When**: 跑 scripts/quick_repro.py
- **Then**: 装依赖 + 跑 A/B 出结果
- **Verification**: `human-judgement`

## Open Questions
- [ ] 默认 config 的 d_model=128 是否足够大？要不要升到 192 或 256（多 30-60 min 训练）
- [ ] 条件式 variant 目前 RL 训练 500 step，是否够？要不要升到 1000
- [ ] 当前 toy 语料完全是模板句，ppl 差异方向可信吗？要不要再加真实句子
