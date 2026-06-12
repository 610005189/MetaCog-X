# MetaCog-X 第三阶段 — Controller 解耦 + 真数据 + 消融研究

## Overview
- **Summary**: MetaCog-X 价值验证（第二阶段）已完成——在 CPU 300 步 short-run 中 MetaCog-X ppl=9.96 战胜 GPT 11.08（-10.1%）。但表征探针揭示"controller 塌陷到常数 0.997、层间 meta 几乎无分化、awareness 贡献了 99% 的优势"三个隐患。本阶段在进入 PPO 之前，用**有监督训练就能完成的修复**把这三个问题解决，确保架构的元认知组件真的在工作，而不是 awareness 辅助损失单独起正则作用。
- **Purpose**: 在下一阶段 PPO/GRPO 上投入算力和工程之前，必须让 controller 能解耦、meta 能分层、并通过消融实验精准回答"到底是哪一个组件在推动 ppl 下降"。
- **Target Users**: 模型架构研究者、论文第一作者。

## Goals
- **Controller 解耦**：加 entropy bonus + layer diversity regularization，让 temp_factor / skip_prob 输出分布的 entropy 明显 > 0。
- **Meta 层间分化**：4 层 meta 的层间 cosine 从现在的 0.999 → 降到 0.85 以下，且随层深递减。
- **真数据 A/B 长训练**：WikiText-2 全量（train 36718 行，valid 3760 行）1000+ 步，再次对比 Meta vs GPT 的 ppl，确认 10.1% 优势不是 820 条 fallback 小样本假象。
- **消融实验**：3 组 A/B 对比——GPT（content only）、MetaCog-awareness-only（三列 attention + awareness 辅助损失、但 meta 分支不参与 controller）、MetaCog-full（三列 + 全部辅助损失 + controller）。定位到底哪部分贡献了 ppl 提升。

## Non-Goals (Out of Scope)
- **不做** PPO/GRPO 强化学习（等 controller 解耦后再上）。
- **不做** 超过 d=256 / L=4 的大模型训练。
- **不做** FlashAttention / Triton kernel 融合。
- **不做** 多 GPU / DeepSpeed。
- **不做** 工具注册表接入真实外部 API。

## Background & Context
- 已交付：Phase1 架构闭环（因果掩码 / pool / controller / trigger / Trainer TotalLoss）✅；Phase2 价值验证（A/B ppl -10.1%）✅；完整测试 23/23 ✅。
- 发现：(a) controller temp_factor std=0.0000，陷在常数 0.997；(b) layer-wise meta cosine off-diag=0.999 说明四层 meta 没分化；(c) awareness real-vs-gibberish ratio=15.4 说明 awareness 是真贡献者。
- 数据：当前 fallback 是手工 820 条模板化文本。需联网或手工下载 WikiText-2 raw 真数据。

## Functional Requirements

### FR-1: Controller 防塌陷正则
- 在 TotalLoss 里新增 `entropy_bonus = -λ * H(controller_logits_dist)`——让 controller 的 softmax 输出保持高熵，避免收敛到常数。
- 在 TotalLoss 里新增 `layer_diversity_loss = contrastive(layer_0_meta, ..., layer_L_meta)`——或简单地"层间 meta cosine 矩阵 off-diag 平均值"当作正则项 pushed down。
- controller 的 logits 才能用于算 entropy：当前 SparseMetaController.forward 只返回 ControlSignals（temp_factor 等），需要在返回前保存 logits，或新增 forward_with_logits 版本。
- ablation 开关：TotalLoss 里 entropy_bonus / layer_diversity 各自独立开关（alpha / beta / gamma / delta 四个超参）。

### FR-2: 真数据 loader 升级
- data/hf_dataset._fetch_wikitext 已有三级 fallback：datasets.load_dataset → 本地文件 → bundled。联网跑就走第一级。
- 需要在联网环境上跑一次，或手工下载：`wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip` → 解压 `data/wikitext-2/train.txt, validation.txt, test.txt`。
- ab_trainer.py 已有 cache_dir 支持。

### FR-3: 消融实验模式
- ab_trainer.py 的 --variant 扩展为 4 种：`gpt | metacog_aware_only | metacog_meta_only | metacog_full`。
  - gpt: enable_metacog=False, attention=content only, loss=CE
  - metacog_aware_only: enable_metacog=True, attention=triple, awareness 辅助损失开, meta 辅助损失开但 controller 不实例化
  - metacog_meta_only: enable_metacog=True, attention=triple, meta 辅助损失开, awareness 辅助损失关闭
  - metacog_full: enable_metacog=True, attention=triple, 全部辅助损失 + controller + bonus + diversity
- 参数量在 4 组里必须一致（否则不可比）。

### FR-4: representation_probe.py 升级
- 新增 entropy_bonus / layer_diversity 关掉前后 controller entropy 对比。
- 新增 4 种 variant 各生成一次 summary。

## Non-Functional Requirements
- **NFR-1**: 防塌陷正则加上后，controller temp_factor std 从 <0.01 升到 ≥ 0.02（有意义分化）。
- **NFR-2**: layer-wise meta off-diag cosine 从 0.999 降到 ≤ 0.95。
- **NFR-3**: 4 组 variant 训练时间 CPU ≤ 每 variant 2 小时（1000 步）。

## Constraints
- **技术**: CPU 为主，可联网环境更好。
- **时间**: 真数据下载时间 + 4 × 训练时间 × 分析时间 ≈ 半天到 1 天。
- **依赖**: 网络（或手工下载 wiki-2 zip）。

## Assumptions
- WikiText-2 真数据能通过 datasets.load_dataset 自动下载或手工放文件。
- 新正则的超参初值 λ=0.02 / γ=0.005 能让 controller 从塌陷中解耦。

## Acceptance Criteria

### AC-1: Controller 解耦
- **Given**: MetaCog-X + entropy_bonus + layer_diversity
- **When**: 训练 300 步
- **Then**: temp_factor std ≥ 0.02，controller entropy ≥ 0.5（以 bits 计）
- **Verification**: `programmatic`

### AC-2: Meta 层间分化
- **Given**: MetaCog-X + layer_diversity
- **When**: 训练 300 步
- **Then**: layer-wise meta cosine off-diag ≤ 0.95（从 0.999 降）
- **Verification**: `programmatic`

### AC-3: 真数据 1000 步 A/B
- **Given**: WikiText-2 train 全量
- **When**: MetaCog-X vs GPT 各训练 1000 步
- **Then**: MetaCog-X ppl < GPT ppl（delta_log_ppl < 0），确认 10.1% 优势可复现
- **Verification**: `programmatic`

### AC-4: 消融实验 4 组对比
- **Given**: 4 种 variant 同等训练步数
- **When**: 比较 valid ppl
- **Then**: metacog_full ≤ metacog_aware_only ≤ metacog_meta_only ≤ gpt（预期顺序；若不是则分析原因）
- **Verification**: `programmatic`

### AC-5: 全部原有测试通过
- **Given**: 改动后
- **When**: run.py --mode full_test + tests/run_tests.py
- **Then**: 7/7 + 23/23 PASS
- **Verification**: `programmatic`

## Open Questions
- [ ] controller 用 entropy bonus 还是用 contrastive objective？（暂定：entropy bonus，实现简单）
- [ ] layer diversity 是在 meta 本身做还是 meta 的 controller 输入做？（暂定：meta 层间 cosine，直接、可解释）
- [ ] 真数据下 ppl 量级会变吗？（WikiText-2 比 fallback 难很多，预期 ppl 更高但相对关系保留）

