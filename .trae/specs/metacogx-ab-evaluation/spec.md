# MetaCog-X 价值验证 — 产品需求文档（PRD）

## Overview
- **Summary**: MetaCog-X 的架构骨架、因果闭环、元认知接线、开悟触发、真实数据接口已全部跑通并通过回归测试。但架构最大的价值命题——"**内嵌元认知的 LLM 在同等参数下能否优于纯 GPT**"——还没有被验证。当前 75% 自我干预有效性是在合成场景上测的，不是真语言任务；controller 的 temp_factor 训练后仍接近 1.0 说明它还没学到有意义的控制信号；fallback 数据只有 820 条、太重复，模型只学会补 EOS。本阶段要在真实 WikiText-2 上把 MetaCog-X 和 GPT 做一次**控制变量的 A/B 对比**，量化回答：元认知到底有没有用？
- **Purpose**: 架构决策点。如果实验表明 metacog 版本 perplexity 不如或等于纯 GPT，就要在投入 PPO/FlashAttention/大算力之前先重新设计 meta/awareness 的监督信号；如果能稳定胜出 2–5%，后续 PPO 微调才值得。
- **Target Users**: 模型架构研究者、论文第一作者、工程负责人。

## Goals
- 在本地 CPU 上跑通 WikiText-2 的真实 train/valid 拆分（不再用 fallback 手工文本）。
- 用完全相同的训练超参（d_model=256, L=4, bs=4, seq=128, steps=5000）训练两个模型：
  - **A 组（MetaCog-X）**：enable_metacog=True, 三重注意力 + meta 一致性损失 + awareness 自预测损失。
  - **B 组（GPT 基线）**：enable_metacog=False，纯 content 单分支注意力，维度 / 层数 / FFN / 优化器 / 学习率完全相同。
- 在 held-out valid split 上比较 perplexity、生成样例质量、训练收敛曲线。
- 分析 A 组中 meta/awareness 两个向量的层间相关性、controller temp_factor 的标准差，回答"meta 表征到底学到了什么"。

## Non-Goals (Out of Scope)
- **不做** PPO/GRPO 强化学习微调（等 A/B 证明价值再说）。
- **不做** 超过 256d / 4 layer 的大模型训练。
- **不做** 多 GPU / DeepSpeed / ZeRO。
- **不做** FlashAttention-3 内核融合。
- **不做** 新架构设计（meta/awareness 维度不变）。
- **不做** 工具注册表接入真实 API。

## Background & Context
- 上一阶段交付物：triple_attention 因果掩码接通 ✅、MetaCogXModel 内部接线 pool + controller + trigger ✅、generate() 每步触发 EnlightenmentTrigger ✅、Trainer 启用 TotalLoss（CE+meta+aware）✅、run.py --real_data + GPT2 tokenizer ✅、8 条验证脚本全绿 ✅。
- 当前"自我干预有效性 75%"来自 tests/test_metrics.py 里构造的 synthetic 场景，不是 perplexity 或 MMLU。
- 当前 fallback 文本每条 30–80 token，max_seq_len=128 大部分被 pad，模型只补 EOS，loss 从 10.7 → 4.3 不是真学到了语言。
- 技术栈：PyTorch 2.x + HuggingFace transformers/datasets，GPU 可用则用，CPU 能跑但 steps 要少。

## Functional Requirements

### FR-1: WikiText-2 真数据可加载（不再依赖 fallback）
- 自动检测 datasets 是否可联网；如果本地已缓存就直接用；如果联网超时，下载 .tar.gz 用 `datasets.load_dataset('wikitext','wikitext-2-raw-v1', cache_dir=...)` 的自动缓存；都失败才 fallback。
- 严格拆分 train / validation / test（按官方 split 而非手工切分）。
- 清洗：去掉 "=" 开头的章节标题行。

### FR-2: GPT 基线模式（enable_metacog=False 且 TripleAttention 退化成单分支）
- MetaCogXModel(enable_metacog=False)：TripleAttention 只跑 content 分支，meta/awareness 分支不参与 attention，controller/trigger/pool 不实例化，TotalLoss 只用 CE（无辅助损失）。
- MetaCogXModel(enable_metacog=True)：三重注意力全开，pool/controller/trigger 实例化，TotalLoss = CE + α·meta + β·aware。
- 两组**参数量**要可比对：enable_metacog=True 的额外参数（meta_mlp / aware_mlp / controller / pool 的 buffer）占总参数 ≤ 5%，否则 B 组可以把 d_model 调到让总参数与 A 组完全相同。

### FR-3: A/B 训练器（ab_trainer.py）
- 一个独立脚本 training/ab_trainer.py，参数化 `--variant {metacog,gpt}`，共享同一份 config，不同的只是 enable_metacog 和辅助损失开关。
- 固定 seeds（torch / random / numpy）以保证可复现。
- 支持 `--steps` 限制训练步数（CPU 上 2000 步约 4–6 小时，d=256 L=4）。
- 周期性评估 valid split 的 perplexity（每 200 步）。
- 训练结束打印两组的 final ppl、best ppl、loss 曲线 CSV。

### FR-4: meta/awareness 表征分析（representation_probe.py）
- 在 A 组训练的模型上，forward 跑出所有层的 meta / awareness，分析：
  - 不同层的 meta 向量两两相似度（cosine）——衡量 meta 表征是否在层间演化。
  - 相邻 batch 之间 meta 变化量（ΔMSE）——是否随训练变稳定。
  - controller 输出 temp_factor 在训练过程中的均值和标准差——是否从 1.0 → 分化。
  - awareness 对不同 prompt（正例 vs 乱码）的区分度。

### FR-5: 评估基线化
- 新增 tests/test_ab.py（或 ab_eval.py）把 A/B 的 ppl 差值作为一个可观察的数（不做硬断言，但打印到 CSV）。

## Non-Functional Requirements
- **NFR-1**: A/B 两个训练的代码路径除了 enable_metacog 和辅助损失开关，不能有其它不同（包括 optim lr、seed、数据顺序、batch size、seq_len）。
- **NFR-2**: 在 CPU 上 2000 步训练耗时 ≤ 6 小时 / variant（d=256 L=4 bs=4）。
- **NFR-3**: 训练结束后必须打印 loss 曲线（每 200 步一行的 CSV），供人眼画图表。

## Constraints
- **技术**: CPU 为主（GPU 可选）；Windows 本地（PowerShell）。
- **时间**: 训练本身耗时不可控（取决于 CPU 速度），至少预留 12 小时可执行时间。代码改动 ≤ 4 小时。
- **依赖**: 新增 `matplotlib` / `pandas` 画图（可选，print CSV 即可不装）。

## Assumptions
- 联网环境可用（或 WikiText-2 本地已缓存）。若不行，继续用 fallback 但数据量要 ≥ 20000 条去重文本（手工粘贴太痛苦，可找公开 dump 下载 wiki-2 train.txt raw 放到 data/ 目录）。
- CPU 上 2000 步（bs=4 seq=128）per variant 可跑完。
- 如果 metacog 版本 ppl 不如或等于 GPT，需要在报告里提出重新设计 meta/awareness 监督信号的候选方案（如把 meta 当 prompt 条件分支做 KL 散度）。

## Acceptance Criteria

### AC-1: WikiText-2 真数据加载
- **Given**: 联网环境
- **When**: `python training/ab_trainer.py --data wikitext`
- **Then**: 打印 train N 条、valid N 条、test N 条（与官方 split 一致），len(train) ≥ 36000
- **Verification**: `programmatic`

### AC-2: GPT 基线 + MetaCog-X 基线都能训练到收敛
- **Given**: 同一 config（d=256, L=4, bs=4, seq=128, steps=2000）
- **When**: 分别训练 A / B
- **Then**: A、B 的 loss 均从 ~11 → ≤ 4（以 CE 分量为准）；valid perplexity 从 ~59000 → ≤ 50（B 组可能更低或更高）
- **Verification**: `programmatic`

### AC-3: A/B ppl 可比较
- **Given**: equal param budget（± 1%）
- **When**: 同训练步数、同数据、同 seed
- **Then**: 输出 `A_ppl=xxx, B_ppl=yyy, delta_ppl=A-B`, `delta_log_ppl = log(A)-log(B)`
- **Verification**: `programmatic`
- **Notes**: 如果 delta_log_ppl < 0 则 MetaCog-X 优于纯 GPT；如果 > 0 则失败。

### AC-4: meta/awareness 表征分析
- **Given**: A 组训练完毕的模型
- **When**: 跑 representation_probe.py
- **Then**: 打印 layer-wise meta cosine matrix、temp_factor mean/histogram、awareness 区分度（正例 vs 乱码 的 L2 距离）
- **Verification**: `programmatic` + `human-judgment`（看数字是否合理）

### AC-5: 全部原有测试继续通过
- **Given**: 本阶段改动后
- **When**: `python run.py --mode full_test` + `python tests/run_tests.py`
- **Then**: 7/7 + 13/13 + 5/5 + 5/5 PASS
- **Verification**: `programmatic`

## Open Questions
- [ ] 如果 metacog 版本 ppl 没赢 GPT，下一步改 meta/awareness 监督信号的候选方案是什么？（预先列出几个：meta 当 prompt condition + KL；awareness 当 auxiliary language head；把 meta/awareness 用 LoRA 注入 FFN）
- [ ] CPU 上跑 2000 步太慢是否用 500 步做早期趋势判断？
- [ ] 如果联网完全不可用，wiki-2 train.txt raw 可以手动下载放到 data/wikitext-2/ 吗？（默认联网，保留 fallback）
