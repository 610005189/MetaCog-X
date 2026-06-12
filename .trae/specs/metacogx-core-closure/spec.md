# MetaCog-X 核心架构闭环 — 产品需求文档（PRD）

## Overview
- **Summary**: MetaCog-X 是一个内嵌元认知 / 觉知 / 开悟机制的自回归语言模型架构。骨架已实现（17 万行 PyTorch），但架构闭环存在 5 处逻辑缺陷：因果掩码未真正启用、元认知控制器输出未注入模型前向、开悟触发器未接入生成循环、Trainer 未使用辅助损失、Tokenizer 和训练数据全是占位符。本阶段目标是补齐这 5 个核心缺口，让 MetaCog-X 真正"按设计文档所说"地运行。
- **Purpose**: 把"玩具原型"变成"架构正确、能学到语言、能展示元认知特色"的可训练模型，为后续 PPO 微调实验和论文级别评估打基础。
- **Target Users**: 模型架构研究者、LLM 训练工程师。

## Goals
- 让 MetaCog-X 的前向 / 反向 / 生成三个阶段都遵守设计方案中的因果 + 元认知反馈回路规范。
- 用真实 tokenizer（HuggingFace GPT2Tokenizer）和真实语料（WikiText-103 子集）跑通一次从加载数据 → 训练 → 生成的端到端流程。
- 让元认知控制信号（temp_factor / skip_prob / mem_strength）和开悟触发信号在每一个 decode step 都实际发挥作用，而不是独立空转。
- 辅助损失（meta 时序一致性 + awareness 自预测）对总损失有可观测的非零贡献，并随训练下降。

## Non-Goals (Out of Scope)
- **不做** FlashAttention / Triton kernel 融合（架构闭环验证完再优化）。
- **不做** 多 GPU / DeepSpeed / ZeRO 优化。
- **不做** 超过 500M 参数量的大模型训练。
- **不做** PPO 强化学习微调、GSM8K 数学环境。
- **不做** 工具注册表接入真实外部 API（搜索、代码执行）。
- **不做** 架构对比基线实验（纯 GPT 对照组）。
- **不做** 权重分享策略对比。

## Background & Context
- 当前架构已实现：认知粒子生成器、三重注意力、Pre-LN Transformer 层、觉知池、稀疏元认知控制器、开悟触发器、TotalLoss（CE + meta + aware）、Trainer（仅 CE）、PPO/GRPO、对抗任务生成器、模仿学习（空壳）。
- 当前 18 个单元/集成测试全部通过，但这些测试用的是随机 tensor，**未暴露**因果掩码缺失和控制器/触发器未接线的问题。
- TripleAttention 有 `_causal_mask()` 方法定义但从未调用；MetaCogXModel.generate() 完全没接控制器/触发器；Trainer.train_step() 只传 labels 计算 CE，没开启 return_meta。
- 技术栈：PyTorch，Python ≥ 3.10，HuggingFace tokenizer + datasets。

## Functional Requirements

### FR-1: 因果掩码必须生效
- TripleAttention 三个注意力分支（content / meta / awareness）的 logits 在 softmax 前必须加上 causal mask，使位置 i 只能看到位置 j ≤ i 的 token。
- 同时正确处理 padding mask（attention_mask 从外部传入）。

### FR-2: 元认知控制器输出必须注入前向
- MetaCogXModel 内部实例化 SparseMetaController + 觉知池 +（可选）触发器。
- 每个 MetaCogXLayer 的 forward 接受 temp_factor 参数，并正确用它缩放 attention scale。
- MetaCogXModel.forward() 逐层跑完后，把每一层的 awareness 送入觉知池，再用最后一层的 meta + 觉知池 stats 调用控制器，拿到 temp_factor，在下一层传入。

### FR-3: 开悟触发器必须接入生成循环
- MetaCogXModel.generate() 每步 decode 之后把当前 logits / 生成序列 tokens 传给 EnlightenmentTrigger。
- 若触发：RESET → 清空觉知池；TOOL → 打印提示并继续（第一版不接真实工具）。
- 自省日志（trigger 原因、confidence、entropy、repeat_count）按可配置频率打印。

### FR-4: Trainer 必须启用辅助损失 + 觉知池更新
- Trainer.train_step() 调用模型时开启 return_meta=True，拿到 meta_per_layer 与 aware_per_layer。
- 用 TotalLoss（CE + α·meta + β·aware）计算总损失。
- 同时调用觉知池的 update(aware_per_layer)。

### FR-5: 真实 tokenizer + 真实数据 + 端到端可训练
- 用 `transformers.AutoTokenizer.from_pretrained("gpt2")` 替换 DummyTokenizer。
- 用 `datasets.load_dataset("wikitext", "wikitext-103-v1")` 或 WikiText-2 子集。
- run.py 新增 `--real_data` 模式，训练后能打印 loss 曲线 + 生成样例。

## Non-Functional Requirements
- **NFR-1**: 加入因果掩码后，模型前向/反向耗时增量 ≤ 5%（相对于双向版本）。
- **NFR-2**: 训练 1 epoch（WikiText-103 train split、batch_size=4、seq_len=128、d=256、L=4）可在单 CPU（或 M1 MacBook）2 小时内完成。
- **NFR-3**: 代码改动要尽量局部，不破坏现有单元/集成测试。
- **NFR-4**: 所有新增参数通过 config dataclass 暴露，无硬编码魔法数。

## Constraints
- **技术**: PyTorch 2.x，Windows 本地开发环境，CPU 训练为主（无 CUDA）。
- **依赖**: 新增 `transformers` 和 `datasets` 两个 HuggingFace 库，需要联网下载 tokenizer 词表。
- **时间**: 本阶段预计 6–10 小时完成（不含下载数据时间）。

## Assumptions
- 用户接受安装 `transformers` / `datasets`（如果没联网可以离线 fallback 到本地保存的 GPT2 tokenizer.json）。
- WikiText-103 train split 约 500MB，下载一次即可。
- 第一版 generate() 的 TOOL 触发只打印提示，不真的联网执行。
- 因果掩码用标准下三角 mask（GPT style），非 RoPE。

## Acceptance Criteria

### AC-1: 因果掩码正确工作
- **Given**: 一个序列长度为 16 的输入
- **When**: 对 TripleAttention 做前向，单独比较位置 i 的 attention 权重是否在 j > i 处全为 0
- **Then**: 所有 j > i 的 logits 被 mask 为 -inf，softmax 后 attention 权重为 0
- **Verification**: `programmatic`

### AC-2: 元认知控制器参与每层的 temp_factor 计算
- **Given**: MetaCogXModel 前向返回 meta / awareness
- **When**: 逐层跑完后检查每层实际收到的 temp_factor 值
- **Then**: temp_factor 来自 SparseMetaController.forward 的输出，范围 ∈ [0.8, 1.2]，且不同 batch 样本值不同
- **Verification**: `programmatic`

### AC-3: generate() 每步检查触发器
- **Given**: generate() 运行
- **When**: 故意构造一个会重复生成的提示（短 prompt、高温度）
- **Then**: 观察到 EnlightenmentTrigger 至少触发一次 RESET 动作（日志可见）
- **Verification**: `programmatic`（可配置触发阈值以便测试）

### AC-4: 辅助损失非零并随训练下降
- **Given**: Trainer 跑 5 步 batch
- **When**: 收集每步的 total / ce / meta / aware 三个分量
- **Then**: meta_loss > 0 且 aware_loss > 0，5 步后 total_loss 与 ce_loss 都下降至少 5%
- **Verification**: `programmatic`

### AC-5: 真实 tokenizer 能把文本编码为真实 token IDs
- **Given**: "The quick brown fox" 一句
- **When**: AutoTokenizer 编码
- **Then**: 输出 shape 正确、vocab_size = 50257（GPT2），decoder 能还原出近似原句
- **Verification**: `programmatic`

### AC-6: 真实数据训练后 perplexity 下降
- **Given**: WikiText-103 train split，batch_size=4，seq_len=128，d=256，L=4
- **When**: 训练 200 步（约 1–2 小时）
- **Then**: loss 从 ~11.0 降到 ≤ 8.0，perplexity 从 e^11 ≈ 59000 降到 ≤ e^8 ≈ 3000
- **Verification**: `programmatic`（loss/ppl 曲线对比）

### AC-7: 生成样例可读
- **Given**: 训练 200 步后的模型，prompt = "The meaning of life is"
- **When**: generate(max_new_tokens=30, temperature=0.7, top_k=30)
- **Then**: 输出是可读的英文片段（非随机乱码），至少包含 2 个真实英文单词
- **Verification**: `human-judgment`

### AC-8: 所有原有测试通过（回归）
- **Given**: 本阶段改动后
- **When**: 跑 python run.py --mode full_test + python tests/run_tests.py
- **Then**: 7 个快速测试 + 13 单元 + 5 集成 + 4 评估全部通过；评估模块的自我干预有效性 ≥ 60%（可以通过调默认阈值优化）
- **Verification**: `programmatic`

## Open Questions
- [ ] 是否需要保留"meta/awareness 跳过 temp_factor 缩放"的设计，还是两者都用同一个 temp_factor？（暂定：三者都用同一个，最简单）
- [ ] generate() 触发 RESET 后是否应该从当前步继续而不是从头？（暂定：当前步继续 + 清空觉知池历史）
- [ ] 真实数据下载失败时是否允许自动 fallback 到 DummyTokenizer + 真实随机文本（WikiText 的前 100 条手工粘贴）？（暂定：允许）
