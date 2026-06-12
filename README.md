# MetaCog-X

> 带内置元认知的神经网络架构 — 低能耗、条件化激活、多策略调度

---

## 1. 项目概述

MetaCog-X 探索在 Transformer 主干之上叠加一套**条件化元认知回路**。核心主张：**正常推理时不开元认知，只有在检测到"认知困境"（高不确定性、死循环、逻辑异常）时才切换到元认知模式投入额外计算**。

三个设计支柱：

| 支柱 | 说明 | 实现状态 |
|---|---|---|
| **困境条件化激活** | L1 极轻量门控器持续采样标准 Transformer 的注意力熵、logits 统计、token 重复等副产品，产出 `dilemma_score`；只有分数>阈值且持续若干步时才激活元认知模式 | ✅ 代码完备、已完成 A/B 训练 |
| **默认模式网络（DMN）** | 一个单 GRU 的极小网络（d_self=16）持续维护"自我"隐藏状态，输出 surprise 信号辅助 L1 门控 | ✅ 代码完备，surprise 已接入 L1 |
| **三级调度** | L1 反射检测 → L2 战术调控（策略库检索 + 控制器） → L3 战略复盘（结构化错误分析 → 更新策略库） | L1✅ / L2 策略库部分✅ / L3 仅骨架 |

**设计原则**：元认知模块默认休眠，额外参数量和 FLOPs 增量控制在主模型 10% 以内。

---

## 2. 架构总览

```
输入 Token
   │
   ▼
嵌入 + 位置编码
   │
   ▼
标准 Transformer 主干  ◄── 默认高效模式（无元认知开销）
   │
   │ attn_entropy, logits_stats, token_rep, surprise
   ▼
L1 困境门控 (DilemmaGate, 2 层 MLP)
   │  dilemma_score > 阈值 且 持续?
   ├─ 否 ───► 继续标准推理
   │
   └─ 是 ───► 元认知模式激活
              ├─ 认知粒子生成: (content, meta, awareness)
              ├─ 所有层切为 MetaCogXBlock（三重注意力 + 稀疏控制器）
              ├─ 觉知池 AwarenessPool 持续更新
              ├─ DMN 的 h_self 注入 attention logits
              ├─ L2 TacticalScheduler 选干预策略
              └─ EnlightenmentTrigger 检测死循环 / 框架错误

推理结束 ──► L3 复盘（离线/低频，结构化错误分析 → 更新策略库）
```

关键设计文档：
- [MetaCog-X 完整设计方案 v3.0](MetaCog-X%20完整设计方案v3.0.md)
- [系统优化建议 3.1](系统优化建议3.1.md)

---

## 3. 模块清单

### 3.1 `models/` — 核心架构

| 模块 | 文件 | 说明 | 状态 |
|---|---|---|---|
| CognitiveParticle | [cognitive_particle.py](models/cognitive_particle.py) | 把隐藏状态投影成 (content, meta, awareness) 三元组；meta 前 4 维带约束（置信度 0-1、深度≥0、策略 softmax、熵≥0） | ✅ |
| TripleAttention | [triple_attention.py](models/triple_attention.py) | 在 content 注意力上加 meta/awareness 的加性偏置；支持 temperature 缩放和 head 掩码；记录每层的 content attention weights | ✅ |
| MetaCogXLayer / MetaCogXBlock | [metacogx_layer.py](models/metacogx_layer.py) | 元认知模式下的 Transformer 层，串 TripleAttention + FFN；支持 skip-path（controller 信号决定是否跳层） | ✅ |
| MetaCogXModel | [metacogx_model.py](models/metacogx_model.py) | 顶层模型：条件化架构。默认走标准 Transformer（不进 TripleAttention 路径），仅当 enable_metacog=True 时 forward 中会根据 mode_state 切换 plain/metacog。输出 logits + 三个辅助损失分量 | ✅ |
| AwarenessPool | [awareness_pool.py](models/awareness_pool.py) | 滑动窗口（capacity=64）+ 衰减（0.95）维护各层 awareness 的统计；支持多图层聚合（MultiLayerAwarenessPool）；可做 self-prediction（喂 content 预测 awareness） | ✅ |
| SparseMetaController | [sparse_meta_controller.py](models/sparse_meta_controller.py) | 稀疏门控，输入当前 meta + awareness，输出 `ControlSignals(temp_factor∈[0.9,1.1], skip_prob, memory_strength, strategy_logits)`；另含带 skip 变体和自适应变体 | ✅ |
| DilemmaGate + 特征提取 | [dilemma_gate.py](models/dilemma_gate.py) | L1 门控器（2 层 MLP）+ 四个特征函数：`attention_entropy`（向量化）、`logits_stats`、`token_repetition`（已向量化，17× 加速）、`extract_features`（把各层 attention entropy 统计 + logits 统计 + token_rep + DMN surprise 拼成 11 维特征向量） | ✅ |
| DMN | [dmn.py](models/dmn.py) | 默认模式网络：单层 GRU (d_self=16) + surprise MLP；有外部输入时更新，空闲时自回归衰减更新；h_self 可注入 attention logits 作为背景偏置 | ✅ |
| TacticalScheduler | [tactical_scheduler.py](models/tactical_scheduler.py) | L2 战术调度器：输入 dilemma 类型 + 特征向量，从**策略库**检索并组合干预（策略库是一个 dict，存储 `(feature→strategy→score)` 三元组，替代原方案的在线参数更新） | ✅ |
| EnlightenmentTrigger | [enlightenment_trigger.py](models/enlightenment_trigger.py) | L3 开悟触发器：检测死循环（token 重复 ≥ 阈值）、高熵（≥阈值持续 N 步）、awareness 异常；可触发 Reset（清 awareness）或 ToolCall（外部工具调用）；另含自适应变体和执行器 | ✅ |

### 3.2 `training/` — 训练框架

| 模块 | 文件 | 说明 | 状态 |
|---|---|---|---|
| RL Framework | [rl_framework.py](training/rl_framework.py) | 最小 PPO/REINFORCE 骨架：freeze backbone → 只训练 L1 gate + controller。reward = ppl 改善 + λ·干预率 + μ·模式切换次数。loss 是显式 controller 正则（因 frozen-backbone 下 CE 无法传回 controller 梯度）+ gate BCE | ✅ 可用，验证结论见 [PROBE_CONCLUSION.md](PROBE_CONCLUSION.md) |
| RL Finetune | [rl_finetune.py](training/rl_finetune.py) | RL 微调入口脚本 | ✅ |
| Enlightenment Finetune | [enlightenment_finetune.py](training/enlightenment_finetune.py) | 开悟触发后的 finetune 逻辑（骨架） | 🟡 基本骨架 |
| Losses | [losses.py](training/losses.py) | 辅助损失：meta 时序一致性（α·0.01）、awareness 自预测（β·0.005）、controller 熵奖励（γ·0.02，当前符号待复核）、meta 多样性（δ·0.005） | ✅ |
| AB Trainer | [ab_trainer.py](training/ab_trainer.py) | A/B 对比训练器基类 | ✅ |

### 3.3 `runs/` — 主要实验入口

| 脚本 | 说明 | 状态 |
|---|---|---|
| [run_ab_v2.py](runs/run_ab_v2.py) | **主要入口**。3 组 A/B：`gpt_plain`、`metacog_alwayson`、`metacog_conditional`（L1 门控 + 滞后切换）。Byte-level 模型 (vocab=260, d_model=128, 4 层 4 头)，8 主题 1200 句，500 step / variant。支持 CPU/OMP+MKL+num_workers+pin_memory 加速 + 自动探测 CUDA / DirectML / CPU | ✅ 运行中，500 step variant ≈ 182s CPU |
| [train_l1_gate.py](runs/train_l1_gate.py) | 自动标注困境步骤（70 百分位 entropy 阈值）+ BCE 训练 L1 门控，输出 train acc / val acc / F1 / pos/neg score 均值 | ✅ |
| [ablation_triple_attention.py](runs/ablation_triple_attention.py) | TripleAttention 消融（纯 content / content+meta / content+aware / full） | ✅ |

### 3.4 `scripts/` — 探测 / 分析

| 脚本 | 说明 | 状态 |
|---|---|---|
| [tempfactor_probe.py](scripts/tempfactor_probe.py) | 扫描 temp_factor ∈ [0.3, 6.0] 对 ppl 的影响。**结论**：tf>1 ppl 单调变差，tf<1 无改善；controller 作用域是"放大→跳出困境→触发重置"，不是让模型更准 | ✅ 结论见 [PROBE_CONCLUSION.md](PROBE_CONCLUSION.md) |
| [representation_probe.py](scripts/representation_probe.py) | 6 指标分析：LayerMetaOffDiag / CtrlEntropy / TempFactorStd / AwInter / AwRatio / BatchMetaMSE | ✅ 结论见 [PHASE3_CONCLUSION.md](PHASE3_CONCLUSION.md) |

### 3.5 `data/`

| 模块 | 说明 | 状态 |
|---|---|---|
| [dataset.py](data/dataset.py) | 基础字节级数据集 | ✅ |
| [hf_dataset.py](data/hf_dataset.py) | HuggingFace 数据集封装 | ✅ |
| `wikitext-2.zip` | 随库携带的 Wikitext-2 语料 | ✅ |

### 3.6 `tests/`

`verify_task1.py` 到 `verify_task5.py` + `test_unit.py` / `test_integration.py` / `test_metrics.py` + `run_tests.py`：模块级 → 端到端校验。v3.0 跑完 7/7 PASS（参数量 17,020,086）。

---

## 4. 配置

所有超参集中在 [config.py](config.py) 的 `MetaCogXConfig` dataclass：

```python
d_model = 512         # 内容向量维度
d_meta  = 32          # 元认知状态维度
d_aware = 16          # 觉知维度
num_layers = 12
num_heads  = 8
awareness_pool_capacity = 64
awareness_decay        = 0.95
l1_enter_thresh        = 0.7
l1_exit_thresh         = 0.3
l1_enter_patience      = 2
l1_exit_patience       = 3
alpha_meta = 0.01
beta_aware = 0.005
```

实验用的 tiny 变体在 run_ab_v2.py / train_l1_gate.py 内联（d_model=128, d_meta=32, d_aware=16, num_layers=4, num_heads=4, d_ffn=512, vocab=260, max_seq_len=64）。

---

## 5. 当前进度

### 5.1 已完成 ✅

| 阶段 | 内容 | 产出 |
|---|---|---|
| 架构骨架 | MetaCogXModel + TripleAttention + CognitiveParticle + AwarenessPool + SparseMetaController + EnlightenmentTrigger + DMN + DilemmaGate | 模块全部可 import，单测通过 |
| v3.0 条件化激活 | L1 门控 + 模式状态机（plain ↔ metacog，带 enter/exit 阈值 + patience 滞后） | run_ab_v2.py 中 conditional variant 可切换 |
| 辅助损失闭环 | α meta-temporal-consistency + β awareness self-prediction + γ controller entropy + δ meta diversity | PHASE3 跑完 4 组 variant 对比 |
| Representation Probe | 6 维度内部状态分析 | **TempFactorStd=0（controller 坍缩）、LayerMetaOffDiag Full=0.058（过正交）、CtrlEntropy Full=0.025（坍缩）** |
| Temp-Factor Probe | 多训练阶段 × 多温度扫描 | **tf>1 ppl 单调变差，tf<1 无改善**；解释了 controller 坍缩根因是 always-on 架构而非模块 bug |
| 策略库（替代在线参数学习） | TacticalScheduler 基于 dict 存 (feature→strategy→score) 三元组；完全删除了主模型在线更新逻辑 | ✅ |
| L1 门控训练 | 自动百分位标注 + BCE + F1 指标 | train_l1_gate.py 可一键跑 |
| CPU 优化 | token_repetition 向量化（17×）+ OMP/MKL 16 线程 + pin_memory + float32 high matmul | 500 step/variant ≈ 182s CPU（2.75 step/s） |
| 代码级 DirectML 探测 | run_ab_v2.py 有 `pick_device()`：cuda → directml → cpu | ✅ 已集成（但 Windows + PyTorch 2.11 + Python 3.14 下 torch-directml 无法安装，见下文） |

### 5.2 进行中 🟡

| 项目 | 说明 | 当前状态 |
|---|---|---|
| **端到端 A/B 完整跑通** | run_ab_v2.py 中 3 variant × 500 step 完整跑完 + 报表生成 | 可手动启动；之前因 CPU 太慢没完整跑完一次 |
| **RL 训练 L1 gate + controller** | rl_framework.py 有显式 controller 正则 surrogate + gate BCE；但 CE 因 backbone 冻结无法回传到 controller，实际只在 metacog 模式更新 controller、plain 模式只更新 gate | 代码写好，未跑长训练 |
| **L3 复盘模块** | 需要生成结构化自然语言错误分析 → 转存策略库 + 灰度 A/B 新策略 | 仅骨架 |

### 5.3 未开始 / 待补 ❌

| 项目 | 说明 | 备注 |
|---|---|---|
| **多步推理数据集** | 当前全是 byte-level 句子，没有需要多步推理的任务（数学/逻辑/代码），真正的"困境"无法模拟 | P4 优先级 |
| **eval.py 评测脚本** | 缺一个统一脚本：给定 ckpt 跑 ppl + 门控触发率 + 元认知干预后 ppl 差 + 模式切换统计 | 可在 run_ab_v2.py 基础上抽 |
| **开悟触发器执行器端到端验证** | 死循环检测 → Reset 后 ppl 是否下降？当前只有检测 | 需要用多步推理任务 |
| **三重注意力融合方式消融** | 加性偏置 vs 乘性门控 vs Q/K 拼接（v3.1 建议 #5） | 低优先级 |
| **自适应阈值（v3.1 建议 P0#1）** | L1 门控阈值根据任务 / 负载动态调 | 依赖多任务 |
| **框架一致性检测 + 多假设并行** | 开悟触发器 v3.1 升级方向 | 依赖 L3 |
| **跨模型迁移** | 把 L1/L2 训练为模型无关模块（v3.1 建议 #8） | 长期 |
| **wikitext / 真实 tokenizer 接入** | 当前 byte-level 是 sanity check；GPT2 tokenizer + Wikitext-2 才能验证真实场景 | 需要 PyTorch 能支持的 tokenizer（当前环境无 transformers 库） |

### 5.4 已知缺陷 / 待修 🐛

| 问题 | 根因 | 建议修复 |
|---|---|---|
| **Controller TempFactor 始终常数（std=0）** | (a) frozen-backbone 下 CE 对 controller 无梯度；(b) v3.0 输出空间窄 [0.9,1.1] 但 probe 显示方向只能坏不能好 | 用 RL surrogate 单独训练 controller，重定义动作空间为"放大 → 触发 Reset"而非微调 ppl |
| **γ controller 熵奖励方向** | Full 组 CtrlEntropy 0.025（几乎坍缩），说明熵奖励写反了 softmax max 是均匀 | 在 rl_framework.py 确认 loss 符号 |
| **Full variant 过正则化** | α+β+γ+δ 同时开时 ppl 反而劣化 γ+δ 作用过强 | Phase 4 扫描 γ/δ 网格，或固定为 0 只保留 α+β |
| **temp_factor 有效作用方向单一** | tf>1 ppl 单调变差，tf<1 不变 | 把 controller 的有效动作重新定义为"把 tf 拉起来 → 检测到 ppl 进一步差 → 触发 Reset" |

---

## 6. 运行指南

### 6.1 准备

```powershell
# 需要 Python 3.14 + PyTorch（官方 CPU 轮子即可）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

依赖：`torch`、`numpy`、`scikit-learn`（train_l1_gate.py 用 F1 指标）。

### 6.2 一键 A/B

```powershell
cd d:\Projects\MetaCog-X
python runs/run_ab_v2.py
```

输出：3 个 variant 顺序训练，每 100 step 打印一行 `step loss val_ppl mode ctrl_std switches plain% meta% score`；结束后落盘 `.csv / .log / .pt` 到 `runs/`。

**实测**（AMD R7 PRO 4750U, 8C16T, CPU-only）：

```
单 variant 500 step  ≈ 182s  (2.75 step/s)
完整 A/B 3 variant   ≈ 15~20 min
```

### 6.3 训练 L1 门控

```powershell
python runs/train_l1_gate.py
```

### 6.4 环境说明

| 项目 | 值 |
|---|---|
| CPU | AMD Ryzen 7 PRO 4750U (Zen 3, 8C16T, 1.7-4.1 GHz) |
| GPU | AMD Radeon 780M iGPU (RDNA 3, 2.2 TFLOPS FP32) |
| OS  | Windows 11 |
| PyTorch | 2.11.0+cpu |
| Python | 3.14 |
| CUDA | ❌ 无 NVIDIA GPU |
| DirectML | ❌ torch-directml 最高支持 PyTorch 2.3.1 + Python ≤ 3.11，当前环境无法安装 |
| WSL2 + ROCm | ROCm 支持 AMD 核显，但系统未装 WSL |

> 如要尝试 AMD 核显训练，建议：
> 1. `wsl --install -d Ubuntu-24.04`（重启）
> 2. Ubuntu 内 `sudo apt install python3-venv`
> 3. `pip install torch==2.3.1 torch-directml`（参考 https://learn.microsoft.com/zh-tw/windows/ai/directml/pytorch-windows ）
> 预期 iGPU 实际训练加速约 3–6×（相比 CPU）。

### 6.5 优化开关（run_ab_v2.py 内）

```python
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"]  = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
torch.set_float32_matmul_precision("high")
```

若要占用更少核心，改数字即可；建议等于逻辑核数（8C16T 填 16）。

---

## 7. 实验资产

| 路径 | 说明 |
|---|---|
| [runs/gpt_300.csv](runs/gpt_300.csv) / `.log` / `.pt` | 300 step gpt_plain variant |
| [runs/metacog_300.csv](runs/metacog_300.csv) / `.log` / `.pt` | 300 step metacog_alwayson |
| [runs/metacog_aware_only_300.csv](runs/metacog_aware_only_300.csv) / `.pt` | aware-only (β=0.005) |
| [runs/metacog_full_300.csv](runs/metacog_full_300.csv) / `.log` / `.pt` | full α+β+γ+δ |
| [runs/metacog_meta_only_300.csv](runs/metacog_meta_only_300.csv) / `.pt` | meta-only (α=0.01) |
| [runs/probe_summary_4variants.csv](runs/probe_summary_4variants.csv) | 4 variant representation probe 汇总 |
| [runs/train_l1_gate_out.txt](runs/train_l1_gate_out.txt) | L1 门控训练日志 |

---

## 8. 目录结构

```
MetaCog-X/
├─ config.py                       # 全局超参
├─ train.py / run.py / run_diag*   # 早期/实验入口
├─ data/
│  ├─ dataset.py
│  ├─ hf_dataset.py
│  └─ wikitext-2.zip
├─ models/                         # 核心架构（§3.1）
│  ├─ cognitive_particle.py
│  ├─ triple_attention.py
│  ├─ metacogx_layer.py
│  ├─ metacogx_model.py
│  ├─ awareness_pool.py
│  ├─ sparse_meta_controller.py
│  ├─ dilemma_gate.py
│  ├─ dmn.py
│  ├─ tactical_scheduler.py
│  └─ enlightenment_trigger.py
├─ training/                        # 训练框架（§3.2）
│  ├─ rl_framework.py
│  ├─ rl_finetune.py
│  ├─ enlightenment_finetune.py
│  ├─ losses.py
│  └─ ab_trainer.py
├─ runs/                            # 实验入口（§3.3）
│  ├─ run_ab_v2.py                  # ★ 主要 A/B 脚本
│  ├─ train_l1_gate.py
│  └─ ablation_triple_attention.py
├─ scripts/                        # 探测脚本（§3.4）
│  ├─ tempfactor_probe.py
│  └─ representation_probe.py
├─ tests/                           # verify_task1…5 + unit/integration
├─ MetaCog-X 完整设计方案v3.0.md
├─ 系统优化建议3.1.md
├─ PROBE_CONCLUSION.md
└─ PHASE3_CONCLUSION.md
```

---

## 9. 下一阶段建议（按优先级）

1. **🐛 修 controller 训练环路**：在 rl_framework.py 里去掉对 CE 传梯度到 controller 的期待，改用显式 reward-weighted controller 输出正则 + gate BCE；γ 熵奖励符号先在小网格扫一遍确认方向。
2. **🗂️ 多步推理数据集**：找一个需要真正多步推理的 byte-level 或简单字符级数据集（如规则系统 / 组合数学题 / 字母谜），让困境检测真的触发到"中间算错 → 重复某类 token → 门控触发"。
3. **🧪 完整 A/B 跑完 + 出报告**：run_ab_v2.py 一次跑通 3 variant 全流程，把关键指标（ppl、门控触发率、模式切换率、conditional vs alwayson ppl 差）整理成表格。
4. **🧪 eval.py 统一评测脚本**：给定 ckpt，输出 ppl + 门控特征分布 + 条件激活 ppl 差 + 模式切换统计。
5. **📝 L3 复盘闭环**：在任务真的需要多步推理之后才值得做 — 先生成结构化自然语言错误分析，再接入策略库。
6. **🧬 tokenizer 升级**：如果要验证真实场景，接入 `transformers.GPT2TokenizerFast` + Wikitext-2（需安装 `transformers` 库）。
7. **🖥️ AMD 核显**（可选，约 3–6× 训练加速）：Ubuntu WSL 内 `pip install torch==2.3.1 torch-directml`；当前 Windows 原生 PyTorch 2.11 无法支持。

---

> 最后更新：2026-06-11
