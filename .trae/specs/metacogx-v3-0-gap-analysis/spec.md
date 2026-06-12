# MetaCog-X v3.0 Gap Analysis & Revival - Product Requirement Document

## Overview
- **Summary**: 分析 v3.0 完整设计方案与当前实现之间的差距，评估"系统优化建议3.1"文档适用性，制定基于 v3.0 回归的下一阶段开发计划。
- **Purpose**: 解决当前核心问题（控制器坍缩、always-on 架构无有效梯度、temp_factor action space 单向退化），将项目从"裸原型"推进到 v3.0 困境条件激活架构。
- **Target Users**: MetaCog-X 项目开发者（当前单人项目）。

## 两份文档评估结论

### 系统优化建议 3.1 —— 与本项目无关 ❌
该文档是一份**通用 Web 产品优化模板**，内容涉及：
- 注册流程、支付 bug、首页加载、按钮位置、CDN、缓存、移动端适配、推荐算法、数据安全合规、RICE 优先级模型……
- 适用于 XX 电商平台 / XX 管理系统 / XX APP 等互联网产品。
- **与 MetaCog-X（神经网络架构研究项目）完全无关**，应忽略此文档。

### MetaCog-X 完整设计方案 v3.0 —— 核心蓝图 ✅
这是经过深思熟虑的架构蓝图，包含：
- L1 困境门控（条件激活）、DMN 默认模式网络（持续自我感）、L2 战术调度器（在线学习+工具调用）、L3 战略复盘
- 三重注意力（meta+awareness 加性偏置）、稀疏控制器（temp_factor / skip_prob / mem_strength）
- 三级训练策略（预训练 / 元认知微调 PPO / 复盘训练）
- temp_factor 范围窄 [0.8, 1.2]—— 与 probe 结论高度一致（tf<1 无收益）

## 关键背景：temp_factor probe 实验结论

| 训练阶段 | base ppl | tf=0.3 变化 | tf=1.0 | tf=2.0 变化 | tf=3.0 变化 |
|---|---|---|---|---|---|
| step 50（未收敛）| 19.47 | 0% | 19.47 | 0% | 0% |
| step 200（中期）| 2.71 | **+8.9%** | 2.73 | **+16.5%** | **+33.3%** |
| step 400（基本收敛）| 1.15 | +2.0% | 1.15 | **+5.2%** | **+17.3%** |
| step 600（过拟合）| 1.18 | +1.6% | 1.18 | +3.3% | +10.2% |

**发现**：
1. temp_factor **>1.0** 使 attention 分布变平 → ppl 单调变差（这是控制器唯一"可控"方向，但方向是**退化**）
2. temp_factor **<1.0**（sharpens attention）→ ppl 无改善甚至略上升（模型 attention 要么已饱和、要么无尖模式可收敛）
3. 动作空间只有**半个维度可用**（放大→退化→触发重置；缩小→无效）
4. 有效幅度取决于训练阶段：中期 ±62%，后期 ±36%

## 当前实现 vs v3.0 设计的差距清单

| v3.0 模块 | 当前状态 | 问题 |
|---|---|---|
| 标准模式 / 元认知模式 二选一 | ❌ 始终运行 TripleAttention + Controller | always-on 导致训练样本大部分不是"困境时刻"，controller 无有意义梯度 → 坍缩 |
| L1 困境门控 | ❌ 未实现 | 无法检测困境，无法条件激活 |
| DMN 默认模式网络 | ❌ 未实现 | 无持续自我向量、无 surprise 信号 |
| 认知粒子生成器（content/meta/awareness 分裂）| ✅ 已实现 | 可用 |
| 三重注意力（meta+awareness 加性偏置）| ✅ 已实现 | 可用 |
| 稀疏元认知控制器（三层：temp_factor / skip_prob / mem_strength）| ⚠️ 已实现但 always-on + 范围过宽 [0.3, 3.0] | 需改范围到 [0.8, 1.2]，或离散动作 |
| 觉知池 Awareness Pool | ✅ 已实现 | 可用 |
| L2 战术调度器（PPO 训练在线学习+工具调用）| ❌ 未实现 | 当前 controller 只有 temp_factor 一个动作 |
| 开悟触发器（重置框架切换）| ⚠️ 有框架但未与门控串联 | 需依赖 L1 门控信号 |
| L3 战略复盘 | ❌ 未实现 | 可延后 |
| 三级训练策略 | ❌ 当前只有监督式 A/B，无 PPO | 需冻结主干参数，RL 训练控制器 |
| temp_factor 范围 | ❌ 当前 [0.3, 3.0] | v3.0 设计 [0.8, 1.2]，与 probe 结论一致 |

## 核心根因分析

**Controller 坍缩 ≠ 控制器模块 bug**
- 根因：always-on 架构 + 无困境条件 → controller 在非困境步骤也被迫输出信号，但这些步骤 attention 已在软模式下收敛到 tf=1.0 附近最优，任何偏离都是负面 → 梯度告诉 controller "永远输出 tf=1.0" → 坍缩到 std=0.0000
- v3.0 设计里控制器只在"困境时刻"激活 → 此时 attention 处于混乱态，tf 缩放才有意义 → 梯度有信号 → 不会坍缩

**temp_factor action space 问题 ≠ 维度设计错误**
- 根因：probe 显示"缩小 attention（tf<1）无收益"，说明在字节级小模型上 attention 尖度已饱和
- 但 v3.0 设计的窄范围 [0.8,1.2] 其实隐含了这个结论——它根本没让 controller 去探索 tf<0.8
- 真正问题：controller 的有用动作不是"连续微调温度让模型更准"，而是"放大温度→让模型跳出困境→触发重置/换路径"

## Goals
- **G1**: 实现 L1 困境门控（特征提取 + MLP + 滞后机制），支持条件激活元认知模式
- **G2**: 实现"标准模式 / 元认知模式"二选一架构（标准模式跑 plain Transformer，元认知模式才激活 TripleAttention + Controller）
- **G3**: 重构控制器——缩窄 temp_factor 范围 [0.8,1.2] 或改为离散动作（sharpen/normal/blur），仅在元认知模式激活
- **G4**: 搭建 RL 训练环境骨架（冻结主干参数 → 用 PPO 训练 L1 + Controller）
- **G5**: 跑新 A/B：有条件激活（L1 门控 + 条件控制器）vs plain GPT vs always-on metacog

## Non-Goals (Out of Scope)
- 不实现 L2 战术调度器（在线学习 + 工具调用）—— 依赖 L1 + 条件激活先跑通
- 不实现 L3 战略复盘
- 不实现 DMN 默认模式网络（次阶段，先跑 L1+条件激活 解决坍缩根因）
- 不接真实工具 / 多步推理数学任务（先在当前小模型上验证条件激活有效）
- 不修改 TripleAttention / Awareness Pool 核心结构（已验证可用）

## Background & Context
- 已有代码：config.py, run.py, train.py, data/*, models/*, training/*, tests/*, scripts/*
- 已验证：TripleAttention 有效（awareness-only 比 GPT ppl 好 4.5%），controller 模块本身可跑
- 已验证：controller always-on 会坍缩（temp_factor std=0.0000）
- 已验证：temp_factor 缩小无效、放大退化（probe 结论）
- v3.0 方案蓝图完整可执行

## Functional Requirements
- **FR-1**: L1 困境门控模块
  - 输入特征：各层 attention 熵、logits 最大概率、logits 熵、重复计数、序列长度/预期长度比
  - 网络：2层 MLP，输出 sigmoid dilemma_score
  - 滞后机制：连续 2 步 >0.7 才激活元认知模式；连续 3 步 <0.3 才退出
  - 标准模式下不激活 TripleAttention，也不运行控制器
- **FR-2**: 模式切换逻辑集成到 MetaCogXModel
  - 默认 forward：运行 plain Transformer blocks（无 metacog 开销）
  - 激活时：切换到 TripleAttention + Controller + Awareness Pool
  - 切换是 token-by-token 的状态机
- **FR-3**: Controller 重构
  - 仅在元认知模式激活
  - temp_factor 缩窄到 [0.8, 1.2]（或离散三值 {0.9, 1.0, 1.1}）
  - skip_prob ∈ [0, 0.3]（v3.0 原意），mem_strength ∈ [0.5, 1.0] 保留
- **FR-4**: 门控训练数据
  - 自动标注：用人工规则（attention 熵 > 阈值、重复 token、ppl 突变、logit 置信度 < 阈值）在训练集上标注"困境步骤"
  - 训练 L1 门控做二分类 BCE
- **FR-5**: RL 训练骨架（后续阶段，先搭框架）
  - 冻结主干参数 + L1 门控 + 控制器可训练
  - 奖励 = ppl 改善 + 干预次数惩罚 + 模式切换惩罚
  - PPO 环境搭建

## Non-Functional Requirements
- **NFR-1**: 标准模式下额外开销 <5% FLOPs（纯特征采集 + 极轻量 2 层 MLP）
- **NFR-2**: 模式切换延迟 ≤ 2 步响应（滞后机制固有限制）
- **NFR-3**: 条件激活后 controller temp_factor std 应不再为 0（>0.05 说明没坍缩）

## Constraints
- **Technical**: 当前只有 CPU 环境，模型必须是 tiny（≤500K params）、byte-level 或 tiny vocab
- **Time**: 单人项目，每个阶段 ≤ 2-3 天
- **Dependencies**: PyTorch, 当前已有 models 模块可用

## Assumptions
- v3.0 设计的 L1 条件激活 + 窄 temp_factor 范围足以解决当前问题
- 无需 DMN 也能让 L1 门控有效（先纯熵/统计特征验证）
- RL 训练骨架比真实 PPO 训练更重要——先有环境再谈算法

## Acceptance Criteria

### AC-1: L1 门控能有效标注困境步骤
- **Given**: 一段训练序列 + 各层 attention 熵 + logits 统计
- **When**: 运行 L1 门控 forward
- **Then**: dilemma_score 输出在"混乱步"接近 1，在"平稳步"接近 0；有明确阈值
- **Verification**: `programmatic` —— 可以写单元测试

### AC-2: 模式切换逻辑正确
- **Given**: 连续多步输入，其中若干步是困境
- **When**: 运行 MetaCogXModel forward
- **Then**: 前 N 步（平稳）跑 plain blocks；连续 2 步困境后切换到 TripleAttention；困境解除连续 3 步后切回
- **Verification**: `programmatic` —— 打印每层是 plain 还是 metacog

### AC-3: 条件激活后 controller 不再坍缩
- **Given**: 模型跑 A/B 训练（条件激活版本）
- **When**: 打印 controller temp_factor 的 std
- **Then**: std > 0.05（不再是 0.0000）
- **Verification**: `programmatic`

### AC-4: A/B 比较有条件激活 vs always-on vs plain GPT
- **Given**: 相同超参、相同数据
- **When**: 跑 3 组训练
- **Then**: 有条件激活版本 ppl ≤ always-on 版本，且 temp_factor std > 0.05
- **Verification**: `programmatic`

### AC-5: 系统优化建议 3.1 文档不影响后续开发
- **Given**: 本 PRD 已明确标注"忽略此文档"
- **When**: 后续开发
- **Then**: 开发者不再参考系统优化建议 3.1
- **Verification**: `human-judgment`

## Open Questions
- [ ] temp_factor 用连续 [0.8,1.2] 还是离散三值？—— 建议先试离散 {0.9,1.0,1.1}，更容易训练
- [ ] skip_prob 在标准模式下恒为 0，在元认知模式下开启 [0,0.3]—— 是否真的帮助？—— 先跑 A/B 看
- [ ] DMN 真的需要现在做吗？—— 建议延后到 L1 + 条件激活跑通后
- [ ] RL PPO 还是监督式微调 L1 + Controller？—— 先监督式（自动标注 BCE + ppl 直接优化），PPO 放骨架
