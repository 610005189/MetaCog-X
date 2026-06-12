# MetaCog-X v3.0 Gap Analysis - Implementation Plan

## [x] Task 0: Gap Analysis & Decision (本轮已完成)
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 阅读 v3.0 完整设计方案 vs 当前实现
  - 分析"系统优化建议 3.1"适用性（结论：完全无关）
  - 整理 gap 清单与根因分析
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `human-judgement` TR-0.1: 文档明确标注"系统优化建议 3.1 为通用模板，与项目无关"
  - `human-judgement` TR-0.2: 文档列出当前实现 vs v3.0 模块对比表

---

## [ ] Task 1: L1 困境门控实现
- **Priority**: P0
- **Depends On**: Task 0
- **Description**:
  - 新建 `models/dilemma_gate.py`
  - 输入特征：各层 attention 熵均值、logits 最大概率、logits 熵、重复 token 计数（前向时采集）
  - 结构：2 层 MLP → [layers+3] → 32 → 1 → Sigmoid
  - 输出 `dilemma_score ∈ [0,1]`
  - 前向模式：接收 backbone 的 attention entropy list + logits stats，输出 score
  - 导出 `extract_features(attention_weights, logits, token_ids)` 供外部调用做自动标注
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 单元测试：给定 attention entropy + logits entropy，输出 dilemma_score 是 [0,1] 标量
  - `programmatic` TR-1.2: 单元测试：extract_features 在伪造 attention/logits 上返回特征维度正确

## [ ] Task 2: 自动标注困境步骤 + L1 门控训练
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 新建 `runs/train_l1_gate.py`
  - 自动标注规则：attention_entropy > 阈值 或 logit_confidence < 阈值 或 token 重复 ≥ 3 → 正样本 (1)
  - 其余 → 负样本 (0)
  - 在现有 tiny byte-level 数据集上跑 backbone forward，采集特征，生成标注
  - 用 BCE 训练 L1 门控（epoch 5）
  - 输出 train/val 准确率和 F1
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-2.1: 运行脚本后输出 L1 门控的 F1 分数（应为 >0.6）
  - `programmatic` TR-2.2: val 集上 dilemma_score 在正样本均值 > 0.7，负样本均值 < 0.4

## [ ] Task 3: 模式切换逻辑集成到 MetaCogXModel
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 修改 `models/metacogx_model.py` 和 `models/metacogx_layer.py`
  - 增加状态机：`self.mode_state = 'plain' | 'metacog'`
  - plain 模式：运行标准 attention（无 TripleAttention），不激活 Controller
  - metacog 模式：运行 TripleAttention + Controller + Awareness Pool
  - 切换滞后：连续 2 步 dilemma_score > 0.7 → 激活；连续 3 步 < 0.3 → 退出
  - 每 forward 调用时先跑 L1 门控打分，更新 mode_state
  - 采集 attention entropy + logits stats 为 L1 输入
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-3.1: 伪造一段全"平稳"序列，forward 后记录 mode=plain
  - `programmatic` TR-3.2: 伪造一段困境序列（attention entropy 人为拉高），连续 2 步后 mode 切换到 metacog
  - `programmatic` TR-3.3: 在 metacog 模式下 TripleAttention 被调用（可 monkey-patch 计数）

## [ ] Task 4: Controller 重构（条件激活 + 窄范围 + 离散动作）
- **Priority**: P0
- **Depends On**: Task 3
- **Description**:
  - 修改 `models/sparse_meta_controller.py`
  - temp_factor 从连续 [0.3, 3.0] 改为离散三值 {0.9, 1.0, 1.1}，用 gumbel-softmax 或简单 argmax+straight-through 训练
  - 或简化为连续 [0.9, 1.1]（用 sigmoid * 0.2 + 0.9）
  - skip_prob ∈ [0, 0.2]（连续 sigmoid * 0.2）
  - mem_strength ∈ [0.5, 1.0] 保留
  - Controller 只有在 mode_state=='metacog' 时才输出非默认信号；plain 模式输出恒默认值
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-4.1: 在 plain 模式 forward 后 controller temp_factor 输出恒 ≈1.0
  - `programmatic` TR-4.2: 在 metacog 模式 forward 后 controller temp_factor std > 0.05
  - `programmatic` TR-4.3: temp_factor 输出范围严格在 [0.9, 1.1]

## [ ] Task 5: 新 A/B 训练脚本（条件激活 vs always-on vs plain）
- **Priority**: P0
- **Depends On**: Task 3, Task 4
- **Description**:
  - 新建 `runs/run_ab_v2.py`
  - 3 个 variant：
    - gpt_plain: 纯 plain Transformer，无任何 metacog 模块
    - metacog_alwayson: TripleAttention + Controller always-on（旧版基线）
    - metacog_conditional: L1 门控 + 条件激活 + 窄 Controller（新版）
  - 相同超参 tiny byte-level 模型
  - 输出 ppl、mode 切换次数统计、controller temp_factor std（条件激活版本）
  - 训练 2000 step，每 500 step 打印 val ppl + 诊断信息
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-5.1: metacog_conditional 组 controller temp_factor std > 0.05（不坍缩）
  - `programmatic` TR-5.2: metacog_conditional 组 ppl ≤ metacog_alwayson 组
  - `programmatic` TR-5.3: 打印的 mode 切换统计 ≥ 0（说明至少切换过）

## [ ] Task 6: RL 训练骨架（冻结主干 + 训练 L1+Controller 的 PPO 框架）
- **Priority**: P1
- **Depends On**: Task 5
- **Description**:
  - 新建 `training/rl_framework.py`
  - 冻结 backbone 参数（requires_grad=False）
  - 可训练：L1 DilemmaGate + SparseMetaController
  - 写 PPO 更新骨架（不需要真跑完整 RL，只要框架搭好）
  - 奖励 = (-val ppl) - λ1 * intervention_count - λ2 * switch_count
  - 输出每轮 reward + ppl 变化
  - 注意：先跑监督式微调 ppl 直接优化（Task 5 已覆盖），PPO 放骨架
- **Acceptance Criteria Addressed**: AC-4（延伸）
- **Test Requirements**:
  - `programmatic` TR-6.1: 冻结 backbone 后 named_parameters() 里只有 L1 + Controller 可训练
  - `programmatic` TR-6.2: PPO 骨架 forward-backward 不报错（即使 reward 是假的）

## [x] Task 7: 清理临时 probe 脚本 + 更新 PHASE3_CONCLUSION
- **Priority**: P2
- **Depends On**: Task 0
- **Description**:
  - 清理 temp_probe.py、temp_probe2.py、temp_probe3.py（临时验证脚本）
  - 将 probe 结论写入 PHASE3_CONCLUSION.md 或新建 PROBE_CONCLUSION.md
- **Acceptance Criteria Addressed**: 文档整理
- **Test Requirements**:
  - `human-judgement` TR-7.1: 仓库根目录无 .py 临时脚本残留
  - `human-judgement` TR-7.2: probe 结论有地方可查

---

## 任务总览（按优先级 + 依赖排序）
```
Task 0 (完成)
  └─► Task 1 L1门控 ─► Task 2 自动标注+训练
  └─► Task 3 模式切换 ◄── (依赖 Task 1)
  └─► Task 4 Controller重构 ◄── (依赖 Task 3)
  └─► Task 5 新A/B ◄── (依赖 Task 3+4)
  └─► Task 6 RL骨架 ◄── (依赖 Task 5)
Task 7 清理 (随时可做)
```
