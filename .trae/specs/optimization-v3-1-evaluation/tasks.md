# MetaCog-X v3.1 后续工作分解

## [ ] Task A: 跑 Task 5 A/B 训练验证（最高优先级，必须先完成）
- **Priority**: P0
- **Depends On**: None（脚本 runs/run_ab_v2.py 已存在）
- **Description**:
  - 运行 runs/run_ab_v2.py，训练 3 组：gpt_plain / metacog_alwayson / metacog_conditional
  - 收集每 500 step 的 val ppl + controller temp_factor std + mode 切换次数
  - 若 CPU 太慢（>60分钟），先跑 500 step 轻量验证 + 观察趋势
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-A.1: metacog_conditional controller temp_factor std > 0.05
  - `programmatic` TR-A.2: metacog_conditional ppl ≤ metacog_alwayson ppl
  - `programmatic` TR-A.3: 3 组均无训练 crash

## [ ] Task B: 采纳 P0-2 — 更新 v3.0 设计方案 L2/L3 为策略库替代参数在线学习
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 修改 MetaCog-X 完整设计方案v3.0.md：
    - 3.5 L2 战术调度器：删除"参数在线更新/EWC"描述，改为"策略库检索（困境特征→干预策略→效果评分）"
    - 3.7 L3 战略复盘：改为"复盘输出存入策略库，不修改模型参数"
    - 5.2 元认知微调：删除 EWC 相关，策略库检索用简单查找表
    - 风险表：删除 EWC 缓解项，改为"策略库灰箱可解释"
  - 在文末加 "v3.1 修订记录" 标注采纳 P0-2、否决项、延后项
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `human-judgement` TR-B.1: 文档中 EWC/参数在线更新字样全部替换或标注删除
  - `human-judgement` TR-B.2: v3.1 修订记录明确列出采纳（P0-2）、延后（P0-1自监督/P1-3/P1-4/P2-6/P3-7）、否决（P0-1语义特征/P3-8）

## [ ] Task C: P2-5 执行 — 三重注意力融合方式消融实验
- **Priority**: P1
- **Depends On**: Task A
- **Description**:
  - 新建 models/triple_attention.py 增加 3 种融合模式：
    - `fusion='additive_bias'`（当前默认）：attn_logits + meta_bias + aware_bias
    - `fusion='multiplicative_gate'`：attn_logits * (1 + sigmoid(meta_gate)) * (1 + sigmoid(aware_gate))
    - `fusion='concat_proj'`：attn_weights concat meta/aware → linear 投影融合
  - 新建 runs/ablation_triple_attention.py 并排跑 3 种模式
  - 输出 ppl + mode 切换统计 + controller std 对比表
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-C.1: TripleAttention 构造器接受 fusion 参数，默认 'additive_bias'
  - `programmatic` TR-C.2: 三种模式各自 forward-backward 不报错
  - `programmatic` TR-C.3: ablation 脚本输出三列对比表

## [ ] Task D: 实现 DMN 默认模式网络（v3.0 下一个核心未实现模块）
- **Priority**: P1
- **Depends On**: Task A
- **Description**:
  - 新建 models/dmn.py：单层 GRU(d_self=16)，输入 = 注意力熵均值 + logits 熵 + controller temp_factor + 上步 h_self
  - forward(self_features) → h_self, surprise
  - surprise = MLP([h_self, 当前特征]) → [0,1]
  - 修改 dilemna_gate.py：extract_features 增加 surprise 作为第 N+1 维输入特征
  - 修改 metacogx_model.py：每 forward 一步都调用 DMN 更新（plain 模式下也运行，低开销）
  - 注入：plain 模式每层 QKV 加 linear(h_self) 偏置；meta 模式与 awareness 拼接
  - 吸收 P0-1 建议：surprise 作为 L1 额外输入特征
  - 吸收 P3-7 建议基础版：surprise > 阈值时可在日志中记录"好奇"
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-D.1: DMN 每步 forward 输出 h_self shape=[B,16], surprise 标量∈[0,1]
  - `programmatic` TR-D.2: dilemna_gate.extract_features 返回维度 = n_layer + 3 + 1（原） + 1（surprise）
  - `programmatic` TR-D.3: plain 模式 forward 后 DMN 隐藏状态已更新
  - `programmatic` TR-D.4: DMN 参数量 ≤ 主模型 10%

## [ ] Task E: L2 调度器骨架（基于策略库，不做参数更新）
- **Priority**: P2
- **Depends On**: Task B, Task D
- **Description**:
  - 新建 models/tactical_scheduler.py（L2）
  - 输入：最近 T=10 步的特征序列 [d_seq]
  - 编码器：单层 LSTM → last_hidden
  - 策略库（Python 字典初始即可，先不做持久化）：
    - key = (困境类型特征哈希, DMN surprise 级别)
    - value = { strategy_id, confidence, expected_improvement, cost }
  - 输出：strategy_id + confidence + activate_online_learning=False + call_tool=False
  - 在线学习动作改为"查策略库"，不做参数梯度更新
  - 先预置 3 条策略：[调温度0.95, 调温度1.05, 触发重置]
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `programmatic` TR-E.1: 输入伪造特征序列 → 输出 strategy_id 为已注册策略之一
  - `programmatic` TR-E.2: L2 不包含任何梯度更新代码路径
  - `programmatic` TR-E.3: 策略库可 dict 形式保存/加载（json dump）

## [ ] Task F: 实现 L1 v2.0 自监督预训练（P0-1 后续）
- **Priority**: P2
- **Depends On**: Task D
- **Description**:
  - 在 train_l1_gate.py 现有规则标注基础上，增加自监督标注分支：
    - next_loss[N] = 未来 N 步 ppl 增量
    - N 步 ppl 增量 > 阈值 → 正样本
    - 与规则标注合并取并集作为正样本
  - 重新训练 L1 门控
  - 对比纯规则标注 vs 规则+自监督标注的验证集 F1
- **Acceptance Criteria Addressed**: AC-3 延伸
- **Test Requirements**:
  - `programmatic` TR-F.1: 输出"规则标注 F1" vs "规则+自监督 F1" 对比
  - `programmatic` TR-F.2: 自监督正样本数 ≥ 规则正样本数的 50%

---

## 任务依赖 DAG
```
Task A (A/B验证) ──────► Task C (三重注意力消融)
    │
    └─────────────────► Task D (DMN) ──► Task F (L1 v2.0)

Task B (v3.0文档修订：采纳P0-2策略库替代参数) ──► Task E (L2策略库骨架)
```

## 建议执行顺序
1. **Task A**（A/B验证，必须先完成）
2. **Task B**（文档修订，架构决策）
3. **Task D**（DMN，核心下一个模块）
4. **Task C**（三重注意力消融）
5. **Task E**（L2策略库骨架）
6. **Task F**（L1 v2.0）
