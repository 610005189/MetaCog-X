# MetaCog-X v3.1 后续工作验证清单

## Task A: A/B 训练验证
- [ ] 3 组 variant（gpt_plain, metacog_alwayson, metacog_conditional）均训练完 2000 step 或轻量版 500 step
- [ ] metacog_conditional controller temp_factor std > 0.05
- [ ] metacog_conditional ppl ≤ metacog_alwayson ppl
- [ ] 打印了 mode 切换次数统计
- [ ] 无训练 crash

## Task B: v3.0 文档修订（采纳 P0-2）
- [ ] 3.5 L2 战术调度器：已删除 EWC / 参数在线更新描述
- [ ] 3.7 L3 战略复盘：已改为写入策略库而非改模型参数
- [ ] 风险表：已用"策略库灰箱可解释"替代"EWC 缓解"
- [ ] 5.2 微调：已删 EWC 相关
- [ ] 文末 v3.1 修订记录清晰列出 采纳 / 延后 / 否决 三类
- [ ] 采纳项：P0-2 策略库替代参数在线学习（有明确修订段落）
- [ ] 延后项：P0-1 自监督预训练 / P1-3 开悟升级 / P1-4 L3闭环 / P2-6 L2灵活性 / P3-7 DMN深化（有标注）
- [ ] 否决项：P0-1 语义逻辑一致性特征 / P3-8 跨模型迁移（有明确否决理由）

## Task C: 三重注意力融合消融
- [ ] TripleAttention 接受 fusion 参数（'additive_bias' / 'multiplicative_gate' / 'concat_proj'）
- [ ] 三种模式各自 forward-backward 不报错
- [ ] 输出 ppl + mode切换次数 + controller std 三列对比表
- [ ] 脚本路径 runs/ablation_triple_attention.py

## Task D: DMN 实现
- [ ] models/dmn.py 存在
- [ ] GRU hidden_dim=16
- [ ] forward → (h_self ∈ [B,16], surprise ∈ [0,1])
- [ ] DMN 在 plain 模式每 forward 都更新（低开销，无额外注意力模块）
- [ ] surprise 接入 dilemna_gate.extract_features 作为额外输入特征（维度+1）
- [ ] plain 模式 h_self 投影后加到 QKV 作为偏置（或加到 V）
- [ ] meta 模式 h_self 与 awareness 拼接
- [ ] DMN 参数量 ≤ 主模型 10%
- [ ] 不引入新外部依赖

## Task E: L2 策略库骨架
- [ ] models/tactical_scheduler.py 存在
- [ ] 输入序列 T=10，编码器 LSTM
- [ ] 策略库为 Python dict（先内存，非持久化）
- [ ] 预置 ≥ 3 条策略
- [ ] 代码中**没有**任何梯度更新 / 参数微调 / EWC 路径
- [ ] 支持策略库 dict json 序列化

## Task F: L1 v2.0 自监督预训练
- [ ] train_l1_gate.py 增加 next_loss[N] 自监督标注分支
- [ ] 输出纯规则 vs 规则+自监督 F1 对比
- [ ] 自监督正样本 ≥ 规则正样本 50%

## 全局验收
- [ ] 不再出现 controller temp_factor std = 0.0000（验证 A/B 后确认）
- [ ] v3.0 文档与实际实现一致（无 EWC 误导）
- [ ] DMN surprise 真正进入 L1 输入链
- [ ] 没有任何未使用 / 否决的建议项在代码中残留
