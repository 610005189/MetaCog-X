# MetaCog-X v3.0 Gap Analysis - Verification Checklist

## 决策验证
- [x] 确认"系统优化建议 3.1"为通用 Web 产品模板，与 MetaCog-X 项目无关
- [x] 确认 v3.0 完整设计方案为后续开发蓝图
- [x] 确认 temp_factor probe 核心结论：tf<1 无收益、tf>1 单调退化；有效幅度中期 ±62%、后期 ±36%

## 根因验证
- [x] 确认 controller 坍缩根因：always-on 架构 + 非困境步骤无有意义梯度 → 输出恒 tf=1.0
- [x] 确认 temp_factor 宽范围 [0.3, 3.0] 本身就是错误设计方向
- [x] 确认 v3.0 设计的窄范围 [0.8, 1.2] 与 probe 结论一致

## Task 1: L1 困境门控
- [x] models/dilemma_gate.py 文件存在
- [x] DilemmaGate 类：__init__(input_dim, hidden=32)、forward(features) → [B,1] sigmoid
- [x] extract_features(attn_entropy_list, logits, token_ids) → 特征向量
- [x] 单元测试：输入伪造特征 → 输出维度和范围正确
- [x] 单元测试：Sigmoid 输出在 [0,1]

## Task 2: 自动标注 + L1 训练
- [x] runs/train_l1_gate.py 脚本存在
- [x] 自动标注规则：attention 熵 / logit 置信度 / token 重复 → 正样本
- [x] val 集 L1 F1 ≥ 0.6（实际 0.995）
- [x] 正样本 score 均值 > 0.7（实际 0.828）；负样本 score 均值 < 0.4（实际 0.017）

## Task 3: 模式切换逻辑
- [x] MetaCogXModel 有 mode_state 属性，初始 'plain'
- [x] plain 模式下 forward 跑标准 attention（temp_factor=1.0，不激活 controller）
- [x] 连续 2 步 score>0.7 → 切 metacog
- [x] 连续 3 步 score<0.3 → 切 plain
- [x] 切换计数正确（不是一步就切）
- [x] TripleAttention 保存 _last_attn_c 缓存

## Task 4: Controller 重构
- [x] temp_factor 范围缩窄到 [0.9, 1.1]（0.9 + 0.2*sigmoid）
- [x] skip_prob ∈ [0, 0.2]（0.2*sigmoid）
- [x] mem_strength ∈ [0.5, 1.0]（0.5 + 0.5*sigmoid）
- [x] plain 模式 controller 不被调用，temp_factor=1.0

## Task 5: 新 A/B 训练
- [ ] 3 个 variant：gpt_plain / metacog_alwayson / metacog_conditional
- [ ] 每 500 step 打印 val ppl + mode 切换次数 + controller std
- [ ] metacog_conditional controller temp_factor std > 0.05（不坍缩）
- [ ] metacog_conditional ppl ≤ metacog_alwayson ppl
- [ ] 无训练 crash
> 注：脚本 runs/run_ab_v2.py 已创建，但用户要求跳过训练验证，未运行

## Task 6: RL 骨架
- [x] 冻结 backbone 参数后 named_parameters() 显示只有 L1 + Controller 可训练（实际 trainable=10, frozen=80）
- [x] RL forward-backward 闭环不报错
- [x] reward 计算包含 ppl + 干预惩罚 + 切换惩罚

## Task 7: 清理
- [x] 根目录无 temp_probe.py / temp_probe2.py / temp_probe3.py（已删除）
- [x] PROBE_CONCLUSION.md 归档 probe 结论

## 全局验收
- [ ] 不再出现 controller temp_factor std = 0.0000（需 Task 5 A/B 训练后验证）
- [x] 系统优化建议 3.1 不再被开发者当作设计参考
- [x] 后续开发完全基于 v3.0 完整设计方案
- [ ] 新 A/B 能跑通且 ppl 结果比旧 always-on 方案不退化（需 Task 5）
