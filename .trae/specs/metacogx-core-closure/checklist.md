# MetaCog-X 核心架构闭环 — 验证清单

## 架构正确性（必须 100% 通过）

- [ ] VC-1: TripleAttention 因果掩码在所有三个注意力分支生效（assert attention[:, :, i, j>i] == 0）。
- [ ] VC-2: TripleAttention padding mask 在所有三个分支生效（assert attention[:, :, :, pad_pos] == 0）。
- [ ] VC-3: MetaCogXModel.forward(enable_metacog=True) 内部确实实例化了 AwarenessPool / SparseMetaController。
- [ ] VC-4: 每一层 MetaCogXLayer 收到的 temp_factor ∈ [0.8, 1.2] 且不同样本值不同。
- [ ] VC-5: MetaCogXModel.generate() 每步 decode 都调用 EnlightenmentTrigger。
- [ ] VC-6: generate() 触发 RESET 后 awareness_pool 的 buffer_len == 0。
- [ ] VC-7: Trainer.train_step() 中 meta_loss > 0 且 aware_loss > 0（以 5 步平均）。
- [ ] VC-8: Trainer 5 步后 total_loss 和 ce_loss 至少下降 5%。

## 真实数据端到端（必须 100% 通过）

- [ ] VC-9: HuggingFace "gpt2" tokenizer 能 encode + decode 一句常见英文。
- [ ] VC-10: WikiText-2 train split 能加载为 datasets.Dataset，len(train) > 1000。
- [ ] VC-11: run.py --real_data 训练 200 步后 loss ≤ 8.0（d=256, L=4, bs=4, seq=128）。
- [ ] VC-12: 训练后 generate("The meaning of life is", max_new_tokens=30) 输出可读英文片段。

## 回归测试（必须 100% 通过）

- [ ] VC-13: python run.py --mode full_test 7/7 PASS（前向、生成、觉知池、控制器、触发器、损失、RL）。
- [ ] VC-14: python tests/run_tests.py 13/13 单元测试 PASS。
- [ ] VC-15: python tests/run_tests.py 5/5 集成测试 PASS。
- [ ] VC-16: python tests/run_tests.py 评估模块至少 3/4 PASS（自我干预有效性可以记录实际值，不作为阻塞）。

## 无魔法数 / 可配置（代码审查）

- [ ] VC-17: 所有新增的数值超参（因果 mask 的 -1e9、padding mask 值、awareness decay、trigger 阈值）要么在 config dataclass 中暴露，要么用具名常量并给出注释。
- [ ] VC-18: 不出现 `if "real" in ...` 或其他判断字符串开启逻辑的黑魔法；用 enable_* 布尔开关。

## 性能观测（记录实际值，不做硬阻塞）

- [ ] VC-19: Task 1 后前向耗时 vs Task 1 前前向耗时的增量 ≤ 5%（在同一 CPU 上对比）。
- [ ] VC-20: Task 5 训练 200 步的 wall time 记录值（作为后续优化基线）。
