# MetaCog-X 第三阶段 — 验证清单

## 正则正确性
- [ ] VC-1: SparseMetaController 返回 ctrl 时能同时返回 logits（或 ControlSignals 含 logits）
- [ ] VC-2: TotalLoss 支持 alpha / beta / gamma / delta 四个权重独立开关
- [ ] VC-3: ab_trainer.py 接受 --gamma / --delta 参数并传入 TotalLoss
- [ ] VC-4: variant=gpt 时 gamma=delta=0 且 controller 不实例化
- [ ] VC-5: variant=aware_only 时 meta 分支不参与 controller（或 controller 输入只有 awareness）
- [ ] VC-6: variant=meta_only 时 awareness 辅助损失关闭
- [ ] VC-7: variant=full 时全部辅助损失 + controller 正则全开

## Controller 解耦
- [ ] VC-8: variant=full 训练 300 步后 temp_factor std ≥ 0.02（旧值 std=0.0000）
- [ ] VC-9: controller entropy ≥ 0.5 bits（softmax 分布不是常数）
- [ ] VC-10: 关掉 gamma/delta 后 controller 再一次塌陷（gamma/delta 确实起作用）

## Meta 层间分化
- [ ] VC-11: layer-wise meta cosine off-diag ≤ 0.95（旧值 0.999）
- [ ] VC-12: delta=0 时 off-diag 回到 0.999 附近（证明 delta 是因果）

## 真数据 + 消融
- [ ] VC-13: WikiText-2 train 全量（或等价规模）加载成功，source ≠ bundled-fallback
- [ ] VC-14: gpt / aware_only / meta_only / full 4 组都有 best_valid_ppl 和 final_ppl
- [ ] VC-15: full ppl < gpt ppl（10.1% 优势在真数据上可复现）
- [ ] VC-16: delta_log_ppl full vs aware_only 为负（full > aware_only）则 meta 分支有独立贡献；否则 awareness 是主要贡献者

## 回归
- [ ] VC-17: run.py --mode full_test 7/7 PASS
- [ ] VC-18: tests/run_tests.py 23/23 PASS

## 文档
- [ ] VC-19: representation_probe.py 4 组输出完整
- [ ] VC-20: PHASE3_CONCLUSION.md 含 4× ppl 对比表 + WINNER + 下一步 PPO 计划

