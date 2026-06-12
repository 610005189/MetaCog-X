# MetaCog-X 价值验证 — 验证清单

## 架构正确性（保留上一阶段全部绿）

- [ ] VC-1: TripleAttention causal + padding mask 生效（上一阶段已 PASS，此阶段不应回归）
- [ ] VC-2: MetaCogXModel.enable_metacog=True 时 meta/awareness 被正确返回和利用
- [ ] VC-3: generate() 每步触发 EnlightenmentTrigger（上一阶段已 PASS）
- [ ] VC-4: Trainer 4 个 loss 分量（total/ce/meta/aware）正确打印
- [ ] VC-5: datasets.load_dataset("wikitext") 或 fallback 加载成功
- [ ] VC-6: variant=metacog 时 total_loss = ce + α·meta + β·aware（α=0.01, β=0.005）
- [ ] VC-7: variant=gpt 时 total_loss ≡ ce（meta/aware 分量为 0 或不计算）
- [ ] VC-8: 两组 config 参数量差 ≤ 5%（否则调 d_model 让相等）

## 训练 + A/B 比较

- [ ] VC-9: variant=gpt 训练 2000 步后 valid ppl ≤ 50（或 ≥100 且比 baseline 明显收敛）
- [ ] VC-10: variant=metacog 训练 2000 步后 valid ppl ≤ 50（或同 gpt 量级）
- [ ] VC-11: valid ppl 曲线平滑递减（非振荡发散）
- [ ] VC-12: summary 表能同时读 GPT 和 MetaCog-X 的 final_ppl / best_ppl / delta_log_ppl / winner
- [ ] VC-13: 同一 prompt（"The meaning of life is"）两个模型各生成一段可粘贴对比

## 表征分析

- [ ] VC-14: layer-wise meta cosine matrix 对角=1，非对角 > 0（层间有相似性）
- [ ] VC-15: temp_factor min ≥ 0.8，max ≤ 1.2，std > 0（否则 controller 没学到）
- [ ] VC-16: 乱码 vs 正例 awareness L2 距离 > 0.1（可区分）
- [ ] VC-17: 如果 temp_factor std < 0.01（几乎常数 1.0），在 CONCLUSION.txt 中明确标注"controller 信号塌缩"

## 回归

- [ ] VC-18: python run.py --mode full_test 7/7 PASS
- [ ] VC-19: python tests/run_tests.py 单元 13/13 + 集成 5/5 + 评估 5/5 PASS
- [ ] VC-20: CONCLUSION.txt 存在且包含 winner 字段和下一步建议
