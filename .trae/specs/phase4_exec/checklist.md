# Phase IV 验证清单

## P0-基线
- [ ] Checkpoint 0.1: `git log --oneline -1` 有 commit `342d97d`
- [ ] Checkpoint 0.2: `git status` clean

## P0-T1 修复 run_ab_v2.py
- [ ] Checkpoint 1.1: `python runs/run_ab_v2.py --quick` 能跑完全部 3 variant
- [ ] Checkpoint 1.2: JSON 里 3 个 variant 全在 `variants` 数组
- [ ] Checkpoint 1.3: plain / alwayson / conditional 的 `final_ppl` 和 `final_loss` 非 null
- [ ] Checkpoint 1.4: conditional 的 `switches` 非 null 且 ≥ 0
- [ ] Checkpoint 1.5: alwayson variant 的 ppl 逻辑方向正确（应 ≥ plain）

## P0-T2 完整 A/B
- [ ] Checkpoint 2.1: 3 variant 500 steps 总 wall time ≤ 15 分钟
- [ ] Checkpoint 2.2: conditional controller std 在训练过程中 > 0.05
- [ ] Checkpoint 2.3: conditional switches > 0（至少切换过 1 次）
- [ ] Checkpoint 2.4: plain ppl 和 alwayson ppl 都在合理范围（1–15）
- [ ] Checkpoint 2.5: conditional ppl ≤ plain ppl + 0.02（最多差 2%）
- [ ] Checkpoint 2.6: conditional ppl ≤ alwayson ppl（或 ≤ alwayson + 0.02）

## P1-T3 Triple Attention 消融
- [ ] Checkpoint 3.1: `disable_tri_attn=True` 跑 500 steps + ppl 验证通过
- [ ] Checkpoint 3.2: ppl(no_tri_attn) 在合理范围（1–15）
- [ ] Checkpoint 3.3: ppl(no_tri_attn) ≥ ppl(plain)（或记录差异百分比）

## P1-T4 DMN 消融
- [ ] Checkpoint 4.1: `use_dmn=False` 的 conditional variant forward 不崩溃
- [ ] Checkpoint 4.2: ppl(conditional_without_dmn) 在合理范围
- [ ] Checkpoint 4.3: switches 统计正常（0 或正整数）

## P2-T5（可选）加宽到 d_model=256
- [ ] Checkpoint 5.1: d_model=256, 3 variant 跑完整 ≤ 30 分钟
- [ ] Checkpoint 5.2: ppl 差异模式与 d_model=128 一致

## 验收总结
- [ ] Checkpoint A: 有一份包含所有实验 JSON 的汇总表
- [ ] Checkpoint B: 有结论性判断 — "conditional 优于 plain" 或 "需要再调参"
