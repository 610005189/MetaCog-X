# MetaCog-X 阶段 IV — 验收清单

## Task 0: 基线 A/B
- [ ] run_ab_v2.py py3.11+MKL 退出码 0
- [ ] stdout 含 plain / alwayson / conditional 三变体 ppl + best_ce + switch_stats
- [ ] runs/ab_results_v2.json 存在且结构完整
- [ ] 方向解读：conditional vs plain ppl 是升还是降，幅度

## Task 1: Triple Attention + L1 + DMN 消融
- [ ] runs/run_ablation.py 存在且接受 --ablation flag
- [ ] triple_content_only 跑完 ppl 有输出 json
- [ ] l1_skipgate 跑完 ppl 有输出 json
- [ ] dmn_surprise_off 跑完 ppl 有输出 json
- [ ] stdout 有 3 句中文分析

## Task 2: Checkpoint
- [ ] training/checkpoint.py 存在，含 save / load / resume 函数
- [ ] run_ab_v2.py 跑完有 ckpt_plain_v2.pt / ckpt_alwayson_v2.pt / ckpt_conditional_v2.pt
- [ ] 小脚本 load ckpt → model 前向 loss 一致

## Task 3: quick_repro 脚本
- [ ] scripts/quick_repro.py 存在，Windows 专用
- [ ] --dry-run 开关
- [ ] py3.11 不存在时 winget fallback

## Task 4: 探索 config 升级（可选）
- [ ] 仅 Task 0 ppl 差异 <1% 才执行
- [ ] 新 config 训练完成
