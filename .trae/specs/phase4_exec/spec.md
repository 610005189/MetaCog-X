# MetaCog-X Phase IV 实证验证 — 执行计划（2026-06-12）

## 0. 项目总进度

| 阶段 | 目标 | 状态 | 关键交付物 |
|---|---|---|---|
| Phase 0–2 | 理论设计 + 代码实现 | ✅ 完成 | 12 个模块：config, awareness_pool, cognitive_particle, dilemma_gate, dmn, enlightenment_trigger, metacogx_layer, metacogx_model, sparse_meta_controller, tactical_scheduler, triple_attention, dataset, run/train 入口 |
| Phase 3 | 监督训练 + 梯度修复 | ✅ 完成 | 梯度流修复、MSE→CE loss 切换、L1 Gate 单独预训练训练脚本、Triple Attention 消融雏形 |
| Phase 4 | 实证验证（A/B 对比 + 消融 + checkpoint） | 🔧 进行中 | — |

Phase 4 是本项目能否被称为"科学研究"的分水岭。如果 Phase 4 失败，整个 MetaCog-X 就只是个漂亮的工程练习。

## 1. 当前已知的问题

### ❌ 问题 1：run_ab_v2.py 没写完
- `main()` 里只有 `plain` 和 `conditional` 两个 variant 被循环跑
- `alwayson` variant 完全没注册进 `variants` 循环
- `conditional` 里的 `rl_training_steps=500` 和 `l1_gate_pretrain_epochs=4` 只是摆设（`MetaCogXModel.forward` 不接收这些）
- 结果 JSON 里的 ppl/loss/switches 全是 null（因为跑完了 plain 但是没把数据写回）

### ❌ 问题 2：Windows PowerShell + torch-directml AdamW lerp fallback 拖慢速度
- `aten::lerp.Scalar_out` 在 DML 没实现 → 每次 optimizer.step() 都 CPU fallback
- 实测 DML 比 CPU MKL 还要慢 3–5x（参数太少，kernel launch 开销压过计算）
- **临时解法**：`pick_device()` 强制返回 "cpu"；最终 A/B 全部在 CPU MKL 上跑

### ⚠️ 问题 3：DMN 的 GRU 在 DML 不支持
- 已修复：`dmn.forward()` 内部强制把 GRU/h_proj/surprise_mlp 移到 CPU，输入先 `.to(cpu)`，输出再 `.to(orig_device)`
- tensor 尺寸只有 [B,4]/[B,16]，拷贝开销可忽略

### ⚠️ 问题 4：模型规模太小（d_model=128, ~1.1M）
- d_model=128 + seq_len=64 + vocab=260（byte-level）→ 每个 step 纯前向 < 2ms，但是 AdamW + CE loss backward 仍要 50–100ms/step

### ⚠️ 问题 5：Windows PowerShell + Python 3.11 有时静默退出
- `python -c "..."` 超过 30 行时 PowerShell 会截断或直接不执行
- 解法：把 benchmark 写到 `.py` 文件里，而不是单行 -c 字符串

## 2. 性能预算（实测）

| 设定 | 平台 | batch32 step | 500 steps | 整个 A/B（plain+alwayson+conditional = 1900+ steps） |
|---|---|---|---|---|
| d_model=128, 4 layers, AdamW, OMP=16, MKL=16 | CPU (Intel i9-13900K) | ~80–120 ms | ~45–60 s | **~8–12 分钟** |
| d_model=256（2× 宽） | 同上 | ~160–200 ms | ~90–110 s | ~16–22 分钟 |
| DirectML | AMD RX 6800 | ~400 ms（含 lerp fallback） | ~200 s | **~60+ 分钟 ❌** |

**结论**：当前硬件，d_model=128 在 CPU MKL 跑最合算。如果结果不错，可以后续加宽到 d_model=256 再做第二轮。

## 3. Phase IV 任务清单（按优先级排序）

### P0-T1：修复 run_ab_v2.py — 补全 alwayson variant + 写回 ppl/loss/switches
- 现状：main() 只跑 plain 和 conditional；alwayson 没注册；结果 JSON 全部 null
- 必须补：
  ```python
  variants = [
      ("gpt_plain",       dict(enable_metacog=False)),
      ("alwayson_meta",   dict(enable_metacog=True, mode="alwayson")),
      ("conditional",     dict(enable_metacog=True, mode="conditional")),
  ]
  ```
  注意 `MetaCogXModel(mode="alwayson")` 和 `MetaCogXModel(mode="conditional")` 实际在 `enable_metacog=True` 的情况下 forward 已经会根据 L1 Gate 的阈值动态切换（conditional）或强制开启 metacog head（alwayson）。需要确认 `metacogx_model.py` 的 forward 是否支持 `mode` 参数，如果不支持，在 A/B 脚本里分别设置 `gate.enable = False / gate.threshold = float('inf') / gate.threshold = 原阈值` 作为替代。
- 补：验证 perplexity 函数（validate_ppl）正确；每个 variant 训练完跑 validate_ppl 并写回 final_ppl/final_loss
- 补：conditional 跑 switches 统计（在验证集 forward 中计数 mode 被激活的次数）

**测试**：跑 50 steps × 3 variants → 约 5 × 3 = 15 秒，然后 ppl 验证 3 × 5 s = 15 s，总共 30 s 左右。看 JSON 里 ppl/loss 非 null 即可。

### P0-T2：跑完整 A/B 对比（plain 500 + alwayson 500 + conditional backbone 400 + controller 500 RL）
- 预计耗时：~10 分钟（d_model=128, batch32, CPU MKL 16 线程）
- 成功标准：
  - conditional ppl ≤ plain ppl（或至多 +2%）
  - alwayson ppl 应 ≥ plain ppl（"强行开元认知"是代价）
  - conditional 的 controller std 明显 > 0（参数在学，不是常数）
  - switches > 0（conditional 真的切换了）

### P0-T3：根据 A/B 结果决定是否加宽模型到 d_model=256
- 如果 P0-T2 三个 ppl 接近，说明模型还不够 capacity → 加宽重跑
- 如果 conditional 已经优于 plain，d_model=128 足够发表 → 跳过

### P1-T4：Triple Attention 消融（去掉 tri_attn，只留 QKV 自注意力）
- 预计耗时：和 P0-T2 类似，~3–5 分钟（只有 backbone，没 metacog）
- 对比：plain with/without triple_attn 的 ppl

### P1-T5：L1 Gate 消融（始终关闭 metacog，或始终开启）
- 实际上就是 alwayson + plain 已经跑了；conditional 就是"开启 Gate"
- 所以 P0-T2 本身就是 L1 Gate 消融的核心实验

### P1-T6：DMN 消融（去掉 surprise 分支）
- 预计耗时：同 P1-T4，~3–5 分钟
- 对比：conditional variant with/without dmn 的 ppl 和 switch 行为

### P2-T7：Checkpoint 模块（保存/加载）
- 重要但不阻塞验证结论；先把 P0–P1 跑完再说

### P2-T8：小规模预训练 checkpoint（d_model=256, wikitext-2）
- 这个是 Phase V 的事情（投稿前再加）。Phase IV 只要 byte-level tiny 跑通就行。

## 4. 执行顺序与总时间预估

```
T1 修复 run_ab_v2.py          →  10 min (编码 + 小测试)
T2 完整 A/B 3 变体 (d=128)   →  10 min
    └─→ 看结果，决定 T3 要不要加宽
T4 Triple Attn 消融           →  5 min (d=128)
T5 L1 Gate 消融               →  [已包含在 T2]
T6 DMN 消融                   →  5 min
───────────────────────────────────────
合计（d_model=128 一轮）        ~ 1 小时
如果 T3 加宽 → 再 +1.5 小时
```

## 5. 对"整个 A/B 要多久"这个问题的回答

以当前硬件 + 已修好的代码：
- **d_model=128 一轮完整 A/B（plain + alwayson + conditional）** ≈ **10–15 分钟**
- 如果 Triple Attn + DMN 两个消融也算"整个 A/B" → **~30 分钟**
- 如果要加宽到 d_model=256 再跑一轮 → **再加 ~1 小时**

## 6. 验收门槛（Phase IV 通过条件）

- [ ] A/B JSON 中 plain / alwayson / conditional 三个 ppl 都非 null 且可排序
- [ ] conditional 的 controller std > 0.05（不是常数控制信号）
- [ ] conditional 的 switches > 0（至少切换过 1 次）
- [ ] 消融：去掉 triple attention 后 ppl 明显 ↑（triple attention 有贡献）
- [ ] 消融：去掉 DMN 后 surprise 分支不再影响 forward（确认 forward 不再读取 dmn 输出）

## 7. 已修/已知代码摘要

| 文件 | 改了什么 |
|---|---|
| [dmn.py](file:///d:/Projects/MetaCog-X/models/dmn.py) | forward 内强制 CPU 执行 GRU，输入/输出跨设备拷贝 |
| [run_ab_v2.py](file:///d:/Projects/MetaCog-X/runs/run_ab_v2.py) | pick_device() 强制 CPU；加 JSON 写出；main() 目前只跑 plain+conditional（**待补 alwayson + ppl 写回**） |
| [metacogx_model.py](file:///d:/Projects/MetaCog-X/models/metacogx_model.py) | 支持 `enable_metacog` 和 `mode` 参数；forward 根据 L1 Gate threshold 动态切换 metacog head（conditional 模式） |
| [metacogx_layer.py](file:///d:/Projects/MetaCog-X/models/metacogx_layer.py) | 每层 metacog head 受 gate 控制 |
| [config.py](file:///d:/Projects/MetaCog-X/config.py) | d_model/d_meta/d_aware/num_layers/num_heads/d_ffn 等全可配 |

## 8. 下一步执行

1. **立即**：让子 agent 修复 `run_ab_v2.py` — 补全 alwayson variant、补 ppl 验证、写回 JSON
2. **然后**：跑完整 A/B（后台 ≈ 10 分钟）
3. **看结果再决定**：加宽 or 直接进消融
