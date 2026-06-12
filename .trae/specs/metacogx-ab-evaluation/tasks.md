# MetaCog-X 价值验证 — 实施计划

## [ ] Task 1: 改造 data/hf_dataset.py 支持 WikiText-2 真实 train/valid/test 官方拆分
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - data/hf_dataset.py 新增 `load_wikitext(split="train|validation|test", cache_dir=None, timeout_sec=30, fallback_path=None)` 函数：
    1. 先试 `datasets.load_dataset("wikitext","wikitext-2-raw-v1", split=split, cache_dir=...)`，主线程 join(timeout)。
    2. 超时 → 尝试 fallback 路径 `fallback_path/"train.txt"`, `"validation.txt"`（用户可手工下载 wiki-2 raw 的 train/valid 放在这）。
    3. 再失败 → 用 bundled 820 条 fallback（保留现有）。
  - 返回 HFDataset(tokenizer, texts, max_length)。
  - 打印实际走了哪条路径，len(train)/len(valid)/len(test)。
  - run.py --real_data 和 training/ab_trainer.py 统一调用这一个 loader。
- **Acceptance Criteria Addressed**: AC-1, AC-5
- **Test Requirements**:
  - `programmatic` TR-1.1: python -c "from data.hf_dataset import load_wikitext; ds=load_wikitext('train', tokenizer=...); print(len(ds))" 输出 > 1000。
  - `programmatic` TR-1.2: run.py --real_data 显示 train 条数 ≥ 1000 或 fallback 已启用并打印 len。
  - `programmatic` TR-1.3: python run.py --mode full_test 7/7 PASS（回归）。
- **Notes**: PowerShell 线程 join(timeout) 需要用 threading.Event + Timer 或直接忽略 timeout。简单点：只试 datasets.load_dataset 一次，5 秒内没返回就走 fallback。

## [ ] Task 2: 写 training/ab_trainer.py — MetaCog-X vs GPT 基线 A/B 训练器
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - argparse: `--variant {metacog,gpt}`、`--d_model`(256)、`--d_meta`(32)、`--d_aware`(16)、`--num_layers`(4)、`--num_heads`(4)、`--batch_size`(4)、`--max_seq_len`(128)、`--steps`(2000)、`--lr`(2e-4)、`--device`、`--seed`(42)、`--eval_every`(200)、`--save_csv`。
  - 固定 torch.manual_seed + random.seed + numpy.random.seed + cudnn.deterministic=True（如果 GPU）。
  - config = MetaCogXConfig(d_model, d_meta, d_aware, num_layers, num_heads, vocab_size=50257, d_ffn=args.d_model*4, ...)。
  - variant=gpt → enable_metacog=False, alpha=0, beta=0（TotalLoss 退化为仅 CE）。
  - variant=metacog → enable_metacog=True, alpha=0.01, beta=0.005。
  - 训练循环：每 eval_every 步跑 valid split，计算 ppl = exp(CE)；把 step, train_loss, valid_ppl 追加到 CSV。
  - 训练结束：打印 final CE、final ppl、best valid ppl、训练总秒数。
  - 两个 variant 共用同一套 optimizer / lr / weight_decay=0.01 / AdamW。
- **Acceptance Criteria Addressed**: AC-2, AC-3
- **Test Requirements**:
  - `programmatic` TR-2.1: variant=gpt 训练 10 步，loss 从 ~11 下降至少 5%。
  - `programmatic` TR-2.2: variant=metacog 训练 10 步，loss 分量有 meta>0 且 aware>0（≥ 1e-6）。
  - `programmatic` TR-2.3: 输出 CSV 文件至少含 step / train_loss / valid_ppl 三列。
- **Notes**: CPU 上 2000 步 4 层 256d 的 transformer，step 数 * bs * seq_len = 2000*4*128 ≈ 1M token，可接受。如果太慢，默认 2000，可 --steps 改小。

## [ ] Task 3: 跑 A/B 两个变体 + 输出对比表
- **Priority**: P0
- **Depends On**: Task 2
- **Description**:
  - 实际执行两次：
    ```
    python training/ab_trainer.py --variant gpt       --steps 2000 --save_csv runs/gpt.csv
    python training/ab_trainer.py --variant metacog   --steps 2000 --save_csv runs/metacog.csv
    ```
  - 比较 valid ppl（同 checkpoint 步数下）。
  - 打印 summary：
    ```
    GPT         final_ppl=X   best_ppl=Y
    MetaCog-X   final_ppl=X   best_ppl=Y
    delta_ppl = Meta - GPT
    delta_log_ppl = log(Meta_ppl) - log(GPT_ppl)
    winner = 'metacog' if delta_log_ppl < 0 else 'gpt'
    ```
  - 生成样例（同一 prompt）side by side 打印。
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-3.1: summary 表格两列都能读。
  - `human-judgement` TR-3.2: 人眼对比两条生成样例，判断哪个更像英文。
- **Notes**: CPU 上实际训练需要很久，可能一次要跑几小时。可以 --steps 500 先看趋势，再上 2000。

## [x] Task 4: representation_probe.py — meta/awareness 表征分析（awareness 区分度 15.4x，但 controller 塌陷）
- **Priority**: P1
- **Depends On**: Task 3
- **Description**:
  - 加载 Task 3 中 metacog variant 的 checkpoint（如果没存 checkpoint 就再 forward 一次训练后的模型）。
  - 拿真实 data 的一个 batch forward(return_meta=True, enable_metacog=True)。
  - 分析：
    1. layer-wise meta cosine similarity matrix (L x L)。
    2. 不同 batch 之间 meta MSE（连续 5 个 batch）。
    3. controller temp_factor mean / std / histogram（min/max 必须在 0.8-1.2）。
    4. awareness 正例 vs 乱码 prompt 的 L2 距离（"Artificial intelligence is" vs "asdf jkl; foo bar"）。
  - 打印每个分析的数字 + 一句解释。
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `programmatic` TR-4.1: cosine 矩阵所有值在 [-1,1]。
  - `programmatic` TR-4.2: temp_factor min >= 0.8, temp_factor max <= 1.2。
  - `programmatic` TR-4.3: 乱码 vs 正例 awareness L2 距离 > 0（区分度）。

## [ ] Task 5: 完整回归 + 文档结论
- **Priority**: P1
- **Depends On**: Task 1..4
- **Description**:
  - python run.py --mode full_test 7/7 PASS。
  - python tests/run_tests.py 单元 13/13 + 集成 5/5 + 评估 5/5 PASS。
  - 写一份 concise 的 CONCLUSION.txt：
    - 数据来源（WikiText-2 / fallback）。
    - 训练步数、valid ppl 对比。
    - winner = metacog | gpt | draw（|delta_log_ppl| < 0.02 算 draw）。
    - meta 表征分析结论。
    - 下一步建议（如果 win → 上 PPO；如果 draw → 调 alpha/beta；如果 lose → 重新设计 meta 监督信号）。
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-5.1: 两条回归命令 exit_code=0。
  - `human-judgement` TR-5.2: CONCLUSION.txt 让下一个人能知道接下来该做什么。
