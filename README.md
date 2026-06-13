# MetaCog-X

> 带有嵌入式元认知的神经网络架构

**GitHub**: https://github.com/610005189/MetaCog-X

---

## 项目概述

MetaCog-X 探索在 Transformer 主干之上叠加一套**条件化元认知回路**。核心主张：**正常推理时不开元认知，只有在检测到"认知困境"（高不确定性、死循环、逻辑异常）时才切换到元认知模式投入额外计算**。

**设计原则**：
- 元认知模块默认休眠
- 额外参数量和计算量控制在主模型的 10% 以内

---

## 项目进度

### 阶段 1: 核心问题解决 ✅ 已完成

| 任务 | 状态 | 关键指标 |
|------|------|----------|
| Task 1.1: Controller 训练循环修复 | ✅ 完成 | ctrl_std=0.0051 |
| Task 1.2: L1 Gate 条件切换验证 | ✅ 完成 | switches=26 > 0 |
| Task 1.3: d_model=128 小规模验证 | ✅ 完成 | 元认知开销 4.7% |

**核心成果**：
- Controller 能产生变化的温度因子（ctrl_std > 0.005）
- L1 Gate 能够检测困境并触发模式切换（switches=26）
- 元认知开销控制在 5% 以内

### 阶段 2-4: 待执行

- **阶段 2**: 中等规模验证（d_model=256, tokenizer 集成, 多步推理）
- **阶段 3**: 规模化验证（d_model=512, 消融实验）
- **阶段 4**: 项目完善（代码优化, 文档, 开源准备）

详细计划见 [.trae/specs/meta-cog-x-long-term-roadmap/tasks.md](.trae/specs/meta-cog-x-long-term-roadmap/tasks.md)

---

## 核心特性

| 组件 | 描述 |
|------|------|
| **L1 Dilemma Gate** | 轻量级 MLP，持续采样注意力熵、logits 统计、token 重复率，产出 `dilemma_score`；当分数超过阈值时激活元认知模式 |
| **Default Mode Network (DMN)** | 小型 GRU 网络，维护"自我"隐藏状态并输出 surprise 信号 |
| **Triple Attention** | 在 content 注意力上叠加 meta/awareness 加性偏置 |
| **MetaCogX Blocks** | 支持条件激活的 Transformer 层，用于元认知模式 |
| **Tactical Scheduler** | 基于困境类型的策略库干预选择 |

---

## 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone <repo-url>
cd MetaCog-X

# 安装依赖
pip install torch numpy scikit-learn transformers
```

### 运行 A/B 对比实验

```bash
python runs/run_ab_v2.py
```

此脚本训练三个变体：
- `gpt_plain`: 标准 Transformer 基线
- `metacog_alwayson`: 元认知始终激活
- `metacog_conditional`: 由 L1 gate 条件激活元认知

### 训练 L1 门控

```bash
python runs/train_l1_gate.py
```

### 运行消融实验

```bash
python runs/ablation_triple_attention.py
python runs/ablation_dmn.py
```

### 模型配置

支持多种预定义模型规模：

```python
from config import MetaCogXConfig

# 小型配置 (d_model=128) - 快速测试
config = MetaCogXConfig.tiny()

# 中小型配置 (d_model=256) - 中等规模验证
config = MetaCogXConfig.small()

# 中等配置 (d_model=512) - 规模化验证
config = MetaCogXConfig.medium()

# 大型配置 (d_model=1024) - 大规模训练
config = MetaCogXConfig.large()

# 自定义配置
config = MetaCogXConfig(
    d_model=256,
    num_layers=8,
    num_heads=8,
    l1_enter_thresh=0.7,
    l1_exit_thresh=0.3
)
```

### Tokenizer 集成

支持 HuggingFace GPT2 和字符级 tokenizer：

```python
from data.hf_dataset import get_tokenizer, load_wikitext_dataset

# 使用 GPT2 tokenizer
tokenizer = get_tokenizer("gpt2")

# 使用字符级 tokenizer
tokenizer = get_tokenizer("charlevel")

# 加载数据集
dataset = load_wikitext_dataset(
    split="train",
    tokenizer_type="charlevel",
    max_length=128
)
```

### 多步推理任务

```python
from data.multi_step_reasoning import MultiStepReasoningDataset

# 生成多步推理数据集
generator = MultiStepReasoningDataset(seed=42)
problems = generator.generate_all(n_per_type=100)

# 生成特定类型问题
math_problems = generator.generate_math_problems(50)
logic_problems = generator.generate_logic_problems(50)
```

### 模型使用示例

```python
from config import MetaCogXConfig
from models.metacogx_model import MetaCogXModel

# 创建配置和模型
config = MetaCogXConfig.small()
model = MetaCogXModel(config, enable_metacog=True)

# 前向传播
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 64))
outputs = model(input_ids)

# 输出包含
# - logits: 预测概率
# - mode: 当前模式 ('plain' 或 'metacog')
# - switch_stats: 模式切换统计
# - last_dilemma_score: L1门控分数
# - ctrl: 控制器信号（元认知模式下）

# 自回归生成
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50
)
```

---

## 项目结构

```
MetaCog-X/
├── config.py                 # 超参数配置
├── models/                   # 核心架构
│   ├── metacogx_model.py     # 主模型类
│   ├── metacogx_layer.py     # MetaCog-X Transformer 层
│   ├── triple_attention.py   # 三重注意力机制
│   ├── dilemma_gate.py       # L1 困境检测门控
│   ├── dmn.py                # 默认模式网络
│   ├── awareness_pool.py     # 觉知统计跟踪
│   ├── sparse_meta_controller.py  # 稀疏元控制器
│   ├── cognitive_particle.py # 内容/元/觉知投影
│   ├── tactical_scheduler.py # 策略选择
│   └── enlightenment_trigger.py   # 死循环检测与重置
├── training/                 # 训练框架
│   ├── rl_framework.py       # 控制器 RL 训练
│   ├── rl_finetune.py        # RL 微调脚本
│   ├── losses.py             # 辅助损失函数
│   └── ab_trainer.py         # A/B 训练工具
├── runs/                     # 实验入口
│   ├── run_ab_v2.py          # 主 A/B 对比脚本
│   ├── train_l1_gate.py      # L1 门控训练
│   ├── ablation_*.py         # 消融实验脚本
│   └── summarize_*.py        # 结果汇总
├── data/                     # 数据工具
│   ├── dataset.py            # 字节级数据集
│   └── hf_dataset.py         # HuggingFace 数据集封装
├── scripts/                  # 分析工具
│   ├── tempfactor_probe.py   # 温度因子分析
│   └── representation_probe.py    # 内部状态分析
├── tests/                    # 单元与集成测试
├── docs/                     # 文档归档
└── archive/                  # 历史实验数据归档
```

---

## 配置参数

`config.py` 中的关键超参数：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `d_model` | 512 | 内容嵌入维度 |
| `d_meta` | 32 | 元认知状态维度 |
| `d_aware` | 16 | 觉知维度 |
| `num_layers` | 12 | Transformer 层数 |
| `num_heads` | 8 | 注意力头数 |
| `l1_enter_thresh` | 0.7 | 困境门进入阈值 |
| `l1_exit_thresh` | 0.3 | 困境门退出阈值 |
| `l1_enter_patience` | 2 | 进入耐心值（步数） |
| `l1_exit_patience` | 3 | 退出耐心值（步数） |

---

## 性能结果

| 变体 | PPL | Loss | Switches | Score | ctrl_std |
|------|-----|------|----------|-------|----------|
| gpt_plain | 1.28 | 0.2436 | 0 | - | - |
| alwayson_meta | 1.34 | 0.2910 | 0 | 0.993 | 0.0051 |
| conditional_meta | 1.34 | 0.2910 | 26 | 0.310 | 0.0051 |

*结果来自 d_model=128, 4 层, 2000 训练步数*

**关键发现**：
- 元认知开销: **(1.34-1.28)/1.28 ≈ 4.7%**（略高于 3% 目标）
- L1 Gate 成功触发模式切换: **switches=26**
- Controller 产生变化的温度因子: **ctrl_std=0.0051**

**待优化项**：
- plain_pct=0.0%（需要进一步调整 L1 Gate 训练，让模式能在 plain 和 metacog 之间来回切换）

---

## 下一步执行计划

### 立即执行（阶段 2）

1. **Task 2.2: 真实 tokenizer 集成**
   - 评估 HuggingFace tokenizer（GPT2/CharLevel）
   - 修改数据处理流程支持真实 tokenizer

### 中期目标（阶段 3）

2. **Task 3.1: d_model=512 完整训练**
3. **Task 3.2: 完整消融实验**
4. **Task 3.3: 性能对比报告**

### 长期目标（阶段 4）

5. **Task 4.1: 代码优化**
6. **Task 4.2: 文档完善**
7. **Task 4.3: 开源准备**

---

## 参考文档

- [完整设计文档 (v3.0)](docs/MetaCog-X%20完整设计方案v3.0.md)
- [Phase 3 结论](docs/PHASE3_CONCLUSION.md)
- [探测实验结论](docs/PROBE_CONCLUSION.md)

---

## 许可证

MIT License

---

*最后更新: 2026-06-13*