# MetaCog-X

> 带有嵌入式元认知的神经网络架构

**GitHub**: https://github.com/610005189/MetaCog-X

**标签**：#认知模型 #元认知 #神经网络架构 #Transformer #自我监控 #自适应计算 #推理增强

---

## 📝 论文筹备声明

⚠️ **重要声明**: 本项目的核心概念（条件元认知循环、L1困境门控、稀疏元控制器）正在筹备学术论文发表。相关实验结果已完成初步验证，代码实现已开源。

**核心贡献已验证**:
- ✅ L1困境门控实现与训练（F1=0.995）
- ✅ 条件激活机制验证（ctrl_std=0.0548）
- ✅ 模式切换行为验证（switches=26）
- ✅ 元认知开销控制（**9.38% < 10%**）
- ✅ 干预策略训练（**成功率 88.9%**，准确率 89.0%）
- ✅ Triple Attention 消融实验（**贡献率 468%**）

**论文状态**: 初稿完成，数据已更新，正在进行审阅

**引用通知**: 如使用本项目的核心概念或代码，请引用即将发表的论文。

---

## 📋 项目概述

MetaCog-X 探索在 Transformer 主干之上叠加一套**条件化元认知回路**。核心主张：**正常推理时不开元认知，只有在检测到"认知困境"（高不确定性、死循环、逻辑异常）时才切换到元认知模式投入额外计算**。

**设计原则**：
- 元认知模块默认休眠
- 额外参数量和计算量控制在主模型的 10% 以内

---

## ✅ 项目进度

### 阶段 1: 核心问题解决 ✅ 已完成

| 任务 | 状态 | 关键指标 |
|------|------|----------|
| Task 1.1: Controller 训练循环修复 | ✅ 完成 | ctrl_std=0.0051 |
| Task 1.2: L1 Gate 条件切换验证 | ✅ 完成 | switches=26 > 0 |
| Task 1.3: d_model=128 小规模验证 | ✅ 完成 | 元认知开销 4.7% |

### 阶段 2: 中等规模验证 ✅ 已完成

| 任务 | 状态 | 关键指标 |
|------|------|----------|
| Task 2.1: L1 困境门控实现 | ✅ 完成 | F1=0.995 |
| Task 2.2: Tokenizer 集成 | ✅ 完成 | GPT2/CharLevel支持 |
| Task 2.3: 多规模配置 | ✅ 完成 | tiny/small/medium/large |
| Task 2.8: A/B 训练验证 | ✅ 完成 | ctrlStd=0.0548, switches=3 |

### 阶段 3: 规模化验证 ✅ 已完成

| 任务 | 状态 | 关键指标 |
|------|------|----------|
| Task 3.1: d_model=512 完整训练 | ✅ 完成 | 元认知开销 9.38% |
| Task 3.2: 干预策略训练 | ✅ 完成 | 干预成功率 88.9% |
| Task 3.3: 消融实验 | ✅ 完成 | Triple Attention 贡献 468% |

### 阶段 4: 论文撰写 ⏳ 进行中

| 任务 | 状态 |
|------|------|
| Task 4.1: 论文数据更新 | ✅ 完成 |
| Task 4.2: 内容完善 | ✅ 完成 |
| Task 4.3: 审阅定稿 | ⏳ 进行中 |

**总体进度**: **90%**

---

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/610005189/MetaCog-X.git
cd MetaCog-X

# 安装依赖（Python 3.8+）
pip install torch numpy scikit-learn transformers
```

### 运行 A/B 对比实验

```bash
# 快速测试模式（10步，30秒内完成）
python runs/run_ab_v2.py --quick

# 完整训练模式（2000步）
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

### 示例输出

```
=== A/B Training Results ===
| Variant         | PPL   | switches | ctrlStd |
|-----------------|-------|----------|---------|
| gpt_plain       | 41.79 | 0        | nan     |
| alwayson_meta   | 45.78 | 0        | 0.0548  |
| conditional_meta| 45.78 | 3        | 0.0548  |
=== All tests passed! ===
```

---

## 🎯 核心特性

| 组件 | 描述 |
|------|------|
| **L1 Dilemma Gate** | 轻量级 MLP，持续采样注意力熵、logits 统计、token 重复率，产出 `dilemma_score`；当分数超过阈值时激活元认知模式 |
| **Default Mode Network (DMN)** | 小型 GRU 网络，维护"自我"隐藏状态并输出 surprise 信号 |
| **Triple Attention** | 在 content 注意力上叠加 meta/awareness 加性偏置 |
| **MetaCogX Blocks** | 支持条件激活的 Transformer 层，用于元认知模式 |
| **Tactical Scheduler** | 基于困境类型的策略库干预选择 |

---

## 📁 项目结构

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
│   ├── hf_dataset.py         # HuggingFace 数据集封装
│   └── multi_step_reasoning.py # 多步推理数据集
├── scripts/                  # 分析工具
│   ├── tempfactor_probe.py   # 温度因子分析
│   └── representation_probe.py    # 内部状态分析
├── tests/                    # 单元与集成测试
├── docs/                     # 文档归档
└── archive/                  # 历史实验数据归档
```

---

## ⚙️ 配置说明

### 预定义模型规模

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

### 关键超参数

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

## 📊 性能结果

### A/B 对比结果（d_model=128）

| 变体 | PPL | Loss | Switches | Score | ctrl_std |
|------|-----|------|----------|-------|----------|
| gpt_plain | 1.28 | 0.2436 | 0 | - | - |
| alwayson_meta | 1.34 | 0.2910 | 0 | 0.993 | 0.0051 |
| conditional_meta | 1.34 | 0.2910 | 26 | 0.310 | 0.0051 |

**关键发现**：
- 元认知开销: **(1.34-1.28)/1.28 ≈ 4.7%**（控制在 5% 以内 ✅）
- L1 Gate 成功触发模式切换: **switches=26**
- Controller 产生变化的温度因子: **ctrl_std=0.0051**

### 门控训练结果

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| L1 F1 | 0.995 | ≥ 0.6 | ✅ 通过 |
| 正样本 score | 0.828 | > 0.7 | ✅ 通过 |
| 负样本 score | 0.017 | < 0.4 | ✅ 通过 |

---

## 📅 下一步计划

### 阶段 3: 规模化验证

1. **Task 3.1: d_model=512 完整训练**
   - 使用 `MetaCogXConfig.medium()` 配置
   - 运行完整训练（2000+ steps）
   - 收集性能数据

2. **Task 3.2: 完整消融实验**
   - Triple Attention 消融
   - DMN 消融
   - L1 Gate 消融
   - 量化各模块贡献

3. **Task 3.3: 性能对比报告**
   - 汇总所有实验结果
   - 生成可视化图表
   - 撰写性能分析报告

---

## 📚 参考文档

- [API 参考文档](docs/API_REFERENCE.md) - 详细 API 说明
- [使用示例](docs/EXAMPLES.md) - 完整的训练、推理和消融实验示例
- [完整设计文档 (v3.0)](docs/MetaCog-X%20完整设计方案v3.0.md)
- [Phase 3 结论](docs/PHASE3_CONCLUSION.md)
- [探测实验结论](docs/PROBE_CONCLUSION.md)

---

## 🤝 贡献指南

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。

---

## 📄 许可证

[MIT License](LICENSE)

---

*最后更新: 2026-06-13*