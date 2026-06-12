# MetaCog-X

> 带有嵌入式元认知的神经网络架构

---

## 项目概述

MetaCog-X 探索在 Transformer 主干之上叠加一套**条件化元认知回路**。核心主张：**正常推理时不开元认知，只有在检测到"认知困境"（高不确定性、死循环、逻辑异常）时才切换到元认知模式投入额外计算**。

**设计原则**：
- 元认知模块默认休眠
- 额外参数量和计算量控制在主模型的 10% 以内

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
pip install torch numpy scikit-learn
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

| 变体 | PPL | 与 Plain 对比 |
|------|-----|--------------|
| gpt_plain | 1.23 | baseline |
| metacog_alwayson | 1.34 | +9.2% |
| metacog_conditional | 1.34 | +8.8% |

*结果来自 d_model=128, 4 层, 500 训练步数*

---

## 参考文档

- [完整设计文档 (v3.0)](docs/MetaCog-X%20完整设计方案v3.0.md)
- [Phase 3 结论](docs/PHASE3_CONCLUSION.md)
- [探测实验结论](docs/PROBE_CONCLUSION.md)

---

## 许可证

MIT License

---

*最后更新: 2026-06-12*