# MetaCog-X 完整设计方案

## 1. 概述

MetaCog-X 是一个内嵌元认知、觉知与开悟机制的神经网络架构。它将“自我监控”、“自我调节”和“框架质疑”作为基本计算原语，而非外挂模块。本方案面向AI自动开发，提供完整的技术规范。

**核心特性**：
- 认知粒子（Cognitive Particle）：每个token携带内容（content）、元认知状态（meta）、全局觉知（awareness）三部分信息。
- 三重注意力：同时建模内容、元认知状态和觉知之间的相互作用。
- 觉知池：维护历史觉知统计，用于检测异常。
- 稀疏元认知控制器：轻量级门控网络，动态调节注意力温度、推理路径、记忆强度。
- 开悟触发器：当系统陷入无效循环或高不确定性时，自动重置上下文或调用外部工具。

**设计原则**：
- 模块化：每个组件可独立实现和测试。
- 低开销：额外计算量控制在10%以内，能耗收益超过30%。
- 端到端可训练：所有参数均可通过损失函数优化。

---

## 2. 整体架构

```text
                           ┌──────────────┐
                           │  输入 Token  │
                           └──────┬───────┘
                                  ▼
                      ┌─────────────────────┐
                      │   嵌入层 + 位置编码  │
                      └─────────┬───────────┘
                                ▼
               ┌────────────────────────────────┐
               │       认知粒子生成器            │
               │  (content, meta, awareness)    │
               └─────────┬──────────┬──────────┘
                         │          │
                         ▼          ▼
               ┌──────────────┐  ┌──────────────┐
               │  N层认知Transformer │  │ 觉知池(全局) │
               │ (TripleAttn + FFN) │  └──────┬───────┘
               └─────────┬────────┘         │
                         │                  │
                         ▼                  │
               ┌──────────────────┐         │
               │ 稀疏元认知控制器  │◄────────┘
               │ (门控信号生成)    │
               └─────────┬────────┘
                         │
                         ▼
               ┌──────────────────┐
               │   开悟触发器      │
               │ (条件检测与干预)  │
               └─────────┬────────┘
                         │
                         ▼
               ┌──────────────────┐
               │   输出层(语言头)  │
               └──────────────────┘
```

**数据流**：
1. 输入序列 → 嵌入 → 生成初始认知粒子 `(C, M, A)`。
2. 逐层经过认知Transformer块，每块更新 `(C, M, A)`。
3. 每层后，`A` 送入全局觉知池（维护最近N步统计）。
4. 元认知控制器基于当前层的 `M` 和池化后的 `A` 产生控制信号（温度、跳过概率、记忆强度），回传给当前层和后续层。
5. 开悟触发器监测 `A` 统计和输出的logits，当满足条件时，发送干预信号（重置历史或调用工具）。
6. 最终输出logits和可选的自省日志。

---

## 3. 模块详细设计

### 3.1 认知粒子生成器

**输入**: `x_emb`  [batch, seq_len, d_model]
**输出**: `content`, `meta`, `awareness` 三个张量，形状均为 `[batch, seq_len, d_dim]`，其中 `d_dim` 分别为 `d_model`, `d_meta`, `d_aware`。

**实现**:
```python
self.to_particle = nn.Linear(d_model, d_model + d_meta + d_aware)
z = self.to_particle(x_emb)
content, meta, awareness = torch.split(z, [d_model, d_meta, d_aware], dim=-1)
```

**参数**:
- `d_model`: 256-1024（根据任务复杂度）
- `d_meta`: 32-64
- `d_aware`: 8-16

---

### 3.2 认知Transformer块（每层）

#### 3.2.1 三重注意力

**功能**: 同时计算基于 `content`、`meta`、`awareness` 的注意力。

**输入**:
- `content` [B, L, d_model]
- `meta` [B, L, d_meta]
- `awareness` [B, L, d_aware]
- `mask` (可选) [B, L] 填充掩码

**输出**: `content_out` [B, L, d_model]

**伪代码**:
```python
# content-based attention (标准多头)
Qc = linear_c(content).view(B, L, H, D_h)
Kc = linear_c(content).view(B, L, H, D_h)
Vc = linear_v(content).view(B, L, H, D_h)
attn_c = softmax(Qc @ Kc.transpose(-2,-1) / sqrt(D_h) + mask)
out_c = (attn_c @ Vc).transpose(1,2).reshape(B, L, d_model)

# meta-based attention
Qm = linear_meta(meta).view(B, L, H, D_h)
Km = linear_meta(meta).view(B, L, H, D_h)
attn_m = softmax(Qm @ Km.transpose(-2,-1) / sqrt(D_h) + mask)
out_m = (attn_m @ Vc).transpose(1,2).reshape(B, L, d_model)   # value reuse

# awareness-based attention
Qa = linear_aware(awareness).view(B, L, H, D_h)
Ka = linear_aware(awareness).view(B, L, H, D_h)
attn_a = softmax(Qa @ Ka.transpose(-2,-1) / sqrt(D_h) + mask)
out_a = (attn_a @ Vc).transpose(1,2).reshape(B, L, d_model)

# 融合
out = linear_fusion(torch.cat([out_c, out_m, out_a], dim=-1))
```

**参数**:
- 注意力头数 `H` = 8 (当 d_model=512)
- 融合层输出维度 `d_model`

#### 3.2.2 前馈网络 (FFN)

标准两层MLP，激活函数GELU，输出维度 `d_model`。

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, 4*d_model),
    nn.GELU(),
    nn.Linear(4*d_model, d_model)
)
```

#### 3.2.3 残差与层归一化

每子层后使用Pre-LN或Post-LN，建议Pre-LN：

```python
content = content + dropout(attention(layernorm(content), meta, awareness))
content = content + dropout(ffn(layernorm(content)))
# meta, awareness 也需要更新（通过简单线性或保持）
```

#### 3.2.4 meta/awareness 更新

为简化，可令每层中 `meta` 和 `awareness` 经过一个独立的轻量级MLP（可选），或直接保持不变（让后续层仍能利用）。推荐添加可选的更新：

```python
meta = meta + dropout(linear_meta(meta))
awareness = awareness + dropout(linear_aware(awareness))
```

---

### 3.3 觉知池 (Awareness Pool)

**功能**: 维护最近 `capacity` 步的全局 `awareness` 向量的滑动窗口，实时计算统计量。

**接口**:
```python
class AwarenessPool:
    def __init__(self, capacity=32, feature_dim=8, decay=0.95):
        self.capacity = capacity
        self.buffer = []   # list of [B, d_aware] tensors (需处理多batch)
        self.exp_avg = None
        self.exp_std = None
        self.decay = decay

    def update(self, aware_tensor):
        # aware_tensor: [B, L, d_aware] -> 取序列均值 [B, d_aware]
        mean_aware = aware_tensor.mean(dim=1).detach()
        self.buffer.append(mean_aware)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        # 更新指数滑动均值和方差
        new_batch_mean = mean_aware.mean(dim=0)
        if self.exp_avg is None:
            self.exp_avg = new_batch_mean
            self.exp_std = torch.zeros_like(new_batch_mean)
        else:
            self.exp_avg = self.decay * self.exp_avg + (1 - self.decay) * new_batch_mean
            # 使用Welford在线算法更新方差（略）

    def get_stats(self):
        if len(self.buffer) == 0:
            return None
        # 计算当前窗口内均值、方差、趋势（最后一步减第一步）
        stacked = torch.stack(self.buffer, dim=0)   # [T, B, d_aware]
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        trend = stacked[-1] - stacked[0] if len(stacked) > 1 else torch.zeros_like(mean)
        return {"mean": mean, "std": std, "trend": trend, "buffer_len": len(self.buffer)}
```

**说明**: 在实际多GPU训练中，需同步buffer或使用分布式存储。

---

### 3.4 稀疏元认知控制器 (Sparse Meta Controller)

**功能**: 基于当前层的 `meta` 和池化后的 `awareness` 统计，输出调控信号。

**输入**:
- `meta_avg`: [B, d_meta] (当前层 meta 的序列平均)
- `aware_stats`: dict from pool

**输出**:
- `temp_factor`: [B, 1] (控制注意力 softmax 温度，范围0.8~1.2)
- `skip_prob`: [B] (跳过后续计算的概率，用于随机正则)
- `mem_strength`: [B] (记忆强度，控制遗忘门)

**实现**:
```python
class SparseMetaController(nn.Module):
    def __init__(self, d_meta, d_aware, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_meta + d_aware, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, meta_avg, aware_stats):
        # aware_stats 需转换为张量 [B, d_aware]
        x = torch.cat([meta_avg, aware_stats], dim=-1)
        logits = self.net(x)  # [B,3]
        temp_factor = 0.8 + 0.4 * torch.sigmoid(logits[:, 0])
        skip_prob = torch.sigmoid(logits[:, 1])
        mem_strength = torch.sigmoid(logits[:, 2])
        return temp_factor, skip_prob, mem_strength
```

**使用**: 在每层注意力之前，将 `temp_factor` 作用于注意力 logits 缩放，`skip_prob` 用于随机跳过层（训练时采样，推理时可选择关闭），`mem_strength` 可用于控制从觉知池中读取历史信息的权重。

---

### 3.5 开悟触发器 (Enlightenment Trigger)

**功能**: 检测推理是否陷入无效循环或高熵不确定性，决定是否干预。

**输入**:
- `logits` [B, L, vocab] (当前输出)
- `aware_stats` (来自觉知池)
- `repeat_count` (连续重复token数)
- `time_step` (当前推理步数)

**输出**: `trigger` (bool)，`action` (str: "reset"/"tool")

**实现**:
```python
class EnlightenmentTrigger(nn.Module):
    def __init__(self, entropy_thresh=2.0, repeat_thresh=3, entropy_patience=5):
        self.entropy_thresh = entropy_thresh
        self.repeat_thresh = repeat_thresh
        self.counter = 0
    def forward(self, logits, aware_stats, repeat_count, step):
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
        trigger = False
        action = None
        if repeat_count >= self.repeat_thresh:
            trigger = True
            action = "reset"
        elif entropy > self.entropy_thresh:
            self.counter += 1
            if self.counter >= self.entropy_patience:
                trigger = True
                action = "tool"   # 调用外部工具
                self.counter = 0
        else:
            self.counter = max(0, self.counter-1)
        return trigger, action
```

**干预实现**:
- `reset`: 清除觉知池历史，清空部分注意力缓存，从当前步重新开始推理。
- `tool`: 调用外部工具（搜索、代码执行等），将结果编码后注入 `awareness` 或作为新token输入。

---

## 4. 训练策略

### 4.1 预训练阶段

目标：学习语言建模 + 辅助元认知信号。

**损失**:
- 主损失: 交叉熵 `L_ce`
- 辅助损失1: meta时序一致性（鼓励相邻步的meta相似，除非content突变）
- 辅助损失2: awareness自预测（用前k步的awareness预测当前步，提高表征质量）

**总损失**:
```python
L_total = L_ce + α * L_meta_consistency + β * L_aware_pred
```

其中α=0.01, β=0.005（需调优）。

**训练数据**: 大规模无监督文本（如The Pile, CommonCrawl）。

### 4.2 元认知微调（强化学习）

目标：学习使用控制信号（温度、跳过、记忆）来提升任务效率。

**任务集**: 多步推理（GSM8K）、逻辑谜题、工具使用等。

**奖励函数**:
```python
R = (task_success) - λ1 * (能耗) - λ2 * (冗余步数) - λ3 * (误触发开悟次数)
```
能耗可使用模型FLOPs或实际推理时间。

**算法**: PPO或GRPO，训练元认知控制器和开悟触发器的参数，主模型权重可冻结或微调。

### 4.3 开悟微调

**数据**: 构造需要“框架切换”的对抗任务（例如初始条件错误，需质疑假设）。通过模仿学习让模型学会触发开悟后执行重置或工具调用。

---

## 5. 推理与部署

### 5.1 标准推理模式

- 使用预训练+微调后的模型。
- 逐token生成，每步更新觉知池。
- 元认知控制器根据当前meta和aware_stats输出控制信号，调整生成过程（如温度采样）。
- 开悟触发器在后台检测，若触发则按动作干预（如清空历史、调用工具）。

### 5.2 性能优化

- 三重注意力可使用FlashAttention-3实现，融合三个分支。
- 稀疏控制器仅在每层运行一次，开销极小。
- 觉知池使用指数加权移动平均，无需维护长队列（可选）。

---

## 6. 评估指标

- **推理效率**: 单位任务成功率的FLOPs下降百分比。
- **自我干预有效性**: 干预后任务成功率提升 vs 干预次数。
- **开悟解脱率**: 触发后最终解决任务的比率。
- **觉知召回率**: 正确检测到陷入循环/高熵状态的准确率。

---

## 7. 实现计划（AI自动开发）

建议按以下顺序实现模块：

1. 基础骨架：`CognitiveParticle`, `TripleAttention`, `MetaCogXLayer`, 可运行前向。
2. 觉知池和元认知控制器（不参与训练，仅统计）。
3. 开悟触发器（规则版）。
4. 训练循环（语言建模 + 辅助损失）。
5. 强化学习微调（环境构造）。
6. 集成测试。

**参考代码仓库结构** (已提供过类似README，此处略)。

---

## 8. 附录：接口规范（供AI自动生成）

### 8.1 模型配置
```python
config = {
    "vocab_size": 50257,
    "d_model": 512,
    "d_meta": 32,
    "d_aware": 16,
    "num_layers": 12,
    "num_heads": 8,
    "capacity": 64,
    "entropy_threshold": 2.5,
    "repeat_threshold": 3
}
```

### 8.2 输入输出

**前向**:
```python
output = model(
    input_ids, 
    attention_mask=None,
    return_meta=True,
    enable_metacog=True
)
# output.logits, output.meta, output.awareness, output.stats, output.trigger
```

### 8.3 工具调用集成

当开悟触发器返回 `action="tool"` 时，调用外部API，将结果编码后作为新token输入（或作为awareness的一部分）。

---

**本方案提供了足够的细节，以便AI自动开发实现完整的MetaCog-X原型。**