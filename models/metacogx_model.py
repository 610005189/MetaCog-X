"""MetaCog-X 完整模型

整合所有组件：
- 嵌入层 + 位置编码
- 认知粒子生成器
- N层认知Transformer（含可选元认知链路：AwarenessPool → SparseMetaController → temp_factor 逐层回传）
- L1 困境门控 + 条件激活架构（v3.0）
- 输出层（语言建模头）
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, List
from .cognitive_particle import CognitiveParticle
from .metacogx_layer import MetaCogXLayer
from .awareness_pool import AwarenessPool, AwarenessStats
from .sparse_meta_controller import SparseMetaController, ControlSignals
from .enlightenment_trigger import EnlightenmentTrigger, TriggerAction
from .dilemma_gate import DilemmaGate, attention_entropy, extract_features
from .dmn import DMN
from config import MetaCogXConfig


class PositionalEncoding(nn.Module):
    """旋转位置编码 (RoPE) - 简化为标准正弦位置编码"""

    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MetaCogXModel(nn.Module):
    """MetaCog-X 完整模型

    v3.0 条件激活架构数据流：
    1. 输入序列 -> 嵌入 + 位置编码
    2. 生成初始认知粒子 (content, meta, awareness)
    3. 逐层经过认知 Transformer 块 + 采集每层 attention entropy
    4. 收集完所有层 entropy 后喂入 L1 DilemmaGate -> dilemma_score
    5. 模式切换（plain ↔ metacog，带 patience hysteresis）
    6. plain 模式：temp_factor ≡ 1.0，controller / pool 不激活
       metacog 模式：AwarenessPool.update → stats → Controller → temp_factor ∈ [0.9, 1.1]
    7. 最终输出 logits + mode + switch_stats + last_dilemma_score + ctrl(可选)
    """

    def __init__(self, config: MetaCogXConfig, enable_metacog: bool = True):
        super().__init__()
        self.config = config
        self.enable_metacog = enable_metacog

        # 词嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            dropout=config.resid_dropout
        )

        # 认知粒子生成器
        self.cognitive_particle = CognitiveParticle(
            d_model=config.d_model,
            d_meta=config.d_meta,
            d_aware=config.d_aware,
            init_method="split"
        )

        # N层认知Transformer
        self.layers = nn.ModuleList([
            MetaCogXLayer(
                d_model=config.d_model,
                d_meta=config.d_meta,
                d_aware=config.d_aware,
                num_heads=config.num_heads,
                d_ffn=config.d_ffn,
                dropout=config.resid_dropout,
                attn_dropout=config.attn_dropout,
                ffn_dropout=config.ffn_dropout
            )
            for _ in range(config.num_layers)
        ])

        # 输出层归一化
        self.final_norm = nn.LayerNorm(config.d_model)

        # 语言建模头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定（Embedding和LM头共享权重）
        self.lm_head.weight = self.token_embedding.weight

        # DMN —— plain 与 metacog 两条路径都调用
        self.dmn = DMN(input_size=4, hidden_size=16)
        self._prev_surprise: Optional[torch.Tensor] = None
        self._last_h_self: Optional[torch.Tensor] = None
        self._last_surprise: Optional[torch.Tensor] = None

        # 元认知组件（条件初始化）
        if self.enable_metacog:
            self.awareness_pool = AwarenessPool(
                capacity=config.awareness_pool_capacity,
                feature_dim=config.d_aware,
                decay=config.awareness_decay,
                device="cpu"
            )
            self.meta_controller = SparseMetaController(
                d_meta=config.d_meta,
                d_aware=config.d_aware,
                hidden_dim=64
            )
            self.enlightenment_trigger = EnlightenmentTrigger(
                entropy_thresh=config.entropy_threshold,
                repeat_thresh=config.repeat_threshold,
                entropy_patience=config.entropy_patience
            )
            # v3.0 L1 困境门控 (+ DMN surprise 额外维度)
            input_dim = 2 * config.num_layers + 4
            self.l1_gate = DilemmaGate(
                input_dim=input_dim,
                hidden_dim=32,
                dropout=0.1,
            )
            self._mode_state = 'plain'  # 'plain' | 'metacog'
            self._plain_countdown = 0
            self._meta_countdown = 0
            self.enter_thresh = getattr(config, 'l1_enter_thresh', 0.7)
            self.exit_thresh = getattr(config, 'l1_exit_thresh', 0.3)
            self.enter_patience = getattr(config, 'l1_enter_patience', 2)
            self.exit_patience = getattr(config, 'l1_exit_patience', 3)
            self._switch_stats = {
                'switches': 0,
                'total_forward': 0,
                'plain_steps': 0,
                'meta_steps': 0,
            }
            self._last_ctrl_signals: Optional[ControlSignals] = None
            self._last_ctrl_logits: Optional[torch.Tensor] = None
            self._last_dilemma_score: Optional[float] = None
        else:
            self.awareness_pool = None
            self.meta_controller = None
            self.enlightenment_trigger = None
            self.l1_gate = None
            self._mode_state = 'plain'
            self._plain_countdown = 0
            self._meta_countdown = 0
            self._switch_stats = {
                'switches': 0,
                'total_forward': 0,
                'plain_steps': 0,
                'meta_steps': 0,
            }
            self._last_ctrl_signals = None
            self._last_ctrl_logits = None
            self._last_dilemma_score = None

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 初始化嵌入层
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.config.d_model ** -0.5)

        # 初始化所有线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _move_stats_to(stats: Optional[AwarenessStats], device: torch.device) -> Optional[AwarenessStats]:
        """把 AwarenessStats 所有字段搬到指定 device"""
        if stats is None:
            return None
        return AwarenessStats(
            mean=stats.mean.to(device),
            std=stats.std.to(device),
            trend=stats.trend.to(device),
            buffer_len=stats.buffer_len,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_meta: bool = False,
        enable_metacog: Optional[bool] = None,
        reset_pool: bool = False,
    ) -> Dict[str, Any]:
        """
        v3.0 条件激活 forward

        Args:
            input_ids: [B, L] 输入token IDs
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 标签（用于计算损失）
            return_meta: 是否返回meta和awareness
            enable_metacog: 运行时是否启用元认知（None=使用构造值）
            reset_pool: 是否强制重置 awareness pool

        Returns:
            dict with:
                - logits: [B, L, vocab_size] 输出logits
                - loss: 交叉熵损失（如果提供labels）
                - meta: [num_layers, B, L, d_meta] 每层的meta状态
                - awareness: [num_layers, B, L, d_aware] 每层的awareness
                - mode: 'plain' | 'metacog'
                - switch_stats: 模式切换统计
                - last_dilemma_score: 上一次 L1 打分
                - ctrl: 最后一次 ControlSignals（仅 metacog 模式下存在）
        """
        B, L = input_ids.shape
        device = input_ids.device

        use_metacog = self.enable_metacog if enable_metacog is None else bool(enable_metacog)

        if reset_pool and self.awareness_pool is not None:
            self.awareness_pool.reset()

        # 确保元认知组件（如果启用）在正确设备
        if use_metacog:
            self.meta_controller.to(device)
            self.l1_gate.to(device)
            self.awareness_pool.device = str(device)
            if self.awareness_pool.buffer:
                self.awareness_pool.buffer = [t.to(device) for t in self.awareness_pool.buffer]
            if self.awareness_pool.exp_avg is not None:
                self.awareness_pool.exp_avg = self.awareness_pool.exp_avg.to(device)
                self.awareness_pool.exp_var = self.awareness_pool.exp_var.to(device)

        # 1. 嵌入 + 位置编码
        x_emb = self.token_embedding(input_ids)
        x_emb = self.position_encoding(x_emb)

        # 2. 生成认知粒子
        content, meta, awareness = self.cognitive_particle(x_emb)

        # 存储每层的meta和awareness
        all_meta: Optional[List[torch.Tensor]] = [] if return_meta else None
        all_awareness: Optional[List[torch.Tensor]] = [] if return_meta else None

        # 每层的 attention entropy 列表（L1 gate 特征采集）
        entropy_list: List[torch.Tensor] = []

        # temp_factor：传给下一层 TripleAttention content 分支的 scale
        temp_factor: Optional[torch.Tensor] = None
        self._last_ctrl_signals = None
        self._last_ctrl_logits = None
        self._last_dilemma_score = None
        if self._prev_surprise is None or self._prev_surprise.size(0) != B:
            self._prev_surprise = torch.zeros(B, device=device)

        # 3. DMN h_self / surprise 管理（单 pass 方案，用上一步缓存，避免双 pass 计算图冲突）
        #    - 第一步（没有缓存）：h_self=零投影，用零熵初始化 DMN
        #    - 后续步：用上一步 cached h_self 注入
        #    - 当前步 entropy + logits_entropy + temp + prev_surprise → 算 当前 surprise（有梯度）
        #    - 本步结束后，detach surprise 存为 prev_surprise；本步 h_self detach 作为下步注入

        if self._last_h_self is None or self._last_h_self.size(0) != B:
            h_self = torch.zeros(B, 16, device=device)
            with torch.no_grad():
                self_features0 = torch.stack([
                    torch.zeros(B, device=device),
                    torch.zeros(B, device=device),
                    torch.ones(B, device=device),
                    torch.zeros(B, device=device),
                ], dim=-1)
                h_self_init, surprise_init = self.dmn(self_features0)
                h_self = h_self_init.detach()
                surprise_for_step = surprise_init.detach()
        else:
            h_self = self._last_h_self.to(device)
            surprise_for_step = self._prev_surprise

        # 4. 逐层 forward（单 pass，h_self 注入每层 TripleAttention 的 V 路径）
        for layer in self.layers:
            content, meta, awareness = layer(
                content=content,
                meta=meta,
                awareness=awareness,
                mask=attention_mask,
                temp_factor=temp_factor,
                h_self=h_self,
            )

            if use_metacog:
                try:
                    attn_w = getattr(layer.triple_attn, '_last_attn_c', None)
                except AttributeError:
                    attn_w = None
                if attn_w is None:
                    attn_w = getattr(layer.triple_attn, '_last_attn', None)
                if attn_w is not None and isinstance(attn_w, torch.Tensor):
                    if attn_w.dim() == 4:
                        e = attention_entropy(attn_w)
                    elif attn_w.dim() == 3:
                        e = attn_w.mean(dim=-1)
                    elif attn_w.dim() == 2:
                        e = attn_w.unsqueeze(1)
                    else:
                        e = None
                    if e is None or e.dim() != 3:
                        e = torch.zeros(B, layer.triple_attn.num_heads, L, device=device)
                    elif e.size(-1) != L or e.size(0) != B:
                        pad = torch.zeros(B, layer.triple_attn.num_heads, L, device=device)
                        bh = min(e.size(0), B), min(e.size(1), layer.triple_attn.num_heads)
                        pad[:bh[0], :bh[1], :min(e.size(-1), L)] = e[:bh[0], :bh[1], :min(e.size(-1), L)]
                        e = pad
                    entropy_list.append(e)
                else:
                    entropy_list.append(
                        torch.zeros(B, layer.triple_attn.num_heads, L, device=device)
                    )

            if return_meta:
                assert all_meta is not None and all_awareness is not None
                all_meta.append(meta)
                all_awareness.append(awareness)

        # 5. 最终归一化 + logits
        content = self.final_norm(content)
        logits = self.lm_head(content)

        # 6. 构造 DMN self_features（用本步的 entropy + logits）
        if entropy_list:
            all_means = []
            for e in entropy_list:
                if e.dim() == 3:
                    all_means.append(e.mean(dim=(1, 2)))
                else:
                    all_means.append(torch.zeros(B, device=device))
            attn_entropy_mean = torch.stack(all_means, dim=0).mean(dim=0)
        else:
            attn_entropy_mean = torch.zeros(B, device=device)

        p = torch.softmax(logits[:, -1, :], dim=-1)
        logits_entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        temp_factor_scalar = torch.ones(B, device=device)

        self_features = torch.stack(
            [
                attn_entropy_mean,
                logits_entropy,
                temp_factor_scalar,
                surprise_for_step,
            ],
            dim=-1,
        )  # [B, 4]

        # DMN 前向（有梯度，加入正式计算图）
        h_self_next, surprise = self.dmn(self_features)
        self._last_h_self = h_self_next.detach()
        self._last_surprise = surprise.detach()

        # 7. L1 打分 + 模式切换（+ DMN surprise 作为额外特征维度）
        if use_metacog and len(entropy_list) == len(self.layers):
            feats_base = extract_features(entropy_list, logits=None, token_ids=input_ids)
            feats = torch.cat([feats_base, surprise.unsqueeze(-1)], dim=-1)
            score = float(self.l1_gate.forward(feats).mean().item())
            self._last_dilemma_score = score

            if self._mode_state == 'plain':
                if score > self.enter_thresh:
                    self._meta_countdown += 1
                else:
                    self._meta_countdown = 0
                if self._meta_countdown >= self.enter_patience:
                    self._mode_state = 'metacog'
                    self._switch_stats['switches'] += 1
                    self._meta_countdown = 0
                    if self.awareness_pool is not None:
                        self.awareness_pool.reset()
            else:
                if score < self.exit_thresh:
                    self._plain_countdown += 1
                else:
                    self._plain_countdown = 0
                if self._plain_countdown >= self.exit_patience:
                    self._mode_state = 'plain'
                    self._switch_stats['switches'] += 1
                    self._plain_countdown = 0
                    if self.awareness_pool is not None:
                        self.awareness_pool.reset()

            self._switch_stats['total_forward'] += 1
            if self._mode_state == 'plain':
                self._switch_stats['plain_steps'] += 1
            else:
                self._switch_stats['meta_steps'] += 1

            if self._mode_state == 'metacog':
                assert self.awareness_pool is not None and self.meta_controller is not None
                self.awareness_pool.update(awareness)
                stats = self.awareness_pool.get_stats()
                stats = self._move_stats_to(stats, device)
                ctrl, ctrl_logits = self.meta_controller(meta, stats, return_logits=True)
                self._last_ctrl_signals = ctrl
                self._last_ctrl_logits = ctrl_logits
                temp_factor = ctrl.temp_factor
            else:
                temp_factor = torch.ones(B, 1, device=device)
                self._last_ctrl_signals = None
                self._last_ctrl_logits = None
        elif not use_metacog:
            temp_factor = None

        # 8. DMN 状态更新（下一步用当前 surprise 的 detach 版作为 prev）
        self._prev_surprise = surprise.detach()

        output: Dict[str, Any] = {'logits': logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            output['loss'] = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if return_meta:
            output['meta'] = torch.stack(all_meta, dim=0) if all_meta else None  # type: ignore[arg-type]
            output['awareness'] = torch.stack(all_awareness, dim=0) if all_awareness else None  # type: ignore[arg-type]

        output['mode'] = self._mode_state
        output['switch_stats'] = dict(self._switch_stats)
        output['last_dilemma_score'] = self._last_dilemma_score
        output['surprise'] = surprise.detach()
        output['h_self'] = h_self.detach()
        if self._last_ctrl_signals is not None:
            output['ctrl'] = self._last_ctrl_signals
            tf_raw = getattr(self._last_ctrl_signals, 'temp_factor_raw_logit', None)
            if tf_raw is not None:
                output['ctrl_tf_raw_logit'] = tf_raw
            tf = self._last_ctrl_signals.temp_factor
            if isinstance(tf, torch.Tensor):
                output['ctrl_tf'] = tf.detach()
        output['mode_had_metacog'] = self._mode_state == 'metacog'

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: int = 2,
        verbose: bool = False,
        max_enlightenment_steps: int = 3,
    ) -> torch.Tensor:
        """简单的自回归生成，步中接 EnlightenmentTrigger 自省日志"""
        self.eval()
        if self.enable_metacog and self.awareness_pool is not None:
            self.awareness_pool.reset()
            if self.enlightenment_trigger is not None:
                self.enlightenment_trigger.reset()
        step_count = 0
        with torch.no_grad():
            for _ in range(max_new_tokens):
                step_count += 1
                outputs = self.forward(
                    input_ids,
                    enable_metacog=self.enable_metacog,
                )
                logits = outputs["logits"]

                if self.enable_metacog and self.enlightenment_trigger is not None:
                    aware_stats = None
                    if self.awareness_pool is not None:
                        aware_stats = self.awareness_pool.get_stats()
                    trigger_result = self.enlightenment_trigger(
                        logits=logits,
                        aware_stats=aware_stats,
                        tokens=input_ids,
                        step=step_count,
                    )
                    if trigger_result.triggered and step_count <= max_enlightenment_steps:
                        if verbose:
                            print(
                                f"[Enlightenment step={step_count}] "
                                f"action={trigger_result.action.value} "
                                f"conf={trigger_result.confidence:.2f} "
                                f"entropy={trigger_result.current_entropy:.3f} "
                                f"repeat={trigger_result.repeat_count} "
                                f"reason={trigger_result.reason}"
                            )
                        if trigger_result.action == TriggerAction.RESET:
                            if self.awareness_pool is not None:
                                self.awareness_pool.reset()
                            self.enlightenment_trigger.reset()
                        elif trigger_result.action == TriggerAction.TOOL:
                            if verbose:
                                print("[Enlightenment] TOOL trigger: external tool skipped in v1")

                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

                if (next_token == eos_token_id).all():
                    break

        return input_ids

    def get_num_params(self) -> int:
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
