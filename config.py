"""MetaCog-X 配置参数"""

from dataclasses import dataclass


@dataclass
class MetaCogXConfig:
    """MetaCog-X 模型配置"""

    # 词汇表
    vocab_size: int = 50257

    # 模型维度
    d_model: int = 512       # 内容向量维度
    d_meta: int = 32         # 元认知状态维度
    d_aware: int = 16        # 觉知维度

    # Transformer结构
    num_layers: int = 12
    num_heads: int = 8
    d_head: int = 64         # 每个头的维度 (d_model / num_heads)
    d_ffn: int = 2048        # 前馈网络维度

    # 注意力dropout
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    ffn_dropout: float = 0.1

    # 觉知池
    awareness_pool_capacity: int = 64
    awareness_decay: float = 0.95

    # 开悟触发器
    entropy_threshold: float = 2.5
    repeat_threshold: int = 3
    entropy_patience: int = 5

    # 辅助损失权重
    alpha_meta: float = 0.01   # meta一致性损失权重
    beta_aware: float = 0.005  # awareness自预测损失权重

    # L1 困境门控（v3.0 条件激活架构）
    l1_enter_thresh: float = 0.7
    l1_exit_thresh: float = 0.3
    l1_enter_patience: int = 2
    l1_exit_patience: int = 3

    # 推理
    max_seq_len: int = 2048
    use_flash_attn: bool = False  # 是否使用FlashAttention

    def __post_init__(self):
        # 自动计算d_head
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除")
        self.d_head = self.d_model // self.num_heads

    @classmethod
    def tiny(cls):
        """小型配置 (d_model=128) - 用于快速测试和验证"""
        return cls(
            d_model=128,
            d_meta=16,
            d_aware=8,
            num_layers=4,
            num_heads=4,
            d_ffn=512,
            max_seq_len=512
        )

    @classmethod
    def small(cls):
        """小型配置 (d_model=256) - 中等规模验证"""
        return cls(
            d_model=256,
            d_meta=24,
            d_aware=12,
            num_layers=8,
            num_heads=8,
            d_ffn=1024,
            max_seq_len=1024
        )

    @classmethod
    def medium(cls):
        """中等配置 (d_model=512) - 规模化验证"""
        return cls(
            d_model=512,
            d_meta=32,
            d_aware=16,
            num_layers=12,
            num_heads=8,
            d_ffn=2048,
            max_seq_len=2048
        )

    @classmethod
    def large(cls):
        """大型配置 (d_model=1024) - 大规模训练"""
        return cls(
            d_model=1024,
            d_meta=64,
            d_aware=32,
            num_layers=16,
            num_heads=16,
            d_ffn=4096,
            max_seq_len=2048
        )

    def __str__(self):
        return f"MetaCogXConfig(d_model={self.d_model}, num_layers={self.num_layers}, num_heads={self.num_heads})"
