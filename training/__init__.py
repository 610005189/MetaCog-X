"""MetaCog-X 训练模块"""
from .losses import (
    TotalLoss,
    MetaConsistencyLoss,
    AwarenessPredictionLoss,
    AuxiliaryLossCalculator
)
from .rl_finetune import (
    Trajectory,
    MetaControllerPPO,
    GRPO,
    RewardCalculator
)
from .enlightenment_finetune import (
    AdversarialTask,
    AdversarialTaskGenerator,
    ImitationLearning,
    EnlightenmentFineTuner
)

__all__ = [
    # 损失函数
    "TotalLoss",
    "MetaConsistencyLoss",
    "AwarenessPredictionLoss",
    "AuxiliaryLossCalculator",
    # 强化学习
    "Trajectory",
    "MetaControllerPPO",
    "GRPO",
    "RewardCalculator",
    # 开悟微调
    "AdversarialTask",
    "AdversarialTaskGenerator",
    "ImitationLearning",
    "EnlightenmentFineTuner",
]
