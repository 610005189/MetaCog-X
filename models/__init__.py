"""MetaCog-X 模型模块"""
from .cognitive_particle import CognitiveParticle
from .triple_attention import TripleAttention
from .metacogx_layer import MetaCogXLayer, MetaCogXBlock
from .metacogx_model import MetaCogXModel
from .awareness_pool import AwarenessPool, MultiLayerAwarenessPool, AwarenessStats
from .sparse_meta_controller import (
    SparseMetaController,
    ControlSignals,
    MetaControllerWithSkip,
    AdaptiveMetaController
)
from .enlightenment_trigger import (
    EnlightenmentTrigger,
    TriggerAction,
    TriggerResult,
    AdaptiveEnlightenmentTrigger,
    EnlightenmentExecutor
)

__all__ = [
    # 基础组件
    "CognitiveParticle",
    "TripleAttention",
    "MetaCogXLayer",
    "MetaCogXBlock",
    "MetaCogXModel",
    # 觉知与控制
    "AwarenessPool",
    "MultiLayerAwarenessPool",
    "AwarenessStats",
    "SparseMetaController",
    "ControlSignals",
    "MetaControllerWithSkip",
    "AdaptiveMetaController",
    # 开悟触发
    "EnlightenmentTrigger",
    "TriggerAction",
    "TriggerResult",
    "AdaptiveEnlightenmentTrigger",
    "EnlightenmentExecutor",
]
