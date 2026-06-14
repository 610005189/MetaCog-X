# MetaCog-X Core

Conditional Metacognition Detection and Intervention Framework.

## Installation

```bash
pip install metacogx-core
```

## Quick Start

```python
from metacogx_core import DilemmaDetector, StrategyRouter

detector = DilemmaDetector()
signal = detector.detect(tokens, logits)

router = StrategyRouter()
# ...
```

## Features

- **DilemmaDetector**: 检测文本生成过程中的困境状态
- **StrategyRouter**: 根据信号选择并执行干预策略
- 支持多种困境类型：语义卡顿、句法异常、模式重复、生成停滞

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
