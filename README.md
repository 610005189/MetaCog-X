# MetaCog-X

> Neural Network Architecture with Embedded Metacognition

---

## Overview

MetaCog-X explores adding a **conditional metacognitive loop** on top of a Transformer backbone. The core proposition: **run standard inference without metacognition by default; only activate metacognitive mode when "cognitive dilemmas" are detected** (high uncertainty, loops, logical anomalies).

**Design Principles**:
- Metacognitive modules are dormant by default
- Additional parameters and FLOPs are controlled within 10% of the main model

---

## Key Features

| Component | Description |
|-----------|-------------|
| **L1 Dilemma Gate** | Lightweight MLP that continuously samples attention entropy, logits statistics, and token repetition to produce a `dilemma_score`; activates metacognition when score exceeds threshold |
| **Default Mode Network (DMN)** | Tiny GRU network maintaining "self" hidden state and outputting surprise signals |
| **Triple Attention** | Augments content attention with meta/awareness additive biases |
| **MetaCogX Blocks** | Transformer layers with conditional activation for metacognitive mode |
| **Tactical Scheduler** | Strategy library for intervention selection based on dilemma type |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd MetaCog-X

# Install dependencies
pip install torch numpy scikit-learn
```

### Run A/B Comparison

```bash
python runs/run_ab_v2.py
```

This trains three variants:
- `gpt_plain`: Standard Transformer baseline
- `metacog_alwayson`: Metacognition always active
- `metacog_conditional`: Metacognition activated by L1 gate

### Train L1 Gate

```bash
python runs/train_l1_gate.py
```

### Run Ablation Studies

```bash
python runs/ablation_triple_attention.py
python runs/ablation_dmn.py
```

---

## Project Structure

```
MetaCog-X/
├── config.py                 # Hyperparameter configuration
├── models/                   # Core architecture
│   ├── metacogx_model.py     # Main model class
│   ├── metacogx_layer.py     # MetaCog-X transformer layer
│   ├── triple_attention.py   # Triple attention mechanism
│   ├── dilemma_gate.py       # L1 dilemma detection gate
│   ├── dmn.py                # Default Mode Network
│   ├── awareness_pool.py     # Awareness statistics tracking
│   ├── sparse_meta_controller.py  # Sparse meta controller
│   ├── cognitive_particle.py # Content/meta/awareness projection
│   ├── tactical_scheduler.py # Strategy selection
│   └── enlightenment_trigger.py   # Loop detection & reset
├── training/                 # Training framework
│   ├── rl_framework.py       # RL training for controller
│   ├── rl_finetune.py        # RL finetuning script
│   ├── losses.py             # Auxiliary loss functions
│   └── ab_trainer.py         # A/B training utilities
├── runs/                     # Experiment entry points
│   ├── run_ab_v2.py          # Main A/B comparison script
│   ├── train_l1_gate.py      # L1 gate training
│   ├── ablation_*.py         # Ablation study scripts
│   └── summarize_*.py        # Result summarization
├── data/                     # Data utilities
│   ├── dataset.py            # Byte-level dataset
│   └── hf_dataset.py         # HuggingFace dataset wrapper
├── scripts/                  # Analysis tools
│   ├── tempfactor_probe.py   # Temperature factor analysis
│   └── representation_probe.py    # Internal state analysis
├── tests/                    # Unit & integration tests
├── docs/                     # Documentation archives
└── archive/                  # Archived experiment data
```

---

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Content embedding dimension |
| `d_meta` | 32 | Metacognition state dimension |
| `d_aware` | 16 | Awareness dimension |
| `num_layers` | 12 | Number of transformer layers |
| `num_heads` | 8 | Number of attention heads |
| `l1_enter_thresh` | 0.7 | Dilemma gate entry threshold |
| `l1_exit_thresh` | 0.3 | Dilemma gate exit threshold |
| `l1_enter_patience` | 2 | Entry patience (steps) |
| `l1_exit_patience` | 3 | Exit patience (steps) |

---

## Performance

| Variant | PPL | Δ vs Plain |
|---------|-----|------------|
| gpt_plain | 1.23 | baseline |
| metacog_alwayson | 1.34 | +9.2% |
| metacog_conditional | 1.34 | +8.8% |

*Results from d_model=128, 4 layers, 500 training steps*

---

## References

- [Complete Design Document (v3.0)](docs/MetaCog-X%20完整设计方案v3.0.md)
- [Phase 3 Conclusion](docs/PHASE3_CONCLUSION.md)
- [Probe Conclusion](docs/PROBE_CONCLUSION.md)

---

## License

MIT License

---

*Last updated: 2026-06-12*