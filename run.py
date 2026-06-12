# -*- coding: utf-8 -*-
import torch
import argparse
from config import MetaCogXConfig
from models import (
    MetaCogXModel,
    AwarenessPool,
    MultiLayerAwarenessPool,
    SparseMetaController,
    EnlightenmentTrigger,
    TriggerAction,
)
from data.dataset import SimpleTextDataset, DummyTokenizer, create_dataloader
from train import Trainer
from training import (
    TotalLoss,
    MetaControllerPPO,
    GRPO,
    RewardCalculator,
    AdversarialTaskGenerator,
)


FALLBACK_WIKITEXT_SAMPLES = [
    "Natural language processing is a subfield of linguistics computer science and artificial intelligence concerned with the interactions between computers and human language.",
    "A neural network is a network or circuit of neurons or in a modern sense an artificial neural network composed of artificial neurons or nodes.",
    "The transformer architecture uses a self attention mechanism to weight the significance of each part of the input sequence differently.",
    "Machine learning is the study of computer algorithms that improve automatically through experience and by the use of data.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Recurrent neural networks are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.",
    "Long short term memory is a type of recurrent neural network architecture designed to overcome the vanishing gradient problem.",
    "The attention mechanism allows the model to focus on the most relevant parts of the input sequence when producing each part of the output.",
    "Backpropagation is a method used in artificial neural networks to calculate the gradient of the loss function with respect to the weights.",
    "Stochastic gradient descent is an iterative optimization algorithm for minimizing an objective function with smoothness properties.",
    "Convolutional neural networks are a class of deep feedforward networks that learn features via filters applied across spatial data.",
    "The GPT architecture stacks transformer decoder blocks and learns a next token prediction objective on a large text corpus.",
    "BERT uses transformer encoder layers and is trained with masked language modeling and next sentence prediction objectives.",
    "The perplexity of a language model measures how surprised it is by a sequence averaged per token.",
    "Cross entropy loss is the standard training objective when optimizing a neural language model over a vocabulary.",
    "Flash attention is an algorithmic optimization that reorders the attention computation to reduce memory access and increase throughput.",
    "Layer normalization stabilizes training by normalizing activations across the feature dimension before each sublayer.",
    "Residual connections enable deep networks to learn by letting gradients flow directly through additive skip paths.",
    "Weight decay is an L2 penalty on model parameters that discourages overfitting in large neural networks.",
    "Dropout randomly zeros a fraction of units during training to reduce co adaptation between neurons.",
    "Gradient clipping bounds the global norm of gradients before the optimizer step to prevent exploding gradients.",
    "AdamW decouples weight decay from the adaptive learning rates computed by the Adam optimizer.",
    "Learning rate schedules such as cosine annealing or linear warmup adapt the step size during training.",
    "A tokenizer converts raw text into subword or word level indices that a language model can process.",
    "Byte pair encoding starts from bytes and iteratively merges frequent pairs producing a compact vocabulary.",
    "The end of text token marks the boundary between independent documents when concatenating data.",
    "Padding tokens are appended to short sequences so examples within a batch share the same length.",
    "An attention mask tells the transformer which positions should be ignored when computing softmax attention weights.",
    "Causal masking ensures a decoder only transformer can only attend to positions at or before the current token.",
    "Token embeddings and positional embeddings are added together before entering the first transformer block.",
    "The feed forward sublayer in each transformer block typically expands to four times the model dimension then projects back.",
    "GELU and SwiGLU are common activation functions used in modern transformer feed forward networks.",
    "Training throughput is often reported in tokens per second and strongly influences the cost of pre training runs.",
    "Gradient accumulation simulates larger batch sizes by delaying the optimizer step over multiple forward backward passes.",
    "Mixed precision training uses sixteen bit floating point where safe to halve memory and accelerate tensor cores.",
    "Data parallelism shards a batch across devices while model parallelism shards layers or tensors across devices.",
    "Pipeline parallelism assigns groups of layers to different devices and schedules microbatches to keep GPUs busy.",
    "LoRA freezes the base weights and injects small low rank trainable matrices inside attention and feed forward layers.",
    "Prompt tuning prepends a small learned prefix to the input sequence while keeping the base model frozen.",
    "Retrieval augmented generation augments the language model with a retriever over an external corpus.",
    "Reinforcement learning from human preference trains a reward model then fine tunes the language model with PPO.",
] * 20


def parse_args():
    parser = argparse.ArgumentParser(description="MetaCog-X 训练与推理")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "full_test"],
                        help="运行模式")
    parser.add_argument("--real_data", action="store_true",
                        help="use real WikiText data + HuggingFace tokenizer")
    parser.add_argument("--max_train_samples", type=int, default=5000,
                        help="max real data samples to keep after loading")
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training, only load model and run generation")
    parser.add_argument("--verbose", action="store_true",
                        help="verbose logging")
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")
    parser.add_argument("--d_meta", type=int, default=32, help="元认知维度")
    parser.add_argument("--d_aware", type=int, default=16, help="觉知维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--max_seq_len", type=int, default=128, help="最大序列长度")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    return parser.parse_args()


def _resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_flag


def create_test_data(tokenizer, num_samples=100, max_length=64):
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neural networks.",
        "The transformer architecture uses self-attention mechanisms.",
        "Deep learning has revolutionized computer vision and NLP.",
    ]
    return texts * (num_samples // len(texts) + 1)


def test_forward_pass(model, device, max_seq_len=32):
    print("\n=== 测试1: 前向传播 ===")
    model.eval()
    batch_size = 2
    seq_len = max_seq_len
    input_ids = torch.randint(4, model.config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=None, return_meta=True)
    print(f"  Logits形状: {outputs['logits'].shape}")
    if outputs.get("meta") is not None:
        print(f"  Meta形状: {outputs['meta'].shape}")
    if outputs.get("awareness") is not None:
        print(f"  Awareness形状: {outputs['awareness'].shape}")
    print("  [PASS] 前向传播测试通过!")


def test_generation(model, device):
    print("\n=== 测试2: 文本生成 ===")
    model.eval()
    input_ids = torch.tensor([[4, 5, 6, 7, 8]])
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_new_tokens=20, temperature=1.0, top_k=10,
        )
    print(f"  生成序列长度: {output_ids.shape[1]}")
    print("  [PASS] 生成测试通过!")


def test_awareness_pool(device):
    print("\n=== 测试3: 觉知池 ===")
    pool = AwarenessPool(capacity=10, feature_dim=16, device=device)
    for _ in range(15):
        aware = torch.randn(2, 8, 16)
        pool.update(aware)
    stats = pool.get_stats()
    print(f"  Buffer长度: {stats.buffer_len}")
    print(f"  均值形状: {stats.mean.shape}")
    print(f"  趋势形状: {stats.trend.shape}")
    pool.reset()
    stats_after = pool.get_stats()
    print(f"  重置后Buffer长度: {stats_after.buffer_len if stats_after else 0}")
    print("  [PASS] 觉知池测试通过!")


def test_meta_controller(device, d_meta=32, d_aware=16):
    print("\n=== 测试4: 元认知控制器 ===")
    controller = SparseMetaController(d_meta=d_meta, d_aware=d_aware, hidden_dim=64)
    controller.to(device)
    meta = torch.randn(2, 10, d_meta).to(device)
    from models import AwarenessStats
    aware_stats = AwarenessStats(
        mean=torch.randn(d_aware).to(device),
        std=torch.randn(d_aware).to(device),
        trend=torch.randn(d_aware).to(device),
        buffer_len=10,
    )
    with torch.no_grad():
        ctrl = controller(meta, aware_stats)
    print(f"  temp_factor形状: {ctrl.temp_factor.shape}")
    print(f"  skip_prob形状: {ctrl.skip_prob.shape}")
    print(f"  mem_strength形状: {ctrl.mem_strength.shape}")
    print("  [PASS] 元认知控制器测试通过!")


def test_enlightenment_trigger(device):
    print("\n=== 测试5: 开悟触发器 ===")
    trigger = EnlightenmentTrigger(
        entropy_thresh=2.5, repeat_thresh=3, entropy_patience=5,
    )
    logits = torch.randn(2, 10, 50257).to(device)
    tokens = torch.randint(4, 1000, (2, 10)).to(device)
    result = trigger(logits, tokens=tokens, step=0)
    print(f"  触发状态: triggered={result.triggered}, action={result.action.value}")
    print(f"  当前熵: {result.current_entropy:.4f}")
    print("  [PASS] 开悟触发器测试通过!")


def test_loss_function(device, config):
    print("\n=== 测试6: 损失函数 ===")
    loss_fn = TotalLoss(alpha=0.01, beta=0.005)
    logits = torch.randn(2, 10, config.vocab_size).to(device)
    labels = torch.randint(4, config.vocab_size, (2, 10)).to(device)
    meta_per_layer = torch.randn(
        config.num_layers, 2, 10, config.d_meta,
    ).to(device)
    aware_per_layer = torch.randn(
        config.num_layers, 2, 10, config.d_aware,
    ).to(device)
    loss, components = loss_fn(logits, labels, meta_per_layer, aware_per_layer)
    print(f"  总损失: {loss.item():.4f}")
    print(f"  CE损失: {components['loss_ce'].item():.4f}")
    print(f"  Meta损失: {components['loss_meta'].item():.4f}")
    print(f"  Aware损失: {components['loss_aware'].item():.4f}")
    print("  [PASS] 损失函数测试通过!")


def test_rl_components(device):
    print("\n=== 测试7: 强化学习组件 ===")
    reward_calc = RewardCalculator()
    reward = reward_calc.compute_reward(
        task_success=True, energy_cost=0.5, num_steps=10,
        optimal_steps=5, enlightenment_count=2, control_complexity=0.3,
    )
    print(f"  奖励计算: {reward:.4f}")
    task_gen = AdversarialTaskGenerator()
    task = task_gen.generate_task("false_premise")
    print(f"  对抗任务: {task.problem[:50]}...")
    print(f"  需要开悟: {task.requires_enlightenment}")
    print("  [PASS] 强化学习组件测试通过!")


def run_full_test(args):
    print("=" * 60)
    print("MetaCog-X 完整测试")
    print("=" * 60)
    device = _resolve_device(args.device)
    print(f"使用设备: {device}")

    config = MetaCogXConfig(
        d_model=args.d_model, d_meta=args.d_meta, d_aware=args.d_aware,
        num_layers=args.num_layers, num_heads=args.num_heads,
        max_seq_len=args.max_seq_len, d_ffn=args.d_model * 4,
    )
    print(
        f"\n模型配置: d_model={config.d_model}, d_meta={config.d_meta}, "
        f"d_aware={config.d_aware}"
    )
    print(
        f"           num_layers={config.num_layers}, num_heads={config.num_heads}"
    )
    model = MetaCogXModel(config)
    model.to(device)
    print(f"\n模型参数量: {model.get_num_params():,}")
    test_forward_pass(model, device, args.max_seq_len)
    test_generation(model, device)
    test_awareness_pool(device)
    test_meta_controller(device, config.d_meta, config.d_aware)
    test_enlightenment_trigger(device)
    test_loss_function(device, config)
    test_rl_components(device)
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


def _load_hf_tokenizer(verbose: bool):
    from transformers import AutoTokenizer
    if verbose:
        print("  [hf] AutoTokenizer.from_pretrained('gpt2', local_files_only=True)")
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def _load_wikitext_fallback(verbose: bool):
    import threading
    import time

    result = {}

    def _worker():
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            result["ok"] = ds
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=15)

    ds = result.get("ok")
    if ds is not None:
        try:
            train_texts = [
                line.strip()
                for line in ds["train"]["text"]
                if line.strip() and not line.startswith("=")
            ]
            if train_texts:
                if verbose:
                    print(f"  [ok] WikiText-2 train lines: {len(train_texts)}")
                return train_texts, False
        except Exception as e:
            if verbose:
                print(f"  [warn] iterating dataset failed ({type(e).__name__}: {e})")

    if verbose:
        print(f"  [fallback] datasets unavailable ({result.get('err', 'timeout')}); using bundled samples (len={len(FALLBACK_WIKITEXT_SAMPLES)})")
    return list(FALLBACK_WIKITEXT_SAMPLES), True


def run_real_data(args):
    from data.hf_dataset import load_wikitext_dataset, HFDataset

    device = _resolve_device(args.device)
    print(f"使用设备: {device}")

    print("Loading GPT2 tokenizer...")
    tokenizer = _load_hf_tokenizer(args.verbose)
    print(f"  vocab_size = {tokenizer.vocab_size}, pad_token = '{tokenizer.pad_token}'")

    print("Loading WikiText-2 via load_wikitext_dataset...")
    train_ds = load_wikitext_dataset(
        split="train",
        cache_dir="data",
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        max_train_samples=args.max_train_samples,
    )
    valid_ds = load_wikitext_dataset(
        split="validation",
        cache_dir="data",
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )

    config = MetaCogXConfig(
        d_model=args.d_model,
        d_meta=args.d_meta,
        d_aware=args.d_aware,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        d_ffn=args.d_model * 4,
        vocab_size=tokenizer.vocab_size,
    )
    print(f"\nModel vocab_size = {config.vocab_size}")

    train_loader = create_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True,
    )
    print(f"Train dataset: {len(train_ds)}, Valid dataset: {len(valid_ds)}, batches/epoch: {len(train_loader)}")

    model = MetaCogXModel(config, enable_metacog=True)
    model.to(device)
    print(f"Total params: {model.get_num_params():,}")

    prompt = "The meaning of life is"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)])
    print(f"\n--- Baseline generate (before training) ---")
    with torch.no_grad():
        out_baseline = model.generate(
            prompt_ids.to(device), max_new_tokens=30,
            temperature=0.7, top_k=30, verbose=args.verbose,
        )
    print(f"[baseline] {tokenizer.decode(out_baseline[0].tolist())}")

    if not args.eval_only:
        train_config = {
            "lr": args.lr,
            "weight_decay": 0.01,
            "alpha_meta": config.alpha_meta,
            "beta_aware": config.beta_aware,
        }
        trainer = Trainer(
            model=model, config=train_config,
            train_loader=train_loader, device=device,
        )
        trainer.train(num_epochs=args.epochs)
    else:
        print("  --eval_only: skipping training")

    print(f"\n--- Generate (after training) ---")
    with torch.no_grad():
        out = model.generate(
            prompt_ids.to(device), max_new_tokens=30,
            temperature=0.7, top_k=30, verbose=args.verbose,
        )
    gen_text = tokenizer.decode(out[0].tolist())
    print(f"[final] {gen_text}")

    prompt2 = "Artificial intelligence"
    prompt_ids2 = torch.tensor([tokenizer.encode(prompt2)])
    with torch.no_grad():
        out2 = model.generate(
            prompt_ids2.to(device), max_new_tokens=30,
            temperature=0.7, top_k=30, verbose=args.verbose,
        )
    print(f"[final] {tokenizer.decode(out2[0].tolist())}")


def run_legacy_train(args):
    device = _resolve_device(args.device)
    print(f"使用设备: {device}")
    config = MetaCogXConfig(
        d_model=args.d_model, d_meta=args.d_meta, d_aware=args.d_aware,
        num_layers=args.num_layers, num_heads=args.num_heads,
        max_seq_len=args.max_seq_len, d_ffn=args.d_model * 4,
    )
    model = MetaCogXModel(config)
    model.to(device)
    print("\n=== 准备训练数据 ===")
    tokenizer = DummyTokenizer(vocab_size=config.vocab_size)
    texts = create_test_data(tokenizer, num_samples=100)
    dataset = SimpleTextDataset(texts, tokenizer, max_length=args.max_seq_len)
    train_loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"训练数据: {len(dataset)} 条")
    train_config = {"lr": args.lr, "weight_decay": 0.01}
    trainer = Trainer(
        model=model, config=train_config,
        train_loader=train_loader, device=device,
    )
    trainer.train(num_epochs=args.epochs)
    print("\n训练完成!")


def run_legacy_test(args):
    device = _resolve_device(args.device)
    config = MetaCogXConfig(
        d_model=args.d_model, d_meta=args.d_meta, d_aware=args.d_aware,
        num_layers=args.num_layers, num_heads=args.num_heads,
        max_seq_len=args.max_seq_len, d_ffn=args.d_model * 4,
    )
    model = MetaCogXModel(config)
    model.to(device)
    print("=== MetaCog-X 快速测试 ===")
    test_forward_pass(model, device, args.max_seq_len)
    test_generation(model, device)
    print("\n快速测试完成!")


def main():
    args = parse_args()
    if args.real_data:
        run_real_data(args)
    elif args.mode == "full_test":
        run_full_test(args)
    elif args.mode == "train":
        run_legacy_train(args)
    else:
        run_legacy_test(args)


if __name__ == "__main__":
    main()
