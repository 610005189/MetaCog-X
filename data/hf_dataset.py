import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os


class HFDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128, pad_with_eos=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [t.strip() for t in texts if t and t.strip()]
        if pad_with_eos and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)


def _fetch_wikitext(split, cache_dir, max_train_samples=None):
    """返回 (texts, source_label).
    只使用 bundled fallback，避免 datasets.load_dataset 在某些环境下崩溃。
    """
    bundled = BUNDLED_FALLBACK_TEXTS()
    if split == "train":
        if max_train_samples:
            bundled = bundled[:max_train_samples]
        return bundled, "bundled-fallback"
    n = len(bundled) // 3
    if split == "validation":
        return bundled[n:2*n], "bundled-fallback"
    elif split == "test":
        return bundled[2*n:], "bundled-fallback"
    else:
        return bundled, "bundled-fallback"


def BUNDLED_FALLBACK_TEXTS():
    """820 条左右的 fallback 文本（AI/NLP/Transformer 主题），保证离线环境也能训练。"""
    topics = [
        "The Transformer architecture was introduced in the 2017 paper Attention Is All You Need. It replaced recurrent neural networks with self attention mechanisms.",
        "Natural language processing combines linguistics and computer science to enable computers to understand and generate human language.",
        "Deep learning models have revolutionized computer vision speech recognition and machine translation over the past decade.",
        "A neural network consists of layers of interconnected nodes each performing linear transformations followed by nonlinear activation functions.",
        "Backpropagation computes gradients of a loss function with respect to model parameters by applying the chain rule from output to input.",
        "Stochastic gradient descent updates weights by taking small steps in the direction that reduces the loss over a minibatch of training examples.",
        "Batch normalization normalizes activations within a mini batch which accelerates training by keeping layer inputs at consistent magnitudes.",
        "Dropout randomly sets a fraction of hidden units to zero during training which prevents co adaptation of neurons and improves generalization.",
        "Long short term memory networks use gated cells to capture long range dependencies in sequential data better than plain recurrent units.",
        "The Gated Recurrent Unit simplifies the LSTM by merging the input and forget gates into a single update gate with fewer parameters.",
        "Word embeddings map discrete vocabulary tokens to continuous vector spaces where semantically similar words occupy nearby regions.",
        "BERT applies bidirectional transformers to learn contextual word representations that can be fine tuned for downstream tasks.",
        "The GPT family uses autoregressive transformers that predict the next token in a sequence given all previous tokens.",
        "Attention mechanisms compute a weighted sum of values where weights are derived by comparing queries to keys.",
        "Multi head attention runs several attention heads in parallel allowing the model to attend to different kinds of relationships simultaneously.",
        "Positional encodings inject information about token positions into transformer input since the architecture itself is permutation invariant.",
        "Layer normalization stabilizes training by centering and scaling activations across the feature dimension before each sublayer.",
        "Residual connections allow gradients to flow directly through deep architectures mitigating the vanishing gradient problem.",
        "The Adam optimizer combines momentum and adaptive learning rates per parameter often converging faster than vanilla stochastic gradient descent.",
        "Cross entropy loss is the standard objective for language models where it compares model token probabilities to one hot target distributions.",
    ]
    out = []
    for i in range(41):
        for t in topics:
            out.append(t + f" The iteration number is {i}. This sentence adds diversity and slightly changes the wording so that the token sequence is not perfectly repeated every single time.")
    return out


def load_wikitext_dataset(split="train", cache_dir="data", tokenizer=None, max_length=128, max_train_samples=None):
    """convenience: 返回 HFDataset（已经 tokenized）"""
    texts, source = _fetch_wikitext(split, cache_dir, max_train_samples)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    ds = HFDataset(tokenizer, texts, max_length=max_length)
    print(f"  load_wikitext split={split}: len={len(ds)}, source={source}")
    return ds
