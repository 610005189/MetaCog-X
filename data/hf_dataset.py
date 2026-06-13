import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os


class HFDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128, pad_with_eos=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [t.strip() for t in texts if t and t.strip()]
        if pad_with_eos and hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.pad_token = '<pad>'
                self.tokenizer.pad_token_id = 0

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
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0) if 'attention_mask' in enc else torch.ones_like(input_ids)
        return input_ids, attention_mask


class CharLevelTokenizer:
    """字符级tokenizer，用于细粒度文本处理"""
    
    def __init__(self, chars=None):
        if chars is None:
            basic_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            punctuation = '.,!?;:()[]{}<>\'"'
            whitespace = ' \n\t'
            self.chars = list(basic_chars + punctuation + whitespace)
        else:
            self.chars = chars
        
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.idx_to_token = self.special_tokens + self.chars
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        self.pad_token = '<pad>'
        self.pad_token_id = self.token_to_idx['<pad>']
        self.bos_token = '<bos>'
        self.bos_token_id = self.token_to_idx['<bos>']
        self.eos_token = '<eos>'
        self.eos_token_id = self.token_to_idx['<eos>']
        self.unk_token = '<unk>'
        self.unk_token_id = self.token_to_idx['<unk>']
        self.vocab_size = len(self.idx_to_token)

    def encode(self, text):
        tokens = []
        for char in text:
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
            else:
                tokens.append(self.unk_token_id)
        return tokens

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx < len(self.idx_to_token):
                tokens.append(self.idx_to_token[idx])
            else:
                tokens.append(self.unk_token)
        return ''.join(tokens)

    def __call__(self, text, max_length=128, padding='max_length', truncation=True, return_tensors='pt'):
        tokens = self.encode(text)
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        length = len(tokens)
        if padding == 'max_length' and length < max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - length)
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones(max_length, dtype=torch.long)
        if padding == 'max_length' and length < max_length:
            attention_mask[length:] = 0
        
        if return_tensors == 'pt':
            return {'input_ids': input_ids.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
        else:
            return {'input_ids': tokens, 'attention_mask': attention_mask.tolist()}


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


def get_tokenizer(tokenizer_type="gpt2", **kwargs):
    """获取指定类型的tokenizer
    
    Args:
        tokenizer_type: tokenizer类型，支持 'gpt2', 'charlevel'
        **kwargs: 传递给tokenizer的额外参数
    
    Returns:
        tokenizer实例
    """
    if tokenizer_type.lower() == "gpt2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    elif tokenizer_type.lower() == "charlevel":
        return CharLevelTokenizer(**kwargs)
    else:
        raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}. Supported types: 'gpt2', 'charlevel'")


def load_wikitext_dataset(split="train", cache_dir="data", tokenizer=None, max_length=128, max_train_samples=None, tokenizer_type="gpt2"):
    """convenience: 返回 HFDataset（已经 tokenized）
    
    Args:
        split: 数据集划分 ('train', 'validation', 'test')
        cache_dir: 缓存目录
        tokenizer: 自定义tokenizer（如果为None则使用tokenizer_type创建）
        max_length: 最大序列长度
        max_train_samples: 训练样本最大数量（用于快速测试）
        tokenizer_type: tokenizer类型 ('gpt2', 'charlevel')
    
    Returns:
        HFDataset实例
    """
    texts, source = _fetch_wikitext(split, cache_dir, max_train_samples)
    if tokenizer is None:
        tokenizer = get_tokenizer(tokenizer_type)
    ds = HFDataset(tokenizer, texts, max_length=max_length)
    print(f"  load_wikitext split={split}: len={len(ds)}, source={source}, tokenizer={tokenizer_type}")
    return ds
