"""数据加载器"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class SimpleTextDataset(Dataset):
    """简单文本数据集（用于测试）"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False
) -> DataLoader:
    """创建DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available()
    )


class DummyTokenizer:
    """虚拟tokenizer（用于快速测试）"""

    def __init__(self, vocab_size: int = 50257, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0

    def __call__(self, text: str, max_length: int = 512, padding: str = 'max_length',
                 truncation: bool = True, return_tensors: str = 'pt'):
        # 简单模拟：将文本转为随机token
        import random
        length = min(len(text.split()), max_length) if truncation else max_length
        input_ids = torch.randint(4, self.vocab_size, (length,))  # 从4开始避免特殊token
        attention_mask = torch.ones(length, dtype=torch.long)

        # padding
        if length < max_length:
            pad_length = max_length - length
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])

        return {'input_ids': input_ids.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
