"""训练循环"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import math

from training import TotalLoss


class Trainer:
    """MetaCog-X 训练器"""

    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.device = device
        self.model.to(device)

        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.get("lr", 1e-4),
                weight_decay=config.get("weight_decay", 0.01)
            )

        self.scheduler = None

        self.global_step = 0
        self.current_epoch = 0

        alpha = config.get("alpha_meta", 0.01)
        beta = config.get("beta_aware", 0.005)
        self.total_loss_fn = TotalLoss(alpha=alpha, beta=beta)

    def train_step(self, batch) -> Dict[str, float]:
        """单步训练（启用辅助损失）"""
        self.model.train()
        input_ids, attention_mask = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_meta=True,
            enable_metacog=True,
        )
        logits = outputs["logits"]
        meta_per_layer = outputs.get("meta")
        aware_per_layer = outputs.get("awareness")

        loss, loss_components = self.total_loss_fn(
            logits,
            input_ids,
            meta_per_layer,
            aware_per_layer,
            aware_pool_buffer=None,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.global_step += 1

        metrics = {"loss": loss.item()}
        metrics.update({
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in loss_components.items()
        })
        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        assert self.train_loader is not None, "train_loader is required for train_epoch()"
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_meta = 0.0
        total_aware = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch)
            total_loss += metrics.get("loss", 0.0)
            total_ce += metrics.get("loss_ce", 0.0)
            total_meta += metrics.get("loss_meta", 0.0)
            total_aware += metrics.get("loss_aware", 0.0)

            if batch_idx % 10 == 0:
                print(
                    f"  Step {batch_idx}/{num_batches} | "
                    f"total={metrics.get('loss', 0.0):.4f} "
                    f"ce={metrics.get('loss_ce', 0.0):.4f} "
                    f"meta={metrics.get('loss_meta', 0.0):.6f} "
                    f"aware={metrics.get('loss_aware', 0.0):.6f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        avg_ce = total_ce / max(num_batches, 1)
        avg_meta = total_meta / max(num_batches, 1)
        avg_aware = total_aware / max(num_batches, 1)

        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "loss_ce": avg_ce,
            "loss_meta": avg_meta,
            "loss_aware": avg_aware,
        }

    def train(
        self,
        num_epochs: int,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print(f"总参数量: {self.model.get_num_params():,}")
        print(f"可训练参数量: {self.model.get_trainable_params():,}")

        history = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            metrics = self.train_epoch()
            history.append(metrics)

            print(
                f"  Avg: loss={metrics['loss']:.4f} perplex={metrics['perplexity']:.2f} "
                f"ce={metrics.get('loss_ce', float('nan')):.4f} "
                f"meta={metrics.get('loss_meta', float('nan')):.6f} "
                f"aware={metrics.get('loss_aware', float('nan')):.6f}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

        return {"history": history}
