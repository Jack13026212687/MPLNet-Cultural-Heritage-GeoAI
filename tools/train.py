"""
Training script for MPLNet and MCPNet.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_mplnet, get_mcpnet
from datasets import get_dataloader, get_train_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoAI segmentation model")
    parser.add_argument("--model", type=str, default="mplnet", choices=["mplnet", "mcpnet"])
    parser.add_argument("--data_root", type=str, default="./data/TV-RSI-413")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=22)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


class SegmentationTrainer:
    """Trainer for segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_dir: str,
        log_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.best_miou = 0.0
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            if isinstance(outputs, tuple):
                seg_logits, edge_logits = outputs
                edges = self._compute_edges(masks)
                loss = self.criterion(seg_logits, masks, edge_logits, edges)
            else:
                loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}")

        return {"train_loss": total_loss / num_batches}

    def _compute_edges(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute edge map from masks."""
        edges = torch.zeros_like(masks, dtype=torch.float)

        edges[:, 1:, :] += (masks[:, 1:, :] != masks[:, :-1, :]).float()
        edges[:, :-1, :] += (masks[:, :-1, :] != masks[:, 1:, :]).float()
        edges[:, :, 1:] += (masks[:, :, 1:] != masks[:, :, :-1]).float()
        edges[:, :, :-1] += (masks[:, :, :-1] != masks[:, :, 1:]).float()

        edges = (edges > 0).float()
        return edges

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        total_mpa = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            outputs = self.model(images)

            if isinstance(outputs, tuple):
                seg_logits, _ = outputs
            else:
                seg_logits = outputs

            loss = self.criterion(seg_logits, masks)
            total_loss += loss.item()

            preds = seg_logits.argmax(dim=1)

            for pred, target in zip(preds, masks):
                miou, mpa = self._compute_metrics(pred, target)
                total_miou += miou
                total_mpa += mpa

        metrics = {
            "val_loss": total_loss / num_batches,
            "val_miou": total_miou / len(self.val_loader.dataset),
            "val_mpa": total_mpa / len(self.val_loader.dataset),
        }

        return metrics

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute IoU and pixel accuracy."""
        mask = (target >= 0)
        correct = (pred == target).sum().item()
        total = mask.sum().item()
        mpa = correct / total if total > 0 else 0

        classes = torch.unique(target[mask])
        ious = []
        for cls in classes:
            pred_cls = (pred == cls) & mask
            target_cls = (target == cls) & mask
            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()
            if union > 0:
                ious.append(intersection / union)

        miou = sum(ious) / len(ious) if ious else 0
        return miou, mpa

    def train(self, epochs: int):
        """Full training loop."""
        for epoch in range(epochs):
            self.epoch = epoch
            print(f"\nEpoch [{epoch + 1}/{epochs}]")

            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()

            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"mIoU: {val_metrics['val_miou']:.4f}, "
                  f"mPA: {val_metrics['val_mpa']:.4f}")

            self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Metrics/mIoU", val_metrics["val_miou"], epoch)
            self.writer.add_scalar("Metrics/mPA", val_metrics["val_mpa"], epoch)

            if val_metrics["val_miou"] > self.best_miou:
                self.best_miou = val_metrics["val_miou"]
                self.save_checkpoint("best_model.pth")
                print(f"Saved best model with mIoU: {self.best_miou:.4f}")

        self.writer.close()
        print(f"\nTraining complete. Best mIoU: {self.best_miou:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_miou": self.best_miou,
        }
        torch.save(checkpoint, self.save_dir / filename)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "mplnet":
        model = get_mplnet(num_classes=args.num_classes, pretrained=True)
    else:
        model = get_mcpnet(num_classes=args.num_classes, pretrained=True)

    model = model.to(device)

    train_transforms = get_train_transforms(img_size=(args.img_size, args.img_size))
    val_transforms = get_val_transforms(img_size=(args.img_size, args.img_size))

    train_loader = get_dataloader(
        root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        transforms=train_transforms,
        shuffle=True,
    )

    val_loader = get_dataloader(
        root=args.data_root,
        split="val",
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        transforms=val_transforms,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.epoch = checkpoint["epoch"]
        trainer.best_miou = checkpoint["best_miou"]
        print(f"Resumed from epoch {trainer.epoch}")

    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
