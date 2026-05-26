"""
Evaluation script for GeoAI segmentation models.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_mplnet, get_mcpnet
from datasets import get_dataloader, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoAI segmentation model")
    parser.add_argument("--model", type=str, default="mplnet", choices=["mplnet", "mcpnet"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data/TV-RSI-413")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=22)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./results")
    return parser.parse_args()


class SegmentationEvaluator:
    """Evaluator for segmentation models."""

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader,
        device: torch.device,
        num_classes: int,
        save_dir: str = None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def _fast_hist(self, pred: np.ndarray, target: np.ndarray, num_classes: int):
        """Compute confusion matrix for one image."""
        mask = (target >= 0) & (target < num_classes)
        hist = np.bincount(
            num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        return hist

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.model.eval()

        for batch in tqdm(self.data_loader, desc="Evaluating"):
            images = batch["image"].to(self.device)
            masks = batch["mask"].numpy()
            names = batch["name"]

            outputs = self.model(images)

            if isinstance(outputs, tuple):
                seg_logits, _ = outputs
            else:
                seg_logits = outputs

            preds = seg_logits.argmax(dim=1).cpu().numpy()

            for i, (pred, mask, name) in enumerate(zip(preds, masks, names)):
                hist = self._fast_hist(pred, mask, self.num_classes)
                self.confusion_matrix += hist

                if self.save_dir:
                    self._save_prediction(pred, name)

            self.total_samples += len(preds)

        metrics = self._compute_metrics()
        return metrics

    def _save_prediction(self, pred: np.ndarray, name: str):
        """Save prediction as image."""
        import cv2
        pred_image = (pred * 10).astype(np.uint8)
        cv2.imwrite(str(self.save_dir / f"{name}_pred.png"), pred_image)

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics from confusion matrix."""
        hist = self.confusion_matrix

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

        pixel_acc = np.diag(hist).sum() / hist.sum()

        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        mean_f1 = np.nanmean(f1)

        total = hist.sum()
        po = np.diag(hist).sum() / total
        pe = (hist.sum(axis=1) @ hist.sum(axis=0)) / (total * total)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0

        metrics = {
            "mIoU": mean_iu,
            "pixel_acc": pixel_acc,
            "mF1": mean_f1,
            "kappa": kappa,
        }

        return metrics

    def print_report(self, class_names: List[str] = None):
        """Print detailed evaluation report."""
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        print("\n" + "=" * 80)
        print("Evaluation Report")
        print("=" * 80)

        print(f"\n{'Class':<25} {'IoU':>10} {'PA':>10} {'Precision':>12} {'Recall':>10}")
        print("-" * 70)

        for i, (name, iou_val) in enumerate(zip(class_names, iu)):
            if not np.isnan(iou_val):
                tp = hist[i, i]
                fp = hist[:, i].sum() - tp
                fn = hist[i, :].sum() - tp
                pa = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0

                print(f"{name:<25} {iou_val:>10.4f} {pa:>10.4f} {prec:>12.4f} {rec:>10.4f}")

        print("-" * 70)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "mplnet":
        model = get_mplnet(num_classes=args.num_classes, pretrained=False)
    else:
        model = get_mcpnet(num_classes=args.num_classes, pretrained=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint from {args.checkpoint}")

    val_transforms = get_val_transforms(img_size=(args.img_size, args.img_size))

    data_loader = get_dataloader(
        root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        transforms=val_transforms,
        shuffle=False,
    )

    evaluator = SegmentationEvaluator(
        model=model,
        data_loader=data_loader,
        device=device,
        num_classes=args.num_classes,
        save_dir=args.save_dir if args.save_predictions else None,
    )

    metrics = evaluator.evaluate()

    print("\n" + "=" * 50)
    print("Final Metrics")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name:>15}: {value:.4f}")

    evaluator.print_report()

    if args.save_predictions:
        print(f"\nPredictions saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
