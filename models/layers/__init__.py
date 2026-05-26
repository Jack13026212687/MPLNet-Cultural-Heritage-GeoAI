"""
Custom layers for GeoAI segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware segmentation loss combining CE with edge predictions.
    """

    def __init__(
        self,
        num_classes: int,
        edge_weight: float = 0.3,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        edge_logits: torch.Tensor = None,
        edge_targets: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] segmentation logits
            targets: [B, H, W] class labels
            edge_logits: [B, 1, H, W] edge prediction logits (optional)
            edge_targets: [B, H, W] edge labels (optional)
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
        )

        total_loss = ce_loss

        if edge_logits is not None and edge_targets is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits.squeeze(1).float(),
                edge_targets.float(),
            )
            total_loss = (1 - self.edge_weight) * ce_loss + self.edge_weight * edge_loss

        return total_loss


class BoundaryRefinement(nn.Module):
    """Post-processing module for boundary refinement."""

    def __init__(self, channels: int, iterations: int = 2):
        super().__init__()
        self.iterations = iterations

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature map
            boundaries: [B, 1, H, W] edge probability map
        """
        refined = features.clone()

        for _ in range(self.iterations):
            boundary_weight = 1 - boundaries
            refined = self.conv(refined * boundary_weight + features)

        return refined


__all__ = ["EdgeAwareLoss", "BoundaryRefinement"]
