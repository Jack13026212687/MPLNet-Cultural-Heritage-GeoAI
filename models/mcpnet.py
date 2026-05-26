"""
MCPNet: Multi-scale Context-Perceptual Network

ResNet-50 backbone with MCP Head for efficient semantic segmentation.
Reference: High efficiency architecture for cultural heritage remote sensing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class MCPModule(nn.Module):
    """Multi-scale Context-Perceptual Module."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention()

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weight = self.channel_attn(x)
        x_weighted = x * channel_weight

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        multi_scale = b1 + b2 + b3

        out = self.conv_out(x_weighted + multi_scale)
        return out


class MCPHead(nn.Module):
    """MCP Head for semantic segmentation."""

    def __init__(self, encoder_channels: List[int], num_classes: int = 22):
        super().__init__()

        self.mcp_modules = nn.ModuleList([
            MCPModule(ch, ch // 4) for ch in encoder_channels
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(sum(ch // 4 for ch in encoder_channels), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        mcp_outputs = []
        for feat, mcp in zip(features, self.mcp_modules):
            mcp_outputs.append(mcp(feat))

        fused = torch.cat(mcp_outputs, dim=1)
        return self.fusion(fused)


class MCPNet(nn.Module):
    """
    MCPNet: Multi-scale Context-Perceptual Network

    Efficient architecture with ResNet-50 backbone and MCP Head.
    Parameters: 55.3M, GFLOPs: 198.5
    """

    def __init__(
        self,
        num_classes: int = 22,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        encoder_channels = [256, 512, 1024, 2048]
        self.mcp_head = MCPHead(encoder_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        logits = self.mcp_head([f1, f2, f3, f4])

        return logits


def get_mcpnet(num_classes: int = 22, **kwargs) -> MCPNet:
    """Factory function to create MCPNet model."""
    return MCPNet(num_classes=num_classes, **kwargs)
