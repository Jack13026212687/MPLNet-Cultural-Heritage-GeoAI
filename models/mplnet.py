"""
MPLNet: Mamba-driven Prompt Learning Network for Cultural Heritage Semantic Segmentation

This module implements the MPLNet architecture combining:
- Mamba state-space model for long-range dependency modeling
- Prompt learning for domain adaptation
- Multi-scale context perception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MambaBlock(nn.Module):
    """Mamba state-space model block for efficient long-range dependency."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.A_log, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, D, H, W] or [B, C, N]
        """
        original_shape = x.shape
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        else:
            B, N, C = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        ssm_params = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = ssm_params.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())

        y = self.selective_scan(x_conv, dt, A, B_ssm, C_ssm, self.D.float())

        y = y * F.silu(z)
        y = self.dropout(self.out_proj(y))

        if len(original_shape) == 4:
            y = y.permute(0, 2, 1).reshape(B, -1, H, W)

        return y

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified selective scan (SSM) operation."""
        B, L, D_inner = u.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        h = torch.zeros(B, D_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = torch.einsum("bds,bs->bd", h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D

        return y


class PromptGenerator(nn.Module):
    """Learnable prompt embeddings for domain adaptation."""

    def __init__(self, embed_dim: int, num_prompts: int = 16, prompt_dim: int = 64):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim

        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, num_prompts, prompt_dim) * 0.02
        )

        self.prompt_proj = nn.Linear(prompt_dim, embed_dim)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate prompt embeddings.
        Returns:
            prompts: [B, num_prompts, embed_dim]
        """
        prompts = self.prompt_proj(self.prompt_embeddings)
        prompts = prompts.expand(batch_size, -1, -1)
        return prompts


class MCPHead(nn.Module):
    """Multi-scale Context Perception Head for boundary-aware segmentation."""

    def __init__(self, in_channels: List[int], out_channels: int = 22):
        super().__init__()
        self.branches = nn.ModuleList()

        for channels in in_channels:
            branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, out_channels, kernel_size=1),
            )
            self.branches.append(branch)

        self.fusion = nn.Sequential(
            nn.Conv2d(len(in_channels) * out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(
        self, features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            features: List of feature maps from encoder at different scales
        Returns:
            seg_logits: Segmentation logits [B, num_classes, H, W]
        """
        branch_outputs = []
        for feat, branch in zip(features, self.branches):
            branch_outputs.append(branch(feat))

        concat = torch.cat(branch_outputs, dim=1)
        output = self.fusion(concat)

        return output


class MPLNet(nn.Module):
    """
    MPLNet: Mamba-driven Prompt Learning Network

    Architecture:
    1. Encoder (ResNet-50) for feature extraction
    2. Mamba blocks for long-range dependency modeling
    3. Prompt generator for domain adaptation
    4. MCP head for boundary-aware segmentation
    """

    def __init__(
        self,
        num_classes: int = 22,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_mamba: bool = True,
        use_prompts: bool = True,
        num_prompts: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_mamba = use_mamba
        self.use_prompts = use_prompts

        self.backbone_name = backbone

        self.encoder = self._build_encoder(backbone, pretrained)

        encoder_channels = self._get_encoder_channels()

        if use_mamba:
            self.mamba_block = MambaBlock(
                d_model=encoder_channels[-1],
                d_state=16,
                expand=2,
            )

        if use_prompts:
            self.prompt_gen = PromptGenerator(
                embed_dim=encoder_channels[-1],
                num_prompts=num_prompts,
            )

        self.mcp_head = MCPHead(encoder_channels, num_classes)

        self.edge_branch = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _build_encoder(self, backbone: str, pretrained: bool):
        """Build encoder backbone."""
        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights

            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)

            return nn.ModuleDict(
                {
                    "conv1": model.conv1,
                    "bn1": model.bn1,
                    "relu": model.relu,
                    "maxpool": model.maxpool,
                    "layer1": model.layer1,
                    "layer2": model.layer2,
                    "layer3": model.layer3,
                    "layer4": model.layer4,
                }
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def _get_encoder_channels(self) -> List[int]:
        """Get output channels for each encoder stage."""
        if self.backbone_name == "resnet50":
            return [256, 512, 1024, 2048]
        return [256, 512, 1024, 2048]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]
        Returns:
            seg_logits: Segmentation logits [B, num_classes, H/4, W/4]
            edge_logits: Edge prediction logits [B, 1, H/4, W/4]
        """
        B = x.shape[0]

        feat1 = self.encoder["layer1"](self.encoder["maxpool"](
            self.encoder["relu"](self.encoder["bn1"](self.encoder["conv1"](x)))
        ))
        feat2 = self.encoder["layer2"](feat1)
        feat3 = self.encoder["layer3"](feat2)
        feat4 = self.encoder["layer4"](feat3)

        if self.use_mamba:
            feat4 = feat4 + self.mamba_block(feat4)

        if self.use_prompts:
            prompts = self.prompt_gen(B)
            feat4 = feat4 + prompts.mean(dim=1, keepdim=True).permute(0, 2, 1)

        edge_logits = self.edge_branch(feat1)

        seg_logits = self.mcp_head([feat1, feat2, feat3, feat4])

        return seg_logits, edge_logits


def get_mplnet(num_classes: int = 22, **kwargs) -> MPLNet:
    """Factory function to create MPLNet model."""
    return MPLNet(num_classes=num_classes, **kwargs)
