"""
Residual FCN DQN for Minesweeper — v3 Architecture

Key improvements over v2 (MinesweeperFCN, ~52K params):
- Pre-activation residual blocks for stable deep training
- Dilated convolutions in middle blocks for large receptive field (~39x39)
- ~300K parameters — enough capacity for expert boards, trainable on MPS
- Squeeze-and-excitation channel attention for global board reasoning
- Fully convolutional — works for any board size with identical weights
- Dueling DQN with separate value/advantage heads

Input:  [B, 12, H, W]  — 12-channel one-hot state representation
Output: [B, H, W]      — one Q-value per cell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention: lets the network weight which feature channels
    matter based on global board context (e.g., fraction of cells revealed)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip.

    Supports dilation for expanding the receptive field without adding parameters.
    Optional squeeze-and-excitation attention.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dilation: int = 1, use_se: bool = False):
        super().__init__()
        padding = dilation  # maintains spatial dims with 3x3

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=padding, dilation=dilation, bias=False)
        self.se = SqueezeExcitation(out_channels) if use_se else None

        # 1x1 projection for channel mismatch
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.se is not None:
            out = self.se(out)
        return out + residual


class MinesweeperResNet(nn.Module):
    """
    Residual FCN DQN — works for any board size, ~300K parameters.

    Architecture:
        Stem:    Conv 3x3 (12 -> 64)
        Block 0: ResBlock 64 -> 64, d=1         (RF grows: 3 -> 7)
        Block 1: ResBlock 64 -> 64, d=1 + SE    (RF: 7 -> 11)
        Block 2: ResBlock 64 -> 64, d=2         (RF: 11 -> 19)
        Block 3: ResBlock 64 -> 64, d=2 + SE    (RF: 19 -> 27)
        Block 4: ResBlock 64 -> 64, d=4         (RF: 27 -> 43)
        Block 5: ResBlock 64 -> 64, d=1 + SE    (RF: 43 -> 47)
        Dueling heads via 1x1 convolutions

    Effective receptive field ~47x47 covers the entire expert board (16x30).

    Parameter count: ~305K (64ch, 6 blocks)
    """

    def __init__(self, input_channels: int = 12, hidden_channels: int = 64,
                 num_blocks: int = 6):
        super().__init__()

        # Block configs: (dilation, use_se)
        block_configs = [
            (1, False),   # Block 0: local patterns
            (1, True),    # Block 1: local + attention
            (2, False),   # Block 2: medium-range
            (2, True),    # Block 3: medium-range + attention
            (4, False),   # Block 4: long-range (key for big boards)
            (1, True),    # Block 5: refine + attention
        ]
        # Extend or truncate to match num_blocks
        block_configs = block_configs[:num_blocks]

        # Stem: project input channels to hidden
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.blocks = nn.ModuleList()
        for dilation, use_se in block_configs:
            self.blocks.append(
                ResidualBlock(hidden_channels, hidden_channels,
                              dilation=dilation, use_se=use_se)
            )

        # Final BN + ReLU before heads (needed for pre-activation residual)
        self.final_bn = nn.BatchNorm2d(hidden_channels)

        # Dueling heads (fully convolutional)
        # Value stream: V(s) per cell
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        # Advantage stream: A(s, a) per cell
        self.advantage_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, H, W] Q-values — one per cell
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = F.relu(self.final_bn(x))

        value = self.value_head(x)          # [B, 1, H, W]
        advantage = self.advantage_head(x)  # [B, 1, H, W]

        # Dueling: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=(2, 3), keepdim=True)
        return q.squeeze(1)  # [B, H, W]
