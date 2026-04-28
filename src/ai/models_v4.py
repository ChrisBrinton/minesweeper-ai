"""
Residual FCN for Minesweeper — v4 Architecture

Upgraded from v3 (461K params) with:
- 96 channels (up from 64) — 2.25x more capacity per layer
- 8 residual blocks (up from 6) — deeper feature extraction
- More varied dilation schedule for larger effective receptive field (~59x59)
- Dropout in heads for regularization (overfitting was an issue in v3)
- ~1M parameters — should be trainable on MPS in ~3h/iteration

Input:  [B, 12, H, W]  — 12-channel one-hot state representation
Output: [B, H, W]      — P(mine) logit per cell (for supervised) or Q-value (for RL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention via global average pooling."""
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
    """Pre-activation residual block with optional dilation and SE attention."""
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: int = 1, use_se: bool = False):
        super().__init__()
        padding = dilation
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=padding, dilation=dilation, bias=False)
        self.se = SqueezeExcitation(out_channels) if use_se else None
        self.skip = (nn.Conv2d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.se is not None:
            out = self.se(out)
        return out + residual


class MinesweeperResNetV4(nn.Module):
    """
    Residual FCN v4 — deeper and wider than v3.

    Architecture:
        Stem:    Conv 3x3 (12 -> 96)
        Block 0: ResBlock 96, d=1           (RF: 3 -> 7)
        Block 1: ResBlock 96, d=1 + SE      (RF: 7 -> 11)
        Block 2: ResBlock 96, d=2           (RF: 11 -> 19)
        Block 3: ResBlock 96, d=2 + SE      (RF: 19 -> 27)
        Block 4: ResBlock 96, d=4           (RF: 27 -> 43)
        Block 5: ResBlock 96, d=4 + SE      (RF: 43 -> 59)
        Block 6: ResBlock 96, d=2           (RF: 59 -> 67)
        Block 7: ResBlock 96, d=1 + SE      (RF: 67 -> 71)
        Dueling heads with dropout

    Effective receptive field ~71x71 — covers expert board (16x30) with margin.
    Parameter count: ~1.0M
    """

    def __init__(self, input_channels: int = 12, hidden_channels: int = 96,
                 head_dropout: float = 0.1):
        super().__init__()

        block_configs = [
            (1, False),   # Block 0: local patterns
            (1, True),    # Block 1: local + attention
            (2, False),   # Block 2: medium-range
            (2, True),    # Block 3: medium-range + attention
            (4, False),   # Block 4: long-range
            (4, True),    # Block 5: long-range + attention
            (2, False),   # Block 6: refine medium
            (1, True),    # Block 7: refine local + attention
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, d, se)
            for d, se in block_configs
        ])

        self.final_bn = nn.BatchNorm2d(hidden_channels)

        # Dueling heads with dropout for regularization
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 48, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(48, 1, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 48, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_dropout),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.final_bn(x))
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        q = value + advantage - advantage.mean(dim=(2, 3), keepdim=True)
        return q.squeeze(1)
