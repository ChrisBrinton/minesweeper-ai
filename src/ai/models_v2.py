"""
Fully Convolutional DQN for Minesweeper — Phase 2 Architecture

Works for ANY board size with identical weights (no FC layers).
Dueling architecture with per-cell value and advantage heads.
~50K parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinesweeperFCN(nn.Module):
    """
    Fully convolutional DQN — works for any board size, ~50K parameters.
    
    Architecture:
        4 conv layers (3×3, padding=1) with ReLU
        Dueling output via 1×1 convolutions (value + advantage per cell)
    
    Input:  [B, 12, H, W]  — 12-channel one-hot state representation
    Output: [B, H, W]      — one Q-value per cell (reveal action only)
    """
    
    def __init__(self, input_channels: int = 12):
        super().__init__()
        
        # Shared convolutional backbone
        # Keep channels small to stay near ~50K params
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 32, kernel_size=3, padding=1)
        
        # Dueling heads (1×1 convolutions — no FC layers)
        # Value stream: V(s) per cell
        self.value_conv = nn.Conv2d(32, 1, kernel_size=1)
        # Advantage stream: A(s, a) per cell
        self.advantage_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W] input tensor
            
        Returns:
            [B, H, W] Q-values — one per cell
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        value = self.value_conv(x)          # [B, 1, H, W]
        advantage = self.advantage_conv(x)  # [B, 1, H, W]
        
        # Dueling: Q = V + (A - mean(A))
        # Mean over spatial dims (all cells) for advantage centering
        q = value + advantage - advantage.mean(dim=(2, 3), keepdim=True)
        
        return q.squeeze(1)  # [B, H, W]
