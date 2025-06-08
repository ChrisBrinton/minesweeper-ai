"""
Neural Network Models for Minesweeper AI
Includes CNN and DQN architectures optimized for Minesweeper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MinesweeperNet(nn.Module):
    """
    Convolutional Neural Network for Minesweeper board analysis
    """
    
    def __init__(self, board_height: int, board_width: int, input_channels: int = 3):
        """
        Initialize the network
        
        Args:
            board_height: Height of the minesweeper board
            board_width: Width of the minesweeper board  
            input_channels: Number of input channels (default: 3)
        """
        super(MinesweeperNet, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size of flattened features
        self.feature_size = self._calculate_feature_size()
          # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
    
    def _calculate_feature_size(self) -> int:
        """Calculate the size of features after conv layers"""
        # Create dummy input to calculate size
        dummy_input = torch.zeros(1, self.input_channels, self.board_height, self.board_width)
        with torch.no_grad():
            dummy_output = self._forward_conv(dummy_input)
            return dummy_output.numel()
    
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.reshape(x.size(0), -1)  # Flatten
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, 256]
        """
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class DQN(nn.Module):
    """
    Deep Q-Network for Minesweeper
    Uses CNN backbone to process board state and outputs Q-values for actions
    """
    
    def __init__(self, board_height: int, board_width: int, 
                 input_channels: int = 3, num_actions: int = None):
        """
        Initialize the DQN
        
        Args:
            board_height: Height of the minesweeper board
            board_width: Width of the minesweeper board
            input_channels: Number of input channels (default: 3)
            num_actions: Number of possible actions (default: board_size * 3)
        """
        super(DQN, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        self.num_actions = num_actions or (board_height * board_width * 3)
        
        # CNN backbone
        self.backbone = MinesweeperNet(board_height, board_width, input_channels)
        
        # Q-value head
        self.q_head = nn.Linear(256, self.num_actions)
        
        # Value head (for Dueling DQN)
        self.value_head = nn.Linear(256, 1)
        
        # Advantage head (for Dueling DQN)  
        self.advantage_head = nn.Linear(256, self.num_actions)
        
        self.use_dueling = True  # Enable dueling architecture
        
    def forward(self, x: torch.Tensor, use_dueling: bool = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            use_dueling: Whether to use dueling architecture
            
        Returns:
            Q-values [batch_size, num_actions]
        """
        features = self.backbone(x)
        
        if use_dueling or (use_dueling is None and self.use_dueling):
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            
            # Subtract mean advantage for identifiability
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard DQN
            q_values = self.q_head(features)
            
        return q_values
    
    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
                   epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: Current state tensor
            action_mask: Mask of valid actions
            epsilon: Exploration probability
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Random action
            if action_mask is not None:
                valid_actions = torch.nonzero(action_mask).squeeze(-1)
                if len(valid_actions) > 0:
                    return valid_actions[np.random.randint(len(valid_actions))].item()
            return np.random.randint(self.num_actions)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                
                if action_mask is not None:
                    # Mask invalid actions
                    q_values = q_values.clone()
                    q_values[0, ~action_mask] = float('-inf')
                
                return q_values.argmax().item()


class PolicyGradientNet(nn.Module):
    """
    Policy Gradient Network for Minesweeper
    Outputs action probabilities instead of Q-values
    """
    
    def __init__(self, board_height: int, board_width: int, 
                 input_channels: int = 3, num_actions: int = None):
        """
        Initialize the Policy Network
        
        Args:
            board_height: Height of the minesweeper board
            board_width: Width of the minesweeper board
            input_channels: Number of input channels (default: 3)
            num_actions: Number of possible actions (default: board_size * 3)
        """
        super(PolicyGradientNet, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        self.num_actions = num_actions or (board_height * board_width * 3)
        
        # CNN backbone
        self.backbone = MinesweeperNet(board_height, board_width, input_channels)
        
        # Policy head
        self.policy_head = nn.Linear(256, self.num_actions)
        
        # Value head (for Actor-Critic)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            (action_probabilities, state_values)
        """
        features = self.backbone(x)
        
        # Policy probabilities
        logits = self.policy_head(features)
        action_probs = F.softmax(logits, dim=-1)
        
        # State value
        state_value = self.value_head(features)
        
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: Current state tensor
            action_mask: Mask of valid actions
            
        Returns:
            (selected_action, log_probability)
        """
        action_probs, _ = self.forward(state.unsqueeze(0))
        
        if action_mask is not None:
            # Mask invalid actions
            action_probs = action_probs.clone()
            action_probs[0, ~action_mask] = 0.0
            # Renormalize
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        # Sample action
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob


def create_model(model_type: str, board_height: int, board_width: int, 
                input_channels: int = 3, **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('dqn', 'policy', 'backbone')
        board_height: Height of the board
        board_width: Width of the board
        input_channels: Number of input channels
        **kwargs: Additional arguments
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'dqn':
        return DQN(board_height, board_width, input_channels, **kwargs)
    elif model_type.lower() == 'policy':
        return PolicyGradientNet(board_height, board_width, input_channels, **kwargs)
    elif model_type.lower() == 'backbone':
        return MinesweeperNet(board_height, board_width, input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module):
    """Initialize model weights using Xavier/He initialization"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
