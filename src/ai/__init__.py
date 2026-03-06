"""
AI Package for Minesweeper
Provides components for training neural networks to play minesweeper
"""

from .game_api import MinesweeperAPI
from .environment import MinesweeperEnvironment
from .models import DQN, MinesweeperNet, create_model, count_parameters
from .models_v2 import MinesweeperFCN
from .models_v3 import MinesweeperResNet
from .algorithmic_solver import AlgorithmicSolver
from .trainer import DQNTrainer, create_trainer

__all__ = [
    'MinesweeperAPI',
    'MinesweeperEnvironment',
    'DQN',
    'MinesweeperNet',
    'MinesweeperFCN',
    'MinesweeperResNet',
    'AlgorithmicSolver',
    'DQNTrainer',
    'create_trainer',
    'create_model',
    'count_parameters',
]
