"""
Minesweeper Environment for Reinforcement Learning
Provides a gym-like interface for training RL agents
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import random

from .game_api import MinesweeperAPI, Action


class MinesweeperEnvironment:
    """
    Reinforcement Learning environment for Minesweeper
    Provides standardized interface for RL algorithms
    """
    
    def __init__(self, rows: int = 9, cols: int = 9, mines: int = 10, 
                 reward_config: Optional[Dict[str, float]] = None):
        """
        Initialize the environment
        
        Args:
            rows: Number of rows in the game board
            cols: Number of columns in the game board
            mines: Number of mines to place
            reward_config: Custom reward configuration
        """
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.api = MinesweeperAPI(rows, cols, mines)
        
        # Reward configuration
        self.reward_config = reward_config or {
            'win': 100.0,
            'lose': -100.0,
            'reveal_safe': 2.0,
            'reveal_number': 4.0,
            'flag_correct': 5.0,
            'flag_incorrect': -10.0,
            'invalid_action': -1.0,
            'step_penalty': -0.1
        }
        
        # Action space: (row, col, action_type)
        # action_type: 0=reveal, 1=flag, 2=unflag
        self.action_space_size = rows * cols * 3
        
        # Observation space: 3D array [rows, cols, channels]
        self.observation_space_shape = (rows, cols, 3)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial observation
        """
        self.api.reset_game()
        self.steps_taken = 0
        self.last_cells_revealed = 0
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Encoded action (row * cols * 3 + col * 3 + action_type)
            
        Returns:
            observation, reward, done, info
        """
        self.steps_taken += 1
        
        # Decode action
        row, col, action_type = self._decode_action(action)
        action_enum = [Action.REVEAL, Action.FLAG, Action.UNFLAG][action_type]
        
        # Take action
        result = self.api.take_action(row, col, action_enum)
        
        # Calculate reward
        reward = self._calculate_reward(result, row, col, action_enum)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if done
        done = self.api.game_board.game_state.value in ['won', 'lost']
        
        # Additional info
        info = {
            'action_taken': (row, col, action_enum.value),
            'action_success': result['success'],
            'game_state': self.api.game_board.game_state.value,
            'cells_revealed': self.api.game_board.cells_revealed,
            'flags_used': self.api.game_board.flags_used,
            'steps_taken': self.steps_taken,
            'reward_breakdown': self._get_reward_breakdown(result, row, col, action_enum)
        }
        
        if 'error' in result:
            info['error'] = result['error']
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the current game state
        
        Args:
            mode: Rendering mode ('human' or 'string')
            
        Returns:
            String representation if mode='string'
        """
        state = self.api.get_game_state()
        visible_board = state['visible_board']
        
        # Create display
        lines = []
        lines.append(f"Game State: {state['game_state']}")
        lines.append(f"Cells Revealed: {state['cells_revealed']}")
        lines.append(f"Flags Used: {state['flags_used']}/{state['total_mines']}")
        lines.append(f"Steps Taken: {self.steps_taken}")
        lines.append("")
        
        # Board header
        header = "   " + " ".join(f"{c:2d}" for c in range(self.cols))
        lines.append(header)
        lines.append("   " + "---" * self.cols)
        
        # Board rows
        for row in range(self.rows):
            row_str = f"{row:2d}|"
            for col in range(self.cols):
                cell_value = visible_board[row][col]
                if cell_value == -3:  # Hidden
                    char = " ?"
                elif cell_value == -2:  # Flag
                    char = " F"
                elif cell_value == -1:  # Mine
                    char = " *"
                else:  # Number
                    char = f" {cell_value}" if cell_value > 0 else "  "
                row_str += char
            lines.append(row_str)
        
        display = "\n".join(lines)
        
        if mode == 'human':
            print(display)
        elif mode == 'string':
            return display
            
        return None
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions
        
        Returns:
            Boolean array indicating valid actions
        """
        mask = np.zeros(self.action_space_size, dtype=bool)
        valid_actions = self.api.get_valid_actions()
        
        for row, col, action_enum in valid_actions:
            action_type = [Action.REVEAL, Action.FLAG, Action.UNFLAG].index(action_enum)
            action_idx = self._encode_action(row, col, action_type)
            mask[action_idx] = True
            
        return mask
    
    def sample_action(self) -> int:
        """
        Sample a random valid action
        
        Returns:
            Random valid action
        """
        valid_actions = self.api.get_valid_actions()
        if not valid_actions:
            return 0  # Default action if no valid actions
        
        row, col, action_enum = random.choice(valid_actions)
        action_type = [Action.REVEAL, Action.FLAG, Action.UNFLAG].index(action_enum)
        return self._encode_action(row, col, action_type)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as numpy array"""
        return self.api.get_board_array()
    
    def _encode_action(self, row: int, col: int, action_type: int) -> int:
        """Encode action as single integer"""
        return row * self.cols * 3 + col * 3 + action_type
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decode action integer to (row, col, action_type)"""
        action_type = action % 3
        temp = action // 3
        col = temp % self.cols
        row = temp // self.cols
        return row, col, action_type
    
    def _calculate_reward(self, result: Dict[str, Any], row: int, col: int, 
                         action: Action) -> float:
        """Calculate reward for the action taken"""
        reward = 0.0
        
        # Step penalty
        reward += self.reward_config['step_penalty']
        
        if not result['success']:
            # Invalid action penalty
            reward += self.reward_config['invalid_action']
            return reward
        
        game_state = self.api.game_board.game_state.value
        
        if game_state == 'won':
            reward += self.reward_config['win']
        elif game_state == 'lost':
            reward += self.reward_config['lose']
        else:
            # Game still playing
            if action == Action.REVEAL:
                cell = self.api.game_board.get_cell(row, col)
                if cell.is_revealed():
                    if cell.adjacent_mines > 0:
                        reward += self.reward_config['reveal_number']
                    else:
                        reward += self.reward_config['reveal_safe']
            
            elif action == Action.FLAG:
                # We can't know if flag is correct until game ends
                # Give small positive reward for flagging
                reward += 0.5
        
        return reward
    
    def _get_reward_breakdown(self, result: Dict[str, Any], row: int, col: int, 
                             action: Action) -> Dict[str, float]:
        """Get detailed breakdown of reward calculation"""
        breakdown = {}
        
        breakdown['step_penalty'] = self.reward_config['step_penalty']
        
        if not result['success']:
            breakdown['invalid_action'] = self.reward_config['invalid_action']
        else:
            game_state = self.api.game_board.game_state.value
            
            if game_state == 'won':
                breakdown['win'] = self.reward_config['win']
            elif game_state == 'lost':
                breakdown['lose'] = self.reward_config['lose']
            else:
                if action == Action.REVEAL:
                    cell = self.api.game_board.get_cell(row, col)
                    if cell.is_revealed():
                        if cell.adjacent_mines > 0:
                            breakdown['reveal_number'] = self.reward_config['reveal_number']
                        else:
                            breakdown['reveal_safe'] = self.reward_config['reveal_safe']
                elif action == Action.FLAG:
                    breakdown['flag_attempt'] = 0.5
        
        return breakdown
