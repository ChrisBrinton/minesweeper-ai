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
        self.api = MinesweeperAPI(rows, cols, mines)        # Reward configuration
        self.reward_config = reward_config or {
            'win': 150.0,
            'lose': -100.0,
            'reveal_safe': 2.0,
            'reveal_number': 4.0,
            'reveal_multi_safe': 1.5,  # Per additional safe cell revealed in cascade
            'reveal_empty_cell': 8.0,  # Bonus for clicking empty cell that causes cascade
            'flag_correct': 5.0,
            'flag_incorrect': -10.0,
            'unflag_penalty': -2.0,  # Penalty for unflagging to discourage flag/unflag loops
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
        
        # Track cells revealed before action (for multi-cell reveal detection)
        cells_revealed_before = self.api.game_board.cells_revealed
        
        # Take action
        result = self.api.take_action(row, col, action_enum)
        
        # Track cells revealed after action
        cells_revealed_after = self.api.game_board.cells_revealed
        cells_newly_revealed = cells_revealed_after - cells_revealed_before
        
        # Calculate reward
        reward = self._calculate_reward(result, row, col, action_enum, cells_newly_revealed)
        
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
            'cells_newly_revealed': cells_newly_revealed,
            'flags_used': self.api.game_board.flags_used,
            'steps_taken': self.steps_taken,
            'reward_breakdown': self._get_reward_breakdown(result, row, col, action_enum, cells_newly_revealed)
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
                         action: Action, cells_newly_revealed: int) -> float:
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
            if action == Action.REVEAL and cells_newly_revealed > 0:
                cell = self.api.game_board.get_cell(row, col)
                if cell.is_revealed():
                    if cell.adjacent_mines == 0 and cells_newly_revealed > 1:
                        # Clicked on empty cell causing cascade - big bonus!
                        reward += self.reward_config['reveal_empty_cell']
                        # Additional reward for each extra cell revealed
                        reward += (cells_newly_revealed - 1) * self.reward_config['reveal_multi_safe']
                    elif cell.adjacent_mines > 0:
                        # Revealed numbered cell                        reward += self.reward_config['reveal_number']
                        # Smaller bonus if this also revealed other cells (rare but possible)
                        if cells_newly_revealed > 1:
                            reward += (cells_newly_revealed - 1) * self.reward_config['reveal_multi_safe'] * 0.5
                    else:
                        # Single safe cell
                        reward += self.reward_config['reveal_safe']
            
            elif action == Action.FLAG:
                # We can't know if flag is correct until game ends
                # Give small positive reward for flagging
                reward += 0.5
            
            elif action == Action.UNFLAG:
                # Penalty for unflagging to discourage flag/unflag loops
                reward += self.reward_config['unflag_penalty']
        return reward

    def _get_reward_breakdown(self, result: Dict[str, Any], row: int, col: int, 
                             action: Action, cells_newly_revealed: int) -> Dict[str, float]:
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
                if action == Action.REVEAL and cells_newly_revealed > 0:
                    cell = self.api.game_board.get_cell(row, col)
                    if cell.is_revealed():
                        if cell.adjacent_mines == 0 and cells_newly_revealed > 1:
                            breakdown['reveal_empty_cell'] = self.reward_config['reveal_empty_cell']
                            breakdown['reveal_multi_safe'] = (cells_newly_revealed - 1) * self.reward_config['reveal_multi_safe']
                        elif cell.adjacent_mines > 0:
                            breakdown['reveal_number'] = self.reward_config['reveal_number']
                            if cells_newly_revealed > 1:
                                breakdown['reveal_multi_safe'] = (cells_newly_revealed - 1) * self.reward_config['reveal_multi_safe'] * 0.5
                        else:
                            breakdown['reveal_safe'] = self.reward_config['reveal_safe']
                elif action == Action.FLAG:
                    breakdown['flag_attempt'] = 0.5
                elif action == Action.UNFLAG:
                    breakdown['unflag_penalty'] = self.reward_config['unflag_penalty']
        
        return breakdown


class PerfectKnowledgeMinesweeperEnvironment(MinesweeperEnvironment):
    """
    Special environment for Phase 0 training where the agent has perfect knowledge
    The board is not revealed but the agent can see mine locations in the observation
    This teaches proper gameplay mechanics with perfect information
    """
    
    def __init__(self, rows: int = 5, cols: int = 5, mines: int = 3, 
                 reward_config: Optional[Dict[str, float]] = None):
        """
        Initialize the perfect knowledge environment
        
        Args:
            rows: Number of rows in the game board
            cols: Number of columns in the game board
            mines: Number of mines to place
            reward_config: Custom reward configuration
        """
        super().__init__(rows, cols, mines, reward_config)
          # Special reward config for phase 0 - emphasize learning game logic
        if reward_config is None:
            self.reward_config.update({
                'win': 250.0,  # Increased win reward for better learning signal
                'lose': -100.0,  # Standard loss penalty
                'reveal_safe': 2.0,
                'reveal_number': 4.0,
                'flag_correct': 10.0,  # Higher reward for correct flagging
                'flag_incorrect': -15.0,  # Higher penalty for wrong flags
                'step_penalty': -0.1
            })
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment - board starts hidden but observation includes perfect info
        """
        # Standard reset
        observation = super().reset()
          # Return observation with perfect knowledge
        # Note: Mines may not be placed yet - that's handled in _get_perfect_knowledge_observation
        return self._get_perfect_knowledge_observation()
    
    def _get_perfect_knowledge_observation(self) -> np.ndarray:
        """
        Get observation with perfect knowledge of mine locations
        
        Returns observation with 4 channels:
        - Channel 0: Visible state (what player normally sees)
        - Channel 1: Adjacent mine counts (revealed and hidden)
        - Channel 2: Flags
        - Channel 3: Mine locations (perfect knowledge - may be empty if mines not placed yet)
        """
        # Get standard 3-channel observation
        standard_obs = super()._get_observation()
        
        # Create 4th channel with perfect mine knowledge
        mine_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Fill mine channel with mine locations (if mines are placed)
        if self.api.game_board.mines_placed:
            for row in range(self.rows):
                for col in range(self.cols):
                    cell = self.api.game_board.get_cell(row, col)
                    if cell.is_mine:
                        mine_channel[row, col] = 1.0
        
        # Stack the 4 channels
        perfect_obs = np.stack([
            standard_obs[:, :, 0],  # Visible state
            standard_obs[:, :, 1],  # Adjacent counts
            standard_obs[:, :, 2],  # Flags
            mine_channel            # Mine locations (perfect knowledge)
        ], axis=2)
        
        return perfect_obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step with perfect knowledge observation
        """
        # Standard step processing
        observation, reward, done, info = super().step(action)
          # Replace observation with perfect knowledge version
        perfect_observation = self._get_perfect_knowledge_observation()
        
        # Add info about the learning phase
        info['phase'] = 'perfect_knowledge'
        info['has_perfect_info'] = True
        
        return perfect_observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Override to provide perfect knowledge observation"""
        return self._get_perfect_knowledge_observation()
