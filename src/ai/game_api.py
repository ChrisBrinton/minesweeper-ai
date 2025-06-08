"""
Minesweeper Game API for AI Training
Provides a clean interface for AI agents to interact with the game
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game.board import GameBoard, GameState, CellState


class Action(Enum):
    """Available actions for the AI agent"""
    REVEAL = "reveal"
    FLAG = "flag"
    UNFLAG = "unflag"


class MinesweeperAPI:
    """
    API for AI agents to interact with Minesweeper game
    Handles cell coordinates and provides board state updates
    """
    
    def __init__(self, rows: int = 9, cols: int = 9, mines: int = 10):
        """
        Initialize the game API
        
        Args:
            rows: Number of rows in the game board
            cols: Number of columns in the game board 
            mines: Number of mines to place
        """
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.game_board = GameBoard(rows, cols, mines)
        self.action_history: List[Dict[str, Any]] = []
        
    def reset_game(self) -> Dict[str, Any]:
        """
        Reset the game to initial state
        
        Returns:
            Initial game state
        """
        self.game_board = GameBoard(self.rows, self.cols, self.mines)
        self.action_history.clear()
        return self.get_game_state()
    
    def take_action(self, row: int, col: int, action: Action) -> Dict[str, Any]:
        """
        Take an action at the specified coordinates
        
        Args:
            row: Row coordinate (0-indexed)
            col: Column coordinate (0-indexed)
            action: Action to take (REVEAL, FLAG, UNFLAG)
            
        Returns:
            Updated game state with action result
        """
        # Validate coordinates
        if not self._is_valid_coordinate(row, col):
            return {
                'success': False,
                'error': f'Invalid coordinates: ({row}, {col})',
                'state': self.get_game_state()
            }
        
        # Record action
        action_record = {
            'row': row,
            'col': col,
            'action': action.value,
            'game_state_before': self.game_board.game_state.value
        }
        
        success = False
        error = None
        
        try:
            if action == Action.REVEAL:
                success = self._reveal_cell(row, col)
            elif action == Action.FLAG:
                success = self._flag_cell(row, col)
            elif action == Action.UNFLAG:
                success = self._unflag_cell(row, col)
            else:
                error = f"Unknown action: {action}"
                
        except Exception as e:
            error = str(e)
        
        # Update action record
        action_record.update({
            'success': success,
            'error': error,
            'game_state_after': self.game_board.game_state.value
        })
        
        self.action_history.append(action_record)
        
        # Return result
        result = {
            'success': success,
            'action': action.value,
            'coordinates': (row, col),
            'state': self.get_game_state()
        }
        
        if error:
            result['error'] = error
            
        return result
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get the current complete game state
        
        Returns:
            Complete game state information
        """
        board_state = []
        visible_board = []
        
        for row in range(self.rows):
            board_row = []
            visible_row = []
            for col in range(self.cols):
                cell = self.game_board.get_cell(row, col)
                
                # Full cell information (for analysis)
                cell_info = {
                    'row': row,
                    'col': col,
                    'is_mine': cell.is_mine,
                    'state': cell.state.value,
                    'adjacent_mines': cell.adjacent_mines,
                    'is_revealed': cell.is_revealed(),
                    'is_flagged': cell.is_flagged()
                }
                board_row.append(cell_info)
                
                # Visible information (what AI can see)
                if cell.is_revealed():
                    if cell.is_mine:
                        visible_row.append(-1)  # Mine
                    else:
                        visible_row.append(cell.adjacent_mines)  # Number
                elif cell.is_flagged():
                    visible_row.append(-2)  # Flag
                else:
                    visible_row.append(-3)  # Hidden
                    
            board_state.append(board_row)
            visible_board.append(visible_row)
        
        return {
            'board_size': (self.rows, self.cols),
            'total_mines': self.mines,
            'game_state': self.game_board.game_state.value,
            'mines_placed': self.game_board.mines_placed,
            'cells_revealed': self.game_board.cells_revealed,
            'flags_used': self.game_board.flags_used,
            'remaining_mines': self.game_board.get_remaining_mines(),
            'visible_board': visible_board,  # What AI can see (-3=hidden, -2=flag, -1=mine, 0-8=numbers)
            'full_board': board_state,  # Complete information (for analysis)
            'action_count': len(self.action_history),
            'is_game_over': self.game_board.game_state in [GameState.WON, GameState.LOST],
            'is_won': self.game_board.game_state == GameState.WON,
            'is_lost': self.game_board.game_state == GameState.LOST
        }
    
    def get_board_array(self) -> np.ndarray:
        """
        Get the board as a numpy array for neural network input
        
        Returns:
            3D numpy array: [rows, cols, channels]
            Channels:
            0: Visible state (-3=hidden, -2=flag, -1=mine, 0-8=numbers)
            1: Is revealed (0 or 1)
            2: Is flagged (0 or 1)
        """
        state = self.get_game_state()
        visible_board = np.array(state['visible_board'], dtype=np.float32)
        
        # Create additional channels
        revealed_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        flagged_channel = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.game_board.get_cell(row, col)
                revealed_channel[row, col] = 1.0 if cell.is_revealed() else 0.0
                flagged_channel[row, col] = 1.0 if cell.is_flagged() else 0.0
        
        # Stack channels
        board_array = np.stack([visible_board, revealed_channel, flagged_channel], axis=-1)
        return board_array
    
    def get_valid_actions(self) -> List[Tuple[int, int, Action]]:
        """
        Get all valid actions in the current state
        
        Returns:
            List of (row, col, action) tuples
        """
        if self.game_board.game_state in [GameState.WON, GameState.LOST]:
            return []
        
        valid_actions = []
        
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.game_board.get_cell(row, col)
                
                if cell.state == CellState.HIDDEN:
                    # Can reveal or flag hidden cells
                    valid_actions.append((row, col, Action.REVEAL))
                    valid_actions.append((row, col, Action.FLAG))
                elif cell.state == CellState.FLAGGED:
                    # Can unflag flagged cells
                    valid_actions.append((row, col, Action.UNFLAG))
                # Cannot act on revealed cells
        
        return valid_actions
    
    def get_reward(self) -> float:
        """
        Calculate reward for current game state
        
        Returns:
            Reward value
        """
        if self.game_board.game_state == GameState.WON:
            return 100.0  # Large positive reward for winning
        elif self.game_board.game_state == GameState.LOST:
            return -100.0  # Large negative reward for losing
        elif self.game_board.game_state == GameState.PLAYING:
            # Small reward for each safe cell revealed
            return self.game_board.cells_revealed * 1.0
        else:
            return 0.0  # No reward for ready state
    
    def export_game_state(self) -> str:
        """
        Export current game state as JSON string
        
        Returns:
            JSON string of game state
        """
        state = self.get_game_state()
        return json.dumps(state, indent=2)
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all actions taken
        
        Returns:
            List of action records
        """
        return self.action_history.copy()
    
    def _is_valid_coordinate(self, row: int, col: int) -> bool:
        """Check if coordinates are valid"""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _reveal_cell(self, row: int, col: int) -> bool:
        """Reveal a cell"""
        cell = self.game_board.get_cell(row, col)
        if cell.state != CellState.HIDDEN:
            return False  # Can only reveal hidden cells
        
        self.game_board.reveal_cell(row, col)
        return True
    
    def _flag_cell(self, row: int, col: int) -> bool:
        """Flag a cell"""
        cell = self.game_board.get_cell(row, col)
        if cell.state != CellState.HIDDEN:
            return False  # Can only flag hidden cells
        
        self.game_board.toggle_flag(row, col)
        return True
    
    def _unflag_cell(self, row: int, col: int) -> bool:
        """Unflag a cell"""
        cell = self.game_board.get_cell(row, col)
        if cell.state != CellState.FLAGGED:
            return False  # Can only unflag flagged cells
        
        self.game_board.toggle_flag(row, col)
        return True
