"""
Minesweeper Game - Core Game Logic
Implements the game mechanics and board management for minesweeper
"""

from enum import Enum
import random
from typing import List, Tuple, Set


class GameState(Enum):
    """Enumeration for different game states"""
    READY = "ready"
    PLAYING = "playing" 
    WON = "won"
    LOST = "lost"


class CellState(Enum):
    """Enumeration for cell states"""
    HIDDEN = "hidden"
    REVEALED = "revealed"
    FLAGGED = "flagged"


class Cell:
    """Represents a single cell on the minesweeper board"""
    
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.is_mine = False
        self.state = CellState.HIDDEN
        self.adjacent_mines = 0
    
    def place_mine(self):
        """Place a mine in this cell"""
        self.is_mine = True
    
    def reveal(self):
        """Reveal this cell"""
        if self.state == CellState.HIDDEN:
            self.state = CellState.REVEALED
            return True
        return False
    
    def toggle_flag(self):
        """Toggle flag state on this cell"""
        if self.state == CellState.HIDDEN:
            self.state = CellState.FLAGGED
        elif self.state == CellState.FLAGGED:
            self.state = CellState.HIDDEN
    
    def is_revealed(self) -> bool:
        """Check if cell is revealed"""
        return self.state == CellState.REVEALED
    
    def is_flagged(self) -> bool:
        """Check if cell is flagged"""
        return self.state == CellState.FLAGGED


class GameBoard:
    """Manages the minesweeper game board and game logic"""
      # Difficulty presets (rows, cols, mines)
    DIFFICULTIES = {
        'beginner': (9, 9, 10),
        'intermediate': (16, 16, 40),
        'expert': (16, 30, 99)
    }
    
    def __init__(self, rows: int = 9, cols: int = 9, mines: int = 10):
        self.rows = rows
        self.cols = cols
        self.total_mines = mines
        self.board: List[List[Cell]] = []
        self.game_state = GameState.READY
        self.mines_placed = False
        self.flags_used = 0
        self.cells_revealed = 0
        self.clicked_mine_pos = None  # Track position of clicked mine for red display
        
        self._initialize_board()
    def _initialize_board(self):
        """Initialize empty board without mines"""
        self.board = []
        for row in range(self.rows):
            board_row = []
            for col in range(self.cols):
                board_row.append(Cell(row, col))
            self.board.append(board_row)
    
    def _place_mines(self, first_click_row: int, first_click_col: int):
        """Place mines randomly, avoiding the first click area using efficient shuffle algorithm"""
        total_cells = self.rows * self.cols
        
        # Create a set of safe positions
        # For small boards (< 5x5), use only the clicked cell as safe
        # For larger boards, use a 3x3 area around the first click
        safe_positions = set()
        
        if self.rows < 5 or self.cols < 5:
            # Small board: only the clicked cell is safe
            safe_positions.add((first_click_row, first_click_col))
        else:
            # Larger board: 3x3 area around first click is safe
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = first_click_row + dr, first_click_col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        safe_positions.add((nr, nc))
        
        # Create list of all valid positions (excluding safe area)
        valid_positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) not in safe_positions:
                    valid_positions.append((row, col))
        
        available_cells = len(valid_positions)
        
        # Ensure we don't try to place more mines than available positions
        mines_to_place = min(self.total_mines, available_cells)
        
        # If there are no available cells for mines, set mines to 0
        if available_cells <= 0:
            self.total_mines = 0
        else:
            # Use Fisher-Yates shuffle algorithm for O(n) mine placement
            # This is much more efficient than rejection sampling
            random.shuffle(valid_positions)
            
            # Place mines at the first 'mines_to_place' positions
            for i in range(mines_to_place):
                row, col = valid_positions[i]
                self.board[row][col].place_mine()
            
            # Update total_mines to reflect actual mines placed
            self.total_mines = mines_to_place
        
        self._calculate_adjacent_mines()
        self.mines_placed = True
    
    def _calculate_adjacent_mines(self):
        """Calculate the number of adjacent mines for each cell"""
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.board[row][col].is_mine:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = row + dr, col + dc
                            if (0 <= nr < self.rows and 0 <= nc < self.cols 
                                and self.board[nr][nc].is_mine):
                                count += 1
                    self.board[row][col].adjacent_mines = count
    
    def get_cell(self, row: int, col: int) -> Cell:
        """Get cell at specified position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None
    
    def reveal_cell(self, row: int, col: int) -> bool:
        """
        Reveal a cell and handle game logic
        Returns True if game should continue, False if game over
        """
        if self.game_state in [GameState.WON, GameState.LOST]:
            return False
            
        cell = self.get_cell(row, col)
        if not cell or cell.state != CellState.HIDDEN:
            return True
        
        # Place mines on first click
        if not self.mines_placed:
            self._place_mines(row, col)
            self.game_state = GameState.PLAYING
          # Reveal the cell
        if cell.reveal():
            self.cells_revealed += 1
            
            # Check if hit a mine
            if cell.is_mine:
                self.clicked_mine_pos = (row, col)  # Track which mine was clicked
                self.game_state = GameState.LOST
                self._reveal_all_mines()
                return False
            
            # Auto-reveal adjacent cells if no adjacent mines
            if cell.adjacent_mines == 0:
                self._auto_reveal_adjacent(row, col)
            
            # Check win condition
            if self._check_win_condition():
                self.game_state = GameState.WON
                self._flag_all_mines()
        
        return True
    
    def _auto_reveal_adjacent(self, row: int, col: int):
        """Auto-reveal adjacent cells when a cell with 0 adjacent mines is revealed"""
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    adjacent_cell = self.board[nr][nc]
                    if adjacent_cell.state == CellState.HIDDEN:
                        self.reveal_cell(nr, nc)
    
    def toggle_flag(self, row: int, col: int):
        """Toggle flag on a cell"""
        if self.game_state in [GameState.WON, GameState.LOST]:
            return
            
        cell = self.get_cell(row, col)
        if not cell:
            return
            
        old_state = cell.state
        cell.toggle_flag()
        
        # Update flag count
        if old_state == CellState.HIDDEN and cell.state == CellState.FLAGGED:
            self.flags_used += 1
        elif old_state == CellState.FLAGGED and cell.state == CellState.HIDDEN:
            self.flags_used -= 1
    
    def _reveal_all_mines(self):
        """Reveal all mines when game is lost"""
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.board[row][col]
                if cell.is_mine and cell.state != CellState.FLAGGED:
                    cell.state = CellState.REVEALED
    
    def _flag_all_mines(self):
        """Flag all mines when game is won"""
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.board[row][col]
                if cell.is_mine and cell.state == CellState.HIDDEN:
                    cell.state = CellState.FLAGGED
                    self.flags_used += 1
    
    def _check_win_condition(self) -> bool:
        """Check if the player has won"""
        total_safe_cells = self.rows * self.cols - self.total_mines
        return self.cells_revealed == total_safe_cells
    
    def get_remaining_mines(self) -> int:
        """Get the number of remaining mines (total mines - flags used)"""
        return max(0, self.total_mines - self.flags_used)
    
    def reset_game(self, difficulty: str = None):
        """Reset the game to initial state"""
        if difficulty and difficulty in self.DIFFICULTIES:
            self.rows, self.cols, self.total_mines = self.DIFFICULTIES[difficulty]
        self.game_state = GameState.READY
        self.mines_placed = False
        self.flags_used = 0
        self.cells_revealed = 0
        self.clicked_mine_pos = None  # Reset clicked mine position
        self._initialize_board()
