"""
Unit tests for the GameBoard class
Tests game logic, mine placement, and game state management
"""

import pytest
from unittest.mock import patch
from game.board import GameBoard, GameState, CellState, Cell


class TestGameBoardInitialization:
    """Test cases for GameBoard initialization"""
    
    def test_default_initialization(self):
        """Test board initializes with default beginner settings"""
        board = GameBoard()
        
        assert board.rows == 9
        assert board.cols == 9
        assert board.total_mines == 10
        assert board.game_state == GameState.READY
        assert board.mines_placed is False
        assert board.flags_used == 0
        assert board.cells_revealed == 0
    
    def test_custom_initialization(self):
        """Test board initializes with custom settings"""
        board = GameBoard(16, 30, 99)
        
        assert board.rows == 16
        assert board.cols == 30
        assert board.total_mines == 99
        assert board.game_state == GameState.READY
    
    def test_board_creation(self):
        """Test that board creates correct number of cells"""
        board = GameBoard(5, 7, 3)
        
        assert len(board.board) == 5  # rows
        assert len(board.board[0]) == 7  # cols
        
        # Check all cells are initialized
        for row in range(5):
            for col in range(7):
                cell = board.board[row][col]
                assert isinstance(cell, Cell)
                assert cell.row == row
                assert cell.col == col
                assert cell.is_mine is False
                assert cell.state == CellState.HIDDEN
    
    def test_difficulty_presets(self):
        """Test that difficulty presets are correctly defined"""
        assert GameBoard.DIFFICULTIES['beginner'] == (9, 9, 10)
        assert GameBoard.DIFFICULTIES['intermediate'] == (16, 16, 40)
        assert GameBoard.DIFFICULTIES['expert'] == (16, 30, 99)


class TestGameBoardCellAccess:
    """Test cases for cell access methods"""
    
    @pytest.fixture
    def board(self):
        """Create a small test board"""
        return GameBoard(3, 3, 2)
    
    def test_get_valid_cell(self, board):
        """Test getting a valid cell"""
        cell = board.get_cell(1, 2)
        assert cell is not None
        assert cell.row == 1
        assert cell.col == 2
    
    def test_get_cell_out_of_bounds(self, board):
        """Test getting cells outside board boundaries"""
        # Test negative coordinates
        assert board.get_cell(-1, 0) is None
        assert board.get_cell(0, -1) is None
        
        # Test coordinates too large
        assert board.get_cell(3, 0) is None
        assert board.get_cell(0, 3) is None
        assert board.get_cell(5, 5) is None


class TestMinePlacement:
    """Test cases for mine placement logic"""
    
    @pytest.fixture
    def board(self):
        """Create a test board"""
        return GameBoard(5, 5, 5)
    
    def test_mine_placement_on_first_click(self, board):
        """Test that mines are placed when first cell is revealed"""
        assert board.mines_placed is False
        
        board.reveal_cell(2, 2)
        
        assert board.mines_placed is True
        assert board.game_state == GameState.PLAYING
    
    def test_first_click_safety(self, board):
        """Test that first click position never has a mine"""
        first_row, first_col = 2, 2
        
        # Reveal first cell multiple times to test randomness
        for _ in range(10):
            board = GameBoard(5, 5, 5)
            board.reveal_cell(first_row, first_col)
            
            first_cell = board.get_cell(first_row, first_col)
            assert first_cell.is_mine is False
    
    def test_correct_number_of_mines_placed(self, board):
        """Test that correct number of mines are placed"""
        board.reveal_cell(0, 0)
        
        mine_count = 0
        for row in range(board.rows):
            for col in range(board.cols):
                if board.get_cell(row, col).is_mine:
                    mine_count += 1
        
        assert mine_count == board.total_mines
    
    def test_adjacent_mine_calculation(self):
        """Test that adjacent mine counts are calculated correctly"""
        board = GameBoard(3, 3, 0)  # No mines initially
        
        # Manually place mines in specific positions
        board.board[0][0].place_mine()  # Top-left
        board.board[2][2].place_mine()  # Bottom-right
        
        # Calculate adjacent mines
        board._calculate_adjacent_mines()
        
        # Check adjacent counts
        assert board.get_cell(0, 1).adjacent_mines == 1  # Adjacent to (0,0)
        assert board.get_cell(1, 0).adjacent_mines == 1  # Adjacent to (0,0)
        assert board.get_cell(1, 1).adjacent_mines == 2  # Adjacent to both mines
        assert board.get_cell(1, 2).adjacent_mines == 1  # Adjacent to (2,2)
        assert board.get_cell(2, 1).adjacent_mines == 1  # Adjacent to (2,2)


class TestGameLogic:
    """Test cases for core game logic"""
    
    @pytest.fixture
    def board(self):
        """Create a test board with known mine placement"""
        board = GameBoard(3, 3, 1)
        # Place mine manually for predictable testing
        board.board[0][0].place_mine()
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        return board
    
    def test_reveal_safe_cell(self, board):
        """Test revealing a safe cell"""
        initial_revealed = board.cells_revealed
        
        result = board.reveal_cell(1, 1)
        
        assert result is True
        assert board.cells_revealed == initial_revealed + 1
        assert board.get_cell(1, 1).state == CellState.REVEALED
        assert board.game_state == GameState.PLAYING
    
    def test_reveal_mine_cell(self, board):
        """Test revealing a mine cell ends the game"""
        result = board.reveal_cell(0, 0)  # Mine location
        
        assert result is False
        assert board.game_state == GameState.LOST
        assert board.get_cell(0, 0).state == CellState.REVEALED
    
    def test_reveal_already_revealed_cell(self, board):
        """Test revealing an already revealed cell"""
        board.reveal_cell(1, 1)  # First reveal
        initial_revealed = board.cells_revealed
        
        result = board.reveal_cell(1, 1)  # Second reveal
        
        assert result is True
        assert board.cells_revealed == initial_revealed  # No change
    
    def test_reveal_flagged_cell(self, board):
        """Test that flagged cells cannot be revealed"""
        board.toggle_flag(1, 1)
        initial_revealed = board.cells_revealed
        
        result = board.reveal_cell(1, 1)
        
        assert result is True
        assert board.cells_revealed == initial_revealed  # No change
        assert board.get_cell(1, 1).state == CellState.FLAGGED


class TestFlagging:
    """Test cases for flag functionality"""
    
    @pytest.fixture
    def board(self):
        """Create a test board"""
        board = GameBoard(3, 3, 2)
        board.game_state = GameState.PLAYING
        return board
    
    def test_flag_hidden_cell(self, board):
        """Test flagging a hidden cell"""
        board.toggle_flag(1, 1)
        
        assert board.get_cell(1, 1).state == CellState.FLAGGED
        assert board.flags_used == 1
    
    def test_too_many_mines_requested(self):
        """Test board handles requests for more mines than available positions"""
        # 3x3 board = 9 cells, requesting 10 mines (more than possible)
        board = GameBoard(3, 3, 10)
        
        # First click should be safe
        board.reveal_cell(1, 1)
          # Should only place 8 mines (9 cells - 1 for first click safety)
        assert board.total_mines == 8
        assert board.game_state == GameState.WON  # Only 1 safe cell, so immediate win
        
        # Count actual mines placed
        mine_count = 0
        for row in range(board.rows):
            for col in range(board.cols):
                if board.get_cell(row, col).is_mine:
                    mine_count += 1
        
        assert mine_count == 8
    
    def test_unflag_flagged_cell(self, board):
        """Test unflagging a flagged cell"""
        board.toggle_flag(1, 1)  # Flag
        board.toggle_flag(1, 1)  # Unflag
        
        assert board.get_cell(1, 1).state == CellState.HIDDEN
        assert board.flags_used == 0
    
    def test_flag_revealed_cell(self, board):
        """Test that revealed cells cannot be flagged"""
        board.reveal_cell(1, 1)
        board.toggle_flag(1, 1)
        
        assert board.get_cell(1, 1).state == CellState.REVEALED
        assert board.flags_used == 0
    
    def test_flag_when_game_over(self):
        """Test that flagging is disabled when game is over"""
        board = GameBoard(3, 3, 1)
        board.game_state = GameState.LOST
        
        board.toggle_flag(1, 1)
        
        assert board.get_cell(1, 1).state == CellState.HIDDEN
        assert board.flags_used == 0
    
    def test_remaining_mines_calculation(self, board):
        """Test remaining mines calculation"""
        assert board.get_remaining_mines() == 2
        
        board.toggle_flag(0, 0)
        assert board.get_remaining_mines() == 1
        
        board.toggle_flag(0, 1)
        assert board.get_remaining_mines() == 0
        
        board.toggle_flag(0, 2)  # More flags than mines
        assert board.get_remaining_mines() == 0  # Should not go negative


class TestAutoReveal:
    """Test cases for auto-reveal functionality"""
    
    def test_auto_reveal_adjacent_empty_cells(self):
        """Test that empty cells auto-reveal adjacent cells"""
        board = GameBoard(4, 4, 1)
        
        # Place mine in corner
        board.board[0][0].place_mine()
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        
        # Reveal a cell far from mine (should auto-reveal)
        board.reveal_cell(3, 3)
        
        # Check that multiple cells were revealed
        assert board.cells_revealed > 1
        assert board.get_cell(3, 3).state == CellState.REVEALED
        assert board.get_cell(2, 3).state == CellState.REVEALED
        assert board.get_cell(3, 2).state == CellState.REVEALED


class TestWinCondition:
    """Test cases for win condition detection"""
    
    def test_win_condition_simple(self):
        """Test win condition with simple 2x2 board"""
        board = GameBoard(2, 2, 1)
        
        # Manually set up board state
        board.board[0][0].place_mine()
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        
        # Reveal all safe cells
        board.reveal_cell(0, 1)
        board.reveal_cell(1, 0)
        board.reveal_cell(1, 1)
        
        assert board.game_state == GameState.WON
    
    def test_win_condition_flags_all_mines(self):
        """Test that winning automatically flags all mines"""
        board = GameBoard(2, 2, 1)
        
        # Set up board
        board.board[0][0].place_mine()
        board._calculate_adjacent_mines()
        board.mines_placed = True
        board.game_state = GameState.PLAYING
        
        # Reveal all safe cells to win
        board.reveal_cell(0, 1)
        board.reveal_cell(1, 0)
        board.reveal_cell(1, 1)
        
        # Check that mine was automatically flagged
        assert board.get_cell(0, 0).state == CellState.FLAGGED


class TestGameReset:
    """Test cases for game reset functionality"""
    
    def test_reset_game_same_difficulty(self):
        """Test resetting game with same settings"""
        board = GameBoard(5, 5, 3)
        
        # Play some moves
        board.reveal_cell(2, 2)
        board.toggle_flag(0, 0)
        
        # Reset
        board.reset_game()
        
        assert board.game_state == GameState.READY
        assert board.mines_placed is False
        assert board.flags_used == 0
        assert board.cells_revealed == 0
        assert board.rows == 5
        assert board.cols == 5
        assert board.total_mines == 3
    
    def test_reset_game_different_difficulty(self):
        """Test resetting game with different difficulty"""
        board = GameBoard(5, 5, 3)
        
        # Reset to expert difficulty
        board.reset_game('expert')
        
        assert board.rows == 16
        assert board.cols == 30
        assert board.total_mines == 99
        assert board.game_state == GameState.READY
    
    def test_reset_invalid_difficulty(self):
        """Test resetting with invalid difficulty keeps current settings"""
        board = GameBoard(5, 5, 3)
        original_settings = (board.rows, board.cols, board.total_mines)
        
        board.reset_game('invalid_difficulty')
        
        assert (board.rows, board.cols, board.total_mines) == original_settings


class TestEdgeCases:
    """Test cases for edge cases and error conditions"""
    
    def test_reveal_out_of_bounds(self):
        """Test revealing out of bounds coordinates"""
        board = GameBoard(3, 3, 1)
        
        result = board.reveal_cell(-1, 0)
        assert result is True  # Should not crash
        
        result = board.reveal_cell(5, 5)
        assert result is True  # Should not crash
    
    def test_flag_out_of_bounds(self):
        """Test flagging out of bounds coordinates"""
        board = GameBoard(3, 3, 1)
        board.game_state = GameState.PLAYING
          # Should not crash
        board.toggle_flag(-1, 0)
        board.toggle_flag(5, 5)
        assert board.flags_used == 0
    
    def test_zero_mines_board(self):
        """Test board with zero mines"""
        board = GameBoard(3, 3, 0)
        
        # First click should win immediately
        board.reveal_cell(1, 1)
        
        assert board.game_state == GameState.WON
        assert board.cells_revealed == 9  # All cells revealed
    
    def test_all_mines_board(self):
        """Test board where every cell except first click is a mine"""
        board = GameBoard(2, 2, 4)
        
        # Any click should be safe due to first click safety, 
        # but since all other cells will be mines, this should win immediately
        board.reveal_cell(0, 0)
        
        # Should place 3 mines (all positions except first click)
        assert board.total_mines == 3
        assert board.game_state == GameState.WON  # Only 1 safe cell, so immediate win
        
        # Verify first click cell is safe and revealed
        assert board.get_cell(0, 0).is_mine is False
        assert board.get_cell(0, 0).state == CellState.REVEALED
