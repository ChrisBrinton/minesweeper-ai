"""
Test script for minesweeper game logic
Tests core functionality without GUI
"""

import pytest
from game import GameBoard, GameState, CellState


def test_game_board_creation():
    """Test basic game board creation"""
    board = GameBoard(9, 9, 10)  # Beginner difficulty
    assert board.rows == 9
    assert board.cols == 9
    assert board.total_mines == 10
    assert board.game_state == GameState.READY
    assert board.get_remaining_mines() == 10


def test_first_click_safety():
    """Test that first click is always safe"""
    board = GameBoard(9, 9, 10)
    
    # Test first click (should be safe)
    board.reveal_cell(4, 4)  # Click center
    assert board.game_state == GameState.PLAYING
    assert board.mines_placed == True
    
    # The clicked cell should not be a mine
    center_cell = board.get_cell(4, 4)
    assert not center_cell.is_mine


def test_flagging():
    """Test flagging functionality"""
    board = GameBoard(9, 9, 10)
    
    # Test flagging
    board.toggle_flag(0, 0)
    assert board.flags_used == 1
    assert board.get_remaining_mines() == 9
    
    # Test cell state
    cell = board.get_cell(0, 0)
    assert cell.is_flagged()
    
    # Test unflagging
    board.toggle_flag(0, 0)
    assert board.flags_used == 0
    assert board.get_remaining_mines() == 10
    assert not cell.is_flagged()


def test_board_statistics():
    """Test board statistics are correct"""
    board = GameBoard(9, 9, 10)
    
    # Initial statistics
    assert board.rows * board.cols == 81  # Total cells
    assert board.total_mines == 10
    assert board.cells_revealed == 0
    assert board.flags_used == 0
    
    # After first click
    board.reveal_cell(4, 4)
    assert board.cells_revealed > 0


@pytest.mark.parametrize("difficulty,expected", [
    ("beginner", (9, 9, 10)),
    ("intermediate", (16, 16, 40)),
    ("expert", (16, 30, 99))
])
def test_all_difficulties(difficulty, expected):
    """Test all difficulty levels"""
    rows, cols, mines = expected
    board = GameBoard(rows, cols, mines)
    
    assert board.rows == rows
    assert board.cols == cols
    assert board.total_mines == mines
    
    # Test first click
    board.reveal_cell(0, 0)
    assert board.game_state == GameState.PLAYING
    assert board.mines_placed == True


def test_difficulty_constants():
    """Test that difficulty constants are accessible"""
    assert hasattr(GameBoard, 'DIFFICULTIES')
    difficulties = GameBoard.DIFFICULTIES
    
    assert 'beginner' in difficulties
    assert 'intermediate' in difficulties
    assert 'expert' in difficulties
    
    # Test that each difficulty has correct format (rows, cols, mines)
    for difficulty, settings in difficulties.items():
        assert len(settings) == 3
        rows, cols, mines = settings
        assert isinstance(rows, int) and rows > 0
        assert isinstance(cols, int) and cols > 0
        assert isinstance(mines, int) and mines > 0
        assert mines < rows * cols  # Can't have more mines than cells
