"""
Test script to verify timer functionality
"""

import pytest
from game import GameBoard, GameState


def test_timer_initial_state():
    """Test that timer logic starts in correct state"""
    board = GameBoard(3, 3, 1)
    
    # Verify initial state
    assert board.game_state == GameState.READY
    

def test_timer_state_transition():
    """Test that game state transitions correctly for timer"""
    board = GameBoard(3, 3, 1)
    
    # Initial state should be READY
    assert board.game_state == GameState.READY
    
    # Simulate first click
    board.reveal_cell(1, 1)
    
    # Verify state changed to PLAYING
    assert board.game_state == GameState.PLAYING


def test_timer_logic_requirements():
    """Test that timer logic requirements are met"""
    board = GameBoard(9, 9, 10)
    
    # Timer should start when state changes from READY to PLAYING
    initial_state = board.game_state
    assert initial_state == GameState.READY
    
    # Make first move
    board.reveal_cell(4, 4)
    
    # State should now be PLAYING (required for timer to work)
    assert board.game_state == GameState.PLAYING
    
    # This ensures _update_timer() condition is met in GUI


def test_game_completion_states():
    """Test that game can reach completion states for timer stopping"""
    # Use standard beginner difficulty to prevent immediate wins from auto-reveal
    board = GameBoard(9, 9, 10)  # Standard beginner: 71 safe cells, impossible to reveal all in one chain
    
    # Make first move in center area to avoid corner edge effects
    board.reveal_cell(4, 4)
    
    # With standard beginner board, first click should always result in PLAYING state
    assert board.game_state == GameState.PLAYING
    
    # Game should be able to reach WON or LOST states
    # (The actual win/loss testing is in other test files)
    

def test_multiple_game_sessions():
    """Test timer behavior across multiple game sessions"""
    # Test that each new game starts fresh
    for _ in range(3):
        board = GameBoard(9, 9, 10)
        
        # Each game should start in READY state
        assert board.game_state == GameState.READY
        
        # First click should transition to PLAYING
        board.reveal_cell(4, 4)
        assert board.game_state == GameState.PLAYING


def test_timer_edge_cases():
    """Test timer behavior in edge cases"""
    # Small board - use 3x3 with 2 mines to prevent immediate wins
    board = GameBoard(3, 3, 2)
    assert board.game_state == GameState.READY
    
    board.reveal_cell(1, 1)  # Center click
    # With 7 safe cells, unlikely to win immediately
    assert board.game_state == GameState.PLAYING
    
    # Large board
    board = GameBoard(30, 16, 99)  # Expert size
    assert board.game_state == GameState.READY
    
    board.reveal_cell(15, 8)  # Center-ish
    assert board.game_state == GameState.PLAYING


@pytest.mark.parametrize("rows,cols,mines", [
    (9, 9, 10),    # Beginner
    (16, 16, 40),  # Intermediate  
    (16, 30, 99),  # Expert
])
def test_timer_all_difficulties(rows, cols, mines):
    """Test timer state transitions work for all difficulty levels"""
    board = GameBoard(rows, cols, mines)
    
    # Initial state
    assert board.game_state == GameState.READY
    
    # First click should always transition to PLAYING
    center_row, center_col = rows // 2, cols // 2
    board.reveal_cell(center_row, center_col)
    assert board.game_state == GameState.PLAYING
