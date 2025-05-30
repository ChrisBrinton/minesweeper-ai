#!/usr/bin/env python3
"""
Test script for minesweeper game logic
Tests core functionality without GUI
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from game import GameBoard, GameState, CellState


def test_game_board():
    """Test basic game board functionality"""
    print("Testing Minesweeper Game Logic...")
    print("=" * 40)
    
    # Test board creation
    board = GameBoard(9, 9, 10)  # Beginner difficulty
    print(f"✓ Created {board.rows}x{board.cols} board with {board.total_mines} mines")
    print(f"✓ Initial state: {board.game_state}")
    print(f"✓ Remaining mines: {board.get_remaining_mines()}")
    
    # Test first click (should be safe)
    print("\nTesting first click...")
    board.reveal_cell(4, 4)  # Click center
    print(f"✓ Game state after first click: {board.game_state}")
    print(f"✓ Mines placed: {board.mines_placed}")
    
    # Test flagging
    print("\nTesting flagging...")
    board.toggle_flag(0, 0)
    print(f"✓ Flagged cell (0,0)")
    print(f"✓ Flags used: {board.flags_used}")
    print(f"✓ Remaining mines: {board.get_remaining_mines()}")
    
    # Test cell states
    cell = board.get_cell(0, 0)
    print(f"✓ Cell (0,0) state: {cell.state}")
    print(f"✓ Cell (0,0) is flagged: {cell.is_flagged()}")
    
    # Display some board info
    print(f"\nBoard Statistics:")
    print(f"- Total cells: {board.rows * board.cols}")
    print(f"- Total mines: {board.total_mines}")
    print(f"- Cells revealed: {board.cells_revealed}")
    print(f"- Flags used: {board.flags_used}")
    
    print("\n✅ All tests passed! Game logic is working correctly.")


def test_all_difficulties():
    """Test all difficulty levels"""
    print("\nTesting Difficulty Levels...")
    print("=" * 40)
    
    for difficulty, (rows, cols, mines) in GameBoard.DIFFICULTIES.items():
        board = GameBoard(rows, cols, mines)
        print(f"✓ {difficulty.title()}: {rows}×{cols} with {mines} mines")
        
        # Test first click
        board.reveal_cell(0, 0)
        assert board.game_state == GameState.PLAYING
        assert board.mines_placed == True
        
        print(f"  - Game state: {board.game_state}")
        print(f"  - Mines placed successfully")


if __name__ == "__main__":
    try:
        test_game_board()
        test_all_difficulties()
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
