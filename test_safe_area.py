#!/usr/bin/env python3
"""
Test script to verify the 3x3 safe area functionality
"""

from src.game.board import GameBoard

def test_safe_area_small_board():
    """Test safe area on small board (should use 1-cell safe area)"""
    print("Testing 3x3 board with 5 mines...")
    board = GameBoard(3, 3, 5)
    
    # First click should be safe
    board.reveal_cell(1, 1)  # Center cell
    center_cell = board.get_cell(1, 1)
    
    print(f"Center cell is mine: {center_cell.is_mine}")
    print(f"Total mines placed: {board.total_mines}")
    
    # Count actual mines on board
    mine_count = 0
    for row in range(3):
        for col in range(3):
            if board.get_cell(row, col).is_mine:
                mine_count += 1
    
    print(f"Actual mines on board: {mine_count}")
    print()

def test_safe_area_large_board():
    """Test safe area on large board (should use 3x3 safe area)"""
    print("Testing 9x9 board with 10 mines...")
    board = GameBoard(9, 9, 10)
    
    # First click at center
    board.reveal_cell(4, 4)  # Center cell
    
    # Check 3x3 area around center
    safe_area_mines = 0
    print("3x3 safe area around center:")
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            row, col = 4 + dr, 4 + dc
            cell = board.get_cell(row, col)
            is_mine = cell.is_mine
            if is_mine:
                safe_area_mines += 1
            print(f"  ({row},{col}): mine={is_mine}")
    
    print(f"Mines in safe area: {safe_area_mines}")
    print(f"Total mines placed: {board.total_mines}")
    
    # Count actual mines on board
    mine_count = 0
    for row in range(9):
        for col in range(9):
            if board.get_cell(row, col).is_mine:
                mine_count += 1
    
    print(f"Actual mines on board: {mine_count}")
    print()

def test_safe_area_edge_click():
    """Test safe area when clicking near edge"""
    print("Testing 9x9 board with click at edge...")
    board = GameBoard(9, 9, 10)
    
    # First click at edge
    board.reveal_cell(0, 0)  # Top-left corner
    
    # Check safe area around corner (should only include valid cells)
    safe_area_mines = 0
    print("Safe area around top-left corner:")
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            row, col = 0 + dr, 0 + dc
            if 0 <= row < 9 and 0 <= col < 9:
                cell = board.get_cell(row, col)
                is_mine = cell.is_mine
                if is_mine:
                    safe_area_mines += 1
                print(f"  ({row},{col}): mine={is_mine}")
    
    print(f"Mines in safe area: {safe_area_mines}")
    print(f"Total mines placed: {board.total_mines}")
    print()

if __name__ == "__main__":
    test_safe_area_small_board()
    test_safe_area_large_board()
    test_safe_area_edge_click()
