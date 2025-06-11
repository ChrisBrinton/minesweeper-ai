"""
Test script to verify the 3x3 safe area functionality
"""

import pytest
from game.board import GameBoard


def test_safe_area_small_board():
    """Test safe area on small board (should use 1-cell safe area)"""
    board = GameBoard(3, 3, 5)
    
    # First click should be safe
    board.reveal_cell(1, 1)  # Center cell
    center_cell = board.get_cell(1, 1)
    
    assert not center_cell.is_mine, "Center cell should not be a mine after first click"
    assert board.total_mines == 5, "Total mines should be 5"
    
    # Count actual mines on board
    mine_count = 0
    for row in range(3):
        for col in range(3):
            if board.get_cell(row, col).is_mine:
                mine_count += 1
    
    assert mine_count == 5, f"Should have exactly 5 mines, found {mine_count}"


def test_safe_area_large_board():
    """Test safe area on large board (should use 3x3 safe area)"""
    board = GameBoard(9, 9, 10)
    
    # First click at center
    board.reveal_cell(4, 4)  # Center cell
    
    # Check 3x3 area around center should be mine-free
    safe_area_mines = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            row, col = 4 + dr, 4 + dc
            cell = board.get_cell(row, col)
            if cell.is_mine:
                safe_area_mines += 1
    
    assert safe_area_mines == 0, f"Safe area should have no mines, found {safe_area_mines}"
    assert board.total_mines == 10, "Total mines should be 10"
    
    # Count actual mines on board
    mine_count = 0
    for row in range(9):
        for col in range(9):
            if board.get_cell(row, col).is_mine:
                mine_count += 1
    
    assert mine_count == 10, f"Should have exactly 10 mines, found {mine_count}"


def test_safe_area_edge_click():
    """Test safe area when clicking near edge"""
    board = GameBoard(9, 9, 10)
    
    # First click at edge
    board.reveal_cell(0, 0)  # Top-left corner
    
    # Check safe area around corner (should only include valid cells)
    safe_area_mines = 0
    safe_area_cells = []
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            row, col = 0 + dr, 0 + dc
            if 0 <= row < 9 and 0 <= col < 9:
                safe_area_cells.append((row, col))
                cell = board.get_cell(row, col)
                if cell.is_mine:
                    safe_area_mines += 1
    
    assert safe_area_mines == 0, f"Safe area should have no mines, found {safe_area_mines}"
    assert len(safe_area_cells) == 4, f"Corner safe area should have 4 cells, found {len(safe_area_cells)}"
    assert board.total_mines == 10, "Total mines should be 10"


def test_safe_area_different_positions():
    """Test safe area works for different click positions"""
    positions_to_test = [
        (0, 0),    # Top-left corner
        (0, 4),    # Top edge
        (4, 0),    # Left edge
        (4, 4),    # Center
        (8, 8),    # Bottom-right corner
    ]
    
    for row, col in positions_to_test:
        board = GameBoard(9, 9, 10)
        board.reveal_cell(row, col)
        
        # The clicked cell should not be a mine
        clicked_cell = board.get_cell(row, col)
        assert not clicked_cell.is_mine, f"Clicked cell at ({row}, {col}) should not be a mine"
        
        # Count safe area mines
        safe_area_mines = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = row + dr, col + dc
                if 0 <= r < 9 and 0 <= c < 9:
                    cell = board.get_cell(r, c)
                    if cell.is_mine:
                        safe_area_mines += 1
        
        # For boards with enough space, safe area should be mine-free
        # For very small boards or high mine density, this might not always be possible
        if board.total_mines < (board.rows * board.cols) - 9:  # If there's room for a 3x3 safe area
            assert safe_area_mines == 0, f"Safe area around ({row}, {col}) should have no mines"


def test_mine_placement_after_first_click():
    """Test that mines are only placed after first click"""
    board = GameBoard(9, 9, 10)
    
    # Initially, no mines should be placed
    assert not board.mines_placed, "Mines should not be placed initially"
    
    # Count mines before first click (should be 0)
    initial_mine_count = 0
    for row in range(9):
        for col in range(9):
            if board.get_cell(row, col).is_mine:
                initial_mine_count += 1
    
    assert initial_mine_count == 0, "No mines should be present before first click"
    
    # After first click, mines should be placed
    board.reveal_cell(4, 4)
    assert board.mines_placed, "Mines should be placed after first click"
    
    # Count mines after first click
    final_mine_count = 0
    for row in range(9):
        for col in range(9):
            if board.get_cell(row, col).is_mine:
                final_mine_count += 1
    
    assert final_mine_count == 10, f"Should have 10 mines after first click, found {final_mine_count}"
