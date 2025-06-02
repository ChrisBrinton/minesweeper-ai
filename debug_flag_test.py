#!/usr/bin/env python3
"""Debug script to test flagging functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from game.board import GameBoard, GameState, CellState

def debug_flag_test():
    """Debug the flag issue"""
    print("Creating board...")
    board = GameBoard(9, 9, 10)  # Beginner
    
    print(f"Initial state: {board.game_state}")
    print(f"Initial flags_used: {board.flags_used}")
    print(f"Mines placed: {board.mines_placed}")
    
    # First click should start the game
    print("\nClicking center cell...")
    board.reveal_cell(4, 4)  # Click center
    
    print(f"After first click - state: {board.game_state}")
    print(f"After first click - flags_used: {board.flags_used}")
    print(f"After first click - mines_placed: {board.mines_placed}")
    
    # Check cell (0,0) before flagging
    cell_00 = board.get_cell(0, 0)
    print(f"\nCell (0,0) state before flag: {cell_00.state}")
    print(f"Cell (0,0) is_mine: {cell_00.is_mine}")
    
    # Try to flag (0,0)
    print(f"\nFlagging (0,0)...")
    print(f"Game state check: {board.game_state in [GameState.WON, GameState.LOST]}")
    board.toggle_flag(0, 0)
    
    cell_00_after = board.get_cell(0, 0)
    print(f"Cell (0,0) state after flag: {cell_00_after.state}")
    print(f"Flags used after flagging (0,0): {board.flags_used}")
    
    # Try to flag (0,1)
    print(f"\nFlagging (0,1)...")
    board.toggle_flag(0, 1)
    
    cell_01_after = board.get_cell(0, 1)
    print(f"Cell (0,1) state after flag: {cell_01_after.state}")
    print(f"Flags used after flagging (0,1): {board.flags_used}")

if __name__ == "__main__":
    debug_flag_test()
