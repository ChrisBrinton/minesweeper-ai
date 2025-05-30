#!/usr/bin/env python3
"""
Debug script to test GUI click handling
"""

import tkinter as tk
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from game.board import GameBoard, GameState, CellState
from ui.gui import CellButton


def debug_click_callback(row, col):
    print(f"DEBUG: Left click on cell ({row}, {col})")


def debug_right_click_callback(row, col):
    print(f"DEBUG: Right click on cell ({row}, {col})")


def create_test_gui():
    """Create a simple test GUI to debug click events"""
    root = tk.Tk()
    root.title("Click Debug Test")
    
    # Create a simple 3x3 grid of test buttons
    for row in range(3):
        for col in range(3):
            button = CellButton(
                root,
                row, col,
                debug_click_callback,
                debug_right_click_callback
            )
            button.grid(row=row, column=col, padx=1, pady=1)
    
    print("Debug GUI created. Try clicking on the buttons.")
    print("Left clicks should print 'Left click on cell (row, col)'")
    print("Right clicks should print 'Right click on cell (row, col)'")
    
    root.mainloop()


if __name__ == "__main__":
    create_test_gui()
