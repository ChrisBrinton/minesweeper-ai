#!/usr/bin/env python3
"""
Simple demo to test GUI click functionality
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui import MinesweeperGUI


def test_gui_functionality():
    """Test that the GUI starts and basic functionality works"""
    print("Starting Minesweeper GUI test...")
    print("=" * 40)
    print("Instructions:")
    print("1. Left-click on cells to reveal them")
    print("2. Right-click on cells to flag/unflag them")
    print("3. Try clicking on different cells to test auto-reveal")
    print("4. Check that numbers appear correctly")
    print("5. Close the window when done testing")
    print("=" * 40)
    
    try:
        game = MinesweeperGUI()
        game.run()
        print("GUI test completed successfully!")
    except Exception as e:
        print(f"GUI test failed with error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    test_gui_functionality()
