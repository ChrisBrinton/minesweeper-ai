"""
Minesweeper Game - Main Entry Point
Classic Windows 3.1 style minesweeper game
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui.gui import MinesweeperGUI


def main():
    """Main entry point for the minesweeper game"""
    try:
        # Create and run the game
        game = MinesweeperGUI()
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running minesweeper: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
