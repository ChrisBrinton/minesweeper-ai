# Minesweeper Game

A classic Windows 3.1 style Minesweeper game implemented in Python with tkinter. This faithful recreation includes all the original features and classic visual design.

## Features

🎮 **Classic Gameplay**
- Three difficulty levels: Beginner (9×9), Intermediate (16×16), Expert (16×30)
- First click safety (no mine on first click)
- Auto-reveal for empty cells
- Flag/unflag mines with right-click

🎨 **Authentic Windows 3.1 Style**
- Digital mine counter and timer displays
- Expressive smiley face button
- 3D button effects with raised/sunken states
- Classic color scheme and fonts
- Original number colors and mine/flag symbols

🏆 **Game Features**
- Win/lose detection with appropriate messages
- Timer that starts on first click
- Mine counter that updates with flags
- Menu system with difficulty selection
- Help and about dialogs

## Requirements

- Python 3.6 or higher
- tkinter (included with most Python installations)
- No additional dependencies required!

## Installation & Running

1. Clone or download this repository
2. Navigate to the project directory
3. Run the game:

```bash
python main.py
```

## How to Play

### Objective
Find all mines without detonating any of them.

### Controls
- **Left Click**: Reveal a cell
- **Right Click**: Flag/unflag a suspected mine
- **Smiley Button**: Start a new game

### Game Rules
- Numbers indicate how many mines are adjacent to that cell
- Use the numbers to deduce where mines are located
- Flag all suspected mines
- The first click is always safe
- Win by revealing all non-mine cells

### Difficulty Levels
- **Beginner**: 9×9 grid with 10 mines
- **Intermediate**: 16×16 grid with 40 mines  
- **Expert**: 16×30 grid with 99 mines

## Project Structure

```
minesweeper-ai/
├── main.py              # Entry point
├── requirements.txt     # Dependencies (none needed)
├── README.md           # This file
└── src/
    ├── __init__.py
    ├── game/           # Core game logic
    │   ├── __init__.py
    │   └── board.py    # GameBoard, Cell classes
    └── ui/             # User interface
        ├── __init__.py
        └── gui.py      # MinesweeperGUI class
```

## Architecture

The game follows object-oriented design principles:

- **`GameBoard`**: Manages game state, mine placement, and game logic
- **`Cell`**: Represents individual board cells with state management
- **`MinesweeperGUI`**: Handles all UI components and user interactions
- **`DigitalDisplay`**: Custom widget for mine counter and timer
- **`SmileyButton`**: Expressive smiley face that shows game state
- **`CellButton`**: Individual cell buttons with proper event handling

## AI Training Ready

This implementation provides a clean separation between game logic and UI, making it perfect for AI training scenarios. The `GameBoard` class can be used independently for:

- Reinforcement learning environments
- Game state analysis
- Algorithm testing
- Educational purposes

## Screenshots

The game faithfully recreates the classic Windows 3.1 minesweeper experience with:
- Digital displays for mine count and timer
- Expressive smiley face button
- Classic 3D button styling
- Original color scheme and typography

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.
