# Minesweeper Game

[![Tests & Coverage](https://github.com/ChrisBrinton/minesweeper-ai/workflows/Tests%20%26%20Coverage/badge.svg)](https://github.com/ChrisBrinton/minesweeper-ai/actions)
[![codecov](https://codecov.io/gh/ChrisBrinton/minesweeper-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/ChrisBrinton/minesweeper-ai)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## AI Training System

This implementation includes a comprehensive **Deep Q-Network (DQN) AI training system** for learning to play Minesweeper:

### 🤖 AI Training Features
- **Multiple difficulty levels**: Beginner, Intermediate, Expert
- **Curriculum Learning**: Progressive training from simple to complex boards
- **Advanced DQN architecture** with dueling networks
- **Parallel evaluation**: Multiple evaluation methods for performance
- **Resume capability**: Continue training from checkpoints
- **Comprehensive metrics**: Win rates, rewards, training plots

### 🚀 Quick Start Training

#### Traditional Training
```bash
# Primary training entry point
python train_ai.py --help

# Start new training
python train_ai.py --mode new --difficulty beginner --episodes 5000

# Resume from existing checkpoint
python train_ai.py --mode resume --eval-method lightweight

# Benchmark evaluation methods
python train_ai.py --mode benchmark --episodes 50
```

#### Curriculum Learning (Recommended)
```bash
# Start progressive curriculum from micro to expert
python train_curriculum.py --mode new

# Resume curriculum training
python train_curriculum.py --mode resume

# Check curriculum progress
python train_curriculum.py --status
```

### 📊 Evaluation Methods
- **Sequential**: Standard single-threaded evaluation with progress
- **Lightweight**: Thread-based parallel evaluation (recommended)
- **Optimized**: Process-based parallel evaluation for maximum performance

### 🎯 Training Approaches

#### Progressive Curriculum Learning
The system supports curriculum learning where a single model trains on increasingly difficult board configurations:

1. **Micro (3x3)** → **Tiny (5x5)** → **Small (7x7)** → **Mini Beginner (8x8)**
2. **Beginner (9x9)** → **Intermediate (16x16)** → **Expert (16x30)**

Each stage has target win rates and automatic advancement. **The curriculum system is now fully functional and tested.**

**Key Features:**
- ✅ **Automated progression**: Advances through stages based on win rate targets
- ✅ **Knowledge transfer**: Each stage builds on previous learning  
- ✅ **Robust checkpointing**: Resume from any interruption
- ✅ **Comprehensive monitoring**: Real-time progress and detailed logging
- ✅ **Flexible control**: Start from any stage, limit training duration

**Training time**: 8-16 hours for complete curriculum  
**Expected outcome**: Expert-level AI that can play all difficulty levels

See [CURRICULUM_GUIDE.md](CURRICULUM_GUIDE.md) for complete documentation.

#### Traditional Phase-Based Training
For single-difficulty training, the system uses a phased approach:
1. **Foundation**: Extended learning with stable parameters
2. **Stabilization**: Gradual parameter adjustment
3. **Mastery**: Fine-tuning with preserved knowledge

### 🔧 Core Components
- **`MinesweeperEnvironment`**: RL environment wrapper
- **`DQN`**: Deep Q-Network with CNN backbone
- **`DQNTrainer`**: Training orchestration with experience replay
- **`CurriculumLearningTrainer`**: Progressive difficulty training
- **`evaluation.py`**: Parallel evaluation system

## Testing & Coverage

The project includes comprehensive testing with **99.4%** coverage of game logic:

### Test Suite
- **52 unit tests** covering all game functionality
- **Integration tests** for complete game scenarios
- **Edge case testing** for robust behavior
- **pytest** with coverage reporting

### Running Tests
```bash
# Basic tests
python -m pytest tests/ -v

# Tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Using VS Code
# Use "Run Tests with Coverage" launch configuration
```

### Coverage Reports
- **Terminal**: Shows coverage percentages and missing lines
- **HTML**: Interactive report in `htmlcov/index.html`
- **Current Coverage**: 99.4% game logic, 34% overall (GUI excluded)

See `COVERAGE.md` for detailed coverage setup and usage instructions.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.
