# GitHub Copilot Instructions for Minesweeper AI Project

## Project Overview
This is a Minesweeper game implemented in Python with tkinter for the GUI. The project follows a classic Windows 3.1 style interface and includes game logic, UI components, and a leaderboard system.

## Environment Details
- **Operating System**: Windows
- **Shell**: Bash
- **Directory Format**: Use forward slashes (/) for paths when suggesting terminal commands
- **Python Version**: 3.13+

## Project Structure
- `src/game/`: Contains the game logic
  - `board.py`: Implements the game board and cell mechanics
- `src/ui/`: Contains the UI components
  - `gui.py`: Implements the graphical user interface
  - `leaderboard.py`: Handles the leaderboard functionality
- `src/ai/`: Contains AI training components
  - `trainer.py`: Deep Q-Network trainer
  - `models.py`: Neural network architectures
  - `environment.py`: Reinforcement learning environment
  - `game_api.py`: Game interface for AI
- `assets/`: Contains image resources for the game
- `tests/`: Contains unit tests
- `models/`: AI model storage with organized structure
  - `models/<difficulty>/<date>/`: Trained models organized by difficulty and date
    - Example: `models/beginner/20241212/dqn_final.pth`
    - Example: `models/intermediate/20241215/dqn_episode_5000.pth`

### AI Model Storage Convention
Models are stored in a hierarchical structure:
```
models/
├── beginner/
│   ├── 20241212_143022/
│   │   ├── dqn_final.pth
│   │   ├── dqn_episode_1000.pth
│   │   └── training_metrics.json
│   └── 20241213_091545/
│       └── ...
├── intermediate/
│   └── 20241214_162030/
│       └── ...
└── expert/
    └── 20241215_203415/
        └── ...
```

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Include docstrings for classes and methods

### Terminal Commands
When suggesting terminal commands, use Bash syntax:
```bash
# Example: Running the game
python main.py

# Example: Running tests
python -m pytest tests/ -v

# Example: Running tests with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov -v
```

### Key Features
- Classic Minesweeper gameplay
- Three difficulty levels: beginner, intermediate, expert
- Leaderboard system for tracking best times
- Custom sprites resembling Windows 3.1 style
- AI training system with Deep Q-Networks
- Parallel evaluation methods for efficient training
- Organized model storage by difficulty and date

### AI Training System
- **Primary entry point**: `train_ai.py` for all training operations
- **Evaluation methods**: Sequential, lightweight (threading), optimized (multiprocessing)
- **Training modes**: New training, resume from checkpoints, benchmark evaluation
- **Model organization**: Automatic storage in `models/<difficulty>/<date>/` structure

### Recent Updates
- Added feature to show incorrectly flagged cells when a game is lost
- Used `bad_flag_cell.png` to indicate cells that were flagged but didn't contain mines

### Testing
Tests are written using pytest. Run tests using:
```bash
python -m pytest tests/
```

### Common Tasks
- **Running the game**: `python main.py`
- **Running tests**: `python -m pytest tests/ -v`
- **Running tests with coverage**: `python -m pytest tests/ --cov=src --cov-report=term-missing -v`
- **AI training (new)**: `python train_ai.py --mode new --difficulty beginner --episodes 5000`
- **AI training (resume)**: `python train_ai.py --mode resume --eval-method lightweight`
- **Benchmark evaluation**: `python train_ai.py --mode benchmark --episodes 50`
