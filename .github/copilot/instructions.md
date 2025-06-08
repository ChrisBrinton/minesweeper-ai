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
- `assets/`: Contains image resources for the game
- `tests/`: Contains unit tests

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
