# Test Coverage Setup

## 📊 Coverage Status

Your minesweeper project now has comprehensive test coverage setup!

### Current Coverage Results
- **Game Logic** (`src/game/board.py`): **99%** ✅
- **UI Code** (`src/ui/gui.py`): **0%** (GUI not tested automatically)
- **Overall Coverage**: **34%**

### Files with Coverage Configuration

#### 1. `.coveragerc` - Coverage settings
- Source directory: `src/`
- Excludes test files and debugging scripts
- HTML report in `htmlcov/` directory

#### 2. `pytest.ini` - Pytest configuration
- Test discovery settings
- Coverage integration
- Report formatting

#### 3. VS Code Launch Configurations
- **"Run Tests with Coverage"** - Full coverage with HTML report
- **"Debug Tests with Coverage"** - Debug mode with coverage
- **"Run Tests"** - Quick tests without coverage

#### 4. VS Code Tasks
- **Run Tests with Coverage** - Generate coverage reports
- **Open Coverage Report** - Open HTML report in browser
- **Run Tests** - Basic test execution
- **Run Game** - Launch the minesweeper game

## 🚀 How to Use Coverage

### Method 1: VS Code Launch Configurations
1. Open **Run and Debug** panel (Ctrl+Shift+D)
2. Select **"Run Tests with Coverage"**
3. Click the play button
4. Coverage report will be generated in `htmlcov/`

### Method 2: VS Code Tasks
1. Press **Ctrl+Shift+P**
2. Type "Tasks: Run Task"
3. Select **"Run Tests with Coverage"**

### Method 3: Terminal Commands
```bash
# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov -v

# Open coverage report
start htmlcov/index.html
```

### Method 4: Helper Scripts
```bash
# Use the development helper script
./dev.sh coverage        # Run coverage
./dev.sh coverage-open   # Run coverage and open report
```

## 📈 Coverage Reports

### Terminal Report
Shows coverage percentages and missing lines directly in the terminal.

### HTML Report
- Interactive coverage report in `htmlcov/index.html`
- Line-by-line coverage highlighting
- Function and class coverage details
- Branch coverage information

## 🎯 Coverage Goals

### High Coverage Areas ✅
- **Game Logic**: 99% coverage (excellent!)
- **Core Functionality**: Well tested

### Areas to Consider
- **GUI Testing**: Consider adding GUI automation tests
- **Integration Tests**: Good coverage of component interaction
- **Edge Cases**: Line 122 in board.py (mine placement collision)

## 🔧 Improving Coverage

To get the missing 1% in `board.py` line 122:
```python
# This line is in the mine placement loop - rare collision case
if (row, col) in safe_positions or self.board[row][col].is_mine:
    continue  # <- This line (122) needs a test with mine collision
```

Consider adding a test that forces mine placement collisions.

## 📁 Generated Files

The following files are automatically generated (already in `.gitignore`):
- `htmlcov/` - HTML coverage reports
- `.coverage` - Coverage data file
- `*/__pycache__/` - Python cache files

## ✨ Features Added

1. **Coverage Integration** - Full pytest-cov integration
2. **Multiple Report Formats** - Terminal and HTML reports
3. **VS Code Integration** - Launch configs and tasks
4. **Automation Scripts** - Helper scripts for easy execution
5. **Configuration Files** - Proper coverage settings and exclusions
6. **Documentation** - This comprehensive guide

Your minesweeper project now has professional-grade test coverage setup! 🎉
