#!/bin/bash
# Development helper script for minesweeper project

PYTHON="C:/Users/cbrin/AppData/Local/Programs/Python/Python313/python.exe"

case $1 in
    "test")
        echo "🧪 Running tests..."
        "$PYTHON" -m pytest tests/ -v
        ;;
    "coverage")
        echo "📊 Running tests with coverage..."
        "$PYTHON" -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov -v
        echo "📖 Coverage report: htmlcov/index.html"
        ;;
    "coverage-open")
        echo "📊 Running tests with coverage and opening report..."
        "$PYTHON" -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html:htmlcov -v
        if [ -f "htmlcov/index.html" ]; then
            echo "🌐 Opening coverage report..."
            start htmlcov/index.html
        fi
        ;;
    "run")
        echo "🎮 Starting minesweeper game..."
        "$PYTHON" main.py
        ;;
    "debug")
        echo "🐛 Running simple test script..."
        "$PYTHON" test_game.py
        ;;
    "clean")
        echo "🧹 Cleaning cache and coverage files..."
        rm -rf __pycache__ src/__pycache__ src/game/__pycache__ src/ui/__pycache__ tests/__pycache__
        rm -rf htmlcov/
        rm -f .coverage
        echo "✅ Cleaned!"
        ;;
    *)
        echo "🎯 Minesweeper Development Helper"
        echo ""
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  test           - Run all tests"
        echo "  coverage       - Run tests with coverage"
        echo "  coverage-open  - Run coverage and open HTML report"
        echo "  run            - Start the game"
        echo "  debug          - Run simple test script"
        echo "  clean          - Clean cache and coverage files"
        echo ""
        ;;
esac
