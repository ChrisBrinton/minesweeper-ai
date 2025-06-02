#!/usr/bin/env python3
"""
Coverage test runner for minesweeper project
Runs tests with coverage and opens HTML report
"""

import subprocess
import sys
import os
import webbrowser
from pathlib import Path


def run_coverage():
    """Run tests with coverage and generate HTML report"""
    print("🧪 Running tests with coverage...")
    print("=" * 50)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-v"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
        
        # Check if HTML report was generated
        html_report = Path("htmlcov/index.html")
        if html_report.exists():
            print(f"\n📊 Coverage report generated: {html_report.absolute()}")
            
            # Ask if user wants to open the report
            try:
                response = input("\nOpen HTML coverage report in browser? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    webbrowser.open(f"file://{html_report.absolute()}")
                    print("🌐 Coverage report opened in browser")
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Exiting...")
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ Error: Python or pytest not found")
        return False
    except Exception as e:
        print(f"❌ Error running coverage: {e}")
        return False


if __name__ == "__main__":
    success = run_coverage()
    sys.exit(0 if success else 1)
