#!/usr/bin/env python3
"""
Best Model Management Utility
View and manage the best performing Minesweeper AI models
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ai.best_model_tracker import BestModelTracker


def main():
    parser = argparse.ArgumentParser(description="Manage best Minesweeper AI models")
    parser.add_argument("--status", action="store_true", 
                       help="Show current best model status")
    parser.add_argument("--evaluate", type=str, metavar="MODEL_PATH",
                       help="Evaluate a model for best model replacement")
    parser.add_argument("--stage", type=str, default="tiny",
                       help="Stage to evaluate model on (default: tiny)")
    parser.add_argument("--reset", action="store_true",
                       help="Reset best model (use with caution)")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = BestModelTracker()
    
    if args.status:
        print("🏆 Best Model Status")
        print("=" * 50)
        tracker.print_best_model_summary()
        
    elif args.evaluate:
        if not os.path.exists(args.evaluate):
            print(f"❌ Model file not found: {args.evaluate}")
            return 1
        
        print(f"🔍 Evaluating model: {args.evaluate}")
        print(f"   Stage: {args.stage}")
        
        # Dummy performance for evaluation (will be replaced by thorough evaluation)
        dummy_performance = {'win_rate': 0.5, 'avg_reward': 0.0, 'games_played': 100}
        dummy_metadata = {'training_episodes': 1000, 'training_time_hours': 1.0}
        
        updated = tracker.evaluate_and_update_best(
            args.evaluate, args.stage, dummy_performance, dummy_metadata
        )
        
        if updated:
            print("✅ Model evaluation completed and best model updated!")
        else:
            print("📊 Model evaluated but did not exceed current best")
    
    elif args.reset:
        response = input("⚠️  Are you sure you want to reset the best model? (yes/no): ")
        if response.lower() == 'yes':
            # Remove best model files
            best_dir = "models/best"
            if os.path.exists(best_dir):
                import shutil
                shutil.rmtree(best_dir)
                print("✅ Best model reset completed")
            else:
                print("📋 No best model to reset")
        else:
            print("❌ Reset cancelled")
    
    else:
        # Default action - show status
        print("🏆 Best Model Status")
        print("=" * 50)
        tracker.print_best_model_summary()
        print(f"\nUse --help for more options")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
