#!/usr/bin/env python3
"""
Curriculum Learning Training Script for Minesweeper AI

This script implements progressive training where a single model learns
on increasingly difficult board configurations, starting from very simple
boards and working up to expert level.

Usage Examples:
    # Start new curriculum from beginning
    python train_curriculum.py --mode new
    
    # Resume curriculum from last checkpoint
    python train_curriculum.py --mode resume
    
    # Start from specific stage
    python train_curriculum.py --mode new --start-stage small
    
    # Train only a few stages
    python train_curriculum.py --mode new --max-stages 3
    
    # Custom evaluation settings
    python train_curriculum.py --mode resume --eval-method optimized --workers 8
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from src.ai.curriculum import CurriculumLearningTrainer, CurriculumConfig
from src.ai.model_storage import find_latest_model_dir


def print_curriculum_stages():
    """Print all available curriculum stages"""
    print("\n📚 Available Curriculum Stages:")
    print("=" * 60)
    
    for i, stage in enumerate(CurriculumConfig.CURRICULUM_STAGES, 1):
        board_size = stage['rows'] * stage['cols']
        mine_density = stage['mines'] / board_size * 100
        
        print(f"{i}. {stage['name'].upper()}")
        print(f"   📝 {stage['description']}")
        print(f"   🎮 Board: {stage['rows']}x{stage['cols']} ({board_size} cells)")
        print(f"   💣 Mines: {stage['mines']} ({mine_density:.1f}% density)")
        print(f"   🎯 Target: {stage['target_win_rate']:.1%} win rate")
        print(f"   📊 Episodes: {stage['min_episodes']:,} - {stage['max_episodes']:,}")
        print()


def find_curriculum_checkpoint(mode: str) -> str:
    """Find existing curriculum checkpoint directory"""
    if mode != "resume":
        return None
    
    # Look for curriculum directories in models/curriculum/
    curriculum_base_dir = os.path.join("models", "curriculum")
    if not os.path.exists(curriculum_base_dir):
        return None
    
    # Find all timestamped curriculum directories
    curriculum_dirs = []
    for d in os.listdir(curriculum_base_dir):
        full_path = os.path.join(curriculum_base_dir, d)
        if os.path.isdir(full_path):
            progress_file = os.path.join(full_path, "curriculum_progress.json")
            if os.path.exists(progress_file):
                curriculum_dirs.append((d, full_path))
    
    if not curriculum_dirs:
        return None
    
    # Get the most recent one (sort by timestamp in directory name)
    curriculum_dirs.sort(reverse=True, key=lambda x: x[0])
    latest_dir = curriculum_dirs[0][1]
    
    return latest_dir


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning Training for Minesweeper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode new                    # Start new curriculum
  %(prog)s --mode resume                 # Resume existing curriculum
  %(prog)s --mode new --start-stage tiny # Start from 'tiny' stage
  %(prog)s --mode new --max-stages 3     # Train only 3 stages
  %(prog)s --list-stages                 # Show all curriculum stages
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", 
                       choices=["new", "resume"], 
                       default="resume",
                       help="Training mode: start new curriculum or resume existing")
    
    # Curriculum options
    parser.add_argument("--start-stage", 
                       help="Starting curriculum stage (for new mode)")
    
    parser.add_argument("--max-stages", 
                       type=int,
                       help="Maximum number of stages to train")
    
    # Evaluation options
    parser.add_argument("--eval-method", 
                       choices=["sequential", "lightweight", "optimized"],
                       default="lightweight",
                       help="Evaluation method")
    
    parser.add_argument("--workers", 
                       type=int,
                       help="Number of evaluation workers (auto-detect if not specified)")
    
    # Save directory
    parser.add_argument("--save-dir",
                       help="Custom save directory")
    
    # Information
    parser.add_argument("--list-stages", 
                       action="store_true",
                       help="List all curriculum stages and exit")
    
    parser.add_argument("--status", 
                       action="store_true",
                       help="Show curriculum status and exit")
    
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Handle information requests
    if args.list_stages:
        print_curriculum_stages()
        return
    
    if args.status:
        checkpoint_dir = find_curriculum_checkpoint("resume")
        if checkpoint_dir:
            trainer = CurriculumLearningTrainer(
                save_dir=checkpoint_dir,
                evaluation_method=args.eval_method,
                num_eval_workers=args.workers
            )
            status = trainer.get_curriculum_status()
            
            print("\n📊 Curriculum Learning Status:")
            print("=" * 50)
            print(f"Current Stage: {status['current_stage']}")
            print(f"Stages Completed: {status['stages_completed']}")
            print(f"Total Episodes: {status['total_episodes']:,}")
            print(f"Progress: {status['curriculum_progress']['completed']}/{status['curriculum_progress']['total']} ({status['curriculum_progress']['percentage']:.1f}%)")
            
            if status['current_stage_config']:
                config = status['current_stage_config']
                print(f"\nCurrent Stage Details:")
                print(f"  📝 {config['description']}")
                print(f"  🎮 Board: {config['rows']}x{config['cols']} with {config['mines']} mines")
                print(f"  🎯 Target: {config['target_win_rate']:.1%}")
        else:
            print("❌ No curriculum checkpoint found")
        return
    
    # Validate start stage
    if args.start_stage and not CurriculumConfig.get_stage_config(args.start_stage):
        print(f"❌ Invalid start stage: {args.start_stage}")
        print("\nAvailable stages:")
        for stage in CurriculumConfig.CURRICULUM_STAGES:
            print(f"  - {stage['name']}")
        return
    
    print("🎓 Minesweeper AI Curriculum Learning Trainer")
    print("=" * 50)
      # Determine save directory
    save_dir = args.save_dir
    
    if args.mode == "resume":
        if not save_dir:
            save_dir = find_curriculum_checkpoint("resume")
            if not save_dir:
                print("❌ No existing curriculum found to resume")
                print("   Use --mode new to start a new curriculum")
                return
        
        print(f"🔄 Resuming curriculum from: {save_dir}")
        
    elif args.mode == "new":
        if not save_dir:
            from datetime import datetime
            from src.ai.model_storage import get_model_save_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = get_model_save_dir("curriculum", timestamp, allow_custom=True)
        
        print(f"🆕 Starting new curriculum in: {save_dir}")
        
        if args.start_stage:
            print(f"🎯 Starting from stage: {args.start_stage}")
    
    # Initialize trainer
    try:
        trainer = CurriculumLearningTrainer(
            save_dir=save_dir,
            start_stage=args.start_stage if args.mode == "new" else None,
            evaluation_method=args.eval_method,
            num_eval_workers=args.workers
        )
        
        # Show initial status
        status = trainer.get_curriculum_status()
        print(f"\n📊 Initial Status:")
        print(f"   🎯 Current stage: {status['current_stage']}")
        print(f"   ✅ Completed stages: {len(status['stages_completed'])}")
        print(f"   📈 Total episodes so far: {status['total_episodes']:,}")
        
        if args.max_stages:
            print(f"   🛑 Max stages to train: {args.max_stages}")
        
        # Run curriculum training
        print(f"\n🚀 Starting curriculum learning...")
        result = trainer.run_curriculum(max_stages=args.max_stages)
        
        # Show final results
        print(f"\n🎉 Curriculum Training Completed!")
        print(f"   📊 Stages trained: {result['stages_trained']}")
        print(f"   🎮 Total episodes: {result['total_episodes']:,}")
        print(f"   ✅ Curriculum completed: {result['curriculum_completed']}")
        
        if result['stages_completed']:
            print(f"\n📈 Stage Results:")
            for stage_result in result['stages_completed']:
                stage = stage_result['stage']
                win_rate = stage_result['final_win_rate']
                episodes = stage_result['episodes_trained']
                achieved = "✅" if stage_result['target_achieved'] else "❌"
                print(f"   {achieved} {stage}: {win_rate:.1%} win rate in {episodes:,} episodes")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print(f"   💾 Progress has been saved automatically")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
