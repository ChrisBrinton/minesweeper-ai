#!/usr/bin/env python3
"""
Training Monitor for Minesweeper AI
Monitors training progress and displays statistics
"""

import os
import json
import time
import glob
from datetime import datetime


def find_latest_training_dir(base_dir="models"):
    """Find the most recent training directory"""
    if not os.path.exists(base_dir):
        return None
    
    pattern = os.path.join(base_dir, "*_*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    
    # Sort by modification time
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]


def load_training_stats(training_dir):
    """Load training statistics from config and checkpoint files"""
    config_path = os.path.join(training_dir, "config.json")
    
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Look for checkpoint files to determine progress
    checkpoint_files = glob.glob(os.path.join(training_dir, "dqn_episode_*.pth"))
    final_model = os.path.join(training_dir, "dqn_final.pth")
    
    latest_episode = 0
    if checkpoint_files:
        episodes = [int(f.split('_episode_')[1].split('.pth')[0]) for f in checkpoint_files]
        latest_episode = max(episodes)
    
    is_complete = os.path.exists(final_model)
    
    return {
        'config': config,
        'latest_episode': latest_episode,
        'is_complete': is_complete,
        'training_dir': training_dir
    }


def display_progress(stats):
    """Display training progress"""
    if not stats:
        print("No training data found")
        return
    
    config = stats['config']
    
    print(f"\n=== Training Monitor - {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"Training Directory: {stats['training_dir']}")
    print(f"Difficulty: {config.get('difficulty', 'custom')}")
    print(f"Board Size: {config.get('rows', '?')}x{config.get('cols', '?')} with {config.get('mines', '?')} mines")
    print(f"Target Episodes: {config.get('max_episodes', '?')}")
    print(f"Latest Checkpoint: Episode {stats['latest_episode']}")
    
    if stats['is_complete']:
        print("Status: ✅ Training Complete!")
    else:
        progress = (stats['latest_episode'] / config.get('max_episodes', 1)) * 100
        print(f"Status: 🔄 Training in Progress ({progress:.1f}%)")
    
    # Look for training plots
    plot_file = os.path.join(stats['training_dir'], "training_plots.png")
    if os.path.exists(plot_file):
        print(f"Training plots available: {plot_file}")


def main():
    """Main monitoring loop"""
    print("Minesweeper AI Training Monitor")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            training_dir = find_latest_training_dir()
            
            if training_dir:
                stats = load_training_stats(training_dir)
                display_progress(stats)
            else:
                print("\nNo training sessions found in 'models' directory")
            
            print("\n" + "="*50)
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


if __name__ == "__main__":
    main()
