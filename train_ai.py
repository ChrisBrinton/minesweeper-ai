"""
Main training script for Minesweeper AI
Trains a DQN agent to play Minesweeper using the API framework
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai import create_trainer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Minesweeper AI using DQN')
    
    # Game settings
    parser.add_argument('--difficulty', type=str, default='beginner',
                       choices=['beginner', 'intermediate', 'expert'],
                       help='Game difficulty level')
    parser.add_argument('--rows', type=int, help='Custom board height')
    parser.add_argument('--cols', type=int, help='Custom board width')
    parser.add_argument('--mines', type=int, help='Custom number of mines')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Exploration decay rate')
    parser.add_argument('--memory-size', type=int, default=10000,
                       help='Replay buffer size')
    parser.add_argument('--target-update-freq', type=int, default=100,
                       help='Target network update frequency')
    
    # Evaluation and saving
    parser.add_argument('--eval-freq', type=int, default=100,
                       help='Evaluation frequency')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=500,
                       help='Model save frequency')
    parser.add_argument('--save-dir', type=str, default='trained_models',
                       help='Directory to save models')
    
    # Resume training
    parser.add_argument('--resume', type=str,
                       help='Path to model checkpoint to resume from')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting training metrics')
    
    args = parser.parse_args()
    
    # Set random seed
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create training configuration
    config = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'gamma': args.gamma,
        'epsilon_start': args.epsilon_start,
        'epsilon_end': args.epsilon_end,
        'epsilon_decay': args.epsilon_decay,
        'memory_size': args.memory_size,
        'target_update_freq': args.target_update_freq,
        'max_episodes': args.episodes,
        'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes,
        'save_freq': args.save_freq
    }
    
    # Create trainer
    if args.rows and args.cols and args.mines:
        # Custom board size
        from ai.environment import MinesweeperEnvironment
        from ai.trainer import DQNTrainer
        
        env = MinesweeperEnvironment(args.rows, args.cols, args.mines)
        trainer = DQNTrainer(env, config)
        
        difficulty_str = f"{args.rows}x{args.cols}_{args.mines}mines"
    else:
        # Standard difficulty
        trainer = create_trainer(args.difficulty, config)
        difficulty_str = args.difficulty
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{difficulty_str}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_model(args.resume)
        print(f"Resumed training from: {args.resume}")
    
    # Train the agent
    try:
        print(f"Starting training...")
        print(f"Difficulty: {difficulty_str}")
        print(f"Episodes: {args.episodes}")
        print(f"Save directory: {save_dir}")
        
        metrics = trainer.train(save_dir)
        
        print(f"Training completed!")
        print(f"Models saved to: {save_dir}")
        
        # Final evaluation
        print("Running final evaluation...")
        final_reward, final_win_rate = trainer._evaluate()
        print(f"Final evaluation - Avg Reward: {final_reward:.2f}, Win Rate: {final_win_rate:.3f}")
        
        # Plot training metrics
        if not args.no_plot:
            try:
                plot_path = os.path.join(save_dir, "training_plots.png")
                trainer.plot_training_metrics(plot_path)
            except Exception as e:
                print(f"Could not save training plots: {e}")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Difficulty: {difficulty_str}")
        print(f"Episodes trained: {len(metrics['training_rewards'])}")
        print(f"Final average reward (last 100): {np.mean(metrics['training_rewards'][-100:]):.2f}")
        print(f"Final win rate (last 100): {np.mean(metrics['training_wins'][-100:]):.3f}")
        print(f"Best evaluation reward: {max(metrics['eval_rewards']) if metrics['eval_rewards'] else 'N/A'}")
        print(f"Best evaluation win rate: {max(metrics['eval_win_rates']) if metrics['eval_win_rates'] else 'N/A'}")
        print(f"Models saved to: {save_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model...")
        trainer.save_model(os.path.join(save_dir, "dqn_interrupted.pth"))
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
