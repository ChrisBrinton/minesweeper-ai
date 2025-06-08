#!/usr/bin/env python3
"""
GPU-Optimized Training Script for Minesweeper AI
Uses CUDA acceleration and optimized hyperparameters for faster training
"""

import sys
import os
import argparse
import json
import torch
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai import create_trainer


def get_cuda_optimized_config():
    """Get optimized configuration for CUDA training"""
    return {
        # Training parameters optimized for GPU
        'learning_rate': 5e-4,  # Higher learning rate for GPU
        'batch_size': 128,      # Larger batch size for GPU efficiency
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.997,  # Slightly slower decay for more exploration
        
        # Network updates
        'target_update_freq': 50,  # More frequent updates with larger batches
        
        # Memory settings
        'memory_size': 50000,      # Larger replay buffer
        'min_memory_size': 5000,   # Start training sooner
        
        # Training length
        'max_episodes': 3000,
        'max_steps_per_episode': 1000,
        
        # Evaluation and saving
        'save_freq': 250,
        'eval_freq': 50,           # More frequent evaluation
        'eval_episodes': 20,       # More evaluation episodes
    }


def benchmark_training_speed():
    """Benchmark training speed on CPU vs GPU"""
    print("Running training speed benchmark...")
    
    # Test CPU training
    if torch.cuda.is_available():
        print("\nTesting GPU training speed...")
        trainer_gpu = create_trainer('beginner', get_cuda_optimized_config())
        
        import time
        start_time = time.time()
        
        # Run a few training steps to measure speed
        env = trainer_gpu.env
        state = env.reset()
        total_steps = 0
        
        for _ in range(100):  # 100 steps for benchmark
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(trainer_gpu.device)
            action_mask = torch.BoolTensor(env.get_action_mask()).to(trainer_gpu.device)
            
            action = trainer_gpu.q_network.get_action(state_tensor, action_mask, 0.5)
            state, reward, done, info = env.step(action)
            total_steps += 1
            
            if done:
                state = env.reset()
        
        gpu_time = time.time() - start_time
        print(f"GPU: {total_steps} steps in {gpu_time:.2f} seconds ({total_steps/gpu_time:.1f} steps/sec)")
        print(f"Using device: {trainer_gpu.device}")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            print(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")


def main():
    """Main training function with CUDA optimization"""
    parser = argparse.ArgumentParser(description='Train Minesweeper AI using DQN with CUDA acceleration')
    
    # Game configuration
    parser.add_argument('--difficulty', choices=['beginner', 'intermediate', 'expert'], 
                        default='beginner', help='Game difficulty level')
    parser.add_argument('--rows', type=int, help='Custom board height')
    parser.add_argument('--cols', type=int, help='Custom board width') 
    parser.add_argument('--mines', type=int, help='Custom number of mines')
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=3000, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--memory-size', type=int, default=50000, help='Replay buffer size')
    
    # GPU configuration
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU training even if GPU is available')
    parser.add_argument('--benchmark', action='store_true', help='Run training speed benchmark')
    
    # Saving and evaluation
    parser.add_argument('--save-dir', default='models_cuda', help='Directory to save models')
    parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=250, help='Model save frequency')
    
    # Resume training
    parser.add_argument('--resume', help='Path to model checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available() and not args.force_cpu:
        print(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        if args.force_cpu:
            print("💻 Using CPU (forced by --force-cpu flag)")
        else:
            print("💻 CUDA not available, using CPU")
    
    if args.benchmark:
        benchmark_training_speed()
        return
    
    # Create training configuration
    config = get_cuda_optimized_config()
    
    # Override with command line arguments
    if args.episodes:
        config['max_episodes'] = args.episodes
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.memory_size:
        config['memory_size'] = args.memory_size
    if args.eval_freq:
        config['eval_freq'] = args.eval_freq
    if args.save_freq:
        config['save_freq'] = args.save_freq
    
    # Create trainer
    if args.rows and args.cols and args.mines:
        # Custom board size
        from ai.environment import MinesweeperEnvironment
        from ai.trainer import DQNTrainer
        env = MinesweeperEnvironment(args.rows, args.cols, args.mines)
        trainer = DQNTrainer(env, config)
        difficulty_name = f"{args.rows}x{args.cols}_{args.mines}mines"
    else:
        trainer = create_trainer(args.difficulty, config)
        difficulty_name = args.difficulty
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{difficulty_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    training_config = {
        **config,
        'difficulty': difficulty_name,
        'device': str(trainer.device),
        'cuda_available': torch.cuda.is_available(),
        'timestamp': timestamp
    }
    
    if hasattr(trainer.env, 'rows'):
        training_config.update({
            'rows': trainer.env.rows,
            'cols': trainer.env.cols, 
            'mines': trainer.env.mines
        })
    
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.load_model(args.resume)
    
    # Print training info
    print("Starting training...")
    print(f"Difficulty: {difficulty_name}")
    print(f"Episodes: {config['max_episodes']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Device: {trainer.device}")
    print(f"Save directory: {save_dir}")
    
    # Start training
    try:
        metrics = trainer.train(save_dir)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Final average reward: {metrics['training_rewards'][-10:] if metrics['training_rewards'] else 'N/A'}")
        print(f"Final win rate: {metrics['training_wins'][-10:] if metrics['training_wins'] else 'N/A'}")
        
        # Save training plots
        try:
            plot_path = os.path.join(save_dir, "training_plots.png")
            trainer.plot_training_metrics(plot_path)
            print(f"Training plots saved to: {plot_path}")
        except Exception as e:
            print(f"Could not save training plots: {e}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        # Save current model
        interrupt_path = os.path.join(save_dir, "dqn_interrupted.pth")
        trainer.save_model(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
