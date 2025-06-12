#!/usr/bin/env python3
"""
Parallel Evaluation Extension for DQN Trainer
Adds multiprocessing support to speed up evaluation phases during training
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def evaluate_episode_worker(model_state_dict, env_config, device_id, random_seed):
    """
    Worker function to evaluate a single episode in a separate process
    
    Args:
        model_state_dict: Serialized model weights
        env_config: Environment configuration
        device_id: CPU core ID for this worker
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (total_reward, won, steps)
    """
    # Set random seed for this worker
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create environment (import here to avoid pickle issues)
    from src.ai.environment import MinesweeperEnvironment
    from src.ai.models import DQN
    
    env = MinesweeperEnvironment(
        rows=env_config['rows'],
        cols=env_config['cols'], 
        mines=env_config['mines']
    )
    
    # Create model on CPU (workers use CPU to avoid GPU contention)
    device = torch.device('cpu')
    model = DQN(
        env.rows, env.cols,
        input_channels=3,
        num_actions=env.action_space_size
    ).to(device)
    
    # Load model weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Run evaluation episode
    state = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 1000  # Same as training
    
    with torch.no_grad():
        while steps < max_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(device)
            action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
            
            # Get action (no exploration - epsilon = 0)
            action = model.get_action(state_tensor, action_mask, 0.0)
            
            # Take step
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
    
    won = info.get('game_state') == 'won'
    return total_reward, won, steps

class ParallelEvaluator:
    """Parallel evaluation manager for DQN training"""
    
    def __init__(self, trainer, num_workers=None):
        """
        Initialize parallel evaluator
        
        Args:
            trainer: DQNTrainer instance
            num_workers: Number of worker processes (default: CPU cores - 1)
        """
        self.trainer = trainer
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)
        
        print(f"🚀 Parallel evaluator initialized with {self.num_workers} workers")
        
    def evaluate_parallel(self, num_episodes: int) -> Tuple[float, float]:
        """
        Run evaluation episodes in parallel
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (average_reward, win_rate)
        """
        start_time = time.time()
        
        # Prepare model state for workers
        model_state_dict = {k: v.cpu() for k, v in self.trainer.q_network.state_dict().items()}
        
        # Environment configuration
        env_config = {
            'rows': self.trainer.env.rows,
            'cols': self.trainer.env.cols,
            'mines': self.trainer.env.mines
        }
        
        # Generate random seeds for each episode
        random_seeds = np.random.randint(0, 2**31, size=num_episodes)
        
        # Run episodes in parallel
        total_rewards = []
        wins = []
        total_steps = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all episodes
            future_to_episode = {
                executor.submit(
                    evaluate_episode_worker,
                    model_state_dict,
                    env_config,
                    i % self.num_workers,  # Distribute across workers
                    random_seeds[i]
                ): i for i in range(num_episodes)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_episode):
                try:
                    reward, won, steps = future.result()
                    total_rewards.append(reward)
                    wins.append(won)
                    total_steps.append(steps)
                except Exception as e:
                    print(f"⚠️ Evaluation episode failed: {e}")
                    # Use default values for failed episodes
                    total_rewards.append(-100.0)
                    wins.append(False)
                    total_steps.append(0)
        
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        win_rate = np.mean(wins)
        avg_steps = np.mean(total_steps)
        
        eval_time = time.time() - start_time
        
        # Performance info
        episodes_per_second = num_episodes / eval_time
        speedup_estimate = episodes_per_second * self.num_workers / 10  # Rough estimate vs sequential
        
        print(f"📊 Parallel eval: {num_episodes} episodes in {eval_time:.1f}s "
              f"({episodes_per_second:.1f} eps/s, ~{speedup_estimate:.1f}x speedup)")
        
        return avg_reward, win_rate

def patch_trainer_with_parallel_eval(trainer, num_workers=None):
    """
    Patch an existing DQNTrainer to use parallel evaluation
    
    Args:
        trainer: DQNTrainer instance to patch
        num_workers: Number of worker processes
        
    Returns:
        The patched trainer (for chaining)
    """
    # Create parallel evaluator
    parallel_evaluator = ParallelEvaluator(trainer, num_workers)
    
    # Store original evaluation method
    trainer._evaluate_sequential = trainer._evaluate
    
    # Replace with parallel version
    def _evaluate_parallel():
        return parallel_evaluator.evaluate_parallel(trainer.config['eval_episodes'])
    
    trainer._evaluate = _evaluate_parallel
    trainer.parallel_evaluator = parallel_evaluator
    
    print(f"✅ DQNTrainer patched for parallel evaluation")
    
    return trainer

# Convenience function for easy integration
def enable_parallel_evaluation(trainer, num_workers=None):
    """
    Enable parallel evaluation for a DQNTrainer
    
    Args:
        trainer: DQNTrainer instance
        num_workers: Number of worker processes (default: CPU cores - 1)
        
    Returns:
        The trainer with parallel evaluation enabled
    """
    return patch_trainer_with_parallel_eval(trainer, num_workers)

# Performance testing function
def benchmark_evaluation_methods(trainer, num_episodes=100):
    """
    Benchmark sequential vs parallel evaluation performance
    
    Args:
        trainer: DQNTrainer instance
        num_episodes: Number of episodes to test
    """
    print(f"🏁 EVALUATION PERFORMANCE BENCHMARK")
    print(f"Testing with {num_episodes} episodes")
    print("=" * 50)
    
    # Test sequential evaluation
    print("Testing sequential evaluation...")
    start_time = time.time()
    seq_reward, seq_win_rate = trainer._evaluate_sequential()
    seq_time = time.time() - start_time
    
    print(f"Sequential: {seq_time:.2f}s | Reward: {seq_reward:.1f} | Win Rate: {seq_win_rate:.3f}")
    
    # Test parallel evaluation
    print("Testing parallel evaluation...")
    parallel_evaluator = ParallelEvaluator(trainer)
    start_time = time.time()
    par_reward, par_win_rate = parallel_evaluator.evaluate_parallel(num_episodes)
    par_time = time.time() - start_time
    
    print(f"Parallel:   {par_time:.2f}s | Reward: {par_reward:.1f} | Win Rate: {par_win_rate:.3f}")
    
    # Calculate speedup
    speedup = seq_time / par_time
    time_saved = seq_time - par_time
    
    print()
    print(f"🚀 PERFORMANCE RESULTS")
    print(f"Speedup: {speedup:.1f}x faster")
    print(f"Time saved: {time_saved:.1f}s ({time_saved/seq_time*100:.1f}%)")
    print(f"Workers used: {parallel_evaluator.num_workers}")
    
    if speedup > 1.5:
        print(f"✅ Significant speedup achieved!")
    else:
        print(f"⚠️ Limited speedup - consider CPU/memory constraints")

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing parallel evaluation...")
    
    # This would normally be done with an actual trainer
    print("Note: This is a standalone module. Import and use with your DQNTrainer.")
    print()
    print("Usage example:")
    print("```python")
    print("from parallel_evaluation import enable_parallel_evaluation")
    print()
    print("# Enable parallel evaluation")
    print("trainer = enable_parallel_evaluation(trainer, num_workers=4)")
    print()
    print("# Training will now use parallel evaluation automatically")
    print("trainer.train()")
    print("```")
