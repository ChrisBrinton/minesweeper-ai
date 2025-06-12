#!/usr/bin/env python3
"""
Lightweight Parallel Evaluation using Threading
Lower CPU overhead alternative to multiprocessing for evaluation
"""

import torch
import numpy as np
import time
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

def evaluate_episode_thread(model, env_config, random_seed, device):
    """
    Thread worker function to evaluate a single episode
    Uses shared model in memory rather than process copying
    
    Args:
        model: Shared model instance (thread-safe in eval mode)
        env_config: Environment configuration
        random_seed: Random seed for reproducibility
        device: Device to use (should be CPU for threads)
        
    Returns:
        Tuple of (total_reward, won, steps)
    """
    # Set random seed for this thread
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create environment
    from src.ai.environment import MinesweeperEnvironment
    
    env = MinesweeperEnvironment(
        rows=env_config['rows'],
        cols=env_config['cols'], 
        mines=env_config['mines']
    )
    
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

class LightweightParallelEvaluator:
    """Lightweight parallel evaluation using threads instead of processes"""
    
    def __init__(self, trainer, num_threads=None):
        """
        Initialize lightweight parallel evaluator
        
        Args:
            trainer: DQNTrainer instance
            num_threads: Number of threads (default: min(8, CPU cores / 2))
        """
        self.trainer = trainer
        
        # Use conservative thread count - threads are lighter than processes
        # but still want to avoid overwhelming the system
        if num_threads is None:
            self.num_threads = min(8, max(2, os.cpu_count() // 2))
        else:
            self.num_threads = num_threads
            
        print(f"🧵 Lightweight parallel evaluator initialized with {self.num_threads} threads")
        print(f"   💡 Using threading instead of multiprocessing for lower overhead")
        
    def evaluate_parallel(self, num_episodes: int) -> Tuple[float, float]:
        """
        Run evaluation episodes in parallel using threads
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (average_reward, win_rate)
        """
        start_time = time.time()
        
        # Use the existing model (no copying needed for threads)
        model = self.trainer.q_network
        model.eval()
        device = torch.device('cpu')  # Force CPU for evaluation
        
        # Move model to CPU if it's on GPU (for thread safety)
        if next(model.parameters()).device.type == 'cuda':
            # Create a CPU copy for evaluation
            model_cpu = type(model)(
                self.trainer.env.rows, 
                self.trainer.env.cols,
                input_channels=3,
                num_actions=self.trainer.env.action_space_size
            ).to(device)
            model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            model = model_cpu
        
        # Environment configuration
        env_config = {
            'rows': self.trainer.env.rows,
            'cols': self.trainer.env.cols,
            'mines': self.trainer.env.mines
        }
        
        # Generate random seeds for each episode
        random_seeds = np.random.randint(0, 2**31, size=num_episodes)
        
        # Run episodes in parallel using threads
        total_rewards = []
        wins = []
        total_steps = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all episodes
            future_to_episode = {
                executor.submit(
                    evaluate_episode_thread,
                    model,
                    env_config,
                    random_seeds[i],
                    device
                ): i for i in range(num_episodes)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_episode):
                try:
                    reward, won, steps = future.result(timeout=15)  # 15 second timeout
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
        
        print(f"📊 Lightweight eval: {num_episodes} episodes in {eval_time:.1f}s "
              f"({episodes_per_second:.1f} eps/s, {self.num_threads} threads)")
        
        return avg_reward, win_rate

def enable_lightweight_parallel_evaluation(trainer, num_threads=None):
    """
    Enable lightweight parallel evaluation for a DQNTrainer
    
    Args:
        trainer: DQNTrainer instance
        num_threads: Number of threads (default: CPU cores / 2, max 8)
        
    Returns:
        The trainer with lightweight parallel evaluation enabled
    """
    # Create lightweight parallel evaluator
    parallel_evaluator = LightweightParallelEvaluator(trainer, num_threads)
    
    # Store original evaluation method
    trainer._evaluate_sequential = trainer._evaluate
    
    # Replace with lightweight parallel version
    def _evaluate_parallel():
        return parallel_evaluator.evaluate_parallel(trainer.config['eval_episodes'])
    
    trainer._evaluate = _evaluate_parallel
    trainer.parallel_evaluator = parallel_evaluator
    
    print(f"✅ DQNTrainer patched for lightweight parallel evaluation")
    
    return trainer

# Alternative: Sequential evaluation with progress display
def enable_sequential_evaluation_with_progress(trainer):
    """
    Use sequential evaluation but with progress display
    Lowest CPU overhead option
    """
    original_evaluate = trainer._evaluate
    
    def _evaluate_with_progress():
        """Sequential evaluation with progress updates"""
        print(f"🔄 Running {trainer.config['eval_episodes']} evaluation episodes sequentially...")
        start_time = time.time()
        
        total_rewards = []
        wins = []
        
        # Temporary disable exploration
        old_epsilon = trainer.epsilon
        trainer.epsilon = 0.0
        
        for episode in range(trainer.config['eval_episodes']):
            if episode > 0 and episode % 20 == 0:
                print(f"   📊 Episode {episode}/{trainer.config['eval_episodes']} "
                      f"({episode/trainer.config['eval_episodes']*100:.0f}%)")
            
            state = trainer.env.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < trainer.config['max_steps_per_episode']:
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(trainer.device)
                action_mask = torch.BoolTensor(trainer.env.get_action_mask()).to(trainer.device)
                
                action = trainer.q_network.get_action(state_tensor, action_mask, 0.0)
                state, reward, done, info = trainer.env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            wins.append(info.get('game_state') == 'won')
        
        # Restore epsilon
        trainer.epsilon = old_epsilon
        
        eval_time = time.time() - start_time
        avg_reward = np.mean(total_rewards)
        win_rate = np.mean(wins)
        
        print(f"📊 Sequential eval: {trainer.config['eval_episodes']} episodes in {eval_time:.1f}s "
              f"({trainer.config['eval_episodes']/eval_time:.1f} eps/s)")
        
        return avg_reward, win_rate
    
    trainer._evaluate = _evaluate_with_progress
    print("✅ DQNTrainer set to sequential evaluation with progress display")
    
    return trainer

if __name__ == "__main__":
    print("🧵 Lightweight Parallel Evaluation Module")
    print("Features:")
    print("- Uses threading instead of multiprocessing")
    print("- Shared model in memory (no copying)")
    print("- Lower CPU overhead")
    print("- Option for sequential evaluation with progress")
    print("- Good for systems with high multiprocessing overhead")
