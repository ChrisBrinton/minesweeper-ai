#!/usr/bin/env python3
"""
Parallel Evaluation for DQN Trainer
Provides multiple evaluation methods with different performance characteristics
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import os
import psutil

# ============================================================================
# Process-based Evaluation (Optimized)
# ============================================================================

def evaluate_episode_worker_optimized(model_state_dict, env_config, device_id, random_seed, worker_id):
    """
    Optimized worker function with reduced overhead
    
    Args:
        model_state_dict: Serialized model weights
        env_config: Environment configuration
        device_id: CPU core ID for this worker
        random_seed: Random seed for reproducibility
        worker_id: Worker identification
        
    Returns:
        Tuple of (total_reward, won, steps)
    """
    # Set process affinity to specific CPU core if possible
    try:
        p = psutil.Process()
        p.cpu_affinity([device_id % psutil.cpu_count()])
    except:
        pass  # Skip if not supported
    
    # Set random seed for this worker
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Set CPU threads for this worker to 1 to avoid oversubscription
    torch.set_num_threads(1)
    
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
    max_steps = 2000  # Match training configuration
    
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


# ============================================================================
# Thread-based Evaluation (Lightweight)
# ============================================================================

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
    max_steps = 2000  # Match training configuration
    
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


# ============================================================================
# Evaluation Classes
# ============================================================================

class OptimizedParallelEvaluator:
    """Optimized parallel evaluation manager using multiprocessing"""
    
    def __init__(self, trainer, num_workers=None, cpu_limit_percent=80):
        """
        Initialize optimized parallel evaluator
        
        Args:
            trainer: DQNTrainer instance
            num_workers: Number of worker processes (default: CPU cores / 4)
            cpu_limit_percent: Maximum CPU usage percentage to target
        """
        self.trainer = trainer
        
        # More conservative worker count to reduce CPU load
        if num_workers is None:
            # Use fewer workers by default - CPU cores / 4, but at least 2
            default_workers = max(2, os.cpu_count() // 4)
        else:
            default_workers = num_workers
            
        self.num_workers = min(default_workers, os.cpu_count() - 2)  # Leave 2 cores free
        self.cpu_limit_percent = cpu_limit_percent
        
        print(f"🚀 Optimized parallel evaluator initialized with {self.num_workers} workers")
        print(f"   🎯 Target CPU usage: {cpu_limit_percent}%")
        
    def evaluate_parallel(self, num_episodes: int) -> Tuple[float, float]:
        """
        Run evaluation episodes in parallel with optimized resource usage
        
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
        
        # Run episodes in parallel with CPU monitoring
        total_rewards = []
        wins = []
        total_steps = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all episodes
            future_to_episode = {
                executor.submit(
                    evaluate_episode_worker_optimized,
                    model_state_dict,
                    env_config,
                    i % self.num_workers,  # Distribute across workers
                    random_seeds[i],
                    i
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
        episodes_per_second = num_episodes / eval_time
        
        print(f"📊 Optimized eval: {num_episodes} episodes in {eval_time:.1f}s "
              f"({episodes_per_second:.1f} eps/s)")
        
        return avg_reward, win_rate


class LightweightParallelEvaluator:
    """Lightweight parallel evaluation using threading"""
    
    def __init__(self, trainer, num_threads=None):
        """
        Initialize lightweight evaluator
        
        Args:
            trainer: DQNTrainer instance
            num_threads: Number of threads (default: min(8, CPU cores))
        """
        self.trainer = trainer
        # Conservative thread count to avoid contention
        self.num_threads = num_threads or min(8, max(2, os.cpu_count() // 2))
        
        print(f"🧵 Lightweight evaluator initialized with {self.num_threads} threads")
        
    def evaluate_parallel(self, num_episodes: int) -> Tuple[float, float]:
        """
        Run evaluation episodes in parallel using threads
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (average_reward, win_rate)
        """
        start_time = time.time()
        
        # Use CPU for thread-based evaluation to avoid GPU contention
        device = torch.device('cpu')
        
        # Move model to CPU for evaluation
        original_device = next(self.trainer.q_network.parameters()).device
        model_cpu = self.trainer.q_network.to(device)
        model_cpu.eval()
        
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
                    model_cpu,
                    env_config,
                    random_seeds[i],
                    device
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
        
        # Restore model to original device
        self.trainer.q_network.to(original_device)
        
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        win_rate = np.mean(wins)
        avg_steps = np.mean(total_steps)
        
        eval_time = time.time() - start_time
        episodes_per_second = num_episodes / eval_time
        
        print(f"📊 Lightweight eval: {num_episodes} episodes in {eval_time:.1f}s "
              f"({episodes_per_second:.1f} eps/s)")
        
        return avg_reward, win_rate


class SequentialEvaluator:
    """Sequential evaluation with progress tracking"""
    
    def __init__(self, trainer):
        """
        Initialize sequential evaluator
        
        Args:
            trainer: DQNTrainer instance
        """
        self.trainer = trainer
        print("📈 Sequential evaluator initialized")
    
    def evaluate_sequential(self, num_episodes: int) -> Tuple[float, float]:
        """
        Run evaluation episodes sequentially with progress
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (average_reward, win_rate)
        """
        start_time = time.time()
        
        total_rewards = []
        wins = []
        
        # Temporary disable exploration
        old_epsilon = self.trainer.epsilon
        self.trainer.epsilon = 0.0
        
        print(f"🎯 Running {num_episodes} evaluation episodes...")
        
        # Initialize progress bar
        bar_width = 40  # Width of the progress bar
          # Show initial progress
        bar = '░' * bar_width
        print(f"   ⏳ [{bar}] 0/{num_episodes} (0.0%) - Starting...", end='', flush=True)
        
        for i in range(num_episodes):
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (num_episodes - i) / rate if rate > 0 else 0
                
                # Calculate progress bar
                progress = i / num_episodes
                filled_length = int(bar_width * progress)
                bar = '█' * filled_length + '░' * (bar_width - filled_length)
                
                # Update progress bar (overwrite previous line)
                print(f"\r   ⏳ [{bar}] {i}/{num_episodes} ({progress*100:.1f}%) - ETA: {eta:.1f}s", end='', flush=True)
            
            state = self.trainer.env.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < self.trainer.config['max_steps_per_episode']:
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(self.trainer.device)
                action_mask = torch.BoolTensor(self.trainer.env.get_action_mask()).to(self.trainer.device)
                
                action = self.trainer.q_network.get_action(state_tensor, action_mask, 0.0)
                state, reward, done, info = self.trainer.env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            total_rewards.append(total_reward)
            wins.append(info.get('game_state') == 'won')
          # Restore epsilon
        self.trainer.epsilon = old_epsilon
        
        # Show completed progress bar
        bar = '█' * bar_width
        print(f"\r   ⏳ [{bar}] {num_episodes}/{num_episodes} (100.0%) - Complete! ✨")
        
        avg_reward = np.mean(total_rewards)
        win_rate = np.mean(wins)
        
        eval_time = time.time() - start_time
        episodes_per_second = num_episodes / eval_time
        
        print(f"📊 Sequential eval: {num_episodes} episodes in {eval_time:.1f}s "
              f"({episodes_per_second:.1f} eps/s)")
        
        return avg_reward, win_rate


# ============================================================================
# Standalone Model Evaluation Function
# ============================================================================

def evaluate_model(model_path: str, difficulty: str = "beginner", difficulty_custom: tuple = None,
                  num_games: int = 100, method: str = "lightweight", num_workers: int = None,
                  verbose: bool = True) -> dict:
    """
    Standalone function to evaluate a trained model
    
    Args:
        model_path: Path to the saved model file
        difficulty: Standard difficulty ('beginner', 'intermediate', 'expert')
        difficulty_custom: Custom difficulty as (rows, cols, mines) tuple
        num_games: Number of games to evaluate
        method: Evaluation method ('sequential', 'lightweight', 'optimized')
        num_workers: Number of workers for parallel methods
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with evaluation results
    """
    import torch
    from src.ai.models import DQN
    from src.ai.environment import MinesweeperEnvironment
    
    # Determine board configuration
    if difficulty_custom:
        rows, cols, mines = difficulty_custom
    else:
        difficulty_configs = {
            'beginner': (9, 9, 10),
            'intermediate': (16, 16, 40),
            'expert': (16, 30, 99)
        }
        if difficulty not in difficulty_configs:
            raise ValueError(f"Unknown difficulty: {difficulty}. Use one of {list(difficulty_configs.keys())}")
        rows, cols, mines = difficulty_configs[difficulty]
    
    if verbose:
        print(f"🔍 Evaluating model: {model_path}")
        print(f"📋 Configuration: {rows}x{cols} board with {mines} mines")
        print(f"🎮 Games to play: {num_games}")
        print(f"⚙️  Method: {method}")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'q_network_state_dict' in checkpoint:
            model_state_dict = checkpoint['q_network_state_dict']
        elif 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            model_state_dict = checkpoint
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    # Create environment and model
    env = MinesweeperEnvironment(rows, cols, mines)
    model = DQN(rows, cols, input_channels=3, num_actions=env.action_space_size)
    
    try:
        model.load_state_dict(model_state_dict)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    # Set up evaluation method
    if method == "sequential":
        evaluator = SequentialEvaluator(None)  # We'll call evaluate_sequential directly
        if verbose:
            print("🔄 Running sequential evaluation...")
        
        # Run sequential evaluation
        total_reward = 0.0
        wins = 0
        
        for game in range(num_games):
            if verbose and (game + 1) % max(1, num_games // 10) == 0:
                progress = (game + 1) / num_games * 100
                print(f"   Progress: {progress:.1f}% ({game + 1}/{num_games} games)")
            # Run single game
            state = env.reset()
            episode_reward = 0.0
            steps = 0
            max_steps = 2000
            
            with torch.no_grad():
                while steps < max_steps:
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state).permute(2, 0, 1)
                    action_mask = torch.BoolTensor(env.get_action_mask())
                    
                    # Get action (no exploration)
                    action = model.get_action(state_tensor, action_mask, 0.0)
                    
                    # Take step
                    state, reward, done, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        break
            
            total_reward += episode_reward
            if info.get('game_state') == 'won':
                wins += 1
    
    elif method == "lightweight":
        if verbose:
            print("🔄 Running lightweight parallel evaluation...")
        
        # Create a mock trainer for the evaluator
        class MockTrainer:
            def __init__(self, model, env):
                self.q_network = model
                self.env = env
                self.device = 'cpu'
                self.config = {'eval_episodes': num_games}
        
        mock_trainer = MockTrainer(model, env)
        evaluator = LightweightParallelEvaluator(mock_trainer, num_threads=num_workers)
          # Run evaluation
        env_config = {'rows': rows, 'cols': cols, 'mines': mines}
        total_reward, wins = evaluator.evaluate_parallel(num_games)
    
    elif method == "optimized":
        if verbose:
            print("🔄 Running optimized parallel evaluation...")
        
        # Create a mock trainer for the evaluator
        class MockTrainer:
            def __init__(self, model, env):
                self.q_network = model
                self.env = env
                self.device = 'cpu'
                self.config = {'eval_episodes': num_games}
        
        mock_trainer = MockTrainer(model, env)
        evaluator = OptimizedParallelEvaluator(mock_trainer, num_workers=num_workers)
          # Run evaluation
        env_config = {'rows': rows, 'cols': cols, 'mines': mines}
        total_reward, wins = evaluator.evaluate_parallel(num_games)
    
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
    
    # Calculate results
    win_rate = wins / num_games
    avg_reward = total_reward / num_games
    
    results = {
        'win_rate': win_rate,
        'wins': wins,
        'games_played': num_games,
        'avg_reward': avg_reward,
        'total_reward': total_reward,
        'model_path': model_path,
        'board_config': (rows, cols, mines),
        'evaluation_method': method
    }
    
    if verbose:
        print(f"\n📊 Evaluation Results:")
        print(f"   🎯 Win Rate: {win_rate:.1%} ({wins}/{num_games})")
        print(f"   🏆 Average Reward: {avg_reward:.2f}")
        print(f"   📋 Board: {rows}x{cols} with {mines} mines")
    
    return results


# ============================================================================
# Trainer Patching Functions
# ============================================================================

def enable_optimized_parallel_evaluation(trainer, num_workers=None, cpu_limit_percent=80):
    """
    Enable optimized parallel evaluation for a DQNTrainer
    
    Args:
        trainer: DQNTrainer instance
        num_workers: Number of worker processes
        cpu_limit_percent: Maximum CPU usage percentage
        
    Returns:
        The trainer with optimized parallel evaluation enabled
    """
    # Create optimized evaluator
    evaluator = OptimizedParallelEvaluator(trainer, num_workers, cpu_limit_percent)
    
    # Store original evaluation method
    trainer._evaluate_sequential = trainer._evaluate
      # Replace with optimized parallel version
    def _evaluate_parallel(num_episodes=None):
        episodes = num_episodes or trainer.config['eval_episodes']
        return evaluator.evaluate_parallel(episodes)
    
    trainer._evaluate = _evaluate_parallel
    trainer.parallel_evaluator = evaluator
    
    print(f"✅ DQNTrainer patched for optimized parallel evaluation")
    
    return trainer


def enable_lightweight_parallel_evaluation(trainer, num_threads=None):
    """
    Enable lightweight parallel evaluation for a DQNTrainer
    
    Args:
        trainer: DQNTrainer instance
        num_threads: Number of threads
        
    Returns:
        The trainer with lightweight parallel evaluation enabled
    """
    # Create lightweight evaluator
    evaluator = LightweightParallelEvaluator(trainer, num_threads)
    
    # Store original evaluation method
    trainer._evaluate_sequential = trainer._evaluate
      # Replace with lightweight parallel version
    def _evaluate_parallel(num_episodes=None):
        episodes = num_episodes or trainer.config['eval_episodes']
        return evaluator.evaluate_parallel(episodes)
    
    trainer._evaluate = _evaluate_parallel
    trainer.parallel_evaluator = evaluator
    
    print(f"✅ DQNTrainer patched for lightweight parallel evaluation")
    
    return trainer


def enable_sequential_evaluation_with_progress(trainer):
    """
    Enable sequential evaluation with progress tracking
    
    Args:
        trainer: DQNTrainer instance
        
    Returns:
        The trainer with sequential evaluation enabled
    """
    # Create sequential evaluator
    evaluator = SequentialEvaluator(trainer)
    
    # Store original evaluation method
    trainer._evaluate_original = trainer._evaluate
      # Replace with sequential version with progress
    def _evaluate_sequential(num_episodes=None):
        episodes = num_episodes or trainer.config['eval_episodes']
        return evaluator.evaluate_sequential(episodes)
    
    trainer._evaluate = _evaluate_sequential
    trainer.sequential_evaluator = evaluator
    
    print(f"✅ DQNTrainer patched for sequential evaluation with progress")
    
    return trainer


# ============================================================================
# Performance Testing
# ============================================================================

def benchmark_evaluation_methods(trainer, num_episodes=100):
    """
    Benchmark different evaluation performance
    
    Args:
        trainer: DQNTrainer instance
        num_episodes: Number of episodes to test
    """
    print(f"🏁 EVALUATION PERFORMANCE BENCHMARK")
    print(f"Testing with {num_episodes} episodes")
    print("=" * 50)
    
    methods_to_test = [
        ("Sequential", enable_sequential_evaluation_with_progress),
        ("Lightweight", lambda t: enable_lightweight_parallel_evaluation(t, num_threads=4)),
        ("Optimized", lambda t: enable_optimized_parallel_evaluation(t, num_workers=2))
    ]
    
    results = {}
    
    for method_name, enable_func in methods_to_test:
        print(f"\nTesting {method_name} evaluation...")
        
        # Create a copy of trainer config for testing
        test_config = trainer.config.copy()
        test_config['eval_episodes'] = num_episodes
        
        # Enable the evaluation method
        test_trainer = enable_func(trainer)
        test_trainer.config['eval_episodes'] = num_episodes
        
        start_time = time.time()
        reward, win_rate = test_trainer._evaluate()
        eval_time = time.time() - start_time
        
        results[method_name] = {
            'time': eval_time,
            'reward': reward,
            'win_rate': win_rate,
            'eps_per_sec': num_episodes / eval_time
        }
        
        print(f"{method_name}: {eval_time:.2f}s | Reward: {reward:.1f} | Win Rate: {win_rate:.3f}")
    
    # Performance comparison
    print(f"\n🚀 PERFORMANCE COMPARISON")
    baseline_time = results['Sequential']['time']
    for method_name, result in results.items():
        if method_name != 'Sequential':
            speedup = baseline_time / result['time']
            print(f"{method_name}: {speedup:.1f}x speedup over Sequential")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing evaluation methods...")
    
    # This would normally be done with an actual trainer
    print("Note: This is a standalone module. Import and use with your DQNTrainer.")
    print()
    print("Usage examples:")
    print("```python")
    print("from evaluation import enable_optimized_parallel_evaluation")
    print("from evaluation import enable_lightweight_parallel_evaluation")
    print("from evaluation import enable_sequential_evaluation_with_progress")
    print()
    print("# Enable optimized parallel evaluation")
    print("trainer = enable_optimized_parallel_evaluation(trainer, num_workers=4)")
    print()
    print("# Enable lightweight parallel evaluation")
    print("trainer = enable_lightweight_parallel_evaluation(trainer, num_threads=8)")
    print()
    print("# Enable sequential evaluation with progress")
    print("trainer = enable_sequential_evaluation_with_progress(trainer)")
    print()
    print("# Training will now use the selected evaluation method")
    print("trainer.train()")
    print("```")
