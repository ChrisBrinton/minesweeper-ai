#!/usr/bin/env python3
"""
Optimized Parallel Evaluation Extension for DQN Trainer
Reduced CPU usage through better worker management and process reuse
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import threading
from queue import Queue
import psutil

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
    
    # Run evaluation episode with optimized inference
    state = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 1000  # Same as training
    
    with torch.no_grad():
        while steps < max_steps:
            # Convert state to tensor (minimize memory allocations)
            state_tensor = torch.from_numpy(state).permute(2, 0, 1).float()
            action_mask = torch.from_numpy(env.get_action_mask()).bool()
            
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

class OptimizedParallelEvaluator:
    """Optimized parallel evaluation manager for DQN training"""
    
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
        print(f"   💻 Available cores: {os.cpu_count()}")
        
    def evaluate_parallel(self, num_episodes: int) -> Tuple[float, float]:
        """
        Run evaluation episodes in parallel with CPU monitoring
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (average_reward, win_rate)
        """
        start_time = time.time()
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        # Prepare model state for workers (only copy essential weights)
        model_state_dict = {}
        for k, v in self.trainer.q_network.state_dict().items():
            model_state_dict[k] = v.cpu().clone()  # Explicit clone to reduce memory pressure
        
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
        
        # Use context manager to ensure proper cleanup
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit episodes in batches to control CPU load
            batch_size = max(1, num_episodes // 4)  # Process in 4 batches
            
            for batch_start in range(0, num_episodes, batch_size):
                batch_end = min(batch_start + batch_size, num_episodes)
                current_batch_size = batch_end - batch_start
                
                # Submit batch
                future_to_episode = {
                    executor.submit(
                        evaluate_episode_worker_optimized,
                        model_state_dict,
                        env_config,
                        i % self.num_workers,  # Distribute across workers
                        random_seeds[batch_start + i],
                        i  # Worker ID
                    ): batch_start + i for i in range(current_batch_size)
                }
                
                # Collect batch results
                for future in as_completed(future_to_episode):
                    try:
                        reward, won, steps = future.result(timeout=30)  # 30 second timeout
                        total_rewards.append(reward)
                        wins.append(won)
                        total_steps.append(steps)
                    except Exception as e:
                        print(f"⚠️ Evaluation episode failed: {e}")
                        # Use default values for failed episodes
                        total_rewards.append(-100.0)
                        wins.append(False)
                        total_steps.append(0)
                
                # Brief pause between batches to let CPU cool down
                if batch_end < num_episodes:
                    time.sleep(0.1)
        
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        win_rate = np.mean(wins)
        avg_steps = np.mean(total_steps)
        
        eval_time = time.time() - start_time
        final_cpu = psutil.cpu_percent(interval=0.1)
        
        # Performance info
        episodes_per_second = num_episodes / eval_time
        
        print(f"📊 Optimized eval: {num_episodes} episodes in {eval_time:.1f}s "
              f"({episodes_per_second:.1f} eps/s)")
        print(f"   🖥️  CPU: {initial_cpu:.1f}% → {final_cpu:.1f}% (workers: {self.num_workers})")
        
        return avg_reward, win_rate

    def benchmark_worker_counts(self, num_episodes=50):
        """Benchmark different worker counts to find optimal setting"""
        print(f"🔬 BENCHMARKING WORKER COUNTS")
        print("=" * 50)
        
        worker_counts = [1, 2, 4, 8, min(16, os.cpu_count() - 2)]
        results = {}
        
        for workers in worker_counts:
            if workers > os.cpu_count():
                continue
                
            print(f"\nTesting {workers} workers...")
            
            # Temporarily set worker count
            original_workers = self.num_workers
            self.num_workers = workers
            
            # Monitor CPU before
            initial_cpu = psutil.cpu_percent(interval=1)
            
            # Run evaluation
            start_time = time.time()
            avg_reward, win_rate = self.evaluate_parallel(num_episodes)
            eval_time = time.time() - start_time
            
            # Monitor CPU after
            final_cpu = psutil.cpu_percent(interval=1)
            avg_cpu = (initial_cpu + final_cpu) / 2
            
            results[workers] = {
                'time': eval_time,
                'eps_per_sec': num_episodes / eval_time,
                'cpu_usage': avg_cpu,
                'avg_reward': avg_reward,
                'win_rate': win_rate
            }
            
            print(f"   Time: {eval_time:.1f}s | CPU: {avg_cpu:.1f}% | "
                  f"Speed: {num_episodes/eval_time:.1f} eps/s")
            
            # Restore original worker count
            self.num_workers = original_workers
        
        # Find best configuration
        print(f"\n🏆 BENCHMARK RESULTS")
        print("-" * 30)
        
        best_speed = max(results.items(), key=lambda x: x[1]['eps_per_sec'])
        best_cpu = min(results.items(), key=lambda x: x[1]['cpu_usage'])
        
        print(f"Fastest: {best_speed[0]} workers ({best_speed[1]['eps_per_sec']:.1f} eps/s)")
        print(f"Lowest CPU: {best_cpu[0]} workers ({best_cpu[1]['cpu_usage']:.1f}% CPU)")
        
        # Recommend optimal setting
        for workers, data in results.items():
            efficiency = data['eps_per_sec'] / data['cpu_usage']  # eps per second per % CPU
            print(f"{workers} workers: {data['eps_per_sec']:.1f} eps/s, {data['cpu_usage']:.1f}% CPU, efficiency: {efficiency:.3f}")
        
        return results

def enable_optimized_parallel_evaluation(trainer, num_workers=None, cpu_limit_percent=80):
    """
    Enable optimized parallel evaluation for a DQNTrainer
    
    Args:
        trainer: DQNTrainer instance
        num_workers: Number of worker processes (default: CPU cores / 4)
        cpu_limit_percent: Maximum CPU usage percentage to target
        
    Returns:
        The trainer with optimized parallel evaluation enabled
    """
    # Create optimized parallel evaluator
    parallel_evaluator = OptimizedParallelEvaluator(trainer, num_workers, cpu_limit_percent)
    
    # Store original evaluation method
    trainer._evaluate_sequential = trainer._evaluate
    
    # Replace with optimized parallel version
    def _evaluate_parallel():
        return parallel_evaluator.evaluate_parallel(trainer.config['eval_episodes'])
    
    trainer._evaluate = _evaluate_parallel
    trainer.parallel_evaluator = parallel_evaluator
    
    print(f"✅ DQNTrainer patched for optimized parallel evaluation")
    
    return trainer

if __name__ == "__main__":
    print("🧪 Optimized Parallel Evaluation Module")
    print("Features:")
    print("- Reduced CPU usage through conservative worker counts")
    print("- CPU core affinity for better performance")
    print("- Batched processing to control load")
    print("- Process cleanup and timeout handling")
    print("- Built-in benchmarking tools")
