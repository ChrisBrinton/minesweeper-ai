"""
DQN Trainer for Minesweeper AI
Implements Deep Q-Learning with experience replay and target networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any, Optional
import time
import os
import json

from .environment import MinesweeperEnvironment
from .models import DQN


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'action_mask'])


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, action_mask: np.ndarray = None):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done, action_mask)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNTrainer:
    """
    Deep Q-Network trainer for Minesweeper
    """
    
    def __init__(self, env: MinesweeperEnvironment, config: Dict[str, Any] = None):
        """
        Initialize the trainer
        
        Args:
            env: Minesweeper environment
            config: Training configuration
        """
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Default configuration
        default_config = {
            'learning_rate': 1e-4,
            'batch_size': 32,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 100,            
            'memory_size': 10000,
            'min_memory_size': 1000,
            'update_freq': 1000,  # Update network every N steps (not episodes)
            'eval_freq': 5000,  # Evaluate every N steps (not episodes)
            'max_episodes': 5000,
            'max_steps_per_episode': 2000,
            'save_freq': 5000,  # Save every N steps (not episodes)
            'eval_episodes': 100  # More episodes for better evaluation since it's less frequent
        }
        
        self.config = {**default_config, **(config or {})}        # Initialize networks
        # Detect input channels from environment observation
        sample_obs = env.reset()
        input_channels = sample_obs.shape[-1] if len(sample_obs.shape) == 3 else 3
        
        self.q_network = DQN(
            env.rows, env.cols, 
            input_channels=input_channels,
            num_actions=env.action_space_size
        ).to(self.device)
        
        self.target_network = DQN(
            env.rows, env.cols,
            input_channels=input_channels,
            num_actions=env.action_space_size
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = ReplayBuffer(self.config['memory_size'])        # Training state
        self.epsilon = self.config['epsilon_start']
        self.episode = 0
        self.total_steps = 0
        self.steps_since_update = 0  # Track steps since last network update
        self.steps_since_eval = 0    # Track steps since last evaluation
        self.steps_since_save = 0    # Track steps since last save
        self.last_eval_episode = -1  # Track episode of last evaluation
        self.last_training_report_episode = -1  # Track episode of last training report
        
        # Metrics
        self.training_rewards = []
        self.training_steps = []
        self.training_wins = []
        self.eval_rewards = []
        self.eval_win_rates = []
        self.losses = []
        
    def train(self, save_dir: str = "models") -> Dict[str, List[float]]:
        """
        Train the DQN agent
        
        Args:
            save_dir: Directory to save models and metrics
            
        Returns:
            Training metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting DQN training...")
        print(f"Episodes: {self.config['max_episodes']}")
        print(f"Max steps per episode: {self.config['max_steps_per_episode']}")
        print(f"Environment: {self.env.rows}x{self.env.cols} with {self.env.mines} mines")
        start_time = time.time()
        
        for episode in range(self.config['max_episodes']):
            self.episode = episode
            episode_reward, episode_steps, won, network_updated = self._train_episode()            # Record metrics
            self.training_rewards.append(episode_reward)
            self.training_steps.append(episode_steps)
            self.training_wins.append(won)
            
            # Decay epsilon
            self.epsilon = max(self.config['epsilon_end'], 
                             self.epsilon * self.config['epsilon_decay'])            # Update target network
            if episode % self.config['target_update_freq'] == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
              # Report training progress only when network training actually happens
            if network_updated:
                # Calculate statistics since last training report
                episodes_since_report = episode - self.last_training_report_episode
                report_start = max(0, self.last_training_report_episode + 1)
                
                if report_start < len(self.training_rewards):
                    batch_rewards = self.training_rewards[report_start:]
                    batch_steps = self.training_steps[report_start:]
                    batch_wins = self.training_wins[report_start:]
                    
                    avg_reward = np.mean(batch_rewards) if batch_rewards else 0.0
                    avg_steps = np.mean(batch_steps) if batch_steps else 0.0
                    num_wins = sum(1 for win in batch_wins if win is True)
                    num_losses = sum(1 for win in batch_wins if win is False)
                    num_incomplete = sum(1 for win in batch_wins if win is None)
                    
                    print(f"Training | Episode {episode:4d} | Steps: {self.total_steps:6d} | "
                          f"Since last report ({episodes_since_report} eps) - "
                          f"Avg Reward: {avg_reward:7.2f} | Avg Steps: {avg_steps:5.1f} | "
                          f"W/L/I: {num_wins:2d}/{num_losses:2d}/{num_incomplete:2d} | "
                          f"Epsilon: {self.epsilon:.3f}")
                else:
                    # First training report
                    print(f"Training | Episode {episode:4d} | Steps: {self.total_steps:6d} | "
                          f"First training update | Epsilon: {self.epsilon:.3f}")
                
                # Update last training report tracking
                self.last_training_report_episode = episode

            # Evaluation - based on step frequency, not network updates
            if self.steps_since_eval >= self.config['eval_freq']:
                eval_reward, eval_win_rate = self._evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_win_rates.append(eval_win_rate)
                
                # Calculate training statistics since last evaluation
                episodes_since_eval = episode - self.last_eval_episode
                eval_start = max(0, self.last_eval_episode + 1)
                
                if eval_start < len(self.training_rewards):
                    batch_rewards = self.training_rewards[eval_start:]
                    batch_steps = self.training_steps[eval_start:]
                    batch_wins = self.training_wins[eval_start:]
                      # Batch statistics since last evaluation
                    avg_reward = np.mean(batch_rewards) if batch_rewards else 0.0
                    avg_steps = np.mean(batch_steps) if batch_steps else 0.0
                    num_wins = sum(1 for win in batch_wins if win is True)
                    num_losses = sum(1 for win in batch_wins if win is False)
                    num_incomplete = sum(1 for win in batch_wins if win is None)
                    
                    print(f"Evaluation | Episode {episode:4d} | Steps: {self.total_steps:6d} | "
                          f"Since last eval ({episodes_since_eval} eps) - "
                          f"Avg Reward: {avg_reward:7.2f} | Avg Steps: {avg_steps:5.1f} | "
                          f"W/L/I: {num_wins:2d}/{num_losses:2d}/{num_incomplete:2d} | "
                          f"Eval Reward: {eval_reward:.2f} | Eval Win Rate: {eval_win_rate:.3f}")
                else:
                    print(f"Evaluation | Episode {episode:4d} | Steps: {self.total_steps:6d} | "
                          f"Eval Reward: {eval_reward:.2f} | Eval Win Rate: {eval_win_rate:.3f}")
                
                # Update last evaluation tracking
                self.last_eval_episode = episode
                self.steps_since_eval = 0
              # Save model - based on steps, not episodes
            if self.steps_since_save >= self.config['save_freq']:
                self.save_model(os.path.join(save_dir, f"dqn_step_{self.total_steps}.pth"))
                self._save_metrics(os.path.join(save_dir, "training_metrics.json"))
                self.steps_since_save = 0
        
        # Final save
        self.save_model(os.path.join(save_dir, "dqn_final.pth"))
        self._save_metrics(os.path.join(save_dir, "training_metrics.json"))
        
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        return {
            'training_rewards': self.training_rewards,
            'training_steps': self.training_steps,
            'training_wins': self.training_wins,
            'eval_rewards': self.eval_rewards,
            'eval_win_rates': self.eval_win_rates,
            'losses': self.losses
        }
    
    def _train_episode(self) -> Tuple[float, int, bool, bool]:
        """Train for one episode"""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < self.config['max_steps_per_episode']:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(self.device)
            
            # Get action mask
            action_mask = torch.BoolTensor(self.env.get_action_mask()).to(self.device)
            
            # Select action
            action = self.q_network.get_action(state_tensor, action_mask, self.epsilon)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            self.total_steps += 1            # Store experience
            self.memory.push(state, action, reward, next_state, done, 
                           self.env.get_action_mask())
            
            self.steps_since_update += 1
            self.steps_since_eval += 1
            self.steps_since_save += 1
            state = next_state
            
            if done:
                break
          # Update network only after complete episodes, enough experiences, and enough steps
        network_updated = False
        if (len(self.memory) >= self.config['min_memory_size'] and 
            self.steps_since_update >= self.config['update_freq']):
            loss = self._update_network()
            if loss is not None:
                self.losses.append(loss)
                network_updated = True
                self.steps_since_update = 0  # Reset counter after update
        
        # Check game outcome
        if done:
            # Game ended naturally (won or lost)
            won = info.get('game_state') == 'won'
        else:
            # Game ended due to max steps - incomplete game
            won = None
            
        return total_reward, steps, won, network_updated
    
    def _update_network(self) -> Optional[float]:
        """Update the Q-network using a batch of experiences"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.config['batch_size'])
        
        # Prepare batch tensors more efficiently
        states = np.array([e.state for e in batch])
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = np.array([e.next_state for e in batch])
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the current policy"""
        total_rewards = []
        wins = []
        
        # Temporary disable exploration
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        
        for _ in range(self.config['eval_episodes']):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < self.config['max_steps_per_episode']:
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).to(self.device)
                action_mask = torch.BoolTensor(self.env.get_action_mask()).to(self.device)
                
                action = self.q_network.get_action(state_tensor, action_mask, 0.0)
                state, reward, done, info = self.env.step(action)
                
                total_reward += reward
                steps += 1
                if done:
                    break
            
            total_rewards.append(total_reward)
            # Track game outcome: True for won, False for lost, None for incomplete
            if done:
                wins.append(info.get('game_state') == 'won')
            else:
                wins.append(None)  # Incomplete game
          # Restore epsilon
        self.epsilon = old_epsilon
        
        avg_reward = np.mean(total_rewards)
        # Calculate win rate only from completed games (exclude None values)
        completed_games = [w for w in wins if w is not None]
        win_rate = np.mean(completed_games) if completed_games else 0.0
        
        return avg_reward, win_rate
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode': self.episode,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.episode = checkpoint.get('episode', 0)
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        print(f"Model loaded from {filepath}")
    
    def _save_metrics(self, filepath: str):
        """Save training metrics"""
        metrics = {
            'training_rewards': self.training_rewards,
            'training_steps': self.training_steps,
            'training_wins': [bool(w) for w in self.training_wins],
            'eval_rewards': self.eval_rewards,
            'eval_win_rates': self.eval_win_rates,
            'losses': self.losses,
            'config': self.config,
            'episode': self.episode,
            'total_steps': self.total_steps
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_training_metrics(self, save_path: str = None):
        """Plot training metrics"""
        if not self.training_rewards:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Metrics')
        
        # Training rewards
        axes[0, 0].plot(self.training_rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Training steps
        axes[0, 1].plot(self.training_steps)
        axes[0, 1].set_title('Steps per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Win rate (rolling average)
        if len(self.training_wins) > 50:
            win_rate = np.convolve(self.training_wins, np.ones(50)/50, mode='valid')
            axes[1, 0].plot(win_rate)
        axes[1, 0].set_title('Win Rate (Rolling Avg)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        
        # Evaluation metrics
        if self.eval_rewards:
            x_eval = np.arange(0, len(self.eval_rewards) * self.config['eval_freq'], 
                             self.config['eval_freq'])
            axes[1, 1].plot(x_eval, self.eval_rewards, label='Avg Reward')
            axes[1, 1].plot(x_eval, self.eval_win_rates, label='Win Rate')
            axes[1, 1].set_title('Evaluation Metrics')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training plots saved to {save_path}")
        
        plt.show()


def create_trainer(difficulty: str = 'beginner', config: Dict[str, Any] = None) -> DQNTrainer:
    """
    Create a DQN trainer for a specific difficulty
    
    Args:
        difficulty: Game difficulty ('beginner', 'intermediate', 'expert')
        config: Training configuration
        
    Returns:
        Configured DQN trainer
    """
    # Difficulty settings
    difficulty_settings = {
        'beginner': (9, 9, 10),
        'intermediate': (16, 16, 40),
        'expert': (16, 30, 99)
    }
    
    if difficulty not in difficulty_settings:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    rows, cols, mines = difficulty_settings[difficulty]
    env = MinesweeperEnvironment(rows, cols, mines)
    
    return DQNTrainer(env, config)
