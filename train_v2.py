#!/usr/bin/env python3
"""
Phase 1 Training Entry Point — Fixed training harness for Minesweeper AI

Uses:
- 12-channel one-hot state representation
- Reveal-only action space (no flag/unflag)
- Sparse rewards (win=1, lose=-1, reveal=0.01, invalid=-0.1)
- MPS device support
- Huber loss (SmoothL1)
- Soft target updates (tau=0.005)
- Update every 4 steps (not 1000)
- Auto-reveal on reset
"""

import argparse
import os
import sys
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.ai.environment import MinesweeperEnvironment
from src.ai.trainer import DQNTrainer


DIFFICULTIES = {
    'tiny':     (5, 5, 3),
    'small':    (7, 7, 7),
    'beginner': (9, 9, 10),
}


def main():
    parser = argparse.ArgumentParser(description='Minesweeper AI Training v2')
    parser.add_argument('--difficulty', type=str, default='tiny',
                        choices=list(DIFFICULTIES.keys()),
                        help='Board difficulty')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes to train')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--save-dir', type=str, default='models_v2',
                        help='Directory to save models')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    args = parser.parse_args()

    rows, cols, mines = DIFFICULTIES[args.difficulty]
    print(f"=== Minesweeper AI Training v2 ===")
    print(f"Difficulty: {args.difficulty} ({rows}x{cols}, {mines} mines)")
    print(f"Episodes: {args.episodes}")
    print()

    # Create v2 environment
    env = MinesweeperEnvironment(rows, cols, mines, use_v2=True)

    # Resolve device
    device = None
    if args.device == 'auto':
        device = None  # Let trainer auto-detect
    else:
        device = args.device

    # Training config
    config = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.999,
        'target_update_freq': 100,
        'memory_size': 50000,
        'min_memory_size': 500,
        'update_freq': 4,
        'eval_freq': 100000,  # We do our own eval below
        'max_episodes': args.episodes,
        'max_steps_per_episode': rows * cols * 2,
        'save_freq': 100000,  # We do our own saving below
        'eval_episodes': 50,
        'tau': 0.005,
        'use_soft_update': True,
    }
    if device:
        config['device'] = device

    # Create trainer
    trainer = DQNTrainer(env, config)
    
    print(f"Device: {trainer.device}")
    print(f"Action space: {env.action_space_size} (reveal only)")
    print(f"Observation shape: {env.observation_space_shape}")
    print(f"Network params: {sum(p.numel() for p in trainer.q_network.parameters()):,}")
    print()

    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop with custom progress reporting
    start_time = time.time()
    report_interval = 100
    save_interval = 1000
    
    wins_window = []
    loss_window = []

    for episode in range(args.episodes):
        ep_reward, ep_steps, won, network_updated = trainer._train_episode()

        # Record metrics
        trainer.training_rewards.append(ep_reward)
        trainer.training_steps.append(ep_steps)
        trainer.training_wins.append(won)
        trainer.episode = episode

        # Decay epsilon
        trainer.epsilon = max(trainer.config['epsilon_end'],
                              trainer.epsilon * trainer.config['epsilon_decay'])

        # Track wins for reporting
        if won is not None:
            wins_window.append(1.0 if won else 0.0)
        if len(wins_window) > 200:
            wins_window = wins_window[-200:]

        # Track loss
        if trainer.losses:
            loss_window.append(trainer.losses[-1])
        if len(loss_window) > 200:
            loss_window = loss_window[-200:]

        # Progress report
        if (episode + 1) % report_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
            avg_win = np.mean(wins_window) if wins_window else 0.0
            avg_loss = np.mean(loss_window[-50:]) if loss_window else 0.0
            print(f"Episode {episode+1:5d}/{args.episodes} | "
                  f"Win rate: {avg_win:.3f} | "
                  f"Loss: {avg_loss:.5f} | "
                  f"Epsilon: {trainer.epsilon:.3f} | "
                  f"Steps: {trainer.total_steps:7d} | "
                  f"{eps_per_sec:.1f} eps/s")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            path = os.path.join(args.save_dir, f"checkpoint_ep{episode+1}.pth")
            trainer.save_model(path)

    # Final evaluation
    print("\n=== Final Evaluation (100 games, greedy) ===")
    avg_reward, win_rate = trainer._evaluate(num_episodes=100)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Win rate: {win_rate:.3f}")

    # Final save
    trainer.save_model(os.path.join(args.save_dir, "final_model.pth"))
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({args.episodes/elapsed:.1f} eps/s)")


if __name__ == '__main__':
    main()
