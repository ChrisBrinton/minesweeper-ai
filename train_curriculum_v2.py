#!/usr/bin/env python3
"""
Minesweeper AI — Full Curriculum Training Runner (v2)

Trains through progressively harder boards. Each stage trains from scratch
until Phase 2 (FCN architecture) enables weight transfer between sizes.

Usage:
    PYTHONUNBUFFERED=1 python3 train_curriculum_v2.py [--device auto|mps|cpu] [--start-stage 0]

Stages:
    0: Tiny     (5×5,  3 mines) — target 60% win rate
    1: Small    (7×7,  7 mines) — target 45% win rate  
    2: Mini     (8×8,  9 mines) — target 40% win rate
    3: Beginner (9×9, 10 mines) — target 35% win rate
    4: Intermediate (16×16, 40 mines) — target 20% win rate
    5: Expert   (16×30, 99 mines) — target 10% win rate
"""

import os
import sys
import json
import time
import copy
import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.environment import MinesweeperEnvironment
from src.ai.models_v2 import MinesweeperFCN

# ─── Configuration ───────────────────────────────────────────────────────────

CURRICULUM_STAGES = [
    {
        'name': 'tiny',
        'rows': 5, 'cols': 5, 'mines': 3,
        'target_win_rate': 0.60,
        'max_episodes': 15000,
        'eval_every': 500,
        'eval_episodes': 200,
        'patience': 3000,  # episodes without improvement before advancing anyway
    },
    {
        'name': 'small', 
        'rows': 7, 'cols': 7, 'mines': 7,
        'target_win_rate': 0.45,
        'max_episodes': 30000,
        'eval_every': 500,
        'eval_episodes': 200,
        'patience': 5000,
    },
    {
        'name': 'mini',
        'rows': 8, 'cols': 8, 'mines': 9,
        'target_win_rate': 0.40,
        'max_episodes': 40000,
        'eval_every': 500,
        'eval_episodes': 200,
        'patience': 7000,
    },
    {
        'name': 'beginner',
        'rows': 9, 'cols': 9, 'mines': 10,
        'target_win_rate': 0.35,
        'max_episodes': 60000,
        'eval_every': 1000,
        'eval_episodes': 300,
        'patience': 10000,
    },
    {
        'name': 'intermediate',
        'rows': 16, 'cols': 16, 'mines': 40,
        'target_win_rate': 0.20,
        'max_episodes': 150000,
        'eval_every': 2000,
        'eval_episodes': 300,
        'patience': 20000,
    },
    {
        'name': 'expert',
        'rows': 16, 'cols': 30, 'mines': 99,
        'target_win_rate': 0.10,
        'max_episodes': 300000,
        'eval_every': 5000,
        'eval_episodes': 500,
        'patience': 40000,
    },
]

TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'batch_size': 128,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_fraction': 0.3,  # Decay over first 30% of max episodes
    'target_update_tau': 0.005,
    'update_freq': 4,
    'memory_size': 200000,
    'min_memory_size': 1000,
    'max_steps_per_episode': 500,
}

# ─── Replay Buffer ───────────────────────────────────────────────────────────

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ─── Training ────────────────────────────────────────────────────────────────

def get_device(preference='auto'):
    if preference == 'mps' or (preference == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif preference == 'cuda' or (preference == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda')
    return torch.device('cpu')


def create_env(stage):
    """Create environment for a curriculum stage."""
    env = MinesweeperEnvironment(
        rows=stage['rows'], cols=stage['cols'], mines=stage['mines'],
        use_v2=True
    )
    return env


def create_model(device, input_channels=12):
    """Create a fully convolutional DQN model (works for any board size)."""
    model = MinesweeperFCN(input_channels=input_channels).to(device)
    return model


def select_action(state, model, env, epsilon, device):
    """Epsilon-greedy action selection with masking."""
    action_mask = env.get_action_mask()
    valid_indices = np.where(action_mask)[0]
    
    if random.random() < epsilon:
        if len(valid_indices) > 0:
            return random.choice(valid_indices)
        return random.randint(0, env.rows * env.cols - 1)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        q_values = model(state_tensor).squeeze(0)  # [H, W]
        q_values = q_values.reshape(-1)  # [H*W] — flat action space
        
        valid_mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
        q_values[~valid_mask] = float('-inf')
        return q_values.argmax().item()


def train_step(model, target_model, optimizer, memory, config, device):
    """Single training step. Handles variable board sizes in replay buffer."""
    if len(memory) < config['min_memory_size']:
        return None
    
    batch = memory.sample(config['batch_size'])
    
    states = torch.FloatTensor(np.array([e.state for e in batch])).permute(0, 3, 1, 2).contiguous().to(device)
    actions = torch.LongTensor([e.action for e in batch]).to(device)
    rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
    next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).permute(0, 3, 1, 2).contiguous().to(device)
    dones = torch.BoolTensor([e.done for e in batch]).to(device)
    
    # FCN outputs [B, H, W] — flatten to [B, H*W] for action indexing
    current_q_map = model(states)  # [B, H, W]
    B = current_q_map.shape[0]
    current_q_flat = current_q_map.reshape(B, -1)  # [B, H*W]
    current_q = current_q_flat.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q-values (Double DQN)
    with torch.no_grad():
        next_q_map = model(next_states)  # [B, H, W]
        next_q_flat = next_q_map.reshape(B, -1)
        best_actions = next_q_flat.argmax(1)
        
        next_target_map = target_model(next_states)  # [B, H, W]
        next_target_flat = next_target_map.reshape(B, -1)
        next_q = next_target_flat.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        
        target_q = rewards + config['gamma'] * next_q * (~dones)
    
    loss = nn.SmoothL1Loss()(current_q, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    
    return loss.item()


def soft_update(target, source, tau):
    """Polyak averaging."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def evaluate(model, env, num_episodes, device):
    """Evaluate model win rate."""
    wins = 0
    model.eval()
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                q_values = model(state_tensor).squeeze(0)  # [H, W]
                q_values = q_values.reshape(-1)  # [H*W]
                
                action_mask = env.get_action_mask()
                valid_mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
                q_values[~valid_mask] = float('-inf')
                
                action = q_values.argmax().item()
            
            state, reward, done, info = env.step(action)
            steps += 1
        
        if info.get('game_state') == 'won':
            wins += 1
    
    model.train()
    return wins / num_episodes


def train_stage(stage, device, save_dir, log_file, prev_model=None):
    """Train a single curriculum stage. Transfers weights from prev_model if provided."""
    config = TRAINING_CONFIG.copy()
    
    env = create_env(stage)
    
    if prev_model is not None:
        # Transfer weights from previous stage (FCN works for any board size!)
        model = prev_model
    else:
        model = create_model(device)
    
    target_model = copy.deepcopy(model)
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    memory = ReplayBuffer(config['memory_size'])
    
    # Epsilon schedule
    decay_episodes = int(stage['max_episodes'] * config['epsilon_decay_fraction'])
    
    # Tracking
    best_win_rate = 0.0
    best_episode = 0
    total_steps = 0
    episode_start_time = time.time()
    stage_start_time = time.time()
    recent_losses = deque(maxlen=100)
    
    stage_dir = os.path.join(save_dir, stage['name'])
    os.makedirs(stage_dir, exist_ok=True)
    
    msg = f"\n{'='*70}\nStage: {stage['name']} ({stage['rows']}×{stage['cols']}, {stage['mines']} mines)\nTarget: {stage['target_win_rate']*100:.0f}% win rate | Max episodes: {stage['max_episodes']}\nDevice: {device} | Params: {sum(p.numel() for p in model.parameters()):,}\n{'='*70}\n"
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()
    
    for episode in range(1, stage['max_episodes'] + 1):
        # Epsilon decay (linear)
        if episode <= decay_episodes:
            epsilon = config['epsilon_start'] - (config['epsilon_start'] - config['epsilon_end']) * (episode / decay_episodes)
        else:
            epsilon = config['epsilon_end']
        
        # Run episode
        state = env.reset()
        done = False
        episode_steps = 0
        
        while not done and episode_steps < config['max_steps_per_episode']:
            action = select_action(state, model, env, epsilon, device)
            next_state, reward, done, info = env.step(action)
            
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            episode_steps += 1
            
            # Train
            if total_steps % config['update_freq'] == 0:
                loss = train_step(model, target_model, optimizer, memory, config, device)
                if loss is not None:
                    recent_losses.append(loss)
                
                # Soft target update
                soft_update(target_model, model, config['target_update_tau'])
        
        # Periodic evaluation
        if episode % stage['eval_every'] == 0:
            elapsed = time.time() - episode_start_time
            eps_per_sec = stage['eval_every'] / elapsed if elapsed > 0 else 0
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            
            win_rate = evaluate(model, env, stage['eval_episodes'], device)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_episode = episode
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'win_rate': win_rate,
                    'stage': stage['name'],
                }, os.path.join(stage_dir, 'best_model.pth'))
            
            stage_elapsed = time.time() - stage_start_time
            
            msg = (f"[{stage['name']}] Ep {episode:>6}/{stage['max_episodes']} | "
                   f"Win: {win_rate:.1%} (best: {best_win_rate:.1%} @{best_episode}) | "
                   f"Loss: {avg_loss:.5f} | Eps: {epsilon:.3f} | "
                   f"Steps: {total_steps:>8} | {eps_per_sec:.1f} eps/s | "
                   f"Elapsed: {timedelta(seconds=int(stage_elapsed))}")
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()
            
            episode_start_time = time.time()
            
            # Check if target reached
            if win_rate >= stage['target_win_rate']:
                msg = f"🎯 TARGET REACHED! {stage['name']}: {win_rate:.1%} >= {stage['target_win_rate']:.0%}"
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()
                break
            
            # Check patience
            if episode - best_episode >= stage['patience']:
                msg = f"⏸️  PATIENCE EXCEEDED. Best: {best_win_rate:.1%} @ep{best_episode}. Moving on."
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()
                break
        
        # Save checkpoint every 5000 episodes
        if episode % 5000 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'best_win_rate': best_win_rate,
                'stage': stage['name'],
            }, os.path.join(stage_dir, f'checkpoint_ep{episode}.pth'))
    
    stage_time = time.time() - stage_start_time
    msg = (f"\n{'─'*70}\n"
           f"Stage {stage['name']} COMPLETE | Best win rate: {best_win_rate:.1%} @ep{best_episode}\n"
           f"Total steps: {total_steps:,} | Time: {timedelta(seconds=int(stage_time))}\n"
           f"{'─'*70}\n")
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()
    
    return best_win_rate, best_episode, stage_time, model


def main():
    parser = argparse.ArgumentParser(description='Minesweeper AI Curriculum Training v2')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--start-stage', type=int, default=0, help='Stage index to start from (0-5)')
    parser.add_argument('--end-stage', type=int, default=None, help='Stage index to end at (inclusive)')
    args = parser.parse_args()
    
    device = get_device(args.device)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v2', 'curriculum')
    os.makedirs(save_dir, exist_ok=True)
    
    log_path = os.path.join(save_dir, 'training_log.txt')
    
    end_stage = args.end_stage if args.end_stage is not None else len(CURRICULUM_STAGES) - 1
    
    with open(log_path, 'a') as log_file:
        header = (f"\n{'='*70}\n"
                  f"CURRICULUM TRAINING START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"Device: {device}\n"
                  f"Stages: {args.start_stage} → {end_stage}\n"
                  f"{'='*70}\n")
        print(header)
        log_file.write(header + '\n')
        log_file.flush()
        
        results = {}
        curriculum_start = time.time()
        prev_model = None  # Will carry FCN weights between stages
        
        for i in range(args.start_stage, end_stage + 1):
            stage = CURRICULUM_STAGES[i]
            win_rate, best_ep, stage_time, prev_model = train_stage(
                stage, device, save_dir, log_file, prev_model=prev_model
            )
            results[stage['name']] = {
                'win_rate': win_rate,
                'best_episode': best_ep,
                'time_seconds': stage_time,
            }
            
            # Save progress
            with open(os.path.join(save_dir, 'curriculum_progress.json'), 'w') as f:
                json.dump({
                    'results': results,
                    'last_completed_stage': i,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)
        
        total_time = time.time() - curriculum_start
        summary = (f"\n{'='*70}\n"
                   f"CURRICULUM COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Total time: {timedelta(seconds=int(total_time))}\n\n")
        
        for name, r in results.items():
            summary += f"  {name:>15}: {r['win_rate']:.1%} win rate (ep {r['best_episode']}, {timedelta(seconds=int(r['time_seconds']))})\n"
        
        summary += f"\n{'='*70}\n"
        print(summary)
        log_file.write(summary + '\n')
        log_file.flush()


if __name__ == '__main__':
    main()
