#!/usr/bin/env python3
"""
Minesweeper AI — v7 Phase 2: Model-Only RL Fine-Tuning

Takes the Phase 1 supervised model (trained on constraint-implied labels)
and fine-tunes with RL where the model makes ALL moves (no solver).

Reward structure: Win=+1, Lose=-1, Step=0, Invalid=-0.1
No progress scaling, no survival bonus, no cascade incentive.
Pure win-rate maximisation.

Usage:
    python train_v7_rl.py                             # uses models/v7/expert/best.pth
    python train_v7_rl.py --checkpoint best_model.pth
    python train_v7_rl.py --num-episodes 1000 --eval-every 200 --device cpu  # smoke test
"""

import argparse
import json
import os
import random
import signal
import sys
import time
import traceback
import copy
from collections import deque, namedtuple
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.environment import MinesweeperEnvironment
from src.ai.models_v4 import MinesweeperResNetV4
from src.ai.algorithmic_solver import AlgorithmicSolver


# ─── Configuration ───────────────────────────────────────────────────────────

EXPERT_CONFIG = {'rows': 16, 'cols': 30, 'mines': 99}

CONFIG = {
    'learning_rate': 1e-5,
    'batch_size': 128,
    'gamma': 0.99,
    'epsilon_start': 0.02,
    'epsilon_end': 0.005,
    'epsilon_decay_episodes': 50000,
    'target_update_freq': 2000,
    'update_freq': 4,
    'memory_size': 500000,
    'min_memory_size': 2000,
    'max_steps_per_episode': 1000,
    'grad_clip_norm': 1.0,
    'eval_every': 2000,
    'eval_episodes': 500,
    'max_episodes': 500000,
    'patience': 50000,
    'checkpoint_every': 10000,
    'reward_win': 1.0,
    'reward_lose': -1.0,
    'reward_step': 0.0,
    'reward_invalid': -0.1,
}


# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
SAVE_DIR = REPO_ROOT / 'models' / 'v7_rl' / 'expert'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATH_BEST = SAVE_DIR / 'best.pth'
PATH_LATEST = SAVE_DIR / 'latest.pth'
PATH_LOG = SAVE_DIR / 'log.txt'
PATH_CANONICAL = REPO_ROOT / 'best_model.pth'


# ─── Replay buffer ──────────────────────────────────────────────────────────

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


# ─── Atomic save ─────────────────────────────────────────────────────────────

def atomic_save_torch(path: Path, state: Dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(msg: str, logfile=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if logfile:
        try:
            with open(logfile, 'a') as f:
                f.write(line + '\n')
        except IOError:
            pass


# ─── Model-only episode ─────────────────────────────────────────────────────

def play_episode(model, env, device, epsilon, replay_buffer=None):
    """Play one game with model only (no solver). Store transitions."""
    rows = env.rows
    cols = env.cols

    state = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    max_steps = CONFIG['max_steps_per_episode']

    while not done and steps < max_steps:
        action_mask = env.get_action_mask()
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            break

        if random.random() < epsilon:
            action = random.choice(valid_indices)
        else:
            with torch.no_grad():
                st = (torch.from_numpy(state).permute(2, 0, 1)
                      .unsqueeze(0).contiguous().to(device))
                q_values = model(st).squeeze(0).reshape(-1)
                valid_mask = torch.zeros_like(q_values, dtype=torch.bool)
                valid_mask[valid_indices] = True
                q_values[~valid_mask] = float('inf')
                action = q_values.argmin().item()

        next_state, env_reward, done, info = env.step(action)
        steps += 1

        if done:
            game_state = info.get('game_state')
            if game_state == 'won':
                reward = CONFIG['reward_win']
            elif game_state == 'lost':
                reward = CONFIG['reward_lose']
            else:
                reward = CONFIG['reward_step']
        else:
            reward = CONFIG['reward_step']

        total_reward += reward

        if replay_buffer is not None:
            replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

    won = info.get('game_state') == 'won' if info else False
    return won, steps, total_reward


# ─── Training step ──────────────────────────────────────────────────────────

def train_step(model, target_model, optimizer, replay_buffer, device):
    """One DQN training step."""
    if len(replay_buffer) < CONFIG['min_memory_size']:
        return None

    batch = replay_buffer.sample(CONFIG['batch_size'])
    states = np.array([e.state for e in batch], dtype=np.float32)
    actions = np.array([e.action for e in batch], dtype=np.int64)
    rewards = np.array([e.reward for e in batch], dtype=np.float32)
    next_states = np.array([e.next_state for e in batch], dtype=np.float32)
    dones = np.array([e.done for e in batch], dtype=np.float32)

    states_t = torch.from_numpy(states).permute(0, 3, 1, 2).contiguous().to(device)
    next_states_t = torch.from_numpy(next_states).permute(0, 3, 1, 2).contiguous().to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    q_values = model(states_t)
    B = q_values.shape[0]
    rows, cols = q_values.shape[1], q_values.shape[2]
    q_flat = q_values.reshape(B, -1)
    q_selected = q_flat.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_model(next_states_t).reshape(B, -1)
        next_q_min = next_q.min(1)[0]
        target = rewards_t + CONFIG['gamma'] * next_q_min * (1.0 - dones_t)
        target = target.clamp(-10.0, 10.0)

    loss = nn.functional.smooth_l1_loss(q_selected, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
    optimizer.step()

    return loss.item()


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, device, board_cfg, num_episodes):
    """Model-only evaluation."""
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    model.eval()
    wins = 0

    for _ in range(num_episodes):
        env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                      use_v2=True, normalize_rewards=True)
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            action_mask = env.get_action_mask()
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) == 0:
                break
            with torch.no_grad():
                st = (torch.from_numpy(state).permute(2, 0, 1)
                      .unsqueeze(0).contiguous().to(device))
                q = model(st).squeeze(0).reshape(-1)
                valid_t = torch.zeros_like(q, dtype=torch.bool)
                valid_t[valid_indices] = True
                q[~valid_t] = float('inf')
                action = q.argmin().item()
            state, _, done, info = env.step(action)
            steps += 1
        if info.get('game_state') == 'won':
            wins += 1

    model.train()
    return wins / max(num_episodes, 1)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='v7 Phase 2: Model-only RL')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Phase 1 checkpoint to start from')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num-episodes', type=int, default=None)
    parser.add_argument('--eval-every', type=int, default=None)
    parser.add_argument('--eval-episodes', type=int, default=None)
    args = parser.parse_args()

    if args.num_episodes:
        CONFIG['max_episodes'] = args.num_episodes
    if args.eval_every:
        CONFIG['eval_every'] = args.eval_every
    if args.eval_episodes:
        CONFIG['eval_episodes'] = args.eval_episodes

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    log(f"Device: {device}", PATH_LOG)

    # Load Phase 1 model
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if PATH_BEST.with_name('best.pth').exists():
            ckpt_path = str(REPO_ROOT / 'models' / 'v7' / 'expert' / 'best.pth')
        elif PATH_CANONICAL.exists():
            ckpt_path = str(PATH_CANONICAL)
        else:
            raise FileNotFoundError("No Phase 1 checkpoint found. Run train_v7.py first.")

    log(f"Loading Phase 1 model from {ckpt_path}", PATH_LOG)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)

    model = MinesweeperResNetV4().to(device)
    model.load_state_dict(sd)

    target_model = MinesweeperResNetV4().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    replay_buffer = ReplayBuffer(CONFIG['memory_size'])

    best_win_rate = 0.0
    training_steps = 0
    recent_wins = deque(maxlen=100)

    # Signal handling
    stop_requested = [False]
    def handle_signal(signum, frame):
        log(f"Signal {signum}, stopping...", PATH_LOG)
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    rows, cols, mines = EXPERT_CONFIG['rows'], EXPERT_CONFIG['cols'], EXPERT_CONFIG['mines']

    for episode in range(CONFIG['max_episodes']):
        if stop_requested[0]:
            break

        epsilon = max(
            CONFIG['epsilon_end'],
            CONFIG['epsilon_start'] - (CONFIG['epsilon_start'] - CONFIG['epsilon_end'])
            * episode / CONFIG['epsilon_decay_episodes']
        )

        env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                      use_v2=True, normalize_rewards=True)
        won, steps, total_reward = play_episode(
            model, env, device, epsilon, replay_buffer
        )
        recent_wins.append(1 if won else 0)

        if len(replay_buffer) >= CONFIG['min_memory_size']:
            if episode % CONFIG['update_freq'] == 0:
                loss = train_step(model, target_model, optimizer,
                                  replay_buffer, device)
                training_steps += 1

                if training_steps % CONFIG['target_update_freq'] == 0:
                    target_model.load_state_dict(model.state_dict())

        if (episode + 1) % CONFIG['eval_every'] == 0:
            win_rate = evaluate_model(model, device, EXPERT_CONFIG,
                                      CONFIG['eval_episodes'])
            recent_wr = sum(recent_wins) / max(len(recent_wins), 1)
            log(f"Ep {episode+1}: eval={win_rate:.1%}, "
                f"recent100={recent_wr:.1%}, eps={epsilon:.4f}, "
                f"buf={len(replay_buffer)}", PATH_LOG)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                log(f"  New best: {best_win_rate:.1%}", PATH_LOG)
                atomic_save_torch(PATH_BEST, {
                    'model_state_dict': model.state_dict(),
                    'episode': episode,
                    'win_rate': win_rate,
                    'trainer': 'v7_rl',
                    'config': CONFIG,
                })
                atomic_save_torch(PATH_CANONICAL, {
                    'model_state_dict': model.state_dict(),
                    'episode': episode,
                    'win_rate': win_rate,
                    'trainer': 'v7_rl',
                })

        if (episode + 1) % CONFIG['checkpoint_every'] == 0:
            atomic_save_torch(PATH_LATEST, {
                'model_state_dict': model.state_dict(),
                'target_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'best_win_rate': best_win_rate,
                'training_steps': training_steps,
                'config': CONFIG,
            })

    log(f"Training complete. Best win rate: {best_win_rate:.1%}", PATH_LOG)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
