#!/usr/bin/env python3
"""
Minesweeper AI — Hybrid Trainer v6

Fixes from v5 failure analysis (ANALYSIS.md):
  1. Process resilience: signal handlers + try/except + proper log flushing
  2. Epsilon near zero: 0.05->0.01 (don't destroy pre-trained policy)
  3. Fixed reward asymmetry: survival 0.3-1.0, death -0.5, win 1.0
  4. Gamma=0: each guess is independent (contextual bandit)
  5. MPS memory management: periodic cache clearing
  6. Larger replay buffer: 500K to avoid flushing good experiences

Usage:
    nohup python3 -u train_hybrid_v6.py > hybrid_v6_output.log 2>&1 &
    nohup python3 -u train_hybrid_v6.py --checkpoint models_v3/hybrid_v6/expert/checkpoint_ep10000.pth > hybrid_v6_output.log 2>&1 &
    python3 -u train_hybrid_v6.py --device cpu
"""

import os
import sys
import time
import copy
import signal
import traceback
import argparse
from datetime import datetime, timedelta
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.environment import MinesweeperEnvironment
from src.ai.models_v3 import MinesweeperResNet
from src.ai.algorithmic_solver import AlgorithmicSolver

# ─── Configuration ───────────────────────────────────────────────────────────

EXPERT_CONFIG = {
    'rows': 16,
    'cols': 30,
    'mines': 99,
}

TRAINING_CONFIG = {
    'learning_rate': 5e-5,
    'batch_size': 128,
    'gamma': 0.0,                     # Fix 4: no temporal dependency between guesses
    'epsilon_start': 0.05,            # Fix 2: don't destroy pre-trained policy
    'epsilon_end': 0.01,              # Fix 2
    'epsilon_decay_episodes': 50000,  # Fix 2: shorter decay
    'target_update_freq': 1000,       # Hard update every N training steps
    'update_freq': 4,                 # Train every N guess transitions
    'memory_size': 500000,            # Fix 6: larger buffer to keep good experiences
    'min_memory_size': 1000,
    'max_steps_per_episode': 1000,
    'reward_clip': (-1.0, 1.0),
    'target_q_clip': (-10.0, 10.0),
    'grad_clip_norm': 1.0,
    'eval_every': 2000,
    'eval_episodes': 500,
    'max_episodes': 500000,
    'patience': 50000,
    'checkpoint_every': 10000,
    'benchmark_games': 500,
    'mps_cache_clear_every': 100,     # Fix 5: clear MPS cache periodically
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


# ─── Hybrid Agent ────────────────────────────────────────────────────────────

class HybridAgent:
    """Wraps solver + neural net. Solver handles deterministic moves,
    NN handles probabilistic guesses."""

    def __init__(self, model, device, rows, cols, mines):
        self.model = model
        self.device = device
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.solver = AlgorithmicSolver(rows, cols, mines)

    def choose_action(self, state, env, epsilon=0.0):
        """
        1. Run solver to find deterministic moves
        2. If safe moves found: return (action, is_guess=False)
        3. If no safe moves: use NN (epsilon-greedy), return (action, is_guess=True)
        """
        hidden, flagged, revealed = self.solver._parse_state(state)
        safe, mines = self.solver._find_deterministic_moves(hidden, flagged, revealed)

        if safe:
            best = max(safe, key=lambda cell: self.solver._score_random_cell(
                cell[0], cell[1], revealed))
            r, c = best
            action = r * self.cols + c
            return action, False, mines

        action = self._nn_action(state, env, epsilon, mines)
        return action, True, mines

    def _nn_action(self, state, env, epsilon, known_mines):
        """Epsilon-greedy NN action, avoiding known mines."""
        action_mask = env.get_action_mask()

        for (r, c) in known_mines:
            idx = r * self.cols + c
            action_mask[idx] = False

        valid_indices = np.where(action_mask)[0]

        if len(valid_indices) == 0:
            valid_indices = np.where(env.get_action_mask())[0]
            if len(valid_indices) == 0:
                return 0

        if random.random() < epsilon:
            return random.choice(valid_indices)

        with torch.no_grad():
            state_tensor = (torch.FloatTensor(state)
                            .permute(2, 0, 1).unsqueeze(0).contiguous()
                            .to(self.device))
            q_values = self.model(state_tensor).squeeze(0).reshape(-1)

            valid_mask = torch.zeros_like(q_values, dtype=torch.bool)
            valid_mask[valid_indices] = True
            q_values[~valid_mask] = float('-inf')
            return q_values.argmax().item()

    def play_episode(self, env, epsilon, replay_buffer=None):
        """
        Play one full game. Only store transitions where is_guess=True.
        Fix 3: Reward asymmetry fixed — survival=0.3+0.7*progress, death=-0.5, win=1.0
        """
        state = env.reset()
        done = False
        steps = 0
        num_guesses = 0
        num_solver_moves = 0
        guess_survivals = 0
        guess_deaths = 0
        max_steps = TRAINING_CONFIG['max_steps_per_episode']

        pending_guess_state = None
        pending_guess_action = None

        safe_cells = self.rows * self.cols - self.mines

        while not done and steps < max_steps:
            action, is_guess, known_mines = self.choose_action(state, env, epsilon)

            # If we had a pending guess and solver is now making moves,
            # the guess survived — store with positive reward
            if pending_guess_state is not None and not is_guess:
                # Fix 3: meaningful survival reward based on game progress
                progress = env.api.game_board.cells_revealed / safe_cells
                guess_reward = 0.3 + 0.7 * progress  # range 0.3 to 1.0
                if replay_buffer is not None:
                    replay_buffer.push(pending_guess_state, pending_guess_action,
                                       guess_reward, state, False)
                guess_survivals += 1
                pending_guess_state = None
                pending_guess_action = None

            if is_guess:
                pending_guess_state = state.copy()
                pending_guess_action = action
                num_guesses += 1
            else:
                num_solver_moves += 1

            next_state, reward, done, info = env.step(action)
            steps += 1

            if is_guess and done:
                game_state = info.get('game_state')
                if game_state == 'won':
                    guess_reward = 1.0
                    guess_survivals += 1
                elif game_state == 'lost':
                    guess_reward = -0.5  # Fix 3: reduced death penalty
                    guess_deaths += 1
                else:
                    guess_reward = 0.0

                if replay_buffer is not None:
                    replay_buffer.push(pending_guess_state, pending_guess_action,
                                       guess_reward, next_state, True)
                pending_guess_state = None
                pending_guess_action = None

            elif not is_guess and done:
                if pending_guess_state is not None:
                    game_state = info.get('game_state')
                    if game_state == 'won':
                        guess_reward = 1.0
                        guess_survivals += 1
                    else:
                        progress = env.api.game_board.cells_revealed / safe_cells
                        guess_reward = 0.3 + 0.7 * progress
                    if replay_buffer is not None:
                        replay_buffer.push(pending_guess_state, pending_guess_action,
                                           guess_reward, next_state, True)
                    pending_guess_state = None

            state = next_state

        # Handle any remaining pending guess (game hit max steps)
        if pending_guess_state is not None:
            progress = env.api.game_board.cells_revealed / safe_cells
            guess_reward = 0.3 + 0.7 * progress
            if replay_buffer is not None:
                replay_buffer.push(pending_guess_state, pending_guess_action,
                                   guess_reward, state, True)
            guess_survivals += 1

        won = info.get('game_state') == 'won'
        total_moves = num_guesses + num_solver_moves
        return won, num_guesses, num_solver_moves, total_moves, guess_survivals, guess_deaths


# ─── Training Functions ──────────────────────────────────────────────────────

def get_device(preference='auto'):
    if preference == 'mps' or (preference == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif preference == 'cuda' or (preference == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda')
    return torch.device('cpu')


def train_step(model, target_model, optimizer, memory, config, device):
    """Single DQN training step on guess transitions."""
    if len(memory) < config['min_memory_size']:
        return None

    batch = memory.sample(config['batch_size'])

    states = (torch.FloatTensor(np.array([e.state for e in batch]))
              .permute(0, 3, 1, 2).contiguous().to(device))
    actions = torch.LongTensor([e.action for e in batch]).to(device)
    rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
    next_states = (torch.FloatTensor(np.array([e.next_state for e in batch]))
                   .permute(0, 3, 1, 2).contiguous().to(device))
    dones = torch.BoolTensor([e.done for e in batch]).to(device)

    # Clip rewards
    r_lo, r_hi = config['reward_clip']
    rewards = rewards.clamp(r_lo, r_hi)

    # Current Q-values
    current_q_map = model(states)  # [B, H, W]
    B = current_q_map.shape[0]
    current_q_flat = current_q_map.reshape(B, -1)  # [B, H*W]
    current_q = current_q_flat.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values
    # Fix 4: with gamma=0, target is just the reward (no bootstrapping)
    with torch.no_grad():
        if config['gamma'] > 0:
            # Double DQN (kept for flexibility, but gamma=0 by default)
            next_q_map = model(next_states)
            next_q_flat = next_q_map.reshape(B, -1)
            best_actions = next_q_flat.argmax(1)

            next_target_map = target_model(next_states)
            next_target_flat = next_target_map.reshape(B, -1)
            next_q = next_target_flat.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + config['gamma'] * next_q * (~dones)
        else:
            target_q = rewards

        tq_lo, tq_hi = config['target_q_clip']
        target_q = target_q.clamp(tq_lo, tq_hi)

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
    optimizer.step()

    return loss.item()


def benchmark_solver_only(rows, cols, mines, num_games):
    """Benchmark the solver alone (no NN) on the given board size."""
    solver = AlgorithmicSolver(rows, cols, mines)
    wins = 0
    total_guesses = 0
    total_solver_moves = 0

    for _ in range(num_games):
        env = MinesweeperEnvironment(
            rows=rows, cols=cols, mines=mines,
            use_v2=True, normalize_rewards=True
        )
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 1000:
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)

            if safe:
                best = max(safe, key=lambda cell: solver._score_random_cell(
                    cell[0], cell[1], revealed))
                r, c = best
                action = r * cols + c
                total_solver_moves += 1
            else:
                hidden_list = list(hidden - known_mines)
                if not hidden_list:
                    hidden_list = list(hidden)
                if not hidden_list:
                    break
                scores = [solver._score_random_cell(r, c, revealed)
                          for r, c in hidden_list]
                min_score = min(scores)
                candidates = [cell for cell, s in zip(hidden_list, scores)
                              if s == min_score]
                r, c = random.choice(candidates)
                action = r * cols + c
                total_guesses += 1

            state, reward, done, info = env.step(action)
            steps += 1

        if info.get('game_state') == 'won':
            wins += 1

    win_rate = wins / num_games
    avg_guesses = total_guesses / num_games
    avg_solver = total_solver_moves / num_games
    return win_rate, avg_guesses, avg_solver


def benchmark_hybrid(agent, rows, cols, mines, num_games):
    """Benchmark the hybrid agent (solver + NN, epsilon=0)."""
    wins = 0
    total_guesses = 0
    total_solver_moves = 0
    total_guess_survivals = 0
    total_guess_deaths = 0

    agent.model.eval()
    for _ in range(num_games):
        env = MinesweeperEnvironment(
            rows=rows, cols=cols, mines=mines,
            use_v2=True, normalize_rewards=True
        )
        won, ng, ns, tm, gs, gd = agent.play_episode(env, epsilon=0.0)
        if won:
            wins += 1
        total_guesses += ng
        total_solver_moves += ns
        total_guess_survivals += gs
        total_guess_deaths += gd

    agent.model.train()
    win_rate = wins / num_games
    avg_guesses = total_guesses / num_games
    avg_solver = total_solver_moves / num_games
    total_guess_outcomes = total_guess_survivals + total_guess_deaths
    guess_survival_rate = (total_guess_survivals / total_guess_outcomes
                           if total_guess_outcomes > 0 else 0.0)
    return win_rate, avg_guesses, avg_solver, guess_survival_rate


def evaluate_hybrid(agent, rows, cols, mines, num_episodes):
    """Evaluate hybrid agent. Returns dict of metrics."""
    wins = 0
    total_guesses = 0
    total_solver_moves = 0
    total_moves = 0
    total_guess_survivals = 0
    total_guess_deaths = 0

    agent.model.eval()
    for _ in range(num_episodes):
        env = MinesweeperEnvironment(
            rows=rows, cols=cols, mines=mines,
            use_v2=True, normalize_rewards=True
        )
        won, ng, ns, tm, gs, gd = agent.play_episode(env, epsilon=0.0)
        if won:
            wins += 1
        total_guesses += ng
        total_solver_moves += ns
        total_moves += tm
        total_guess_survivals += gs
        total_guess_deaths += gd

    agent.model.train()
    total_guess_outcomes = total_guess_survivals + total_guess_deaths
    return {
        'win_rate': wins / num_episodes,
        'avg_guesses': total_guesses / num_episodes,
        'avg_solver_moves': total_solver_moves / num_episodes,
        'avg_total_moves': total_moves / num_episodes,
        'guess_ratio': total_guesses / max(total_moves, 1),
        'guess_survival_rate': (total_guess_survivals / total_guess_outcomes
                                 if total_guess_outcomes > 0 else 0.0),
    }


# ─── Main Training Loop ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Minesweeper Hybrid Trainer v6')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from scratch (no pre-trained weights)')
    args = parser.parse_args()

    device = get_device(args.device)
    config = TRAINING_CONFIG.copy()
    board = EXPERT_CONFIG

    # Default pre-trained model (v5's best) — used unless --checkpoint or --from-scratch
    pretrained_path = 'models_v3/hybrid/expert/best_model.pth'

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'models_v3', 'hybrid_v6', 'expert')
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'training_log.txt')

    # Create model
    model = MinesweeperResNet(input_channels=12).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    start_episode = 0
    best_win_rate = 0.0
    best_episode = 0
    total_train_steps = 0

    if args.checkpoint and os.path.exists(args.checkpoint):
        # Resume from a v6 training checkpoint
        print(f"Resuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_episode = ckpt.get('episode', 0)
        best_win_rate = ckpt.get('best_win_rate', 0.0)
        best_episode = ckpt.get('best_episode', 0)
        total_train_steps = ckpt.get('total_train_steps', 0)
        print(f"  Resumed: episode={start_episode}, best_win_rate={best_win_rate:.1%}")
    elif not args.from_scratch and os.path.exists(pretrained_path):
        # Load pre-trained weights from v5's best model
        print(f"Loading pre-trained model: {pretrained_path}")
        ckpt = torch.load(pretrained_path, weights_only=True, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        ckpt_info = (f"  win_rate={ckpt.get('win_rate', '?')}, "
                     f"stage={ckpt.get('stage', '?')}, "
                     f"episode={ckpt.get('episode', '?')}")
        print(f"  Loaded: {ckpt_info}")
    elif not args.from_scratch:
        print(f"WARNING: Pre-trained model not found at {pretrained_path}, training from scratch")

    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Load optimizer state if resuming
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    memory = ReplayBuffer(config['memory_size'])
    agent = HybridAgent(model, device, board['rows'], board['cols'], board['mines'])

    # ─── Fix 1: Signal handling for process resilience ────────────────────
    # These variables are shared with the signal handler via closure
    current_episode = [start_episode]  # mutable container for signal handler

    def save_checkpoint(prefix, episode, extra_msg=''):
        """Save a checkpoint with current training state."""
        path = os.path.join(save_dir, f'{prefix}_ep{episode}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'best_win_rate': best_win_rate,
            'best_episode': best_episode,
            'total_train_steps': total_train_steps,
        }, path)
        return path

    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        ep = current_episode[0]
        path = save_checkpoint('interrupted', ep)
        msg = f"\nINTERRUPTED by {sig_name} at ep {ep}. Checkpoint saved: {path}\n"
        print(msg, flush=True)
        try:
            with open(log_path, 'a') as f:
                f.write(msg)
                f.flush()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ─── Training ────────────────────────────────────────────────────────
    with open(log_path, 'a') as log_file:
        header = (f"\n{'='*70}\n"
                  f"HYBRID TRAINING v6 START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"Architecture: MinesweeperResNet ({param_count:,} params)\n"
                  f"Board: {board['rows']}x{board['cols']}, {board['mines']} mines\n"
                  f"Device: {device}\n"
                  f"LR: {config['learning_rate']} | Batch: {config['batch_size']} | "
                  f"Gamma: {config['gamma']} | "
                  f"Epsilon: {config['epsilon_start']}->{config['epsilon_end']}\n"
                  f"Replay: {config['memory_size']} | Target update: every {config['target_update_freq']} steps\n"
                  f"Grad clip: {config['grad_clip_norm']} | Max episodes: {config['max_episodes']}\n"
                  f"Reward: survival=0.3+0.7*progress, death=-0.5, win=1.0\n"
                  f"Resuming from episode: {start_episode}\n"
                  f"{'='*70}\n")
        print(header)
        log_file.write(header)
        log_file.flush()

        # ─── Benchmarks (skip if resuming) ────────────────────────────────
        if start_episode == 0:
            num_bench = config['benchmark_games']

            print(f"\nBenchmark: Solver-only ({num_bench} games on expert)...")
            solver_wr, solver_guesses, solver_moves = benchmark_solver_only(
                board['rows'], board['cols'], board['mines'], num_bench)
            msg = (f"Solver-only baseline: {solver_wr:.1%} win rate | "
                   f"Avg guesses: {solver_guesses:.1f} | Avg solver moves: {solver_moves:.1f}\n")
            print(msg)
            log_file.write(msg)
            log_file.flush()

            print(f"Benchmark: Hybrid agent ({num_bench} games on expert)...")
            hybrid_wr, hybrid_guesses, hybrid_moves, hybrid_gsr = benchmark_hybrid(
                agent, board['rows'], board['cols'], board['mines'], num_bench)
            msg = (f"Hybrid baseline:     {hybrid_wr:.1%} win rate | "
                   f"Avg guesses: {hybrid_guesses:.1f} | Avg solver moves: {hybrid_moves:.1f} | "
                   f"Guess survival: {hybrid_gsr:.1%}\n")
            print(msg)
            log_file.write(msg)
            log_file.flush()
        else:
            solver_wr = 0.0
            hybrid_wr = 0.0

        # ─── Main training loop ──────────────────────────────────────────
        print(f"\n{'='*70}")
        print("Starting hybrid training...")
        print(f"{'='*70}\n")
        log_file.write(f"\n{'='*70}\nStarting hybrid training...\n{'='*70}\n\n")
        log_file.flush()

        episode_start_time = time.time()
        training_start_time = time.time()
        recent_losses = deque(maxlen=100)

        try:
            for episode in range(start_episode + 1, config['max_episodes'] + 1):
                current_episode[0] = episode

                # Epsilon decay (linear)
                if episode <= config['epsilon_decay_episodes']:
                    epsilon = (config['epsilon_start'] -
                               (config['epsilon_start'] - config['epsilon_end']) *
                               (episode / config['epsilon_decay_episodes']))
                else:
                    epsilon = config['epsilon_end']

                # Play one episode
                env = MinesweeperEnvironment(
                    rows=board['rows'], cols=board['cols'], mines=board['mines'],
                    use_v2=True, normalize_rewards=True
                )
                won, ng, ns, tm, gs, gd = agent.play_episode(
                    env, epsilon, replay_buffer=memory)

                # Train on guess transitions
                num_new_guesses = ng
                train_iters = max(1, num_new_guesses // config['update_freq'])
                for _ in range(train_iters):
                    loss = train_step(model, target_model, optimizer, memory,
                                      config, device)
                    if loss is not None:
                        recent_losses.append(loss)
                        total_train_steps += 1

                        # Hard target update
                        if total_train_steps % config['target_update_freq'] == 0:
                            target_model.load_state_dict(model.state_dict())

                # Fix 5: MPS memory management
                if (episode % config['mps_cache_clear_every'] == 0
                        and device.type == 'mps'):
                    torch.mps.empty_cache()

                # Periodic evaluation
                if episode % config['eval_every'] == 0:
                    elapsed = time.time() - episode_start_time
                    eps_per_sec = config['eval_every'] / elapsed if elapsed > 0 else 0
                    avg_loss = np.mean(recent_losses) if recent_losses else 0

                    metrics = evaluate_hybrid(
                        agent, board['rows'], board['cols'], board['mines'],
                        config['eval_episodes'])

                    win_rate = metrics['win_rate']

                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_episode = episode
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'episode': episode,
                            'win_rate': win_rate,
                            'best_win_rate': best_win_rate,
                            'best_episode': best_episode,
                            'total_train_steps': total_train_steps,
                            'metrics': metrics,
                        }, os.path.join(save_dir, 'best_model.pth'))

                    total_elapsed = time.time() - training_start_time

                    msg = (f"Ep {episode:>6}/{config['max_episodes']} | "
                           f"Win: {win_rate:.1%} (best: {best_win_rate:.1%} @{best_episode}) | "
                           f"Loss: {avg_loss:.5f} | Eps: {epsilon:.3f} | "
                           f"Guesses: {metrics['avg_guesses']:.1f} | "
                           f"Solver: {metrics['avg_solver_moves']:.1f} | "
                           f"GR: {metrics['guess_ratio']:.1%} | "
                           f"GSR: {metrics['guess_survival_rate']:.1%} | "
                           f"Buf: {len(memory)} | "
                           f"{eps_per_sec:.1f} eps/s | "
                           f"Elapsed: {timedelta(seconds=int(total_elapsed))}")
                    print(msg)
                    log_file.write(msg + '\n')
                    log_file.flush()

                    episode_start_time = time.time()

                    # Check patience
                    if episode - best_episode >= config['patience']:
                        msg = (f"\nPATIENCE EXCEEDED at ep {episode}. "
                               f"Best: {best_win_rate:.1%} @ep{best_episode}")
                        print(msg)
                        log_file.write(msg + '\n')
                        log_file.flush()
                        break

                # Periodic checkpoint
                if episode % config['checkpoint_every'] == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'episode': episode,
                        'best_win_rate': best_win_rate,
                        'best_episode': best_episode,
                        'total_train_steps': total_train_steps,
                    }, os.path.join(save_dir, f'checkpoint_ep{episode}.pth'))

        except Exception as e:
            # Fix 1: Crash resilience — save checkpoint and log traceback
            ep = current_episode[0]
            path = save_checkpoint('crash', ep)
            crash_msg = (f"\nCRASHED at ep {ep}: {e}\n"
                         f"Checkpoint saved: {path}\n"
                         f"{''.join(traceback.format_exc())}\n")
            print(crash_msg, flush=True)
            log_file.write(crash_msg)
            log_file.flush()
            raise

        # ─── Final Summary ───────────────────────────────────────────────
        total_time = time.time() - training_start_time
        summary = (f"\n{'='*70}\n"
                   f"HYBRID TRAINING v6 COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Total time: {timedelta(seconds=int(total_time))}\n"
                   f"Best win rate: {best_win_rate:.1%} @episode {best_episode}\n"
                   f"Total training steps: {total_train_steps:,}\n"
                   f"{'='*70}\n")
        print(summary)
        log_file.write(summary)
        log_file.flush()


if __name__ == '__main__':
    main()
