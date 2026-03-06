#!/usr/bin/env python3
"""
Minesweeper AI — Curriculum Training v4

Two-phase training with the v3 ResNet architecture:
  Phase 1: Behavioral cloning from algorithmic solver demonstrations
  Phase 2: RL fine-tuning (DQN) with curriculum stages

Usage:
    PYTHONUNBUFFERED=1 python3 train_curriculum_v4.py [--device auto|mps|cpu] [--start-stage 0]
    PYTHONUNBUFFERED=1 python3 train_curriculum_v4.py --skip-pretraining --start-stage 3
    PYTHONUNBUFFERED=1 python3 train_curriculum_v4.py --load-checkpoint models_v3/curriculum/beginner/best_model.pth --start-stage 4
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.environment import MinesweeperEnvironment
from src.ai.models_v3 import MinesweeperResNet
from src.ai.algorithmic_solver import AlgorithmicSolver

# ─── Configuration ───────────────────────────────────────────────────────────

PRETRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'batch_size': 256,
    'epochs': 30,
    'val_fraction': 0.1,
    # Games to generate per board size for demonstration data
    'demo_games': {
        'tiny':    (5, 5, 3, 3000),
        'small':   (7, 7, 7, 3000),
        'mini':    (8, 8, 9, 3000),
        'beginner': (9, 9, 10, 5000),
        'bridge1': (10, 10, 15, 3000),
        'bridge2': (12, 12, 22, 3000),
        'intermediate': (16, 16, 40, 5000),
        'advanced': (16, 20, 60, 3000),
        'expert':  (16, 30, 99, 3000),
    },
}

CURRICULUM_STAGES = [
    {
        'name': 'tiny',
        'rows': 5, 'cols': 5, 'mines': 3,
        'target_win_rate': 0.65,
        'max_episodes': 15000,
        'eval_every': 500,
        'eval_episodes': 200,
        'patience': 5000,
        'learning_rate': 5e-5,
    },
    {
        'name': 'small',
        'rows': 7, 'cols': 7, 'mines': 7,
        'target_win_rate': 0.50,
        'max_episodes': 30000,
        'eval_every': 500,
        'eval_episodes': 200,
        'patience': 8000,
        'learning_rate': 5e-5,
    },
    {
        'name': 'mini',
        'rows': 8, 'cols': 8, 'mines': 9,
        'target_win_rate': 0.45,
        'max_episodes': 40000,
        'eval_every': 500,
        'eval_episodes': 200,
        'patience': 10000,
        'learning_rate': 5e-5,
    },
    {
        'name': 'beginner',
        'rows': 9, 'cols': 9, 'mines': 10,
        'target_win_rate': 0.40,
        'max_episodes': 60000,
        'eval_every': 1000,
        'eval_episodes': 300,
        'patience': 15000,
        'learning_rate': 3e-5,
    },
    {
        'name': 'bridge1',
        'rows': 10, 'cols': 10, 'mines': 15,
        'target_win_rate': 0.32,
        'max_episodes': 80000,
        'eval_every': 1000,
        'eval_episodes': 300,
        'patience': 20000,
        'learning_rate': 3e-5,
    },
    {
        'name': 'bridge2',
        'rows': 12, 'cols': 12, 'mines': 22,
        'target_win_rate': 0.28,
        'max_episodes': 100000,
        'eval_every': 1500,
        'eval_episodes': 300,
        'patience': 25000,
        'learning_rate': 2e-5,
    },
    {
        'name': 'bridge3',
        'rows': 14, 'cols': 14, 'mines': 32,
        'target_win_rate': 0.22,
        'max_episodes': 150000,
        'eval_every': 2000,
        'eval_episodes': 300,
        'patience': 35000,
        'learning_rate': 2e-5,
    },
    {
        'name': 'intermediate',
        'rows': 16, 'cols': 16, 'mines': 40,
        'target_win_rate': 0.30,
        'max_episodes': 300000,
        'eval_every': 2000,
        'eval_episodes': 400,
        'patience': 50000,
        'learning_rate': 2e-5,
    },
    {
        'name': 'advanced',
        'rows': 16, 'cols': 20, 'mines': 60,
        'target_win_rate': 0.15,
        'max_episodes': 500000,
        'eval_every': 3000,
        'eval_episodes': 400,
        'patience': 80000,
        'learning_rate': 1e-5,
    },
    {
        'name': 'expert',
        'rows': 16, 'cols': 30, 'mines': 99,
        'target_win_rate': 0.10,
        'max_episodes': 800000,
        'eval_every': 5000,
        'eval_episodes': 500,
        'patience': 150000,
        'learning_rate': 1e-5,
    },
]

RL_CONFIG = {
    'batch_size': 128,
    'gamma': 0.99,
    'epsilon_start': 0.5,        # Lower epsilon since we have pretrained model
    'epsilon_end': 0.03,
    'epsilon_decay_fraction': 0.4,
    'target_update_tau': 0.005,
    'update_freq': 4,
    'memory_size': 300000,
    'min_memory_size': 2000,
    'max_steps_per_episode': 1000,
    'reward_clip': (-1.0, 1.0),
    'target_q_clip': (-10.0, 10.0),
    'warmup_episodes': 2000,
    'milestone_threshold': 0.05,
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


# ─── Phase 1: Behavioral Cloning ─────────────────────────────────────────────

def generate_all_demonstrations(config, save_dir):
    """Generate solver demonstrations for all board sizes."""
    demo_path = os.path.join(save_dir, 'demonstrations.npz')
    if os.path.exists(demo_path):
        print(f"Loading cached demonstrations from {demo_path}")
        data = np.load(demo_path, allow_pickle=True)
        return data['states'], data['actions'], data['board_rows'], data['board_cols']

    all_states = []
    all_actions = []
    all_board_rows = []
    all_board_cols = []

    print("\n" + "=" * 70)
    print("PHASE 1: Generating solver demonstrations")
    print("=" * 70)

    for name, (rows, cols, mines, num_games) in config['demo_games'].items():
        print(f"\n--- {name} ({rows}x{cols}, {mines} mines, {num_games} games) ---")
        solver = AlgorithmicSolver(rows, cols, mines)
        states, actions, metadata = solver.generate_demonstrations(num_games, verbose=True)

        # We need to pad states to a common shape for batching during training.
        # Instead, we'll store them with their board dimensions and handle
        # variable sizes during training by grouping by board size.
        for s, a in zip(states, actions):
            all_states.append(s)
            all_actions.append(a)
            all_board_rows.append(rows)
            all_board_cols.append(cols)

    # Convert to arrays (states are stored as object array since shapes vary)
    actions_arr = np.array(all_actions, dtype=np.int64)
    board_rows_arr = np.array(all_board_rows, dtype=np.int32)
    board_cols_arr = np.array(all_board_cols, dtype=np.int32)

    # Save demonstrations
    np.savez_compressed(
        demo_path,
        states=np.array(all_states, dtype=object),
        actions=actions_arr,
        board_rows=board_rows_arr,
        board_cols=board_cols_arr,
    )
    print(f"\nSaved {len(all_states):,} demonstrations to {demo_path}")

    return np.array(all_states, dtype=object), actions_arr, board_rows_arr, board_cols_arr


def pretrain_behavioral_cloning(model, device, save_dir, config, log_file):
    """Phase 1: Train model to imitate the algorithmic solver."""
    print("\n" + "=" * 70)
    print("PHASE 1: Behavioral Cloning Pre-training")
    print("=" * 70)
    log_file.write("\nPHASE 1: Behavioral Cloning Pre-training\n")
    log_file.write("=" * 70 + "\n")

    states, actions, board_rows, board_cols = generate_all_demonstrations(config, save_dir)

    # Group demonstrations by board size for efficient batching
    size_groups = {}
    for i in range(len(actions)):
        key = (int(board_rows[i]), int(board_cols[i]))
        if key not in size_groups:
            size_groups[key] = {'states': [], 'actions': []}
        size_groups[key]['states'].append(states[i])
        size_groups[key]['actions'].append(actions[i])

    # Convert to tensors per group
    group_data = {}
    total_demos = 0
    for (rows, cols), group in size_groups.items():
        s = np.array(group['states'], dtype=np.float32)
        a = np.array(group['actions'], dtype=np.int64)
        n = len(a)
        total_demos += n

        # Split train/val
        val_n = max(1, int(n * config['val_fraction']))
        perm = np.random.permutation(n)
        train_idx, val_idx = perm[val_n:], perm[:val_n]

        group_data[(rows, cols)] = {
            'train_states': s[train_idx],
            'train_actions': a[train_idx],
            'val_states': s[val_idx],
            'val_actions': a[val_idx],
        }
        print(f"  {rows}x{cols}: {len(train_idx)} train, {len(val_idx)} val")

    print(f"  Total: {total_demos:,} demonstrations")

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    batch_size = config['batch_size']
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Shuffle and iterate through all board sizes
        for (rows, cols), data in group_data.items():
            s_train = data['train_states']
            a_train = data['train_actions']
            n = len(a_train)
            perm = np.random.permutation(n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]

                # [B, H, W, 12] -> [B, 12, H, W]
                s_batch = torch.FloatTensor(s_train[idx]).permute(0, 3, 1, 2).contiguous().to(device)
                a_batch = torch.LongTensor(a_train[idx]).to(device)

                # Forward pass — Q-values as logits for cross-entropy
                q_values = model(s_batch)  # [B, H, W]
                B = q_values.shape[0]
                q_flat = q_values.reshape(B, -1)  # [B, H*W]

                loss = nn.CrossEntropyLoss()(q_flat, a_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item() * len(idx)
                epoch_correct += (q_flat.argmax(1) == a_batch).sum().item()
                epoch_total += len(idx)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for (rows, cols), data in group_data.items():
                s_val = data['val_states']
                a_val = data['val_actions']
                n = len(a_val)

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    s_batch = torch.FloatTensor(s_val[start:end]).permute(0, 3, 1, 2).contiguous().to(device)
                    a_batch = torch.LongTensor(a_val[start:end]).to(device)

                    q_values = model(s_batch)
                    q_flat = q_values.reshape(s_batch.shape[0], -1)

                    loss = nn.CrossEntropyLoss()(q_flat, a_batch)
                    val_loss += loss.item() * len(a_batch)
                    val_correct += (q_flat.argmax(1) == a_batch).sum().item()
                    val_total += len(a_batch)

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        val_loss_avg = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        lr = optimizer.param_groups[0]['lr']

        msg = (f"[BC] Epoch {epoch:>2}/{config['epochs']} | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
               f"Val Loss: {val_loss_avg:.4f} Acc: {val_acc:.1%} | "
               f"LR: {lr:.1e}")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss_avg,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'pretrained_best.pth'))

    msg = f"\nBehavioral cloning complete. Best val loss: {best_val_loss:.4f}, acc: {best_val_acc:.1%}\n"
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

    # Load best model
    ckpt = torch.load(os.path.join(save_dir, 'pretrained_best.pth'), weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    # Quick eval on each board size
    print("Pre-trained model evaluation:")
    log_file.write("Pre-trained model evaluation:\n")
    for stage in CURRICULUM_STAGES:
        env = MinesweeperEnvironment(
            rows=stage['rows'], cols=stage['cols'], mines=stage['mines'],
            use_v2=True, normalize_rewards=True
        )
        wr = evaluate(model, env, 200, device)
        msg = f"  {stage['name']:>15}: {wr:.1%}"
        print(msg)
        log_file.write(msg + "\n")
    log_file.flush()

    return model


# ─── Phase 2: RL Fine-tuning ─────────────────────────────────────────────────

def get_device(preference='auto'):
    if preference == 'mps' or (preference == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif preference == 'cuda' or (preference == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda')
    return torch.device('cpu')


def create_env(stage):
    return MinesweeperEnvironment(
        rows=stage['rows'], cols=stage['cols'], mines=stage['mines'],
        use_v2=True, normalize_rewards=True
    )


def select_action(state, model, env, epsilon, device):
    action_mask = env.get_action_mask()
    valid_indices = np.where(action_mask)[0]

    if random.random() < epsilon:
        if len(valid_indices) > 0:
            return random.choice(valid_indices)
        return random.randint(0, env.rows * env.cols - 1)

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        q_values = model(state_tensor).squeeze(0).reshape(-1)

        valid_mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
        q_values[~valid_mask] = float('-inf')
        return q_values.argmax().item()


def train_step(model, target_model, optimizer, memory, config, device):
    if len(memory) < config['min_memory_size']:
        return None

    batch = memory.sample(config['batch_size'])

    states = torch.FloatTensor(np.array([e.state for e in batch])).permute(0, 3, 1, 2).contiguous().to(device)
    actions = torch.LongTensor([e.action for e in batch]).to(device)
    rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
    next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).permute(0, 3, 1, 2).contiguous().to(device)
    dones = torch.BoolTensor([e.done for e in batch]).to(device)

    r_lo, r_hi = config['reward_clip']
    rewards = rewards.clamp(r_lo, r_hi)

    current_q_map = model(states)
    B = current_q_map.shape[0]
    current_q_flat = current_q_map.reshape(B, -1)
    current_q = current_q_flat.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_map = model(next_states)
        next_q_flat = next_q_map.reshape(B, -1)
        best_actions = next_q_flat.argmax(1)

        next_target_map = target_model(next_states)
        next_target_flat = next_target_map.reshape(B, -1)
        next_q = next_target_flat.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        target_q = rewards + config['gamma'] * next_q * (~dones)
        tq_lo, tq_hi = config['target_q_clip']
        target_q = target_q.clamp(tq_lo, tq_hi)

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def evaluate(model, env, num_episodes, device):
    wins = 0
    model.eval()

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 500:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                q_values = model(state_tensor).squeeze(0).reshape(-1)

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


def train_stage(stage, model, device, save_dir, log_file):
    """Train a single curriculum stage with RL."""
    config = RL_CONFIG.copy()
    stage_lr = stage.get('learning_rate', 1e-5)

    env = create_env(stage)
    target_model = copy.deepcopy(model)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=stage_lr * 0.1)
    memory = ReplayBuffer(config['memory_size'])

    # Use per-stage epsilon if specified
    eps_start = stage.get('epsilon_start', config['epsilon_start'])
    eps_end = stage.get('epsilon_end', config['epsilon_end'])
    decay_frac = stage.get('epsilon_decay_fraction', config['epsilon_decay_fraction'])
    decay_episodes = int(stage['max_episodes'] * decay_frac)

    warmup_episodes = config['warmup_episodes']

    best_win_rate = 0.0
    best_episode = 0
    total_steps = 0
    episode_start_time = time.time()
    stage_start_time = time.time()
    recent_losses = deque(maxlen=100)
    last_milestone = 0.0

    stage_dir = os.path.join(save_dir, stage['name'])
    os.makedirs(stage_dir, exist_ok=True)

    safe_cells = stage['rows'] * stage['cols'] - stage['mines']
    param_count = sum(p.numel() for p in model.parameters())
    msg = (f"\n{'='*70}\n"
           f"Stage: {stage['name']} ({stage['rows']}x{stage['cols']}, {stage['mines']} mines)\n"
           f"Target: {stage['target_win_rate']*100:.0f}% win rate | Max episodes: {stage['max_episodes']}\n"
           f"LR: {stage_lr} | Safe cells: {safe_cells} | Epsilon: {eps_start}->{eps_end}\n"
           f"Device: {device} | Params: {param_count:,}\n"
           f"{'='*70}\n")
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

    for episode in range(1, stage['max_episodes'] + 1):
        # LR warmup
        if episode <= warmup_episodes:
            warmup_factor = episode / warmup_episodes
            current_lr = stage_lr * max(0.1, warmup_factor)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
        elif episode == warmup_episodes + 1:
            for pg in optimizer.param_groups:
                pg['lr'] = stage_lr

        # Epsilon decay
        if episode <= decay_episodes:
            epsilon = eps_start - (eps_start - eps_end) * (episode / decay_episodes)
        else:
            epsilon = eps_end

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

            if total_steps % config['update_freq'] == 0:
                loss = train_step(model, target_model, optimizer, memory, config, device)
                if loss is not None:
                    recent_losses.append(loss)
                soft_update(target_model, model, config['target_update_tau'])

        # Evaluation
        if episode % stage['eval_every'] == 0:
            elapsed = time.time() - episode_start_time
            eps_per_sec = stage['eval_every'] / elapsed if elapsed > 0 else 0
            avg_loss = np.mean(recent_losses) if recent_losses else 0

            win_rate = evaluate(model, env, stage['eval_episodes'], device)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_episode = episode
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'win_rate': win_rate,
                    'stage': stage['name'],
                }, os.path.join(stage_dir, 'best_model.pth'))

            # Milestone saving
            milestone_step = config['milestone_threshold']
            current_milestone = int(win_rate / milestone_step) * milestone_step
            if current_milestone > last_milestone and win_rate >= milestone_step:
                last_milestone = current_milestone
                milestone_pct = int(current_milestone * 100)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'episode': episode,
                    'win_rate': win_rate,
                    'stage': stage['name'],
                }, os.path.join(stage_dir, f'milestone_{milestone_pct}pct.pth'))
                mile_msg = f"  Milestone saved: {milestone_pct}% win rate"
                print(mile_msg)
                log_file.write(mile_msg + '\n')

            stage_elapsed = time.time() - stage_start_time
            current_lr = optimizer.param_groups[0]['lr']

            msg = (f"[{stage['name']}] Ep {episode:>6}/{stage['max_episodes']} | "
                   f"Win: {win_rate:.1%} (best: {best_win_rate:.1%} @{best_episode}) | "
                   f"Loss: {avg_loss:.5f} | Eps: {epsilon:.3f} | LR: {current_lr:.1e} | "
                   f"Steps: {total_steps:>8} | {eps_per_sec:.1f} eps/s | "
                   f"Elapsed: {timedelta(seconds=int(stage_elapsed))}")
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()

            episode_start_time = time.time()

            if win_rate >= stage['target_win_rate']:
                msg = f"TARGET REACHED! {stage['name']}: {win_rate:.1%} >= {stage['target_win_rate']:.0%}"
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()
                break

            if episode - best_episode >= stage['patience']:
                msg = f"PATIENCE EXCEEDED. Best: {best_win_rate:.1%} @ep{best_episode}. Moving on."
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()
                break

        # Periodic checkpoint
        if episode % 10000 == 0:
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Minesweeper AI Curriculum Training v4')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--start-stage', type=int, default=0)
    parser.add_argument('--end-stage', type=int, default=None)
    parser.add_argument('--skip-pretraining', action='store_true',
                        help='Skip behavioral cloning phase')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Load model from checkpoint before RL training')
    args = parser.parse_args()

    device = get_device(args.device)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v3', 'curriculum')
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, 'training_log.txt')
    end_stage = args.end_stage if args.end_stage is not None else len(CURRICULUM_STAGES) - 1

    model = MinesweeperResNet(input_channels=12).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    with open(log_path, 'a') as log_file:
        header = (f"\n{'='*70}\n"
                  f"CURRICULUM TRAINING v4 START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"Architecture: MinesweeperResNet ({param_count:,} params)\n"
                  f"Device: {device}\n"
                  f"Stages: {args.start_stage} -> {end_stage}\n"
                  f"Pre-training: {'SKIP' if args.skip_pretraining else 'YES'}\n"
                  f"{'='*70}\n")
        print(header)
        log_file.write(header + '\n')
        log_file.flush()

        # Load checkpoint if specified
        if args.load_checkpoint:
            print(f"Loading checkpoint: {args.load_checkpoint}")
            ckpt = torch.load(args.load_checkpoint, weights_only=True, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            msg = f"Loaded checkpoint (win_rate={ckpt.get('win_rate', '?')}, stage={ckpt.get('stage', '?')})\n"
            print(msg)
            log_file.write(msg + '\n')

        # Phase 1: Behavioral cloning
        elif not args.skip_pretraining:
            model = pretrain_behavioral_cloning(model, device, save_dir,
                                                PRETRAINING_CONFIG, log_file)

        # Phase 2: RL fine-tuning
        print("\n" + "=" * 70)
        print("PHASE 2: RL Fine-tuning (DQN)")
        print("=" * 70)
        log_file.write("\nPHASE 2: RL Fine-tuning (DQN)\n")
        log_file.write("=" * 70 + "\n")

        results = {}
        curriculum_start = time.time()

        for i in range(args.start_stage, end_stage + 1):
            stage = CURRICULUM_STAGES[i]
            win_rate, best_ep, stage_time, model = train_stage(
                stage, model, device, save_dir, log_file
            )
            results[stage['name']] = {
                'win_rate': win_rate,
                'best_episode': best_ep,
                'time_seconds': stage_time,
            }

            with open(os.path.join(save_dir, 'curriculum_progress.json'), 'w') as f:
                json.dump({
                    'results': results,
                    'last_completed_stage': i,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)

        total_time = time.time() - curriculum_start
        summary = (f"\n{'='*70}\n"
                   f"CURRICULUM v4 COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Total time: {timedelta(seconds=int(total_time))}\n\n")

        for name, r in results.items():
            summary += f"  {name:>15}: {r['win_rate']:.1%} win rate (ep {r['best_episode']}, {timedelta(seconds=int(r['time_seconds']))})\n"

        summary += f"\n{'='*70}\n"
        print(summary)
        log_file.write(summary + '\n')
        log_file.flush()


if __name__ == '__main__':
    main()
