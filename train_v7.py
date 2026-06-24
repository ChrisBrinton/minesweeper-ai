#!/usr/bin/env python3
"""
Minesweeper AI — Supervised Trainer v7 (Constraint-Implied Labels)

Key changes from v5:
  - Labels are exact P(mine) from the constraint engine, not binary 0/1.
  - Trains on ALL board states (not just guess states) so the model
    learns deterministic logic the solver would handle.
  - Loss: BCE with soft labels on hidden cells.
  - Evaluates model-only win rate (no solver at inference).
  - Desktop-friendly: inherits v5's throttle/scheduling/low-priority.

Usage:
    python train_v7.py                      # resume or fresh start
    python train_v7.py --fresh              # ignore existing checkpoint
    python train_v7.py --num-iterations 1 --num-games 200   # smoke test
"""

import argparse
import json
import os
import random
import signal
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.algorithmic_solver import AlgorithmicSolver
from src.ai.constraint_engine import ConstraintEngine
from src.ai.environment import MinesweeperEnvironment
from src.ai.models_v4 import MinesweeperResNetV4


# ─── Configuration ───────────────────────────────────────────────────────────

EXPERT_CONFIG = {'rows': 16, 'cols': 30, 'mines': 99}

CONFIG = {
    'num_iterations': 30,
    'num_games_per_iteration': 10000,
    'batch_size': 256,
    'learning_rate': 3e-4,
    'max_epochs_per_iteration': 150,
    'patience_epochs': 15,
    'cross_iteration_patience': 12,
    'eval_episodes': 2000,
    'replay_buffer_iters': 3,
    'weight_decay': 1e-5,
    'ema_decay': 0.999,
    'use_bf16': True,
    'grad_clip': 1.0,
    'yield_ms': 0,
    'active_hours': None,
    'low_priority': False,
    'sample_interval': 3,
}


# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
SAVE_DIR = REPO_ROOT / 'models' / 'v7' / 'expert'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATH_BEST = SAVE_DIR / 'best.pth'
PATH_LATEST = SAVE_DIR / 'latest.pth'
PATH_REPLAY = SAVE_DIR / 'replay_buffer.npz'
PATH_LOG = SAVE_DIR / 'log.txt'
PATH_CONFIG = SAVE_DIR / 'config.json'
PATH_CANONICAL_INFERENCE = REPO_ROOT / 'best_model.pth'


# ─── Desktop helpers (from v5) ──────────────────────────────────────────────

def parse_active_hours(spec: str):
    from datetime import time as dt_time
    try:
        start_s, end_s = spec.split('-')
        sh, sm = map(int, start_s.strip().split(':'))
        eh, em = map(int, end_s.strip().split(':'))
        return dt_time(sh, sm), dt_time(eh, em)
    except Exception as e:
        raise SystemExit(f"--active-hours expects HH:MM-HH:MM, got {spec!r} ({e})")


def in_active_window(now, start, end) -> bool:
    t = now.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def wait_until_active(window, log_func=None):
    if window is None:
        return
    start, end = window
    if in_active_window(datetime.now(), start, end):
        return
    if log_func:
        log_func(f"[active-hours] sleeping until {start.strftime('%H:%M')}...")
    while not in_active_window(datetime.now(), start, end):
        time.sleep(60)


def set_low_priority_windows() -> bool:
    if not sys.platform.startswith('win'):
        return False
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        return bool(kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS))
    except Exception:
        return False


# ─── Atomic save ─────────────────────────────────────────────────────────────

def atomic_save_torch(path: Path, state: Dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays):
    tmp = path.with_suffix('.npz.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ─── EMA ─────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self) -> Dict:
        return self.shadow

    def load_state_dict(self, sd: Dict):
        for k, v in sd.items():
            self.shadow[k] = v.clone()

    def apply_to(self, model: nn.Module) -> Dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: nn.Module, backup: Dict):
        model.load_state_dict(backup)


# ─── Sample container ───────────────────────────────────────────────────────

class ConstraintSample:
    """One board state + constraint-engine P(mine) labels for hidden cells."""
    __slots__ = ['state', 'labels', 'mask']

    def __init__(self, state: np.ndarray, labels: np.ndarray, mask: np.ndarray):
        self.state = state    # [H, W, 12] float32
        self.labels = labels  # [H, W] float32: P(mine) in [0, 1]
        self.mask = mask      # [H, W] bool: True for hidden cells


# ─── Data generation ────────────────────────────────────────────────────────

def generate_data(model: nn.Module, device: torch.device, board_cfg: Dict,
                  num_games: int) -> Tuple[List[ConstraintSample], Dict]:
    """Play games with hybrid agent, capture ALL board states with
    constraint-engine labels (not just guess states)."""
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)
    engine = ConstraintEngine(rows, cols, mines)
    use_bf16 = device.type == 'cuda' and CONFIG['use_bf16']
    sample_interval = CONFIG.get('sample_interval', 3)

    samples: List[ConstraintSample] = []
    wins = 0
    total_steps = 0
    total_samples = 0

    model.eval()
    for game_idx in range(num_games):
        env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                      use_v2=True, normalize_rewards=True)
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000
        info: Dict = {}

        while not done and steps < max_steps:
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)

            if steps % sample_interval == 0 and len(hidden) > 0:
                hidden_mask = np.zeros((rows, cols), dtype=bool)
                for (r, c) in hidden:
                    hidden_mask[r, c] = True

                if hidden_mask.any():
                    prob_labels = engine.compute_probabilities(
                        state, hidden=hidden, flagged=flagged, revealed=revealed
                    )
                    label_array = np.where(
                        hidden_mask, np.nan_to_num(prob_labels, nan=0.5), 0.0
                    ).astype(np.float32)
                    samples.append(ConstraintSample(
                        state.copy(), label_array, hidden_mask.copy()
                    ))

            if safe:
                best = max(safe, key=lambda cell: solver._score_random_cell(
                    cell[0], cell[1], revealed))
                action = best[0] * cols + best[1]
            else:
                action_mask = env.get_action_mask()
                for (r, c) in known_mines:
                    action_mask[r * cols + c] = False
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) == 0:
                    valid_indices = np.where(env.get_action_mask())[0]
                    if len(valid_indices) == 0:
                        break

                with torch.no_grad():
                    st = (torch.from_numpy(state).permute(2, 0, 1)
                          .unsqueeze(0).contiguous().to(device))
                    if use_bf16:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = model(st).float()
                    else:
                        logits = model(st)
                    q = logits.squeeze(0).reshape(-1)
                    valid_t = torch.zeros_like(q, dtype=torch.bool)
                    valid_t[valid_indices] = True
                    q[~valid_t] = float('inf')
                    action = q.argmin().item()

            next_state, _, done, info = env.step(action)
            steps += 1
            state = next_state

        if info.get('game_state') == 'won':
            wins += 1
        total_steps += steps

        if (game_idx + 1) % max(1, num_games // 10) == 0:
            wr = wins / (game_idx + 1)
            print(f"  Data gen: {game_idx+1}/{num_games} | "
                  f"Win: {wr:.1%} | Samples: {len(samples):,}", flush=True)
            wait_until_active(CONFIG.get('active_hours'))

    model.train()
    stats = {
        'games': num_games, 'wins': wins,
        'win_rate': wins / max(num_games, 1),
        'samples': len(samples),
        'avg_steps': total_steps / max(num_games, 1),
    }
    return samples, stats


# ─── D4 augmentation ────────────────────────────────────────────────────────

def apply_aug(state, labels, mask, t):
    if t == 0:
        return state, labels, mask
    if t == 1:
        return (np.flip(state, axis=1).copy(),
                np.flip(labels, axis=1).copy(),
                np.flip(mask, axis=1).copy())
    if t == 2:
        return (np.flip(state, axis=0).copy(),
                np.flip(labels, axis=0).copy(),
                np.flip(mask, axis=0).copy())
    return (np.flip(np.flip(state, axis=0), axis=1).copy(),
            np.flip(np.flip(labels, axis=0), axis=1).copy(),
            np.flip(np.flip(mask, axis=0), axis=1).copy())


# ─── Training loop ──────────────────────────────────────────────────────────

def train_epoch(model, optimizer, samples, batch_size, device, ema=None):
    """One epoch with D4 augmentation. BCE with soft labels."""
    model.train()
    use_bf16 = device.type == 'cuda' and CONFIG['use_bf16']

    aug_indices = [(i, t) for i in range(len(samples)) for t in range(4)]
    random.shuffle(aug_indices)

    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(aug_indices), batch_size):
        batch = aug_indices[start:start + batch_size]
        if len(batch) < 2:
            continue

        states_list, labels_list, masks_list = [], [], []
        for (si, ti) in batch:
            s, l, m = apply_aug(samples[si].state, samples[si].labels,
                                 samples[si].mask, ti)
            states_list.append(s)
            labels_list.append(l)
            masks_list.append(m)

        states = np.array(states_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.float32)
        masks = np.array(masks_list)

        states_t = torch.from_numpy(states).permute(0, 3, 1, 2).contiguous().to(device)
        labels_t = torch.from_numpy(labels).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        if use_bf16:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(states_t)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels_t, reduction='none')
            loss = loss.float()
        else:
            logits = model(states_t)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels_t, reduction='none')

        masked = loss * masks_t.float()
        num_valid = masks_t.float().sum()
        if num_valid <= 0:
            continue
        batch_loss = masked.sum() / num_valid

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += batch_loss.item()
        num_batches += 1

        yield_ms = CONFIG.get('yield_ms', 0) or 0
        if yield_ms > 0:
            time.sleep(yield_ms / 1000.0)
        wait_until_active(CONFIG.get('active_hours'))

    return total_loss / max(num_batches, 1)


# ─── Evaluation (model-only + hybrid) ───────────────────────────────────────

def evaluate(model, device, board_cfg, num_episodes, ema=None):
    """Evaluate model-only AND hybrid win rates."""
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)
    use_bf16 = device.type == 'cuda' and CONFIG['use_bf16']

    backup = ema.apply_to(model) if ema is not None else None
    model.eval()

    model_only_wins = 0
    hybrid_wins = 0

    for ep in range(num_episodes):
        # Model-only evaluation
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
                if use_bf16:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(st).float()
                else:
                    logits = model(st)
                q = logits.squeeze(0).reshape(-1)
                valid_t = torch.zeros_like(q, dtype=torch.bool)
                valid_t[valid_indices] = True
                q[~valid_t] = float('inf')
                action = q.argmin().item()
            state, _, done, info = env.step(action)
            steps += 1
        if info.get('game_state') == 'won':
            model_only_wins += 1

        # Hybrid evaluation (same game seed not possible — new game)
        env2 = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                       use_v2=True, normalize_rewards=True)
        state2 = env2.reset()
        done2 = False
        steps2 = 0
        while not done2 and steps2 < 1000:
            hidden, flagged, revealed = solver._parse_state(state2)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)
            if safe:
                best = max(safe, key=lambda c: solver._score_random_cell(c[0], c[1], revealed))
                action2 = best[0] * cols + best[1]
            else:
                action_mask2 = env2.get_action_mask()
                for (r, c) in known_mines:
                    action_mask2[r * cols + c] = False
                valid2 = np.where(action_mask2)[0]
                if len(valid2) == 0:
                    valid2 = np.where(env2.get_action_mask())[0]
                    if len(valid2) == 0:
                        break
                with torch.no_grad():
                    st2 = (torch.from_numpy(state2).permute(2, 0, 1)
                           .unsqueeze(0).contiguous().to(device))
                    if use_bf16:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits2 = model(st2).float()
                    else:
                        logits2 = model(st2)
                    q2 = logits2.squeeze(0).reshape(-1)
                    vt2 = torch.zeros_like(q2, dtype=torch.bool)
                    vt2[valid2] = True
                    q2[~vt2] = float('inf')
                    action2 = q2.argmin().item()
            state2, _, done2, info2 = env2.step(action2)
            steps2 += 1
        if info2.get('game_state') == 'won':
            hybrid_wins += 1

    if backup is not None:
        ema.restore(model, backup)

    return {
        'model_only_win_rate': model_only_wins / max(num_episodes, 1),
        'hybrid_win_rate': hybrid_wins / max(num_episodes, 1),
        'model_only_wins': model_only_wins,
        'hybrid_wins': hybrid_wins,
        'episodes': num_episodes,
    }


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


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Minesweeper v7 supervised trainer')
    parser.add_argument('--fresh', action='store_true', help='Ignore existing checkpoint')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num-iterations', type=int, default=None)
    parser.add_argument('--num-games', type=int, default=None)
    parser.add_argument('--eval-episodes', type=int, default=None)
    parser.add_argument('--yield-ms', type=int, default=0)
    parser.add_argument('--active-hours', type=str, default=None)
    parser.add_argument('--low-priority', action='store_true')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Path to checkpoint for warm start')
    args = parser.parse_args()

    if args.num_iterations:
        CONFIG['num_iterations'] = args.num_iterations
    if args.num_games:
        CONFIG['num_games_per_iteration'] = args.num_games
    if args.eval_episodes:
        CONFIG['eval_episodes'] = args.eval_episodes
    if args.yield_ms:
        CONFIG['yield_ms'] = args.yield_ms
    if args.active_hours:
        CONFIG['active_hours'] = parse_active_hours(args.active_hours)
    if args.low_priority:
        CONFIG['low_priority'] = True
        if set_low_priority_windows():
            print("Set process to below-normal priority")

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

    if device.type != 'cuda':
        CONFIG['use_bf16'] = False

    log(f"Device: {device}, BF16: {CONFIG['use_bf16']}", PATH_LOG)
    log(f"Config: {json.dumps(CONFIG, indent=2, default=str)}", PATH_LOG)

    # Save config
    with open(PATH_CONFIG, 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)

    # Model
    model = MinesweeperResNetV4().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])
    ema = EMA(model, CONFIG['ema_decay'])
    start_iteration = 0
    best_model_only_wr = 0.0
    best_hybrid_wr = 0.0
    replay_buffer: deque = deque(maxlen=CONFIG['replay_buffer_iters'])
    stale_iters = 0

    # Resume or warm start
    if not args.fresh and PATH_LATEST.exists():
        log("Resuming from latest checkpoint...", PATH_LOG)
        ckpt = torch.load(PATH_LATEST, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        start_iteration = ckpt.get('iteration', 0) + 1
        best_model_only_wr = ckpt.get('best_model_only_win_rate', 0.0)
        best_hybrid_wr = ckpt.get('best_hybrid_win_rate', 0.0)
        stale_iters = ckpt.get('stale_iters', 0)

        if PATH_REPLAY.exists():
            buf = np.load(PATH_REPLAY, allow_pickle=True)
            for key in sorted(buf.files):
                data = buf[key].item()
                replay_buffer.append(data)
            log(f"  Loaded replay buffer: {len(replay_buffer)} iteration(s)", PATH_LOG)
        log(f"  Resuming at iteration {start_iteration}, "
            f"best model-only={best_model_only_wr:.1%}, "
            f"best hybrid={best_hybrid_wr:.1%}", PATH_LOG)

    elif args.warm_start:
        log(f"Warm-starting from {args.warm_start}", PATH_LOG)
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd)
        ema = EMA(model, CONFIG['ema_decay'])
    elif PATH_CANONICAL_INFERENCE.exists():
        log(f"Warm-starting from {PATH_CANONICAL_INFERENCE}", PATH_LOG)
        ckpt = torch.load(PATH_CANONICAL_INFERENCE, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd)
        ema = EMA(model, CONFIG['ema_decay'])

    # Signal handling
    stop_requested = [False]
    def handle_signal(signum, frame):
        log(f"Signal {signum} received, stopping after current iteration...", PATH_LOG)
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, handle_signal)

    # Main training loop
    for iteration in range(start_iteration, CONFIG['num_iterations']):
        if stop_requested[0]:
            break

        log(f"\n{'='*60}", PATH_LOG)
        log(f"Iteration {iteration + 1}/{CONFIG['num_iterations']}", PATH_LOG)
        log(f"{'='*60}", PATH_LOG)

        # Generate data with constraint-engine labels
        log("Generating training data with constraint-engine labels...", PATH_LOG)
        t0 = time.time()
        samples, gen_stats = generate_data(
            model, device, EXPERT_CONFIG, CONFIG['num_games_per_iteration']
        )
        log(f"  Data gen: {time.time()-t0:.0f}s | {gen_stats}", PATH_LOG)

        if not samples:
            log("  No samples generated, skipping iteration", PATH_LOG)
            continue

        replay_buffer.append(samples)
        all_samples = [s for buf in replay_buffer for s in buf]
        log(f"  Replay buffer: {len(replay_buffer)} iter(s), "
            f"{len(all_samples):,} total samples", PATH_LOG)

        # Save replay buffer
        buf_dict = {f'iter_{i}': np.array(buf, dtype=object)
                    for i, buf in enumerate(replay_buffer)}
        atomic_save_npz(PATH_REPLAY, **buf_dict)

        # Train
        log("Training on constraint-implied labels...", PATH_LOG)
        best_epoch_loss = float('inf')
        patience_counter = 0

        for epoch in range(CONFIG['max_epochs_per_iteration']):
            if stop_requested[0]:
                break

            loss = train_epoch(model, optimizer, all_samples,
                               CONFIG['batch_size'], device, ema)

            if loss < best_epoch_loss - 1e-5:
                best_epoch_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter >= CONFIG['patience_epochs']:
                log(f"  Epoch {epoch+1}: loss={loss:.6f} "
                    f"(best={best_epoch_loss:.6f}, patience={patience_counter}/"
                    f"{CONFIG['patience_epochs']})", PATH_LOG)

            if patience_counter >= CONFIG['patience_epochs']:
                log(f"  Early stop at epoch {epoch+1}", PATH_LOG)
                break

        # Evaluate
        log("Evaluating...", PATH_LOG)
        eval_result = evaluate(model, device, EXPERT_CONFIG,
                               CONFIG['eval_episodes'], ema)
        mo_wr = eval_result['model_only_win_rate']
        hy_wr = eval_result['hybrid_win_rate']
        log(f"  Model-only: {mo_wr:.1%} ({eval_result['model_only_wins']}/"
            f"{eval_result['episodes']})", PATH_LOG)
        log(f"  Hybrid:     {hy_wr:.1%} ({eval_result['hybrid_wins']}/"
            f"{eval_result['episodes']})", PATH_LOG)

        improved = False
        if mo_wr > best_model_only_wr:
            best_model_only_wr = mo_wr
            improved = True
        if hy_wr > best_hybrid_wr:
            best_hybrid_wr = hy_wr
            improved = True

        if improved:
            stale_iters = 0
            log(f"  New best! Model-only={best_model_only_wr:.1%}, "
                f"Hybrid={best_hybrid_wr:.1%}", PATH_LOG)
            save_sd = ema.state_dict() if ema else model.state_dict()
            atomic_save_torch(PATH_BEST, {
                'model_state_dict': save_sd,
                'iteration': iteration,
                'model_only_win_rate': mo_wr,
                'hybrid_win_rate': hy_wr,
                'config': CONFIG,
            })
            atomic_save_torch(PATH_CANONICAL_INFERENCE, {
                'model_state_dict': save_sd,
                'iteration': iteration,
                'model_only_win_rate': mo_wr,
                'hybrid_win_rate': hy_wr,
                'trainer': 'v7',
            })
        else:
            stale_iters += 1
            log(f"  No improvement ({stale_iters}/"
                f"{CONFIG['cross_iteration_patience']})", PATH_LOG)

        # Save latest checkpoint
        atomic_save_torch(PATH_LATEST, {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_state_dict': ema.state_dict() if ema else None,
            'iteration': iteration,
            'best_model_only_win_rate': best_model_only_wr,
            'best_hybrid_win_rate': best_hybrid_wr,
            'stale_iters': stale_iters,
            'config': CONFIG,
        })

        if stale_iters >= CONFIG['cross_iteration_patience']:
            log(f"Cross-iteration patience exhausted. Stopping.", PATH_LOG)
            break

    log(f"\nTraining complete. Best model-only: {best_model_only_wr:.1%}, "
        f"Best hybrid: {best_hybrid_wr:.1%}", PATH_LOG)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
