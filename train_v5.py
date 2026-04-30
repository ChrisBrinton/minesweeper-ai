#!/usr/bin/env python3
"""
Minesweeper supervised trainer v5 — Blackwell-ready, resumable.

Improvements over train_supervised_v4.py:
  * BF16 mixed-precision training (faster on RTX 5070 Ti / Blackwell).
  * AdamW + EMA weights (small but reliable stability win).
  * No teacher dependency — pure self-play with the warm-started student.
  * Standardized paths:  models/v5/expert/ for run artifacts,
                          ./best_model.pth for canonical inference.
  * Atomic checkpointing (.tmp.pth -> rename) — won't corrupt files even
    on a power-cut mid-write.
  * Auto-resume: saved replay buffer + full optimizer / EMA / scheduler
    state means an interrupted run picks up at the start of the next
    iteration on relaunch.
  * Windows-friendly: no SIGHUP usage.

Layout written by this script:
    models/v5/expert/
        best.pth              # best within v5 (model_state_dict + metadata)
        latest.pth            # full state for resume; written every iteration
        replay_buffer.npz     # replay buffer (last N iterations of samples)
        log.txt               # human-readable training log
        config.json           # the config used for this run

When a new global best is found, ./best_model.pth at the repo root is
also updated atomically so the in-game inference picks it up.

Usage:
    python train_v5.py                    # resume if a latest.pth exists, else fresh
    python train_v5.py --fresh            # ignore any existing checkpoint
    python train_v5.py --num-iterations 1 --num-games 200    # quick smoke test
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
from src.ai.environment import MinesweeperEnvironment
from src.ai.models_v4 import MinesweeperResNetV4


# ─── Configuration ───────────────────────────────────────────────────────────

EXPERT_CONFIG = {'rows': 16, 'cols': 30, 'mines': 99}

CONFIG = {
    'num_iterations': 30,
    'num_games_per_iteration': 10000,
    'batch_size': 256,                  # 256 fits easily in 16GB BF16
    'learning_rate': 3e-4,
    'max_epochs_per_iteration': 150,
    'patience_epochs': 15,
    'cross_iteration_patience': 12,
    'eval_episodes': 2000,
    'replay_buffer_iters': 3,
    'weight_decay': 1e-5,
    'ema_decay': 0.999,
    'use_bf16': True,                   # auto-disabled if device != cuda
    'grad_clip': 1.0,
    # Desktop-friendliness knobs (set by CLI flags; keeps default training
    # behaviour unchanged when omitted)
    'yield_ms': 0,                      # sleep N ms between training batches
    'active_hours': None,               # tuple (dt_time, dt_time) or None
    'low_priority': False,
}


# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
SAVE_DIR = REPO_ROOT / 'models' / 'v5' / 'expert'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATH_BEST = SAVE_DIR / 'best.pth'
PATH_LATEST = SAVE_DIR / 'latest.pth'
PATH_REPLAY = SAVE_DIR / 'replay_buffer.npz'
PATH_LOG = SAVE_DIR / 'log.txt'
PATH_CONFIG = SAVE_DIR / 'config.json'

# Canonical inference target — mirrored from PATH_BEST whenever it improves
PATH_CANONICAL_INFERENCE = REPO_ROOT / 'best_model.pth'


# ─── Desktop-friendliness helpers ────────────────────────────────────────────

def parse_active_hours(spec: str):
    """Parse 'HH:MM-HH:MM' into (start, end) datetime.time. Wraps midnight ok."""
    from datetime import time as dt_time
    try:
        start_s, end_s = spec.split('-')
        sh, sm = map(int, start_s.strip().split(':'))
        eh, em = map(int, end_s.strip().split(':'))
        return dt_time(sh, sm), dt_time(eh, em)
    except Exception as e:
        raise SystemExit(f"--active-hours expects HH:MM-HH:MM, got {spec!r} ({e})")


def in_active_window(now, start, end) -> bool:
    """True if `now` (a datetime) is in [start, end). Handles wrap-around."""
    t = now.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def wait_until_active(window, log_func=None):
    """If we're outside the active window, sleep (60s polls) until inside.
    No-op if window is None."""
    if window is None:
        return
    start, end = window
    if in_active_window(datetime.now(), start, end):
        return
    if log_func:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_func(f"[active-hours] outside window {start.strftime('%H:%M')}-"
                 f"{end.strftime('%H:%M')}, sleeping... (now: {now})")
    while not in_active_window(datetime.now(), start, end):
        time.sleep(60)
    if log_func:
        log_func(f"[active-hours] window opened, resuming at "
                 f"{datetime.now().strftime('%H:%M:%S')}")


def set_low_priority_windows() -> bool:
    """Lower this process's priority so the OS scheduler favours foreground
    apps. Windows-only via the Win32 SetPriorityClass; no-op elsewhere."""
    if not sys.platform.startswith('win'):
        return False
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        kernel32 = ctypes.windll.kernel32
        # Default ctypes signatures assume int return — the HANDLE is a
        # pointer and gets truncated to 32 bits without these. SetPriority
        # then receives a garbage handle and fails.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        return bool(kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS))
    except Exception:
        return False


# ─── Atomic save / load ──────────────────────────────────────────────────────

def atomic_save_torch(path: Path, state: Dict):
    """torch.save with .tmp -> rename so a partial write can't corrupt the
    target file. Survives mid-write power loss."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays):
    # np.savez auto-appends .npz if missing — make tmp end in .npz too
    tmp = path.with_suffix('.npz.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ─── EMA weights ─────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    # int/bool buffers (e.g., BatchNorm num_batches_tracked) — copy
                    self.shadow[k].copy_(v)

    def state_dict(self) -> Dict:
        return self.shadow

    def load_state_dict(self, sd: Dict):
        # Replace contents in-place to keep tensor identity
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()
            else:
                self.shadow[k] = v.clone()

    def apply_to(self, model: nn.Module) -> Dict:
        """Swap EMA weights into model. Returns the original weights so a
        later restore() puts them back."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: nn.Module, backup: Dict):
        model.load_state_dict(backup)


# ─── Sample container ────────────────────────────────────────────────────────

class GuessSample:
    """One guess situation: board state + ground-truth mine map + valid
    targets mask."""
    __slots__ = ['state', 'labels', 'mask']

    def __init__(self, state: np.ndarray, labels: np.ndarray, mask: np.ndarray):
        self.state = state    # [H, W, 12] float32
        self.labels = labels  # [H, W] float32: 1.0=mine, 0.0=safe
        self.mask = mask      # [H, W] bool: True for hidden cells (valid targets)


# ─── Data generation (self-play) ─────────────────────────────────────────────

def generate_data(model: nn.Module, device: torch.device, board_cfg: Dict,
                  num_games: int) -> Tuple[List[GuessSample], Dict]:
    """Play games with the hybrid agent (algorithmic solver + NN), capturing
    one (state, labels, mask) sample at every guess situation."""
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)
    use_bf16 = device.type == 'cuda' and CONFIG['use_bf16']

    samples: List[GuessSample] = []
    wins = 0
    total_guesses = 0
    total_solver_moves = 0

    model.eval()
    for game_idx in range(num_games):
        env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                      use_v2=True, normalize_rewards=True)
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000
        game_guess_states: List[Tuple[np.ndarray, np.ndarray]] = []
        game_guesses = 0
        game_solver_moves = 0
        info: Dict = {}

        while not done and steps < max_steps:
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)

            if safe:
                # Deterministic safe move — pick the one nearest the frontier
                best = max(safe, key=lambda cell: solver._score_random_cell(
                    cell[0], cell[1], revealed))
                action = best[0] * cols + best[1]
                game_solver_moves += 1
            else:
                # Guess situation — capture state for training
                hidden_mask = np.zeros((rows, cols), dtype=bool)
                for (r, c) in hidden:
                    if (r, c) not in known_mines:
                        hidden_mask[r, c] = True
                if hidden_mask.any():
                    game_guess_states.append((state.copy(), hidden_mask.copy()))

                # Use NN to pick the action
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
                    q[~valid_t] = float('inf')   # P(mine) mode: argmin = safest
                    action = q.argmin().item()
                game_guesses += 1

            next_state, _, done, info = env.step(action)
            steps += 1
            state = next_state

        # End of game — record per-cell labels for every captured guess state
        if game_guess_states:
            labels = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    if env.api.game_board.get_cell(r, c).is_mine:
                        labels[r, c] = 1.0
            for (snap_state, snap_mask) in game_guess_states:
                samples.append(GuessSample(snap_state, labels, snap_mask))

        if info.get('game_state') == 'won':
            wins += 1
        total_guesses += game_guesses
        total_solver_moves += game_solver_moves

        if (game_idx + 1) % max(1, num_games // 10) == 0:
            wr = wins / (game_idx + 1)
            print(f"  Data gen: {game_idx+1}/{num_games} | "
                  f"Win: {wr:.1%} | Avg guesses: {total_guesses/(game_idx+1):.1f} | "
                  f"Samples: {len(samples):,}", flush=True)
            # Check active window only at progress checkpoints — once per ~10%
            # of the run — so the polling cost is negligible
            wait_until_active(CONFIG.get('active_hours'))

    model.train()
    stats = {
        'games': num_games,
        'wins': wins,
        'win_rate': wins / max(num_games, 1),
        'samples': len(samples),
        'avg_guesses': total_guesses / max(num_games, 1),
        'avg_solver_moves': total_solver_moves / max(num_games, 1),
    }
    return samples, stats


# ─── D4 augmentation (shape-preserving for 16x30) ────────────────────────────

def apply_aug(state: np.ndarray, labels: np.ndarray, mask: np.ndarray,
              t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-fold D4 transform that preserves H × W (no 90/270° on non-square)."""
    if t == 0:
        return state, labels, mask
    if t == 1:  # h-flip
        return (np.flip(state, axis=1).copy(),
                np.flip(labels, axis=1).copy(),
                np.flip(mask, axis=1).copy())
    if t == 2:  # v-flip
        return (np.flip(state, axis=0).copy(),
                np.flip(labels, axis=0).copy(),
                np.flip(mask, axis=0).copy())
    # 180° rotation
    return (np.flip(np.flip(state, axis=0), axis=1).copy(),
            np.flip(np.flip(labels, axis=0), axis=1).copy(),
            np.flip(np.flip(mask, axis=0), axis=1).copy())


# ─── Training loop ───────────────────────────────────────────────────────────

def train_epoch(model: nn.Module, optimizer: optim.Optimizer,
                samples: List[GuessSample], batch_size: int,
                device: torch.device, ema: Optional[EMA]) -> float:
    """One epoch with 4-fold D4 augmentation. Returns avg masked-BCE loss."""
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
            loss = loss.float()  # accumulate in fp32
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

        # Yield to display rendering so Windows UI stays responsive
        yield_ms = CONFIG.get('yield_ms', 0) or 0
        if yield_ms > 0:
            time.sleep(yield_ms / 1000.0)
        # Pause if we've left the active window (in-process pause; preserves
        # GPU memory but submits no kernels, so display gets full GPU)
        wait_until_active(CONFIG.get('active_hours'))

    return total_loss / max(num_batches, 1)


# ─── Evaluation (uses EMA weights if provided) ───────────────────────────────

def evaluate(model: nn.Module, device: torch.device, board_cfg: Dict,
             num_episodes: int, ema: Optional[EMA] = None) -> Dict:
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)
    use_bf16 = device.type == 'cuda' and CONFIG['use_bf16']

    backup = ema.apply_to(model) if ema is not None else None
    model.eval()

    wins = 0
    total_guesses = 0
    total_solver_moves = 0
    total_guess_survivals = 0
    total_guess_deaths = 0

    try:
        for ep in range(num_episodes):
            env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                          use_v2=True, normalize_rewards=True)
            state = env.reset()
            done = False
            steps = 0
            pending_guess = False
            info: Dict = {}

            while not done and steps < 1000:
                hidden, flagged, revealed = solver._parse_state(state)
                safe, known_mines = solver._find_deterministic_moves(
                    hidden, flagged, revealed)

                if safe:
                    if pending_guess:
                        total_guess_survivals += 1
                        pending_guess = False
                    best = max(safe, key=lambda cell: solver._score_random_cell(
                        cell[0], cell[1], revealed))
                    action = best[0] * cols + best[1]
                    total_solver_moves += 1
                else:
                    if pending_guess:
                        total_guess_survivals += 1
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
                    total_guesses += 1
                    pending_guess = True

                state, _, done, info = env.step(action)
                steps += 1

            if pending_guess:
                if info.get('game_state') == 'won':
                    total_guess_survivals += 1
                else:
                    total_guess_deaths += 1
            if info.get('game_state') == 'won':
                wins += 1
    finally:
        if backup is not None:
            ema.restore(model, backup)

    return {
        'win_rate': wins / max(num_episodes, 1),
        'wins': wins,
        'total': num_episodes,
        'avg_guesses': total_guesses / max(num_episodes, 1),
        'avg_solver_moves': total_solver_moves / max(num_episodes, 1),
        'guess_survival_rate': (total_guess_survivals
                                / max(1, total_guess_survivals + total_guess_deaths)),
    }


# ─── Replay buffer save/load ─────────────────────────────────────────────────

def save_replay_buffer(buffer: deque):
    if not buffer:
        return
    arrays: Dict[str, np.ndarray] = {'num_iters': np.array(len(buffer), dtype=np.int32)}
    for i, samples in enumerate(buffer):
        if not samples:
            continue
        arrays[f'iter{i}_states'] = np.stack([s.state for s in samples], axis=0)
        arrays[f'iter{i}_labels'] = np.stack([s.labels for s in samples], axis=0)
        arrays[f'iter{i}_masks'] = np.stack([s.mask for s in samples], axis=0)
    atomic_save_npz(PATH_REPLAY, **arrays)


def load_replay_buffer(maxlen: int) -> deque:
    buf: deque = deque(maxlen=maxlen)
    if not PATH_REPLAY.exists():
        return buf
    try:
        with np.load(PATH_REPLAY) as data:
            n = int(data['num_iters'])
            for i in range(n):
                key_states = f'iter{i}_states'
                if key_states not in data.files:
                    continue
                states = data[key_states]
                labels = data[f'iter{i}_labels']
                masks = data[f'iter{i}_masks']
                samples = [GuessSample(states[j], labels[j], masks[j])
                           for j in range(states.shape[0])]
                buf.append(samples)
    except Exception as e:
        print(f"WARNING: failed to load replay buffer ({e}); starting fresh", flush=True)
        return deque(maxlen=maxlen)
    return buf


# ─── Checkpoint save/load ────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, ema: EMA, optimizer: optim.Optimizer,
                    iteration: int, best_win_rate: float, best_iteration: int,
                    iters_without_improvement: int, config: Dict):
    state = {
        'model': model.state_dict(),
        'ema': ema.state_dict() if ema is not None else None,
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'best_win_rate': best_win_rate,
        'best_iteration': best_iteration,
        'iters_without_improvement': iters_without_improvement,
        'config': config,
    }
    atomic_save_torch(PATH_LATEST, state)


def load_checkpoint(model: nn.Module, ema: EMA, optimizer: optim.Optimizer,
                    device: torch.device) -> Optional[Dict]:
    if not PATH_LATEST.exists():
        return None
    ckpt = torch.load(PATH_LATEST, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if ema is not None and ckpt.get('ema') is not None:
        ema.load_state_dict(ckpt['ema'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return {
        'iteration': int(ckpt['iteration']),
        'best_win_rate': float(ckpt['best_win_rate']),
        'best_iteration': int(ckpt['best_iteration']),
        'iters_without_improvement': int(ckpt.get('iters_without_improvement', 0)),
        'config': ckpt.get('config', {}),
    }


# ─── Best-model save (writes both within-v5 and canonical inference paths) ───

def save_best(model: nn.Module, ema: Optional[EMA], iteration: int,
              metrics: Dict):
    """Save best.pth (with EMA weights if available) to both:
       - models/v5/expert/best.pth  (within-v5 record)
       - ./best_model.pth            (canonical inference target)
    """
    # Use EMA weights for the saved best (eval was done with EMA too)
    if ema is not None:
        backup = ema.apply_to(model)
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.restore(model, backup)
    else:
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    payload = {
        'model_state_dict': sd,
        'iteration': iteration,
        'win_rate': metrics['win_rate'],
        'metrics': metrics,
    }
    atomic_save_torch(PATH_BEST, payload)
    atomic_save_torch(PATH_CANONICAL_INFERENCE, payload)


# ─── Device picker ───────────────────────────────────────────────────────────

def get_device(preference: str) -> torch.device:
    if preference == 'cpu':
        return torch.device('cpu')
    if preference == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA isn't available")
        return torch.device('cuda')
    if preference == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise SystemExit("--device mps requested but MPS isn't available")
        return torch.device('mps')
    # auto
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Windows default console codec is cp1252; force UTF-8 so any non-ASCII
    # characters in log output (or future strings) don't crash the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure') and (stream.encoding or '').lower() not in ('utf-8', 'utf8'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    parser = argparse.ArgumentParser(description='Minesweeper supervised trainer v5')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num-iterations', type=int, default=None)
    parser.add_argument('--num-games', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore any existing checkpoint and start fresh')
    parser.add_argument('--no-bf16', action='store_true',
                        help='Disable BF16 even on CUDA (use FP32)')
    parser.add_argument('--yield-ms', type=int, default=0, metavar='N',
                        help='Sleep N ms between training batches so the '
                             'Windows UI stays responsive. Try 5-15 if the '
                             'desktop locks up during training. Default: 0.')
    parser.add_argument('--active-hours', metavar='HH:MM-HH:MM', default=None,
                        help='Only train inside this time window (e.g. '
                             '20:00-04:00). Outside the window, the script '
                             'pauses in-process — no kernels submitted, so '
                             'display gets full GPU. Wraps midnight ok.')
    parser.add_argument('--low-priority', action='store_true',
                        help='Set this process priority to BELOW_NORMAL '
                             '(Windows). Helps the OS scheduler favour '
                             'foreground apps.')
    args = parser.parse_args()

    device = get_device(args.device)
    config = CONFIG.copy()
    board = EXPERT_CONFIG.copy()

    if args.num_iterations is not None:
        config['num_iterations'] = args.num_iterations
    if args.num_games is not None:
        config['num_games_per_iteration'] = args.num_games
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.no_bf16:
        config['use_bf16'] = False
    if args.yield_ms:
        config['yield_ms'] = max(0, int(args.yield_ms))
    if args.active_hours:
        config['active_hours'] = parse_active_hours(args.active_hours)
    config['low_priority'] = bool(args.low_priority)
    # In-process write so worker funcs see updated values
    CONFIG.update(config)

    if config['low_priority']:
        ok = set_low_priority_windows()
        print(f"Process priority lowered: {ok}", flush=True)

    # Persist config for the run
    with open(PATH_CONFIG, 'w') as f:
        json.dump({**config, 'board': board, 'device': str(device)}, f, indent=2)

    # Build model + optimizer + EMA
    model = MinesweeperResNetV4(input_channels=12).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(),
                             lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])
    ema = EMA(model, decay=config['ema_decay'])

    # Resume from latest.pth if present (and not --fresh)
    resume_state = None
    if not args.fresh:
        resume_state = load_checkpoint(model, ema, optimizer, device)

    if resume_state is not None:
        start_iteration = resume_state['iteration'] + 1
        best_win_rate = resume_state['best_win_rate']
        best_iteration = resume_state['best_iteration']
        iters_without_improvement = resume_state['iters_without_improvement']
        replay_buffer = load_replay_buffer(config['replay_buffer_iters'])
        resume_msg = (f"Resumed from {PATH_LATEST} at iteration "
                      f"{resume_state['iteration']} (best {best_win_rate:.1%}). "
                      f"Replay buffer: {sum(len(b) for b in replay_buffer):,} samples "
                      f"across {len(replay_buffer)} iterations.")
    else:
        start_iteration = 1
        best_win_rate = 0.0
        best_iteration = 0
        iters_without_improvement = 0
        replay_buffer = deque(maxlen=config['replay_buffer_iters'])

        # Warm-start from canonical best_model.pth if present
        if PATH_CANONICAL_INFERENCE.exists():
            ckpt = torch.load(PATH_CANONICAL_INFERENCE, weights_only=False,
                              map_location=device)
            sd = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(sd)
            # Reset EMA shadow to the warm-started weights
            ema = EMA(model, decay=config['ema_decay'])
            resume_msg = (f"Warm-started from {PATH_CANONICAL_INFERENCE} "
                          f"(prior win_rate: {ckpt.get('win_rate', '?')})")
        else:
            resume_msg = "No warm-start model found; starting from random init."

    # Signal handling — save state on Ctrl-C / SIGTERM. SIGHUP isn't on Windows.
    current_iter_holder = [start_iteration - 1]

    def handle_signal(signum, _frame):
        sig_name = signal.Signals(signum).name
        it = current_iter_holder[0]
        try:
            save_checkpoint(model, ema, optimizer, it, best_win_rate,
                            best_iteration, iters_without_improvement, config)
            save_replay_buffer(replay_buffer)
        except Exception:
            traceback.print_exc()
        msg = f"\n*** Interrupted by {sig_name} at iteration {it}. State saved. ***\n"
        print(msg, flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, handle_signal)

    # ─── Header ──────────────────────────────────────────────────────────────
    with open(PATH_LOG, 'a', encoding='utf-8') as log_file:
        header = (f"\n{'='*70}\n"
                  f"v5 TRAINING START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"Model: MinesweeperResNetV4 ({param_count:,} params)\n"
                  f"Board: {board['rows']}x{board['cols']}, {board['mines']} mines\n"
                  f"Device: {device}  BF16: {config['use_bf16'] and device.type == 'cuda'}\n"
                  f"LR: {config['learning_rate']} (cosine annealing per iter) | "
                  f"Batch: {config['batch_size']} | EMA: {config['ema_decay']}\n"
                  f"Games/iter: {config['num_games_per_iteration']} | "
                  f"Max epochs/iter: {config['max_epochs_per_iteration']} | "
                  f"Patience: {config['patience_epochs']} epochs / "
                  f"{config['cross_iteration_patience']} iters\n"
                  f"Iterations: {config['num_iterations']} (starting at "
                  f"{start_iteration}) | Eval episodes: {config['eval_episodes']}\n"
                  f"Yield: {config['yield_ms']}ms/batch | "
                  f"Active hours: "
                  f"{('-'.join(t.strftime('%H:%M') for t in config['active_hours'])) if config['active_hours'] else 'always'} | "
                  f"Low-priority: {config['low_priority']}\n"
                  f"{resume_msg}\n"
                  f"Save dir: {SAVE_DIR}\n"
                  f"{'='*70}\n")
        print(header)
        log_file.write(header)
        log_file.flush()

        # Baseline eval so we can compare iterations against starting point
        if start_iteration == 1:
            print("Evaluating baseline...", flush=True)
            baseline = evaluate(model, device, board, config['eval_episodes'], ema)
            msg = (f"BASELINE (EMA): Win={baseline['win_rate']:.1%} | "
                   f"GSR={baseline['guess_survival_rate']:.1%} | "
                   f"Guesses={baseline['avg_guesses']:.1f} | "
                   f"Solver={baseline['avg_solver_moves']:.1f}\n")
            print(msg)
            log_file.write(msg)
            log_file.flush()

            if baseline['win_rate'] > best_win_rate:
                best_win_rate = baseline['win_rate']
                best_iteration = 0
                save_best(model, ema, 0, baseline)

        training_start = time.time()

        try:
            for iteration in range(start_iteration, config['num_iterations'] + 1):
                # Pause at the iteration boundary if we're outside active hours
                wait_until_active(config.get('active_hours'),
                                  log_func=lambda m: log_file.write(m + '\n') or print(m, flush=True))
                current_iter_holder[0] = iteration
                iter_start = time.time()
                msg = f"\n{'-'*70}\nITERATION {iteration}/{config['num_iterations']}\n{'-'*70}\n"
                print(msg)
                log_file.write(msg)
                log_file.flush()

                # ── Step 1: self-play data gen with current model ────────────
                print(f"Generating data: {config['num_games_per_iteration']} games "
                      f"(self-play)...", flush=True)
                gen_start = time.time()
                samples, gen_stats = generate_data(
                    model, device, board, config['num_games_per_iteration'])
                gen_time = time.time() - gen_start
                msg = (f"Data gen: {gen_stats['games']} games in "
                       f"{timedelta(seconds=int(gen_time))} | "
                       f"Win: {gen_stats['win_rate']:.1%} | "
                       f"Samples: {gen_stats['samples']:,} | "
                       f"Avg guesses: {gen_stats['avg_guesses']:.1f} | "
                       f"Avg solver: {gen_stats['avg_solver_moves']:.1f}\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()

                if not samples:
                    print("  No guess samples! Skipping training.\n")
                    continue

                # ── Step 2: append to replay buffer + train ──────────────────
                replay_buffer.append(samples)
                combined: List[GuessSample] = []
                for buf in replay_buffer:
                    combined.extend(buf)
                msg = (f"Replay buffer: {len(replay_buffer)} iters, "
                       f"{len(combined):,} total samples (this iter: {len(samples):,})\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()

                # Per-iter cosine schedule
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=config['max_epochs_per_iteration'], T_mult=1)

                best_epoch_loss = float('inf')
                patience = 0
                losses: List[float] = []
                for epoch in range(1, config['max_epochs_per_iteration'] + 1):
                    loss = train_epoch(model, optimizer, combined,
                                       config['batch_size'], device, ema)
                    scheduler.step()
                    losses.append(loss)
                    if loss < best_epoch_loss:
                        best_epoch_loss = loss
                        patience = 0
                    else:
                        patience += 1
                    if epoch == 1 or epoch % 5 == 0 or patience >= config['patience_epochs']:
                        lr = optimizer.param_groups[0]['lr']
                        print(f"  Epoch {epoch}: loss={loss:.5f} "
                              f"(best={best_epoch_loss:.5f}, p={patience}, "
                              f"lr={lr:.2e})", flush=True)
                    if patience >= config['patience_epochs']:
                        print(f"  Early stopping at epoch {epoch}", flush=True)
                        break

                msg = (f"Training: {len(losses)} epochs | "
                       f"Final={losses[-1]:.5f} | Best={best_epoch_loss:.5f}\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()
                del combined

                # ── Step 3: eval (with EMA) ──────────────────────────────────
                print(f"Evaluating ({config['eval_episodes']} games, EMA)...", flush=True)
                metrics = evaluate(model, device, board,
                                   config['eval_episodes'], ema)
                improved = metrics['win_rate'] > best_win_rate
                if improved:
                    best_win_rate = metrics['win_rate']
                    best_iteration = iteration
                    iters_without_improvement = 0
                    save_best(model, ema, iteration, metrics)
                else:
                    iters_without_improvement += 1

                iter_time = time.time() - iter_start
                total_time = time.time() - training_start
                marker = " *** NEW BEST ***" if improved else ""
                msg = (f"ITER {iteration} EVAL: Win={metrics['win_rate']:.1%} | "
                       f"GSR={metrics['guess_survival_rate']:.1%} | "
                       f"Guesses={metrics['avg_guesses']:.1f} | "
                       f"Solver={metrics['avg_solver_moves']:.1f} | "
                       f"Best={best_win_rate:.1%} @iter{best_iteration} | "
                       f"Iter: {timedelta(seconds=int(iter_time))} | "
                       f"Total: {timedelta(seconds=int(total_time))}{marker}\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()

                # ── Step 4: persist resume state ─────────────────────────────
                save_checkpoint(model, ema, optimizer, iteration, best_win_rate,
                                best_iteration, iters_without_improvement, config)
                save_replay_buffer(replay_buffer)

                if iters_without_improvement >= config['cross_iteration_patience']:
                    msg = (f"\n*** EARLY STOP: no improvement for "
                           f"{iters_without_improvement} iterations. "
                           f"Best: {best_win_rate:.1%} @iter{best_iteration} ***\n")
                    print(msg)
                    log_file.write(msg)
                    log_file.flush()
                    break

        except Exception:
            it = current_iter_holder[0]
            print(f"\nCRASHED at iteration {it}", flush=True)
            traceback.print_exc()
            try:
                save_checkpoint(model, ema, optimizer, it, best_win_rate,
                                best_iteration, iters_without_improvement, config)
                save_replay_buffer(replay_buffer)
                print(f"State saved to {PATH_LATEST}; can resume.", flush=True)
            except Exception:
                traceback.print_exc()
            raise

        # Final summary
        total_time = time.time() - training_start
        summary = (f"\n{'='*70}\n"
                   f"v5 TRAINING COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Total time: {timedelta(seconds=int(total_time))}\n"
                   f"Best win rate: {best_win_rate:.1%} @iteration {best_iteration}\n"
                   f"Saved: {PATH_BEST}\n"
                   f"Mirrored to: {PATH_CANONICAL_INFERENCE}\n"
                   f"{'='*70}\n")
        print(summary)
        log_file.write(summary)
        log_file.flush()


if __name__ == '__main__':
    main()
