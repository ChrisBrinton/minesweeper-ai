#!/usr/bin/env python3
"""
Minesweeper AI — Supervised Learning Trainer v2 (Wins-Only Data)

Builds on v1's supervised approach but only trains on data from WON games.
Rationale: v1 trained on all games (72% were losses), polluting training data
with bad guesses. Won games demonstrate successful guess patterns exclusively.

Key changes from v1:
  - Only collect guess samples from won games (higher quality signal)
  - More games per iteration (need volume since ~28% win rate → ~72% filtered out)
  - Cross-iteration early stopping (stop entire run if no improvement for N iters)
  - Starts from best v1 model (28.4% win rate @ iter 9)

Algorithm:
  1. Play N games with hybrid agent (solver + current NN, epsilon=0)
  2. For WON games only: record guess situations + ground-truth labels
  3. Train NN with BCE loss on hidden cells
  4. Evaluate. If improved, save. If no improvement for patience iters, stop.

Usage:
    nohup python3 -u train_supervised_v2.py > supervised_v2_output.log 2>&1 &
    python3 -u train_supervised_v2.py --device cpu
    python3 -u train_supervised_v2.py --num-games 15000 --num-iterations 30
"""

import os
import sys
import time
import signal
import traceback
import argparse
from datetime import datetime, timedelta
from collections import deque

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

SUPERVISED_CONFIG = {
    'num_iterations': 30,
    'num_games_per_iteration': 15000,   # more games since we filter to wins only (~28% rate)
    'batch_size': 64,
    'learning_rate': 5e-5,              # slightly lower LR — we're fine-tuning a good model
    'max_epochs_per_iteration': 50,
    'patience_epochs': 5,               # early stopping within each iteration
    'iteration_patience': 8,            # cross-iteration early stopping
    'eval_episodes': 500,
    'mps_cache_clear_every': 500,       # clear MPS cache every N games during data gen
    'weight_decay': 1e-5,
    'wins_only': True,                  # only collect data from won games
}


# ─── Data Generation ─────────────────────────────────────────────────────────

class GuessSample:
    """One guess situation: board state + mine labels for hidden cells."""
    __slots__ = ['state', 'labels', 'mask']

    def __init__(self, state, labels, mask):
        self.state = state    # [H, W, 12] numpy
        self.labels = labels  # [H, W] numpy float32: 1.0=mine, 0.0=safe
        self.mask = mask      # [H, W] numpy bool: True for hidden cells (valid targets)


def generate_data(model, device, board_cfg, num_games, mps_clear_every=500,
                   use_qvalue_mode=False, wins_only=True):
    """Play games with hybrid agent, collect guess situations with ground-truth labels.

    Args:
        use_qvalue_mode: If True, interpret model output as Q-values (argmax = best).
                         If False, interpret as P(mine) logits (argmin = safest).
        wins_only: If True, only keep samples from won games (higher quality signal).
    Returns list of GuessSample.
    """
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)

    samples = []
    wins = 0
    total_guesses = 0
    total_solver_moves = 0

    model.eval()

    for game_idx in range(num_games):
        env = MinesweeperEnvironment(
            rows=rows, cols=cols, mines=mines,
            use_v2=True, normalize_rewards=True
        )
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000

        # Collect guess snapshots for this game
        game_guess_states = []   # list of (state_copy, hidden_mask)
        game_guesses = 0
        game_solver_moves = 0

        while not done and steps < max_steps:
            # Run solver
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)

            if safe:
                # Deterministic safe move
                best = max(safe, key=lambda cell: solver._score_random_cell(
                    cell[0], cell[1], revealed))
                r, c = best
                action = r * cols + c
                game_solver_moves += 1
            else:
                # Guess situation — record state
                hidden_mask = np.zeros((rows, cols), dtype=bool)
                for (r, c) in hidden:
                    if (r, c) not in known_mines:
                        hidden_mask[r, c] = True

                if hidden_mask.any():
                    game_guess_states.append((state.copy(), hidden_mask.copy()))

                # Use NN to pick action
                action_mask = env.get_action_mask()
                for (r, c) in known_mines:
                    action_mask[r * cols + c] = False
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) == 0:
                    valid_indices = np.where(env.get_action_mask())[0]
                    if len(valid_indices) == 0:
                        break

                with torch.no_grad():
                    state_tensor = (torch.FloatTensor(state)
                                    .permute(2, 0, 1).unsqueeze(0).contiguous()
                                    .to(device))
                    q_values = model(state_tensor).squeeze(0).reshape(-1)
                    valid_mask_t = torch.zeros_like(q_values, dtype=torch.bool)
                    valid_mask_t[valid_indices] = True
                    if use_qvalue_mode:
                        # Pre-trained RL model: higher Q = better action
                        q_values[~valid_mask_t] = float('-inf')
                        action = q_values.argmax().item()
                    else:
                        # Supervised model: lower output = lower P(mine) = safer
                        q_values[~valid_mask_t] = float('inf')
                        action = q_values.argmin().item()

                game_guesses += 1

            next_state, reward, done, info = env.step(action)
            steps += 1
            state = next_state

        # Game over — get ground truth mine locations
        won = info.get('game_state') == 'won'
        if won:
            wins += 1

        # Only collect samples from won games (if wins_only) or all games
        if game_guess_states and (not wins_only or won):
            labels = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    cell = env.api.game_board.get_cell(r, c)
                    if cell.is_mine:
                        labels[r, c] = 1.0

            for (snap_state, snap_mask) in game_guess_states:
                samples.append(GuessSample(snap_state, labels, snap_mask))
        total_guesses += game_guesses
        total_solver_moves += game_solver_moves

        # MPS cache management
        if (game_idx + 1) % mps_clear_every == 0 and device.type == 'mps':
            torch.mps.empty_cache()

        # Progress
        if (game_idx + 1) % max(1, num_games // 10) == 0:
            wr = wins / (game_idx + 1)
            avg_g = total_guesses / (game_idx + 1)
            print(f"  Data gen: {game_idx+1}/{num_games} | "
                  f"Win: {wr:.1%} | Avg guesses: {avg_g:.1f} | "
                  f"Samples: {len(samples):,}", flush=True)

    model.train()
    stats = {
        'games': num_games,
        'wins': wins,
        'win_rate': wins / max(num_games, 1),
        'total_guesses': total_guesses,
        'total_solver_moves': total_solver_moves,
        'samples': len(samples),
        'avg_guesses': total_guesses / max(num_games, 1),
        'avg_solver_moves': total_solver_moves / max(num_games, 1),
    }
    return samples, stats


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, samples, batch_size, device):
    """Train one epoch on guess samples. Returns average BCE loss."""
    model.train()
    indices = list(range(len(samples)))
    random.shuffle(indices)

    total_loss = 0.0
    num_batches = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) < 2:
            continue

        # Build batch tensors
        states = np.array([samples[i].state for i in batch_idx], dtype=np.float32)
        labels = np.array([samples[i].labels for i in batch_idx], dtype=np.float32)
        masks = np.array([samples[i].mask for i in batch_idx])

        # [B, 12, H, W]
        states_t = torch.from_numpy(states).permute(0, 3, 1, 2).contiguous().to(device)
        # [B, H, W]
        labels_t = torch.from_numpy(labels).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        # Forward: model outputs [B, H, W] (Q-values / logits)
        # We interpret as logits for P(mine) — apply sigmoid for BCE
        logits = model(states_t)  # [B, H, W]

        # BCE loss only on hidden cells
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels_t, reduction='none'
        )
        # Mask: only hidden cells contribute
        masked_loss = loss * masks_t.float()
        # Average over valid cells
        num_valid = masks_t.float().sum()
        if num_valid > 0:
            batch_loss = masked_loss.sum() / num_valid
        else:
            continue

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, device, board_cfg, num_episodes, use_qvalue_mode=False):
    """Evaluate the supervised model as a hybrid agent.

    Args:
        use_qvalue_mode: If True, interpret model output as Q-values (argmax = best).
                         If False, interpret as P(mine) logits (argmin = safest).
    """
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)

    wins = 0
    total_guesses = 0
    total_solver_moves = 0
    total_guess_survivals = 0
    total_guess_deaths = 0

    model.eval()

    for ep in range(num_episodes):
        env = MinesweeperEnvironment(
            rows=rows, cols=cols, mines=mines,
            use_v2=True, normalize_rewards=True
        )
        state = env.reset()
        done = False
        steps = 0
        pending_guess = False

        while not done and steps < 1000:
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)

            if safe:
                if pending_guess:
                    total_guess_survivals += 1
                    pending_guess = False
                best = max(safe, key=lambda cell: solver._score_random_cell(
                    cell[0], cell[1], revealed))
                r, c = best
                action = r * cols + c
                total_solver_moves += 1
            else:
                if pending_guess:
                    total_guess_survivals += 1

                # NN guess
                action_mask = env.get_action_mask()
                for (r, c) in known_mines:
                    action_mask[r * cols + c] = False
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) == 0:
                    valid_indices = np.where(env.get_action_mask())[0]
                    if len(valid_indices) == 0:
                        break

                with torch.no_grad():
                    state_tensor = (torch.FloatTensor(state)
                                    .permute(2, 0, 1).unsqueeze(0).contiguous()
                                    .to(device))
                    logits = model(state_tensor).squeeze(0).reshape(-1)
                    if use_qvalue_mode:
                        # Pre-trained RL model: higher Q = better action
                        valid_mask_t = torch.zeros_like(logits, dtype=torch.bool)
                        valid_mask_t[valid_indices] = True
                        logits[~valid_mask_t] = float('-inf')
                        action = logits.argmax().item()
                    else:
                        # Supervised model: lower P(mine) = safer
                        probs = torch.sigmoid(logits)
                        valid_mask_t = torch.zeros_like(probs, dtype=torch.bool)
                        valid_mask_t[valid_indices] = True
                        probs[~valid_mask_t] = float('inf')
                        action = probs.argmin().item()

                total_guesses += 1
                pending_guess = True

            next_state, reward, done, info = env.step(action)
            steps += 1
            state = next_state

        # Resolve last pending guess
        if pending_guess:
            game_state = info.get('game_state')
            if game_state == 'lost':
                total_guess_deaths += 1
            else:
                total_guess_survivals += 1

        if info.get('game_state') == 'won':
            wins += 1

        # MPS cache
        if (ep + 1) % 100 == 0 and device.type == 'mps':
            torch.mps.empty_cache()

    total_outcomes = total_guess_survivals + total_guess_deaths
    return {
        'win_rate': wins / max(num_episodes, 1),
        'avg_guesses': total_guesses / max(num_episodes, 1),
        'avg_solver_moves': total_solver_moves / max(num_episodes, 1),
        'guess_survival_rate': (total_guess_survivals / max(total_outcomes, 1)),
        'wins': wins,
        'total_guesses': total_guesses,
    }


# ─── Utils ───────────────────────────────────────────────────────────────────

def get_device(preference='auto'):
    if preference == 'mps' or (preference == 'auto' and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif preference == 'cuda' or (preference == 'auto' and torch.cuda.is_available()):
        return torch.device('cuda')
    return torch.device('cpu')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Minesweeper Supervised Trainer v2 (Wins-Only)')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu', 'cuda'])
    parser.add_argument('--model', type=str, default=None,
                        help='Path to starting model (default: models_v3/hybrid/expert/best_model.pth)')
    parser.add_argument('--num-games', type=int, default=None,
                        help='Games per data generation iteration')
    parser.add_argument('--num-iterations', type=int, default=None,
                        help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    config = SUPERVISED_CONFIG.copy()
    board = EXPERT_CONFIG

    if args.num_games:
        config['num_games_per_iteration'] = args.num_games
    if args.num_iterations:
        config['num_iterations'] = args.num_iterations
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr

    # Paths — start from best v1 supervised model (iter 9, 28.4%)
    pretrained_path = args.model or 'models_v3/supervised/expert/best_model.pth'
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'models_v3', 'supervised_v2', 'expert')
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'training_log.txt')

    # Create model
    model = MinesweeperResNet(input_channels=12).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    # Load pre-trained weights
    if os.path.exists(pretrained_path):
        print(f"Loading pre-trained model: {pretrained_path}")
        ckpt = torch.load(pretrained_path, weights_only=True, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded: win_rate={ckpt.get('win_rate', '?')}, "
              f"stage={ckpt.get('stage', '?')}")
    else:
        print(f"WARNING: No pre-trained model at {pretrained_path}, starting from scratch")

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    # ─── Signal handling ──────────────────────────────────────────────────
    current_iteration = [0]
    shutdown_requested = [False]

    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        it = current_iteration[0]
        path = os.path.join(save_dir, f'interrupted_iter{it}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': it,
        }, path)
        msg = f"\nINTERRUPTED by {sig_name} at iteration {it}. Saved: {path}\n"
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

    # ─── Training loop ────────────────────────────────────────────────────
    with open(log_path, 'a') as log_file:
        header = (f"\n{'='*70}\n"
                  f"SUPERVISED TRAINING v2 (WINS-ONLY) START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"Architecture: MinesweeperResNet ({param_count:,} params)\n"
                  f"Board: {board['rows']}x{board['cols']}, {board['mines']} mines\n"
                  f"Device: {device}\n"
                  f"LR: {config['learning_rate']} | Batch: {config['batch_size']} | "
                  f"Weight decay: {config['weight_decay']}\n"
                  f"Games/iter: {config['num_games_per_iteration']} | "
                  f"Max epochs/iter: {config['max_epochs_per_iteration']} | "
                  f"Patience: {config['patience_epochs']} epochs\n"
                  f"Iterations: {config['num_iterations']} | "
                  f"Eval episodes: {config['eval_episodes']}\n"
                  f"Wins only: {config['wins_only']} | "
                  f"Iteration patience: {config['iteration_patience']}\n"
                  f"Pre-trained: {pretrained_path}\n"
                  f"{'='*70}\n")
        print(header)
        log_file.write(header)
        log_file.flush()

        # ─── Baseline evaluation ──────────────────────────────────────────
        print("Evaluating baseline model (supervised P(mine) mode)...", flush=True)
        baseline = evaluate(model, device, board, config['eval_episodes'],
                            use_qvalue_mode=False)
        msg = (f"BASELINE: Win={baseline['win_rate']:.1%} | "
               f"GSR={baseline['guess_survival_rate']:.1%} | "
               f"Guesses={baseline['avg_guesses']:.1f} | "
               f"Solver={baseline['avg_solver_moves']:.1f}\n")
        print(msg)
        log_file.write(msg)
        log_file.flush()

        best_win_rate = baseline['win_rate']
        best_iteration = 0
        iters_without_improvement = 0

        # Save baseline as starting best
        torch.save({
            'model_state_dict': model.state_dict(),
            'iteration': 0,
            'win_rate': best_win_rate,
            'metrics': baseline,
        }, os.path.join(save_dir, 'best_model.pth'))

        training_start = time.time()

        try:
            for iteration in range(1, config['num_iterations'] + 1):
                current_iteration[0] = iteration
                iter_start = time.time()

                msg = f"\n{'─'*70}\nITERATION {iteration}/{config['num_iterations']}\n{'─'*70}\n"
                print(msg)
                log_file.write(msg)
                log_file.flush()

                # ── Step 1: Generate data ─────────────────────────────────
                print(f"Generating data: {config['num_games_per_iteration']} games...",
                      flush=True)
                gen_start = time.time()
                # v2 starts from supervised v1 model — always P(mine) mode
                samples, gen_stats = generate_data(
                    model, device, board,
                    config['num_games_per_iteration'],
                    config['mps_cache_clear_every'],
                    use_qvalue_mode=False,
                    wins_only=config['wins_only'],
                )
                gen_time = time.time() - gen_start

                msg = (f"Data gen: {gen_stats['games']} games in {gen_time:.0f}s | "
                       f"Win: {gen_stats['win_rate']:.1%} | "
                       f"Samples: {gen_stats['samples']:,} | "
                       f"Avg guesses: {gen_stats['avg_guesses']:.1f} | "
                       f"Avg solver: {gen_stats['avg_solver_moves']:.1f}\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()

                if len(samples) == 0:
                    msg = "  No guess samples collected! Skipping training.\n"
                    print(msg)
                    log_file.write(msg)
                    log_file.flush()
                    continue

                # ── Step 2: Train ─────────────────────────────────────────
                print(f"Training on {len(samples):,} samples...", flush=True)
                best_epoch_loss = float('inf')
                patience_counter = 0
                epoch_losses = []

                for epoch in range(1, config['max_epochs_per_iteration'] + 1):
                    loss = train_epoch(model, optimizer, samples, config['batch_size'], device)
                    epoch_losses.append(loss)

                    if loss < best_epoch_loss:
                        best_epoch_loss = loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if epoch % 5 == 0 or epoch == 1 or patience_counter >= config['patience_epochs']:
                        print(f"  Epoch {epoch}: loss={loss:.5f} "
                              f"(best={best_epoch_loss:.5f}, patience={patience_counter})",
                              flush=True)

                    if patience_counter >= config['patience_epochs']:
                        print(f"  Early stopping at epoch {epoch}", flush=True)
                        break

                    # MPS cache
                    if device.type == 'mps':
                        torch.mps.empty_cache()

                final_loss = epoch_losses[-1]
                num_epochs = len(epoch_losses)
                msg = (f"Training: {num_epochs} epochs | "
                       f"Final loss: {final_loss:.5f} | "
                       f"Best loss: {best_epoch_loss:.5f}\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()

                # Free sample memory
                del samples

                # ── Step 3: Evaluate ──────────────────────────────────────
                print(f"Evaluating ({config['eval_episodes']} games)...", flush=True)
                metrics = evaluate(model, device, board, config['eval_episodes'])

                improved = metrics['win_rate'] > best_win_rate
                if improved:
                    best_win_rate = metrics['win_rate']
                    best_iteration = iteration
                    iters_without_improvement = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iteration,
                        'win_rate': best_win_rate,
                        'metrics': metrics,
                    }, os.path.join(save_dir, 'best_model.pth'))
                else:
                    iters_without_improvement += 1

                iter_time = time.time() - iter_start
                total_elapsed = time.time() - training_start

                marker = " *** NEW BEST ***" if improved else ""
                msg = (f"ITER {iteration} EVAL: "
                       f"Win={metrics['win_rate']:.1%} | "
                       f"GSR={metrics['guess_survival_rate']:.1%} | "
                       f"Guesses={metrics['avg_guesses']:.1f} | "
                       f"Solver={metrics['avg_solver_moves']:.1f} | "
                       f"Best={best_win_rate:.1%} @iter{best_iteration} | "
                       f"Iter time: {timedelta(seconds=int(iter_time))} | "
                       f"Total: {timedelta(seconds=int(total_elapsed))}"
                       f"{marker}\n")
                print(msg)
                log_file.write(msg)
                log_file.flush()

                # Checkpoint every iteration
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration,
                    'win_rate': metrics['win_rate'],
                    'best_win_rate': best_win_rate,
                    'best_iteration': best_iteration,
                    'metrics': metrics,
                }, os.path.join(save_dir, f'checkpoint_iter{iteration}.pth'))

                # Cross-iteration early stopping
                if iters_without_improvement >= config['iteration_patience']:
                    stop_msg = (f"\n*** EARLY STOPPING: No improvement for "
                                f"{config['iteration_patience']} iterations. "
                                f"Best: {best_win_rate:.1%} @iter{best_iteration} ***\n")
                    print(stop_msg, flush=True)
                    log_file.write(stop_msg)
                    log_file.flush()
                    break

        except Exception as e:
            it = current_iteration[0]
            path = os.path.join(save_dir, f'crash_iter{it}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': it,
            }, path)
            crash_msg = (f"\nCRASHED at iteration {it}: {e}\n"
                         f"Checkpoint saved: {path}\n"
                         f"{''.join(traceback.format_exc())}\n")
            print(crash_msg, flush=True)
            log_file.write(crash_msg)
            log_file.flush()
            raise

        # ─── Final Summary ────────────────────────────────────────────────
        total_time = time.time() - training_start
        summary = (f"\n{'='*70}\n"
                   f"SUPERVISED TRAINING v2 (WINS-ONLY) COMPLETE: "
                   f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Total time: {timedelta(seconds=int(total_time))}\n"
                   f"Baseline win rate: {baseline['win_rate']:.1%}\n"
                   f"Best win rate: {best_win_rate:.1%} @iteration {best_iteration}\n"
                   f"Improvement: {best_win_rate - baseline['win_rate']:+.1%}\n"
                   f"Data filter: wins only\n"
                   f"{'='*70}\n")
        print(summary)
        log_file.write(summary)
        log_file.flush()


if __name__ == '__main__':
    main()
