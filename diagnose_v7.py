#!/usr/bin/env python3
"""
Diagnose the v7 plateau.

Answers two questions, using train_v7.py's own code paths so the numbers
match a real run:

  1. Is loss=~0.386 the irreducible ENTROPY FLOOR of the constraint-engine
     labels?  (If the mean label entropy ~= the training loss, the model has
     learned the labels perfectly and there is no gradient left.)

  2. Has the model MATCHED ITS TEACHER?  Compares:
       - model-only win rate  (no solver/engine at inference)
       - hybrid win rate       (solver logic + model guesses)
       - engine-only win rate  (solver logic + lowest-P(mine) engine guess)
     If model-only ~= engine-only, distillation is done — the ceiling is the
     teacher's policy, not a training bug.

Usage:
    python diagnose_v7.py                          # default sample sizes
    python diagnose_v7.py --entropy-games 100 \
        --eval-episodes 2000 --engine-games 300
    python diagnose_v7.py --model best_model.pth
"""

import argparse
import numpy as np
import torch

# Reuse the EXACT code paths from the trainer.
from train_v7 import (
    EXPERT_CONFIG,
    MinesweeperResNetV4,
    generate_data,
    evaluate,
)
from src.ai.algorithmic_solver import AlgorithmicSolver
from src.ai.constraint_engine import ConstraintEngine
from src.ai.environment import MinesweeperEnvironment


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = MinesweeperResNetV4().to(device)
    model.load_state_dict(sd)
    model.eval()
    # Surface what the checkpoint thinks it scored, if anything.
    if isinstance(ckpt, dict):
        print(f"  checkpoint model_only_win_rate: {ckpt.get('model_only_win_rate')}")
        print(f"  checkpoint hybrid_win_rate:     {ckpt.get('hybrid_win_rate')}")
        print(f"  checkpoint iteration:           {ckpt.get('iteration')}")
    return model


# ── Check 1: label entropy floor ────────────────────────────────────────────

def label_entropy_floor(model, device, num_games):
    """Mean binary entropy of the constraint-engine labels over hidden cells.

    This is the lowest possible value masked BCE can reach. Generated with the
    same hybrid policy train_v7 uses, so the state distribution matches.
    """
    print(f"\n[1] Generating {num_games} games to measure label entropy floor...")
    samples, stats = generate_data(model, device, EXPERT_CONFIG, num_games)
    print(f"    data-gen win rate: {stats['win_rate']:.1%}  "
          f"({stats['wins']}/{stats['games']}), {stats['samples']:,} samples")

    ent_sum = 0.0
    cell_count = 0
    for s in samples:
        p = s.labels[s.mask]                 # P(mine) on hidden cells only
        if p.size == 0:
            continue
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))   # nats
        ent_sum += float(h.sum())
        cell_count += p.size

    floor = ent_sum / max(cell_count, 1)
    return floor, cell_count


# ── Check 2: engine-only (teacher) win rate ─────────────────────────────────

def engine_only_win_rate(num_games):
    """Pure constraint-engine policy: deterministic moves, else guess the
    hidden cell with the lowest engine P(mine). This is the teacher the model
    is being distilled toward — its win rate is the distillation ceiling.
    """
    rows, cols, mines = (EXPERT_CONFIG['rows'], EXPERT_CONFIG['cols'],
                         EXPERT_CONFIG['mines'])
    solver = AlgorithmicSolver(rows, cols, mines)
    engine = ConstraintEngine(rows, cols, mines)
    wins = 0

    print(f"\n[2b] Running {num_games} engine-only games "
          f"(slow: engine probabilities per guess)...")
    for g in range(num_games):
        env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines,
                                     use_v2=True, normalize_rewards=True)
        state = env.reset()
        done = False
        steps = 0
        info = {}
        while not done and steps < 1000:
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(
                hidden, flagged, revealed)

            if safe:
                best = max(safe, key=lambda c: solver._score_random_cell(
                    c[0], c[1], revealed))
                action = best[0] * cols + best[1]
            else:
                if len(hidden) == 0:
                    break
                probs = engine.compute_probabilities(
                    state, hidden=hidden, flagged=flagged, revealed=revealed)
                density = (mines - len(flagged)) / max(len(hidden), 1)
                known = set(known_mines)
                best_cell, best_p = None, 2.0
                for (r, c) in hidden:
                    if (r, c) in known:
                        continue
                    p = probs[r, c]
                    if np.isnan(p):
                        p = density
                    if p < best_p:
                        best_p, best_cell = p, (r, c)
                if best_cell is None:
                    break
                action = best_cell[0] * cols + best_cell[1]

            state, _, done, info = env.step(action)
            steps += 1

        if info.get('game_state') == 'won':
            wins += 1
        if (g + 1) % max(1, num_games // 10) == 0:
            print(f"    engine-only: {g+1}/{num_games} | "
                  f"Win: {wins/(g+1):.1%}", flush=True)

    return wins / max(num_games, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--entropy-games", type=int, default=100,
                    help="games used to estimate the label entropy floor")
    ap.add_argument("--eval-episodes", type=int, default=2000,
                    help="episodes for model-only / hybrid win rate")
    ap.add_argument("--engine-games", type=int, default=300,
                    help="games for the (slow) engine-only teacher policy")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = load_model(args.model, device)

    # Check 1 — entropy floor
    floor, n = label_entropy_floor(model, device, args.entropy_games)

    # Check 2a — model-only + hybrid (train_v7's own evaluate)
    print(f"\n[2a] Evaluating model-only + hybrid over "
          f"{args.eval_episodes} episodes...")
    ev = evaluate(model, device, EXPERT_CONFIG, args.eval_episodes, ema=None)
    mo = ev['model_only_win_rate']
    hy = ev['hybrid_win_rate']

    # Check 2b — engine-only teacher
    eng = engine_only_win_rate(args.engine_games)

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"Label entropy floor (mean over {n:,} hidden cells): "
          f"{floor:.6f} nats")
    print(f"  -> training loss was ~0.3864. If these match, the model is")
    print(f"     sitting on the floor: it learned the labels perfectly and")
    print(f"     there is no gradient left to descend.")
    print()
    print(f"Win rates:")
    print(f"  model-only : {mo:.1%}")
    print(f"  hybrid     : {hy:.1%}")
    print(f"  engine-only: {eng:.1%}   (the teacher / distillation ceiling)")
    print()
    if abs(mo - eng) <= 0.02:
        print("  -> model-only ~= engine-only: the model has MATCHED its")
        print("     teacher. Distillation is done; this objective cannot")
        print("     push higher. Raising the ceiling needs an outcome-based")
        print("     objective (RL on win/loss, or search on guesses).")
    elif mo < eng - 0.02:
        print("  -> model-only is BELOW engine-only: the model has not fully")
        print("     matched the teacher yet — there may still be a little")
        print("     headroom under this objective.")
    else:
        print("  -> model-only is ABOVE engine-only: the model generalizes")
        print("     past greedy-lowest-probability guessing. Interesting!")
    print("=" * 60)


if __name__ == "__main__":
    main()
