#!/usr/bin/env python3
"""
Export training data from recorded game history.

Replays each completed game from ~/.minesweeper/history.json against the
algorithmic solver and records the (state, mask) pair at every guess
situation — every player reveal-click made when the solver had no
deterministic safe move available. The post-hoc mine layout from the
record gives the per-cell mine label.

Output is a single .npz file. Because boards differ in shape across
difficulties, samples are stored as separate arrays per difficulty:

    beginner_states     (N, 9, 9, 12)   float32  one-hot board state
    beginner_labels     (N, 9, 9)       float32  1.0 = mine, 0.0 = safe
    beginner_masks      (N, 9, 9)       bool     valid hidden candidates
    intermediate_states (N, 16, 16, 12) float32
    intermediate_labels (N, 16, 16)     float32
    intermediate_masks  (N, 16, 16)     bool
    expert_states       (N, 16, 30, 12) float32
    expert_labels       (N, 16, 30)     float32
    expert_masks        (N, 16, 30)     bool

Difficulties with no samples are simply absent from the file.

Usage:
    python export_training_data.py
    python export_training_data.py --output my_data.npz
    python export_training_data.py --difficulty expert --result won
    python export_training_data.py --all-reveals  # capture every reveal,
                                                  # not only guesses
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from game.board import GameBoard, GameState, CellState
from ai.game_api import MinesweeperAPI, Action
from ai.algorithmic_solver import AlgorithmicSolver


DEFAULT_HISTORY_PATH = os.path.join(os.path.expanduser('~'), '.minesweeper', 'history.json')
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'training_data.npz')


# ─── Load + filter ────────────────────────────────────────────────────────────


def _load_history(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'games' in data:
        return data['games']
    if isinstance(data, list):
        return data
    return []


def _filter_records(records: List[Dict], difficulty: str, result: str
                    ) -> List[Dict]:
    out = []
    for r in records:
        if difficulty != 'all' and r.get('difficulty') != difficulty:
            continue
        res = r.get('result')
        if result == 'completed' and res not in ('won', 'lost'):
            continue
        if result == 'won' and res != 'won':
            continue
        if result == 'lost' and res != 'lost':
            continue
        if result == 'all' and res is None:
            continue
        out.append(r)
    return out


# ─── Replay one game → samples ────────────────────────────────────────────────


def _build_api_from_record(record: Dict) -> MinesweeperAPI:
    """Construct a MinesweeperAPI with the record's mine layout pre-placed."""
    rows = record['rows']
    cols = record['cols']
    mines = record['mines']
    mine_positions = [tuple(p) for p in record.get('mine_positions', [])]

    api = MinesweeperAPI(rows, cols, mines)
    board = api.game_board
    for (r, c) in mine_positions:
        board.board[r][c].place_mine()
    board._calculate_adjacent_mines()
    board.mines_placed = True
    board.game_state = GameState.PLAYING
    return api


def extract_samples(record: Dict, capture_all_reveals: bool = False
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return (state, mask) pairs captured before each player reveal click.

    By default only guess situations are emitted (solver has no deterministic
    safe move). With capture_all_reveals=True, every reveal click yields a
    sample.
    """
    rows = record['rows']
    cols = record['cols']
    mines = record['mines']
    mine_positions = [tuple(p) for p in record.get('mine_positions', [])]
    moves = record.get('moves', [])
    if not mine_positions or not moves:
        return []

    api = _build_api_from_record(record)
    solver = AlgorithmicSolver(rows, cols, mines)
    samples: List[Tuple[np.ndarray, np.ndarray]] = []

    for move in moves:
        action = move.get('a')
        r, c = move.get('r'), move.get('c')
        if r is None or c is None:
            continue

        if action == 'reveal':
            state = api.get_board_array_v2()
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)

            is_guess = capture_all_reveals or not safe
            if is_guess and hidden:
                mask = np.zeros((rows, cols), dtype=bool)
                for (hr, hc) in hidden:
                    if (hr, hc) not in known_mines:
                        mask[hr, hc] = True
                if mask.any():
                    samples.append((state.copy(), mask.copy()))

            api.take_action(r, c, Action.REVEAL)
            if api.game_board.game_state == GameState.LOST:
                break
        elif action == 'flag':
            api.take_action(r, c, Action.FLAG)
        elif action == 'unflag':
            api.take_action(r, c, Action.UNFLAG)

    return samples


def _labels_for_record(record: Dict) -> np.ndarray:
    rows, cols = record['rows'], record['cols']
    labels = np.zeros((rows, cols), dtype=np.float32)
    for (mr, mc) in record.get('mine_positions', []):
        labels[mr, mc] = 1.0
    return labels


def record_to_arrays(record: Dict, capture_all_reveals: bool = False
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (states, labels, masks) for a single record, or None if no samples.

    Each array has the per-game samples stacked on axis 0:
        states: (N, H, W, 12) float32
        labels: (N, H, W)     float32  (mine map repeated per sample)
        masks:  (N, H, W)     bool
    """
    samples = extract_samples(record, capture_all_reveals=capture_all_reveals)
    if not samples:
        return None
    label_one = _labels_for_record(record)
    states = np.stack([s for (s, _) in samples], axis=0)
    masks = np.stack([m for (_, m) in samples], axis=0)
    labels = np.broadcast_to(label_one, (len(samples),) + label_one.shape).copy()
    return states, labels, masks


# ─── File I/O ─────────────────────────────────────────────────────────────────


def _load_existing(path: str) -> Dict[str, np.ndarray]:
    """Read an existing export .npz into a {key: array} dict."""
    if not os.path.exists(path):
        return {}
    try:
        with np.load(path) as data:
            return {k: data[k].copy() for k in data.files}
    except Exception as e:
        print(f"Could not read existing {path}: {e}")
        return {}


def _save_arrays(path: str, arrays: Dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Ensure the tmp path ends in .npz so np.savez_compressed doesn't append
    # its own .npz suffix (which would break the os.replace below).
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def append_record_to_file(record: Dict, output_path: str,
                          capture_all_reveals: bool = False) -> int:
    """Append one game's samples to the export file. Returns # samples added."""
    arrays_for_game = record_to_arrays(record, capture_all_reveals=capture_all_reveals)
    if arrays_for_game is None:
        return 0
    states, labels, masks = arrays_for_game
    difficulty = record['difficulty']
    keys = (f'{difficulty}_states', f'{difficulty}_labels', f'{difficulty}_masks')

    existing = _load_existing(output_path)
    for key, arr in zip(keys, (states, labels, masks)):
        if key in existing:
            existing[key] = np.concatenate([existing[key], arr], axis=0)
        else:
            existing[key] = arr
    _save_arrays(output_path, existing)
    return len(states)


# ─── Bulk export (CLI) ────────────────────────────────────────────────────────


def export(records: List[Dict], output_path: str, capture_all_reveals: bool
           ) -> Dict[str, Dict]:
    """Build a fresh export file from a list of records (replaces target file).

    Returns per-difficulty summary {difficulty: {games, contributing, samples}}.
    """
    per_difficulty: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    summary: Dict[str, Dict] = {}

    for record in records:
        difficulty = record.get('difficulty')
        if difficulty is None:
            continue
        summary.setdefault(difficulty,
                           {'games': 0, 'contributing': 0, 'samples': 0})
        summary[difficulty]['games'] += 1
        arrays = record_to_arrays(record, capture_all_reveals=capture_all_reveals)
        if arrays is None:
            continue
        per_difficulty.setdefault(difficulty, []).append(arrays)
        summary[difficulty]['contributing'] += 1
        summary[difficulty]['samples'] += arrays[0].shape[0]

    out_arrays: Dict[str, np.ndarray] = {}
    for difficulty, games in per_difficulty.items():
        states = np.concatenate([g[0] for g in games], axis=0)
        labels = np.concatenate([g[1] for g in games], axis=0)
        masks = np.concatenate([g[2] for g in games], axis=0)
        out_arrays[f'{difficulty}_states'] = states
        out_arrays[f'{difficulty}_labels'] = labels
        out_arrays[f'{difficulty}_masks'] = masks

    if out_arrays:
        _save_arrays(output_path, out_arrays)

    summary['_path'] = output_path if out_arrays else None
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Export training data from recorded game history.'
    )
    parser.add_argument('--input', default=DEFAULT_HISTORY_PATH,
                        help=f'Path to history.json (default: {DEFAULT_HISTORY_PATH})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH,
                        help=f'Output .npz file (default: {DEFAULT_OUTPUT_PATH})')
    parser.add_argument('--difficulty', default='all',
                        choices=['all', 'beginner', 'intermediate', 'expert'])
    parser.add_argument('--result', default='completed',
                        choices=['all', 'completed', 'won', 'lost'])
    parser.add_argument('--all-reveals', action='store_true',
                        help='Capture every reveal click, not only guess situations')
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"No history file at {args.input}")
        return 1

    records = _load_history(args.input)
    print(f"Loaded {len(records)} game(s) from {args.input}")

    filtered = _filter_records(records, args.difficulty, args.result)
    if not filtered:
        print(f"No records match (difficulty={args.difficulty}, result={args.result}).")
        return 1
    print(f"After filter: {len(filtered)} game(s) "
          f"(difficulty={args.difficulty}, result={args.result})")

    summary = export(filtered, args.output, args.all_reveals)
    out_path = summary.pop('_path', None)

    print()
    total = 0
    for difficulty, info in summary.items():
        if info['samples']:
            print(f"  {difficulty}: {info['samples']} samples from "
                  f"{info['contributing']}/{info['games']} games")
            total += info['samples']
        else:
            print(f"  {difficulty}: 0 samples (none of {info['games']} games "
                  f"contained guess situations)")
    if out_path:
        print(f"Wrote {total} samples to {out_path}")
    else:
        print("No samples written.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
