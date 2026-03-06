"""
Algorithmic Minesweeper Solver — Constraint-based with demonstration recording

Implements single-cell constraint satisfaction:
1. For each revealed number, count adjacent hidden and flagged cells
2. If hidden == (number - flagged): all hidden neighbors are mines -> flag them
3. If flagged == number: all hidden neighbors are safe -> reveal them
4. If no deterministic move: pick random hidden cell (preferring cells far from numbers)

Performance: ~40-60% beginner, ~15-30% intermediate, ~5-15% expert
(better than random, worse than advanced solvers with coupled constraints)

Records (state, action) pairs for imitation learning / behavioral cloning.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai.environment import MinesweeperEnvironment


class AlgorithmicSolver:
    """Rule-based minesweeper solver that records demonstrations."""

    def __init__(self, rows: int, cols: int, mines: int):
        self.rows = rows
        self.cols = cols
        self.mines = mines

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get valid neighbor coordinates."""
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def _parse_state(self, state: np.ndarray):
        """Parse 12-channel state into useful structures.

        State channels (from get_board_array_v2):
            0: hidden, 1: flagged, 2: revealed, 3-11: number 0-8
        """
        hidden = set()
        flagged = set()
        revealed = {}  # (r,c) -> adjacent_mine_count

        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] > 0.5:  # hidden
                    hidden.add((r, c))
                elif state[r, c, 1] > 0.5:  # flagged
                    flagged.add((r, c))
                elif state[r, c, 2] > 0.5:  # revealed
                    # Find which number channel is active
                    for n in range(9):
                        if state[r, c, 3 + n] > 0.5:
                            revealed[(r, c)] = n
                            break

        return hidden, flagged, revealed

    def _find_deterministic_moves(self, hidden, flagged, revealed):
        """Find cells that are deterministically safe or mines.

        Uses iterative constraint propagation: when we identify new mines,
        we treat them as virtual flags and re-check all constraints. This
        lets us chain deductions (identify mine -> now another cell is safe).

        Returns:
            safe: set of (r,c) guaranteed safe to reveal
            mines: set of (r,c) guaranteed to be mines
        """
        safe = set()
        mines = set()
        # Virtual flags = actual flags + identified mines
        virtual_flags = set(flagged)

        # Iterate until no new deductions
        changed = True
        while changed:
            changed = False
            for (r, c), number in revealed.items():
                if number == 0:
                    continue

                neighbors = self._get_neighbors(r, c)
                # Hidden cells that aren't identified as mines yet
                hidden_neighbors = [
                    (nr, nc) for nr, nc in neighbors
                    if (nr, nc) in hidden and (nr, nc) not in virtual_flags
                ]
                flagged_count = sum(
                    1 for nr, nc in neighbors
                    if (nr, nc) in virtual_flags or (nr, nc) in flagged
                )

                num_hidden = len(hidden_neighbors)
                if num_hidden == 0:
                    continue

                remaining_mines = number - flagged_count

                if remaining_mines == 0:
                    # All mines accounted for — hidden neighbors are safe
                    for cell in hidden_neighbors:
                        if cell not in safe:
                            safe.add(cell)
                            changed = True
                elif remaining_mines == num_hidden:
                    # All hidden neighbors must be mines
                    for cell in hidden_neighbors:
                        if cell not in mines:
                            mines.add(cell)
                            virtual_flags.add(cell)
                            changed = True

        safe -= mines
        return safe, mines

    def _score_random_cell(self, r: int, c: int, revealed) -> float:
        """Score a hidden cell for random selection. Lower = safer guess.

        Prefer cells that:
        - Are far from revealed numbers (frontier is dangerous)
        - Have fewer revealed number neighbors (less constrained = less info = corner/edge)
        """
        neighbors = self._get_neighbors(r, c)
        num_info_neighbors = sum(1 for nr, nc in neighbors if (nr, nc) in revealed)
        return num_info_neighbors  # Lower is better for random guessing

    def choose_action(self, state: np.ndarray) -> int:
        """Choose the best action given current state.

        Returns action as flat index (row * cols + col) for reveal.
        """
        hidden, flagged, revealed = self._parse_state(state)

        if not hidden:
            # No hidden cells — game should be over
            return 0

        safe, _mines = self._find_deterministic_moves(hidden, flagged, revealed)

        if safe:
            # Reveal a deterministically safe cell
            # Prefer cells adjacent to more numbers (more likely to cascade)
            best = max(safe, key=lambda cell: self._score_random_cell(cell[0], cell[1], revealed))
            r, c = best
            return r * self.cols + c

        # No deterministic safe moves — guess
        # Score all hidden cells; prefer cells far from the frontier
        hidden_list = list(hidden)
        scores = [self._score_random_cell(r, c, revealed) for r, c in hidden_list]
        min_score = min(scores)
        # Pick randomly among cells with lowest score (furthest from frontier)
        candidates = [cell for cell, s in zip(hidden_list, scores) if s == min_score]
        r, c = random.choice(candidates)
        return r * self.cols + c

    def play_game(self, record: bool = True):
        """Play one game. Returns (won, demonstrations).

        demonstrations: list of (state_array, action_int) pairs
        """
        env = MinesweeperEnvironment(
            rows=self.rows, cols=self.cols, mines=self.mines,
            use_v2=True, normalize_rewards=True
        )
        state = env.reset()
        demonstrations = []
        done = False
        steps = 0
        max_steps = self.rows * self.cols * 2

        while not done and steps < max_steps:
            action = self.choose_action(state)

            if record:
                demonstrations.append((state.copy(), action))

            state, reward, done, info = env.step(action)
            steps += 1

        won = info.get('game_state') == 'won'
        return won, demonstrations

    def generate_demonstrations(self, num_games: int, verbose: bool = True):
        """Play many games and collect demonstration data.

        Returns:
            states: np.ndarray [N, H, W, 12]
            actions: np.ndarray [N] (flat action indices)
            metadata: dict with stats
        """
        all_states = []
        all_actions = []
        wins = 0

        for i in range(num_games):
            won, demos = self.play_game(record=True)
            if won:
                wins += 1
            # Record all demonstrations (including from lost games — the moves
            # before losing were still reasonable)
            for state, action in demos:
                all_states.append(state)
                all_actions.append(action)

            if verbose and (i + 1) % max(1, num_games // 10) == 0:
                win_rate = wins / (i + 1)
                print(f"  [{i+1}/{num_games}] Win rate: {win_rate:.1%} | "
                      f"Demos: {len(all_states):,}")

        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.int64)

        metadata = {
            'num_games': num_games,
            'wins': wins,
            'win_rate': wins / num_games if num_games > 0 else 0,
            'total_demonstrations': len(all_states),
            'board_size': (self.rows, self.cols, self.mines),
        }

        if verbose:
            print(f"\nSolver stats ({self.rows}x{self.cols}, {self.mines} mines):")
            print(f"  Win rate: {metadata['win_rate']:.1%}")
            print(f"  Total demonstrations: {metadata['total_demonstrations']:,}")

        return states, actions, metadata


def benchmark_solver(rows=9, cols=9, mines=10, num_games=500):
    """Quick benchmark of the algorithmic solver."""
    solver = AlgorithmicSolver(rows, cols, mines)
    wins = 0
    for _ in range(num_games):
        won, _ = solver.play_game(record=False)
        if won:
            wins += 1
    rate = wins / num_games
    print(f"Solver benchmark ({rows}x{cols}, {mines} mines): "
          f"{rate:.1%} win rate ({wins}/{num_games})")
    return rate


if __name__ == '__main__':
    print("Algorithmic Solver Benchmarks:")
    print("=" * 50)
    benchmark_solver(5, 5, 3, num_games=1000)
    benchmark_solver(9, 9, 10, num_games=1000)
    benchmark_solver(16, 16, 40, num_games=500)
    benchmark_solver(16, 30, 99, num_games=200)
