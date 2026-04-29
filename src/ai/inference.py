"""
In-game move-suggestion inference.

Mirrors the hybrid agent the model was trained against:
  - If the algorithmic solver finds a deterministic safe cell, suggest it.
  - Otherwise run the network on the 12-channel state and pick the hidden
    cell with the lowest predicted mine probability.

Architecture detection: tries MinesweeperResNetV4 first (1M params), falls
back to MinesweeperResNet (461K params, v3) on state_dict mismatch.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai.algorithmic_solver import AlgorithmicSolver
from src.ai.models_v3 import MinesweeperResNet
from src.ai.models_v4 import MinesweeperResNetV4
from src.game.board import GameBoard


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _state_from_board(game_board: GameBoard) -> np.ndarray:
    """Build the 12-channel one-hot state from a GameBoard.

    Channels: 0 hidden, 1 flagged, 2 revealed, 3..11 number 0..8 (one-hot
    on revealed cells). Matches MinesweeperAPI.get_board_array_v2.
    """
    rows, cols = game_board.rows, game_board.cols
    state = np.zeros((rows, cols, 12), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cell = game_board.board[r][c]
            if cell.is_revealed():
                state[r, c, 2] = 1.0
                adj = cell.adjacent_mines
                if 0 <= adj <= 8:
                    state[r, c, 3 + adj] = 1.0
            elif cell.is_flagged():
                state[r, c, 1] = 1.0
            else:
                state[r, c, 0] = 1.0
    return state


def _load_checkpoint(path: str) -> Tuple[Dict, Dict]:
    """Returns (state_dict, metadata)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        meta = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    else:
        sd = ckpt
        meta = {}
    return sd, meta


def _build_model_for_state_dict(sd: Dict) -> torch.nn.Module:
    """Try v4 first (1M params), fall back to v3 (461K params)."""
    last_err = None
    for cls in (MinesweeperResNetV4, MinesweeperResNet):
        m = cls()
        try:
            m.load_state_dict(sd)
            return m
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Checkpoint state_dict didn't match v4 or v3 architectures: {last_err}"
    )


class MinesweeperInference:
    """Lazy-loaded model wrapper for the Suggest-move button."""

    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.model_path = model_path
        self.device = device or _pick_device()
        self.model: Optional[torch.nn.Module] = None
        self.metadata: Dict = {}

    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        """Load the model. Idempotent; safe to call repeatedly."""
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        sd, meta = _load_checkpoint(self.model_path)
        self.model = _build_model_for_state_dict(sd).to(self.device).eval()
        self.metadata = meta

    def suggest_move(self, game_board: GameBoard) -> Optional[Dict]:
        """Suggest the next move for the given board.

        Returns a dict like:
            {'row': int, 'col': int, 'source': 'solver'|'model',
             'mine_probability': float | None}
        or None if no hidden cells remain.
        """
        rows, cols = game_board.rows, game_board.cols
        state = _state_from_board(game_board)

        solver = AlgorithmicSolver(rows, cols, game_board.total_mines)
        hidden, flagged, revealed = solver._parse_state(state)
        if not hidden:
            return None

        safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)
        if safe:
            # Among guaranteed-safe cells, pick the one nearest the frontier
            # (most info-revealing neighbors) so the suggestion feels useful.
            r, c = max(
                safe,
                key=lambda cell: solver._score_random_cell(cell[0], cell[1], revealed),
            )
            return {
                'row': int(r), 'col': int(c),
                'source': 'solver',
                'mine_probability': None,
            }

        # No deterministic safe move — fall through to the network
        if self.model is None:
            self.load()

        candidates = [(r, c) for (r, c) in hidden if (r, c) not in known_mines]
        if not candidates:
            candidates = list(hidden)
        if not candidates:
            return None

        with torch.no_grad():
            st = (torch.from_numpy(state)
                  .permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device))
            logits = self.model(st).squeeze(0)  # (rows, cols)
            probs = torch.sigmoid(logits).cpu().numpy()

        best_r, best_c = min(candidates, key=lambda cell: probs[cell])
        return {
            'row': int(best_r), 'col': int(best_c),
            'source': 'model',
            'mine_probability': float(probs[best_r, best_c]),
        }
