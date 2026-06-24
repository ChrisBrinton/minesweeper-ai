"""
Constraint engine for exact mine probability computation.

Given a visible minesweeper board state, computes the true P(mine) for
every hidden cell by enumerating all mine layouts consistent with the
revealed numbers.

Algorithm:
  1. Parse state into hidden/flagged/revealed sets.
  2. Run deterministic constraint propagation to find cells that are
     certainly safe or certainly mines (same logic as AlgorithmicSolver).
  3. Remove deterministic cells from hidden, adjust remaining mine count.
  4. Build constraint graph on the REMAINING uncertain cells.
  5. Decompose into connected components.
  6. Per component: enumerate valid mine assignments (bitmask brute force
     for small components, backtracking for large ones).
  7. Weight by C(unconstrained, remaining_mines) and marginalise.
"""

import numpy as np
from math import comb
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class ConstraintEngine:
    """Compute exact (or approximate) P(mine) for every hidden cell."""

    MAX_EXACT_COMPONENT_SIZE = 24

    def __init__(self, rows: int, cols: int, total_mines: int):
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines

    def compute_probabilities(
        self,
        state: np.ndarray,
        *,
        hidden: Optional[Set] = None,
        flagged: Optional[Set] = None,
        revealed: Optional[Dict] = None,
    ) -> np.ndarray:
        """Compute P(mine) for every cell.

        Returns:
            np.ndarray (rows, cols) of float32:
                NaN  -- revealed/flagged cells
                0.0  -- deterministically safe
                1.0  -- deterministically mine
                (0,1) -- uncertain, true marginal probability
        """
        if hidden is None or flagged is None or revealed is None:
            hidden, flagged, revealed = self._parse_state(state)

        probs = np.full((self.rows, self.cols), np.nan, dtype=np.float32)

        if not hidden:
            return probs

        mines_placed = len(flagged)
        remaining_mines = self.total_mines - mines_placed

        det_safe, det_mines = self._find_deterministic(hidden, flagged, revealed)

        for r, c in det_safe:
            probs[r, c] = 0.0
        for r, c in det_mines:
            probs[r, c] = 1.0

        uncertain = hidden - det_safe - det_mines
        remaining_mines -= len(det_mines)

        if not uncertain:
            return probs
        if remaining_mines <= 0:
            for r, c in uncertain:
                probs[r, c] = 0.0
            return probs
        if remaining_mines >= len(uncertain):
            for r, c in uncertain:
                probs[r, c] = 1.0
            return probs

        virtual_flags = flagged | det_mines
        constraints = self._build_constraints(uncertain, virtual_flags, revealed)
        components = self._find_components(constraints, uncertain)

        constrained_cells = set()
        for comp in components:
            constrained_cells.update(comp)

        unconstrained = uncertain - constrained_cells

        if not components:
            p = remaining_mines / len(uncertain) if uncertain else 0.0
            for r, c in uncertain:
                probs[r, c] = np.float32(p)
            return probs

        component_results = []
        for comp in components:
            comp_constraints = self._constraints_for_component(comp, constraints)
            cells = sorted(comp)
            if len(cells) <= self.MAX_EXACT_COMPONENT_SIZE:
                result = self._enumerate_exact(cells, comp_constraints, remaining_mines)
            else:
                result = self._enumerate_backtrack(cells, comp_constraints, remaining_mines)
            component_results.append((cells, result))

        self._marginalise(
            component_results, unconstrained, remaining_mines, probs
        )

        return probs

    # ── State parsing ───────────────────────────────────────────────────

    def _parse_state(self, state: np.ndarray):
        hidden = set()
        flagged = set()
        revealed = {}

        for r in range(self.rows):
            for c in range(self.cols):
                if state[r, c, 0] > 0.5:
                    hidden.add((r, c))
                elif state[r, c, 1] > 0.5:
                    flagged.add((r, c))
                elif state[r, c, 2] > 0.5:
                    for n in range(9):
                        if state[r, c, 3 + n] > 0.5:
                            revealed[(r, c)] = n
                            break
        return hidden, flagged, revealed

    # ── Deterministic constraint propagation ────────────────────────────

    def _find_deterministic(self, hidden, flagged, revealed):
        """Iterative constraint propagation to find forced cells.

        Same logic as AlgorithmicSolver._find_deterministic_moves.
        """
        safe = set()
        mines = set()
        virtual_flags = set(flagged)

        changed = True
        while changed:
            changed = False
            for (r, c), number in revealed.items():
                neighbors = self._get_neighbors(r, c)
                hidden_neighbors = [
                    (nr, nc) for nr, nc in neighbors
                    if (nr, nc) in hidden
                    and (nr, nc) not in virtual_flags
                    and (nr, nc) not in safe
                ]
                flagged_count = sum(
                    1 for nr, nc in neighbors
                    if (nr, nc) in virtual_flags
                )

                if not hidden_neighbors:
                    continue

                remaining = number - flagged_count

                if remaining == 0:
                    for cell in hidden_neighbors:
                        if cell not in safe:
                            safe.add(cell)
                            changed = True
                elif remaining == len(hidden_neighbors):
                    for cell in hidden_neighbors:
                        if cell not in mines:
                            mines.add(cell)
                            virtual_flags.add(cell)
                            changed = True

        safe -= mines
        return safe, mines

    # ── Constraint graph ────────────────────────────────────────────────

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def _build_constraints(self, hidden, flagged, revealed):
        """Build constraint list from revealed numbers.

        Only includes constraints that touch cells in `hidden`.
        `flagged` includes both real flags and deterministic mines.
        """
        constraints = []
        for (r, c), number in revealed.items():
            neighbors = self._get_neighbors(r, c)
            hidden_nbrs = frozenset(
                (nr, nc) for nr, nc in neighbors if (nr, nc) in hidden
            )
            if not hidden_nbrs:
                continue
            flagged_count = sum(
                1 for nr, nc in neighbors if (nr, nc) in flagged
            )
            effective = number - flagged_count
            if effective < 0 or effective > len(hidden_nbrs):
                continue
            constraints.append((hidden_nbrs, effective))
        return constraints

    def _find_components(self, constraints, hidden):
        if not constraints:
            return []

        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[a] = b

        for hidden_nbrs, _ in constraints:
            cells = list(hidden_nbrs)
            for cell in cells:
                if cell not in parent:
                    parent[cell] = cell
            for i in range(1, len(cells)):
                union(cells[0], cells[i])

        groups = defaultdict(set)
        for cell in parent:
            groups[find(cell)].add(cell)

        return list(groups.values())

    def _constraints_for_component(self, component, constraints):
        comp_set = set(component)
        result = []
        for hidden_nbrs, effective in constraints:
            relevant = hidden_nbrs & comp_set
            if relevant:
                result.append((relevant, effective))
        return result

    # ── Exact enumeration (bitmask brute force) ─────────────────────────

    def _enumerate_exact(self, cells, constraints, remaining_mines):
        """Brute-force enumerate all valid mine assignments for a component.

        Returns dict mapping mine_count -> {config_count, cell_config_counts}.
        """
        n = len(cells)
        cell_to_bit = {cell: i for i, cell in enumerate(cells)}

        constraint_masks = []
        for hidden_nbrs, effective in constraints:
            mask = 0
            for cell in hidden_nbrs:
                if cell in cell_to_bit:
                    mask |= 1 << cell_to_bit[cell]
            constraint_masks.append((mask, effective))

        results = {}

        for config in range(1 << n):
            mine_count = bin(config).count('1')
            if mine_count > remaining_mines:
                continue

            valid = True
            for mask, effective in constraint_masks:
                if bin(config & mask).count('1') != effective:
                    valid = False
                    break
            if not valid:
                continue

            if mine_count not in results:
                results[mine_count] = {
                    'config_count': 0,
                    'cell_config_counts': [0] * n,
                }

            entry = results[mine_count]
            entry['config_count'] += 1
            for i in range(n):
                if config & (1 << i):
                    entry['cell_config_counts'][i] += 1

        return results

    # ── Backtracking enumeration (for large components) ─────────────────

    def _enumerate_backtrack(self, cells, constraints, remaining_mines):
        """Backtracking CSP solver for components too large for brute force.

        Much faster than 2^n enumeration because constraint propagation
        prunes the search tree early.
        """
        n = len(cells)
        cell_to_idx = {cell: i for i, cell in enumerate(cells)}

        parsed = []
        for hidden_nbrs, effective in constraints:
            indices = frozenset(cell_to_idx[c] for c in hidden_nbrs if c in cell_to_idx)
            if indices:
                parsed.append((indices, effective))

        cell_constraints = [[] for _ in range(n)]
        for ci, (indices, effective) in enumerate(parsed):
            for idx in indices:
                cell_constraints[idx].append(ci)

        results = {}
        assignment = [None] * n
        constraint_mine_counts = [0] * len(parsed)
        constraint_assigned_counts = [0] * len(parsed)
        total_mines = [0]

        def backtrack(pos):
            if pos == n:
                mc = total_mines[0]
                if mc not in results:
                    results[mc] = {
                        'config_count': 0,
                        'cell_config_counts': [0] * n,
                    }
                entry = results[mc]
                entry['config_count'] += 1
                for i in range(n):
                    if assignment[i]:
                        entry['cell_config_counts'][i] += 1
                return

            for is_mine in (True, False):
                if is_mine and total_mines[0] >= remaining_mines:
                    continue

                valid = True
                if is_mine:
                    total_mines[0] += 1

                affected = cell_constraints[pos]
                for ci in affected:
                    indices, effective = parsed[ci]
                    if is_mine:
                        constraint_mine_counts[ci] += 1
                    constraint_assigned_counts[ci] += 1

                    mc = constraint_mine_counts[ci]
                    ac = constraint_assigned_counts[ci]
                    tc = len(indices)
                    unassigned = tc - ac

                    if mc > effective:
                        valid = False
                    elif mc + unassigned < effective:
                        valid = False

                if valid:
                    assignment[pos] = is_mine
                    backtrack(pos + 1)

                for ci in affected:
                    if is_mine:
                        constraint_mine_counts[ci] -= 1
                    constraint_assigned_counts[ci] -= 1
                if is_mine:
                    total_mines[0] -= 1

        backtrack(0)
        return results

    # ── Marginalisation ─────────────────────────────────────────────────

    def _marginalise(self, component_results, unconstrained, remaining_mines, probs):
        """Combine per-component results into final P(mine) per cell."""
        n_unc = len(unconstrained)

        comp_dists = []
        for cells, results in component_results:
            dist = {}
            for mc, entry in results.items():
                dist[mc] = entry['config_count']
            if not dist:
                dist[0] = 1
            comp_dists.append(dist)

        combined = self._convolve_count_distributions(comp_dists)

        grand_total = 0.0
        for total_mc, count in combined.items():
            unc_mines = remaining_mines - total_mc
            if unc_mines < 0 or unc_mines > n_unc:
                continue
            grand_total += count * comb(n_unc, unc_mines)

        if grand_total == 0:
            p_fallback = remaining_mines / max(
                sum(len(c) for c, _ in component_results) + n_unc, 1
            )
            for cells, _ in component_results:
                for r, c in cells:
                    probs[r, c] = np.float32(p_fallback)
            for r, c in unconstrained:
                probs[r, c] = np.float32(p_fallback)
            return

        for comp_idx, (cells, results) in enumerate(component_results):
            others = [d for i, d in enumerate(comp_dists) if i != comp_idx]
            other_combined = self._convolve_count_distributions(others) if others else {0: 1}

            cell_mine_weighted = [0.0] * len(cells)
            cell_total_weighted = 0.0

            for mc, entry in results.items():
                for other_mc, other_count in other_combined.items():
                    total_mc = mc + other_mc
                    unc_mines = remaining_mines - total_mc
                    if unc_mines < 0 or unc_mines > n_unc:
                        continue
                    w = entry['config_count'] * other_count * comb(n_unc, unc_mines)
                    cell_total_weighted += w
                    for i in range(len(cells)):
                        cell_mine_weighted[i] += (
                            entry['cell_config_counts'][i]
                            * other_count
                            * comb(n_unc, unc_mines)
                        )

            for i, (r, c) in enumerate(cells):
                if cell_total_weighted > 0:
                    p = cell_mine_weighted[i] / cell_total_weighted
                    probs[r, c] = np.float32(min(1.0, max(0.0, p)))
                else:
                    probs[r, c] = 0.5

        if unconstrained:
            unc_numerator = 0.0
            for total_mc, count in combined.items():
                unc_mines = remaining_mines - total_mc
                if unc_mines < 1 or unc_mines > n_unc:
                    continue
                unc_numerator += count * unc_mines * comb(n_unc, unc_mines)

            p_unc = (unc_numerator / grand_total / n_unc) if grand_total > 0 and n_unc > 0 else 0.0
            p_unc = max(0.0, min(1.0, p_unc))
            for r, c in unconstrained:
                probs[r, c] = np.float32(p_unc)

    @staticmethod
    def _convolve_count_distributions(distributions):
        """Convolve multiple mine-count distributions.

        Each distribution is {mine_count: config_count}.
        Returns {total_mines: total_config_count_product}.
        """
        if not distributions:
            return {0: 1}

        result = distributions[0].copy()
        for dist in distributions[1:]:
            new_result = {}
            for k1, c1 in result.items():
                for k2, c2 in dist.items():
                    k = k1 + k2
                    new_result[k] = new_result.get(k, 0) + c1 * c2
            result = new_result
        return result
