# v6 Design — Constraint-Implied Labels + Mine Probability Heatmap

Two related features that share a single underlying piece (a constraint
engine that estimates per-cell mine probability from a visible board
state):

- **v6 training** — replace the noisy post-hoc 0/1 mine labels with
  constraint-implied probabilities. Likely path past the v5 plateau.
- **UI heatmap** — a button that overlays each cell with a colour from
  bright red (P(mine)≈1) to transparent (no information / fully safe).

The same constraint engine drives both. Build it once for v6, get the
heatmap nearly for free.

---

## 1. The label-noise problem

The v3/v4/v5 supervised pipeline trains on `(state_at_guess, post_hoc
mine_map)` tuples. The mine label for each cell is **1 if it was a
mine, 0 if it was safe** — i.e. ground truth from after the game ended.

This is noisy in a specific, asymmetric way:

- A cell that the constraint solver couldn't deduce, but turned out
  safe, becomes a "0" label. The model learns: situations like this →
  safe. But "this turned out safe" was sometimes lucky — the actual
  P(mine) given the visible info might have been 0.4.
- A cell that turned out to be a mine becomes a "1" label, even if its
  P(mine) given the visible info was only 0.3. The model learns: this
  pattern → mine. Same noise issue, opposite direction.

The model converges to the right thing **on average** (BCE pushes the
output toward the true probability over many samples) but the per-sample
gradient is noisy, slowing convergence and capping ceiling.

## 2. Constraint-implied probability labels

Given a visible state, the **true** P(mine|state) for each hidden cell
is well-defined: among all mine layouts consistent with the visible
numbers and the total mine count, what fraction place a mine at this
cell?

For a small board (beginner 9×9, 10 mines), full enumeration is
tractable — ~10⁶ to 10⁸ consistent layouts in the worst case, often
much fewer when many cells are constrained. For expert (16×30, 99
mines) full enumeration is intractable, but **per-component**
enumeration is.

### Component decomposition

The minesweeper constraint graph (hidden cells + numbered cells they
border) almost always factors into independent components: the cells
near the top-left have nothing to do with cells near the bottom-right
when there are no shared constraint edges.

For each connected component:
1. Enumerate all configurations of mine counts within that component
   that are consistent with the local numbers.
2. Weight each configuration by `C(remaining_unconstrained_cells,
   remaining_mines_after_this_component)` — the number of ways to
   distribute the rest of the mines among the unconstrained cells.
3. Per-cell P(mine) = (sum of weights where this cell is a mine) /
   (sum of all weights).
4. For unconstrained cells (no number neighbour): P(mine) = (mines
   remaining after constrained components) / (unconstrained cells
   remaining).

Components are usually 5–20 hidden cells in size in late-game expert
boards, so each enumeration is ~2¹⁰–2²⁰ — milliseconds with bitmask
tricks. Total per-state cost should stay well under 50 ms even on
expert with the algorithmic-solver-stuck states we sample.

### What this buys training

- **Denoised gradients**: every cell carries its true P(mine) given
  the local information, not a coin-flip outcome.
- **Information-rich masks**: the loss masks change. Cells where
  P(mine) ∈ {0, 1} can be excluded (the algorithmic solver already
  knows them; no learning needed). The interesting cells are the
  fractional ones.
- **Implicit calibration**: the model output is now a regression
  target, not a classification, which makes things like "show me cells
  with P(mine) > 0.7" directly meaningful.

Estimated win-rate gain: hard to predict, but moving from the v5
plateau (~22-25% expected) to 28-35% would be in line with what
denoised supervision typically yields in similar setups.

### Engineering notes

- Cache labels: a state appears once per game in the buffer; recompute
  isn't needed each epoch. Compute once at data-gen time, store with
  the sample.
- Component enumeration: implement bitmask-based brute force first;
  switch to BDD/exact-cover only if profiling demands it.
- For the unconstrained-pool P(mine), we need
  `total_mines_remaining - sum(component mine counts in this layout)`.
  Track explicitly per-config.
- For boards where a single component is too big (>22 cells, ~4M
  layouts), fall back to Monte-Carlo sampling — ~10K random consistent
  layouts gives 1% precision, plenty for label denoising.

## 3. UI: Mine probability heatmap

A button on the board (next to Suggest / Auto-play) that, when clicked,
overlays every hidden cell with a colour based on P(mine):

- **Bright red, opaque** for P(mine) close to 1
- **Transparent** for P(mine) close to 0 (or no information — i.e.
  unconstrained cells with low mine density)
- Smooth gradient between

Implementation:
- The same constraint engine produces per-cell P(mine).
- For cells the algorithmic solver flagged as **definite mines**, show
  fully red.
- For cells the solver flagged as **definite safe**, show transparent
  (no overlay).
- For everything else, run the constraint engine and colour by output.

Visual approach (matching existing yellow Suggest highlight):

- Add canvas image items with semi-transparent red rectangles, alpha
  = P(mine) × 200 (out of 255). Tk doesn't support alpha on
  rectangles directly, but does on PhotoImages — pre-render a small
  set of red overlay PNGs at quantised alpha levels (e.g. 11 steps
  from 0 to 100%) and `create_image` the right one per cell.
- Or render to a single full-board overlay PIL image and put that on
  the canvas in one call (faster to update on each click that
  changes the constraint state).

Toggle: click once → show heatmap, click again → hide. Refresh on
each board state change while shown.

## 4. Sequencing

1. **v5 first** (in progress): clean training pipeline on the 5070
   Ti, push the current self-play approach as far as it goes
   (probably 23-25% expert).
2. **Constraint engine**: build it standalone, validate against
   small boards via brute force, profile.
3. **v6 training**: integrate constraint-implied labels into the
   training pipeline.
4. **Heatmap UI**: hook the constraint engine into a new button.
   Effort here is mostly the rendering — the engine is the hard
   part, already built.

The constraint engine is the keystone for both. Building it once
unblocks the rest.
