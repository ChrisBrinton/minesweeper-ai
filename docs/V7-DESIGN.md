# v7 Design — Model-Only Play via Constraint-Supervised Training

v6 hybrid reached 49% win rate on expert but the model depends on the
algorithmic solver for all deterministic moves.  Heatmap visualization
reveals the model is "chord hunting" — biased toward cells that cascade
(reveal 0-cells opening large areas) rather than cells that are simply
safest.  v7 aims to produce a model that plays solo (no solver at
inference time) and maximises win rate, not cascade size.

---

## 1. Why the model chord-hunts

The v6 survival reward is progress-scaled:

```python
progress = cells_revealed / safe_cells
guess_reward = 0.3 + 0.7 * progress   # range [0.3, 1.0]
```

When a guess near a 0-cell survives and triggers a cascade,
`cells_revealed` jumps between the guess state and the confirmation
state.  The model gets a higher reward for guesses that open large
areas, regardless of whether those guesses were the safest option.

Additionally, early supervised pre-training used the v1 reward
structure where `reveal_empty_cell = 8.0` and `reveal_multi_safe = 1.5`
per cascaded cell — explicit chord bonuses baked into the initial
weights.

v7 eliminates all cascade/progress incentives.

---

## 2. Strategy: two-phase training

### Phase 1 — Supervised: constraint-implied P(mine) labels

Build the constraint engine from the v6 design doc (component
decomposition + enumeration) and use it to compute **true P(mine|visible
state)** for every hidden cell.  Train the model with MSE against these
probability labels.

Key differences from all prior supervised trainers:
- **Train on ALL board states**, not just guess states.  The model must
  learn what the solver knows (deterministic safe/mine) in addition to
  probabilistic reasoning.
- **Labels are continuous** P(mine) ∈ [0, 1], not binary 0/1.
  Denoised gradients — no more "this cell was safe but P(mine) was
  really 0.4" noise.
- **Loss masking**: only compute loss on hidden cells (revealed cells
  carry no prediction target).

Data generation pipeline:
1. Play games with the current best model (hybrid or solo).
2. At each state where hidden cells exist, run constraint engine →
   P(mine) per hidden cell.
3. Store `(state, probability_labels, hidden_mask)` tuples.
4. D4 augmentation (identity + h-flip + v-flip + 180° rotation) as in
   train_supervised_v4.

Architecture: MinesweeperResNetV4 unchanged (1M params).  The output
head semantics change from Q-value to calibrated P(mine) — same sigmoid
activation, but trained with MSE against true probabilities instead of
BCE against binary labels or TD targets.

### Phase 2 — RL fine-tune: model-only, sparse reward

Once Phase 1 produces a strong P(mine) estimator, fine-tune with RL
where the model makes ALL moves (no solver):

| Signal         | Value | Rationale                        |
|----------------|-------|----------------------------------|
| Win            | +1.0  | The only goal                    |
| Lose           | −1.0  | Symmetric penalty                |
| Step (any)     |  0.0  | No per-move reward → no chords   |
| Invalid action | −0.1  | Discourage clicking revealed     |

- **Gamma = 0.99** — temporal credit matters when the model makes every
  move (unlike v6 where gamma=0 was correct for the bandit setting).
- **Epsilon = 0.02 → 0.005** — the supervised phase gives a strong
  starting policy; don't destroy it.
- **Replay buffer**: 500K transitions, prioritised by |TD error|.
- **Target network**: hard update every 2000 training steps.
- No progress scaling.  No survival bonus.  The model learns that the
  only thing that matters is finishing alive.

### Phase 3 (optional) — Self-play iteration

Use the Phase 2 model to generate new games, re-run constraint engine
for updated P(mine) labels, retrain Phase 1 on the combined dataset.
The model explores different board states than the solver, so the data
distribution shifts toward states the model actually encounters.

---

## 3. Constraint engine

### Algorithm (from v6 design doc, refined)

Given a visible board state:

1. **Parse state** into hidden, flagged, revealed sets (existing
   `AlgorithmicSolver._parse_state`).

2. **Build constraint graph**: each revealed number cell is a constraint
   node. Its hidden neighbours are variable nodes. Edges connect
   constraint to variable.

3. **Find connected components** of the constraint graph. Cells that
   share no constraint edges are independent — their probabilities can
   be computed separately.

4. **Per-component enumeration**: for each component, enumerate all
   mine/safe assignments to its hidden cells that satisfy every
   constraint in the component.
   - Components are typically 5–20 cells → 2^5 to 2^20 configs.
   - For components >22 cells, fall back to Monte Carlo sampling
     (10K random consistent layouts → ~1% precision).

5. **Weight each configuration** by the number of ways to distribute
   remaining mines among unconstrained cells:
   `C(unconstrained_count, remaining_mines_for_this_config)`.

6. **Compute P(mine)** per constrained cell:
   `P(mine) = Σ(weight where cell is mine) / Σ(all weights)`.

7. **Unconstrained cells** (no numbered neighbour):
   `P(mine) = remaining_mines / unconstrained_count` (uniform).

### Implementation: `src/ai/constraint_engine.py`

New standalone module.  The algorithmic solver stays unchanged — the
constraint engine is a superset that computes probabilities, not just
deterministic moves.

```python
class ConstraintEngine:
    def compute_probabilities(state, total_mines) -> np.ndarray:
        """Returns P(mine) array of shape (rows, cols).

        - Revealed cells → NaN
        - Deterministic safe → 0.0
        - Deterministic mine → 1.0
        - Uncertain → float in (0, 1)
        - Unconstrained → uniform probability
        """
```

### Performance budget

Target: <100ms per board state on expert (16×30).  Component
enumeration should be fast because:
- Most components are <15 cells (2^15 = 32K configs).
- Bitmask-based brute force with early pruning.
- Only runs on stuck states (solver already found its deductions).

If any component exceeds 22 cells (4M configs), switch that component
to Monte Carlo sampling with 10K trials.

---

## 4. Inference pipeline changes

### Model-only mode (v7 default at inference)

```
get_mine_probabilities(game_board):
    state = _state_from_board(game_board)
    model_probs = sigmoid(model(state))      # P(mine) per cell
    return model_probs for hidden cells, NaN for revealed

suggest_move(game_board):
    probs = get_mine_probabilities(game_board)
    return hidden cell with min P(mine)
```

No solver call.  The model has learned to produce P(mine) ≈ 0 for cells
the solver would have flagged as safe, and P(mine) ≈ 1 for cells the
solver would have flagged as mines.

### Hybrid mode (backward compat, selectable in settings)

Keep the existing solver-first-then-model pipeline for comparison and
as a fallback while v7 trains.

### Heatmap data source

The heatmap shows model predictions directly.  No change to
`BoardCanvas.show_heatmap()` — it already consumes a `(rows, cols)`
probability array.  The only change is what produces that array.

Add a settings toggle: "Heatmap source" → Model / Constraint Engine /
Both.  "Both" overlays constraint-engine P(mine) as a second layer
for direct visual comparison.

---

## 5. New UI visualizations

### 5a. Confidence indicator

Show the model's confidence in its best guess:
- Next to the Suggest button, display `P(mine): 0.12` for the
  suggested cell.
- Color-code: green (<0.15), yellow (0.15–0.35), red (>0.35).
- Gives the player (and the developer) instant feedback on how
  uncertain the model is about its recommendation.

### 5b. Model vs. constraint engine comparison overlay

When "Both" heatmap source is selected:
- Model predictions shown as the existing red gradient.
- Constraint engine probabilities shown as a blue gradient border or
  corner dot on each cell.
- Cells where model and engine disagree by >0.15 get a yellow warning
  border — these are cells where the model hasn't learned the correct
  probability yet, and are the most informative for debugging.

This directly shows where the model's understanding diverges from
ground truth, making it a powerful tool for spotting training gaps.

### 5c. Move source indicator

During auto-play or after suggest, show whether each move was:
- Solver-determined (existing: no change)
- Model guess (existing: no change)
- Model deterministic (NEW in v7: model predicted P(mine) ≈ 0 without
  solver help)

Color-code the last-move highlight:
- Green border: deterministic (solver or model-confident)
- Purple border: model guess (uncertain)
- Gives visual feedback on how much the model has learned to replicate
  solver logic.

---

## 6. Architecture decisions

### Keep MinesweeperResNetV4

The 1M-param fully convolutional ResNet with SE attention and dueling
heads is well-suited:
- 71×71 receptive field covers the entire expert board.
- SE attention can learn to weight channels differently for different
  board densities.
- Fully convolutional — train on expert, deploy on any size.

### Output head semantics

The dueling DQN heads (value + advantage) were designed for RL.  For
supervised P(mine) training, the dueling structure still works:
- Value head → base mine probability for the board state
- Advantage head → per-cell adjustment from the base

The output passes through sigmoid to get P(mine) ∈ [0, 1].

In Phase 2 RL, the same heads serve as Q-values where lower Q = safer.

### Loss function

Phase 1 (supervised):
```
loss = MSE(sigmoid(model(state)), constraint_P_mine)
       weighted by hidden_mask
```

Alternative: BCE with soft labels.  BCE has better gradient properties
near 0 and 1, which matters for cells that are deterministically safe
or mined.  **Start with BCE, fall back to MSE if calibration suffers.**

Phase 2 (RL): standard DQN loss (Huber / smooth-L1) on TD targets.

---

## 7. Training data considerations

### Sample balance

Prior trainers only collected guess states.  v7 Phase 1 collects ALL
states, which means:
- ~80% of states have deterministic safe moves (solver would handle)
- ~20% are genuine guess states

This imbalance is fine — the model NEEDS to learn the deterministic
patterns.  But we should track loss separately on deterministic vs.
guess states to monitor learning.

### Data generation strategy

Iteration 1: play 10K games with existing v6 hybrid model.  Collect
all states, compute constraint-engine labels.  Train Phase 1.

Iteration 2+: play with the Phase 1 model (solo, no solver).  The
model explores different states → fresh data.  Re-label with constraint
engine.  Retrain.

### Augmentation

D4 group (4 transforms): identity, horizontal flip, vertical flip,
180° rotation.  All preserve the 16×30 rectangle.  90° rotation would
change dimensions and is excluded.

---

## 8. Evaluation protocol

### Primary metric: model-only win rate on expert

The v7 north star.  Evaluate every 2K training steps with 500 games,
model playing solo (no solver).  Report with 95% CI.

### Secondary metrics

| Metric | Purpose |
|--------|---------|
| Hybrid win rate | Backward comparison with v6 |
| P(mine) calibration | Reliability diagram: binned predicted vs actual |
| Deterministic accuracy | % of solver-safe cells where model predicts P(mine) < 0.05 |
| Guess accuracy | % of guess-state cells where model picks the actual safest cell |
| Mean absolute error vs constraint engine | How close model is to true P(mine) |

### Visualization tracking

Log heatmap snapshots at evaluation time:
- Board state + model heatmap + constraint engine heatmap side by side.
- Track how the comparison evolves across training.

---

## 9. Sequencing

1. **Constraint engine** (`src/ai/constraint_engine.py`)
   - Component decomposition
   - Bitmask enumeration
   - Monte Carlo fallback for large components
   - Validation against brute force on small boards
   - Performance profiling on expert boards

2. **Inference pipeline update** (`src/ai/inference.py`)
   - Add constraint engine as heatmap data source
   - Add model-only mode
   - Backward compat with hybrid mode

3. **UI enhancements** (`src/ui/gui.py`, `src/ui/settings.py`)
   - Confidence indicator on Suggest button
   - Heatmap source selector
   - Model vs. constraint engine comparison overlay
   - Move source color coding

4. **Phase 1 trainer** (`train_v7.py`)
   - Data generation: all states + constraint engine labels
   - Supervised training loop with D4 augmentation
   - Evaluation: model-only win rate + calibration metrics
   - Resumable, desktop-friendly (inherit v5 patterns)

5. **Phase 2 trainer** (`train_v7_rl.py`)
   - Model-only RL with sparse reward
   - Load Phase 1 weights as starting point
   - DQN with target network, prioritised replay

---

## 10. Risk and mitigation

| Risk | Mitigation |
|------|------------|
| Constraint engine too slow on expert | Monte Carlo fallback for components >22 cells; profile early |
| Model can't learn solver logic | It should — the receptive field covers the board and the labels are exact. If accuracy plateaus, increase model capacity (wider or deeper) |
| Phase 2 RL destroys Phase 1 calibration | Very low epsilon, small learning rate, short fine-tune. Evaluate calibration throughout |
| Not enough compute for large-scale training | Desktop-friendly scheduler from v5; train overnight |
| Regression in heatmap quality | Constraint engine heatmap is strictly better than model-only heatmap; comparison mode makes regressions visible |
