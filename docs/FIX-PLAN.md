# Minesweeper AI — Fix Plan

**Goal:** Fix the training harness so the model actually learns, then run curriculum training on M4 Mac Mini.

## Approach

Three phases. Each phase is a working checkpoint — we can train after Phase 1 to validate fixes before continuing.

---

## Phase 1: Make It Train (Critical — Do First)

**Branch:** `fix/training-harness`  
**Estimated effort:** 2-3 hours of coding  
**Files changed:** `trainer.py`, `environment.py`, `game_api.py`, `models.py`

### 1.1 Fix Update Frequency (trainer.py)
**Problem:** Network trains once per ~1,000 steps. Should be every 4.  
**Change:** 
- Set `update_freq: 4`
- Move training call inside the step loop (not after episode)
- Track `total_steps` globally, not per-episode

### 1.2 Fix State Representation (game_api.py)
**Problem:** 3 channels with mixed categorical/ordinal values. CNN can't learn.  
**Change:** One-hot encoding → 12 channels:
- Ch 0: hidden (binary)
- Ch 1: flagged (binary)  
- Ch 2: revealed (binary)
- Ch 3-11: number 0-8 (one-hot, only for revealed cells)

### 1.3 Remove Flag/Unflag Actions (environment.py)
**Problem:** 3× action space bloat for zero learning signal.  
**Change:** 
- Action space = `rows × cols` (reveal only)
- Remove flag/unflag reward logic
- Simplify action decoding

### 1.4 Fix Reward Shaping (environment.py)
**Problem:** Complex rewards create shortcuts and perverse incentives.  
**Change:** Sparse rewards:
```python
rewards = {
    'win': 1.0,
    'lose': -1.0,
    'reveal_safe': 0.01,    # tiny progress signal
    'invalid_action': -0.1,  # discourage illegal moves
    'step_penalty': 0.0,     # removed
}
```

### 1.5 Add MPS Support (trainer.py)
**Problem:** Only checks CUDA, ignores Apple Silicon GPU.  
**Change:** MPS → CUDA → CPU detection chain.

### 1.6 Use Huber Loss (trainer.py)
**Problem:** MSE is sensitive to outlier Q-values.  
**Change:** `nn.SmoothL1Loss()`

### 1.7 Fix _evaluate Signature (trainer.py)
**Problem:** Method doesn't accept `num_episodes` parameter but callers pass it.  
**Change:** Add parameter with default.

### Phase 1 Validation
After these changes, run a quick 5×5 training (5,000 episodes):
- ✅ Loss should decrease
- ✅ Win rate should increase from random baseline (~12%)  
- ✅ Training should complete without crashes
- ✅ Should use MPS device

---

## Phase 2: Make It Learn Well (Architecture + Curriculum)

**Branch:** `fix/architecture`  
**Estimated effort:** 3-4 hours of coding  
**Files changed:** `models.py`, `curriculum.py`, `trainer.py`

### 2.1 Fully Convolutional Architecture (models.py)
**Problem:** FC layers break curriculum transfers (size mismatch).  
**Change:** New `MinesweeperFCN` class:
- 4 conv layers (3×3, padding=1) → no FC layers
- Dueling output via 1×1 convolutions (value + advantage per cell)
- ~50K params, works for ANY board size with same weights
- Input: 12 channels (from Phase 1 state representation)
- Output: Q-value per cell [B, H, W]

### 2.2 Fix Epsilon Decay (trainer.py / curriculum.py)
**Problem:** Exponential decay reaches ~0 before network has trained.  
**Change:** Linear decay over total training steps:
```python
epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (step / decay_steps))
```

### 2.3 Soft Target Updates (trainer.py)
**Problem:** Target network updates every 100 episodes (incoherent with training frequency).  
**Change:** Polyak averaging after every training step: `τ = 0.005`

### 2.4 Double DQN (trainer.py)
**Problem:** Standard DQN overestimates Q-values.  
**Change:** Use online network to select action, target network to evaluate.

### 2.5 Mask Target Q-Values (trainer.py)
**Problem:** Target computation includes impossible actions.  
**Change:** Apply action mask to next-state Q-values.

### 2.6 Fix Curriculum Transitions (curriculum.py)
**Problem:** Phase 0 (perfect knowledge) uses 4 channels, rest use 12.  
**Change:** Drop Phase 0 entirely. Start with Tiny (5×5) — the auto-reveal on reset gives enough initial signal.

### 2.7 Auto-Reveal on Reset (environment.py)
**Problem:** Initial board is all hidden, zero information.  
**Change:** Automatically reveal a random cell on `reset()` to place mines and give initial state.

### Phase 2 Validation
Run 5×5 training again:
- ✅ Same weights transfer to 7×7 without crash
- ✅ Win rate on 5×5 should reach 50%+ within 200K steps
- ✅ Epsilon decays smoothly over training
- ✅ Q-values don't diverge

---

## Phase 3: Polish (Optional, After Training Works)

### 3.1 Prioritized Experience Replay
Weight sampling by TD-error. Keep winning episodes in a separate buffer.

### 3.2 Hyperparameter Tuning
- Learning rate scheduling (cosine or step decay)
- Batch size experiments (64 vs 128 vs 256)
- Replay buffer size tuning

### 3.3 Reduce Dropout
From 0.3 → 0.0 (or remove entirely). RL has enough variance.

### 3.4 Lower Learning Rate
Start at 1e-4, potentially decay to 1e-5 for later curriculum stages.

### 3.5 Training Monitoring Dashboard
- Real-time loss/win-rate plots
- Q-value distribution tracking
- Episode length tracking
- Curriculum stage visualization

---

## Refactoring Strategy

**Key principle:** Don't rewrite everything. Surgical changes to existing code.

### New files to create:
- `src/ai/models_v2.py` — New FCN architecture (keeps old models.py intact)
- `src/ai/environment_v2.py` — Updated environment with new state/reward/actions (or modify in-place with feature flags)
- `train_v2.py` — New training entry point using fixed components

### Files to modify:
- `trainer.py` — Update frequency, loss function, target updates, MPS, Double DQN
- `game_api.py` — New `get_board_array_v2()` method (keep old for backward compat)
- `curriculum.py` — Updated stages, remove Phase 0, fix epsilon

### Files NOT touched:
- `src/game/board.py` — Game logic stays as-is
- `src/ui/gui.py` — UI stays as-is
- All test files — Will need new tests for v2 components
- `main.py` — Game entry point unchanged

### Git Strategy
1. Branch `fix/training-harness` from `main`
2. Phase 1 commits → validate → merge to `main`
3. Branch `fix/architecture` from `main`
4. Phase 2 commits → validate → merge to `main`
5. Phase 3 as separate branches

---

## Training Plan (Post-Fix)

### Schedule
Run during Bob's workshop hours (8pm-4am ET) on M4 Mac Mini.

| Stage | Board | Target | Est. Time | Milestone |
|-------|-------|--------|-----------|-----------|
| Tiny | 5×5, 3 mines | 60% win | ~30 min | "Can it learn at all?" |
| Small | 7×7, 7 mines | 45% win | ~1.5 hr | "Does curriculum transfer work?" |
| Mini | 8×8, 9 mines | 40% win | ~2.5 hr | "Approaching real Minesweeper" |
| Beginner | 9×9, 10 mines | 35% win | ~5 hr | "Standard difficulty baseline" |
| Intermediate | 16×16, 40 mines | 20% win | ~15 hr | "Can it generalize?" |
| Expert | 16×30, 99 mines | 10% win | ~30 hr | "Stretch goal" |

**Total: ~55 hours (2-3 days continuous)**

### Monitoring
I'll check training progress periodically and:
- Log metrics to `training_logs/`
- Alert Chris if training stalls or diverges
- Adjust hyperparameters if needed between stages
- Commit checkpoints to git

### Success Criteria
- **Minimum viable:** 35% win rate on Beginner (9×9)
- **Good:** 20% win rate on Intermediate (16×16)
- **Excellent:** Any consistent wins on Expert (16×30)

For reference, perfect algorithmic solvers achieve ~85% on Beginner, ~70% on Intermediate, ~30% on Expert (some boards are genuinely unsolvable without guessing).
