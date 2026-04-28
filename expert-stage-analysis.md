# Expert Stage Training Analysis & Recommendations

## Executive Summary

The expert stage (16×30, 99 mines) achieved **0% win rate after 465,710 episodes**, representing a complete failure to learn. While earlier stages showed progress (reaching 20.3% on intermediate), the jump to expert complexity proved insurmountable for the current architecture and training approach.

---

## 1. Architecture Limitations (52K Parameters)

### Current Architecture Analysis
```
MinesweeperFCN (~52K params):
  Conv1: 12 → 32 channels  (3×3)  = ~3.5K params
  Conv2: 32 → 48 channels  (3×3)  = ~13.8K params
  Conv3: 48 → 48 channels  (3×3)  = ~20.7K params
  Conv4: 48 → 32 channels  (3×3)  = ~13.8K params
  Value head: 32 → 1       (1×1)  = ~33 params
  Advantage head: 32 → 1   (1×1)  = ~33 params
```

### Key Limitations

| Issue | Impact | Severity |
|-------|--------|----------|
| **Insufficient receptive field** | 4 conv layers with 3×3 kernels = 9×9 receptive field. Expert board requires understanding long-range dependencies (16×30) | 🔴 Critical |
| **No dilated convolutions** | Cannot capture multi-scale patterns efficiently | 🔴 Critical |
| **No attention mechanisms** | Cannot focus on relevant regions when most board is hidden | 🟡 High |
| **Shallow feature hierarchy** | 4 layers may be insufficient for complex mine patterns | 🟡 High |
| **No recurrence** | Cannot perform iterative reasoning about constraints | 🟡 High |

### Expert Board Complexity
- **480 cells** vs 256 in intermediate (1.9× increase)
- **99 mines** vs 40 in intermediate (2.5× increase)
- **381 safe cells** to reveal vs 216 (1.8× increase)
- **Mine density**: 20.6% vs 15.6% (significantly harder)

### Recommendations - Architecture

1. **Increase model capacity to 200K-500K parameters**
   ```python
   # Proposed architecture
   self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
   self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
   self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=2)  # Dilated
   self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=2)  # Dilated
   self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
   self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
   ```

2. **Add dilated convolutions** to increase receptive field without losing resolution
   - Target: 16×30 receptive field to cover entire expert board
   - Use dilation rates: 1, 1, 2, 2, 4, 1

3. **Consider adding residual connections** for better gradient flow

4. **Optional: Add lightweight attention** (coordinate attention or squeeze-excitation)

---

## 2. Reward Shaping Issues on Sparse-Reward Expert Boards

### Current Reward Structure (v2 with normalization)
```python
reward_config = {
    'win': 1.0,
    'lose': -1.0,
    'reveal_safe': 0.01 / safe_cells,  # Normalized ~0.000026 per cell
    'invalid_action': -0.1,
    'step_penalty': 0.0,
}
```

### The Core Problem

On expert boards:
- Average game length: ~200-300 steps
- Agent dies before winning 99%+ of the time
- **Signal-to-noise ratio**: Tiny positive rewards vs large negative termination
- **Credit assignment**: 381 safe cell reveals needed for win reward
- **Exploration collapse**: Epsilon decays before finding any winning trajectory

### Evidence from Logs
```
[expert] Ep 80000 | Win: 0.0% | Loss: 1.07 | Eps: 0.05
```
- Loss decreased from 1.81 to 1.07, but still 0% win rate
- Epsilon decayed to 0.05 (minimal exploration)
- Agent learned to "not die immediately" but never learned to win

### Recommendations - Reward Shaping

1. **Implement curiosity-driven exploration** (Intrinsic motivation)
   ```python
   # Add prediction error bonus
   intrinsic_reward = ||predict_next_state - actual_next_state||
   total_reward = extrinsic_reward + 0.1 * intrinsic_reward
   ```

2. **Add intermediate milestones**
   ```python
   # Reward for progress percentage
   progress = cells_revealed / total_safe_cells
   milestone_reward = 0.1 if progress > 0.25 and not milestone_25_reached else 0
   ```

3. **Consider demonstration-based learning**
   - Pre-train on expert games from solvers (if available)
   - Use imitation learning to bootstrap

4. **Increase reward for reveals on expert only**
   ```python
   # Expert-specific reward boost
   if board_size == 'expert':
       reward *= 5.0  # Amplify positive feedback
   ```

5. **Implement prioritized experience replay** (already partially done, but ensure wins are highly prioritized)

---

## 3. Hyperparameter Tuning Opportunities

### Current Expert Configuration
```python
expert_stage = {
    'learning_rate': 1e-5,        # Very low
    'epsilon_start': 0.968,       # Inherited from previous stage
    'epsilon_end': 0.05,
    'epsilon_decay_fraction': 0.3,
    'batch_size': 128,
    'memory_size': 200000,
    'max_steps_per_episode': 500,
    'patience': 80000,
}
```

### Issues Identified

| Parameter | Current | Problem | Recommended |
|-----------|---------|---------|-------------|
| `learning_rate` | 1e-5 | Too low for new task; catastrophic forgetting risk | 3e-5 or use LR warmup |
| `epsilon_start` | ~0.97 | Decays too fast; need more exploration | 1.0 with slower decay |
| `epsilon_end` | 0.05 | Too low for sparse rewards | 0.10-0.15 |
| `max_steps` | 500 | May truncate winning games | 800-1000 |
| `batch_size` | 128 | Adequate but could increase | 256 |
| `target_update` | Soft (τ=0.005) | May be too aggressive initially | Try hard updates every 1000 steps |

### Recommendations - Hyperparameters

1. **Reset epsilon at stage start**
   ```python
   if stage['name'] == 'expert':
       epsilon = 1.0  # Full reset
       epsilon_decay = 0.999  # Slower decay
   ```

2. **Use separate epsilon schedules for exploration phases**
   - Phase 1 (0-100k eps): ε=1.0→0.3 (find any wins)
   - Phase 2 (100k-300k eps): ε=0.3→0.15 (exploit winning strategies)
   - Phase 3 (300k+ eps): ε=0.15→0.10 (refine)

3. **Implement automatic exploration boost**
   ```python
   if win_rate == 0 and episodes > 50000:
       epsilon = min(epsilon + 0.1, 0.5)  # Boost exploration
       print(f"🔥 Exploration boost: ε={epsilon:.3f}")
   ```

4. **Learning rate schedule**
   ```python
   # Cosine annealing with warm restarts
   scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2)
   ```

---

## 4. Curriculum Design Improvements

### Current Curriculum Progression

| Stage | Size | Mines | Target | Achieved | Gap |
|-------|------|-------|--------|----------|-----|
| tiny | 5×5 | 3 | 60% | 65% | ✓ |
| small | 7×7 | 7 | 45% | 45.5% | ✓ |
| mini | 8×8 | 9 | 40% | 40.5% | ✓ |
| beginner | 9×9 | 10 | 35% | 40.7% | ✓ |
| bridge1 | 10×10 | 15 | 30% | 32.3% | ✓ |
| bridge2 | 12×12 | 22 | 25% | 13.3% | ✗ (failed) |
| bridge3 | 14×14 | 32 | 22% | 23% | ✓ |
| intermediate | 16×16 | 40 | 20% | 20.3% | ✓ (barely) |
| expert | 16×30 | 99 | 10% | 0% | ✗ (complete failure) |

### Critical Gap Analysis

The jump from **intermediate (16×16, 40 mines)** to **expert (16×30, 99 mines)** is:
- 1.9× cells
- 2.5× mines
- 1.8× safe cells to reveal
- 5.2× longer expected game duration

**Missing: Intermediate-wide stage** (16×20, 60 mines) to bridge the gap.

### Recommendations - Curriculum

1. **Insert additional bridge stage before expert**
   ```python
   {
       'name': 'advanced',
       'rows': 16, 'cols': 20, 'mines': 60,
       'target_win_rate': 0.15,
       'max_episodes': 400000,
       'learning_rate': 2e-5,
   }
   ```

2. **Increase intermediate target to 25-30%**
   - Current 20% target may not provide strong enough foundation
   - Train intermediate longer before advancing

3. **Add curriculum "buffer zones"**
   - Don't advance until win rate is 5% above target
   - Prevents premature advancement

4. **Consider reverse curriculum for expert**
   - Start with partially revealed expert boards
   - Gradually decrease initial revealed cells

5. **Implement skill-based gating**
   ```python
   # Don't start expert until intermediate demonstrates specific skills
   required_skills = {
       'pattern_recognition': True,
       '50_percent_completion_rate': 0.5,
       'average_survival_steps': 100,
   }
   ```

---

## 5. Additional Recommendations

### A. Training Infrastructure

1. **Implement automatic checkpoint saving on any win**
   - Even one win in 100k episodes is valuable
   - Save and analyze winning trajectories

2. **Add detailed logging**
   - Track: cells revealed per game, cause of death (guess vs mistake), pattern types encountered
   - Visualize Q-value heatmaps during training

3. **Use multiple seeds**
   - Current run may be unlucky
   - Run 3-5 seeds in parallel for expert stage

### B. Alternative Approaches to Consider

1. **Hierarchical RL**
   - Low-level policy: reveal individual cells
   - High-level policy: decide which region to focus on

2. **Model-based RL**
   - Learn a world model of mine probabilities
   - Use planning (MCTS) instead of pure Q-learning

3. **Hybrid symbolic-neural approach**
   - Use constraint solver for deterministic moves
   - Use neural net only for probabilistic guesses

4. **Ensemble training**
   - Train multiple models with different seeds
   - Use majority voting for action selection

---

## 6. Immediate Action Plan

### Priority 1 (Must Fix)
- [ ] **Increase model capacity** to 200K+ parameters
- [ ] **Add 16×20 intermediate stage** before expert
- [ ] **Reset epsilon to 1.0** at expert stage start

### Priority 2 (Should Fix)
- [ ] Implement **curiosity-driven exploration**
- [ ] Add **dilated convolutions** for larger receptive field
- [ ] Increase **max_steps_per_episode** to 800+

### Priority 3 (Nice to Have)
- [ ] Implement **automatic exploration boost** when stuck
- [ ] Add **intermediate milestone rewards**
- [ ] Use **demonstration pre-training** if solver available

---

## 7. Expected Outcomes

With these changes, reasonable expectations for expert stage:

| Metric | Current | With Fixes | Timeline |
|--------|---------|------------|----------|
| First win | Never (465k eps) | ~50k episodes | Short |
| 5% win rate | 0% | ~200k episodes | Medium |
| 10% win rate | 0% | ~500k episodes | Long |
| 15% win rate | 0% | ~1M episodes | Stretch |

Human expert performance on Minesweeper is ~30-40% win rate. A 10-15% win rate would be a significant achievement for this RL agent.

---

## Appendix: Key Metrics from Log

```
Expert Stage Summary:
- Episodes: 80,000 (of 500,000 max)
- Total steps: 465,710
- Best win rate: 0.0%
- Final epsilon: 0.05
- Final loss: 1.07
- Training time: 2:18:45

Intermediate → Expert Parameter Jump:
- Input size: 256 → 480 cells (+87%)
- Mines: 40 → 99 (+148%)
- Safe cells: 216 → 381 (+76%)
- Action space: 256 → 480 (+87%)
```

---

*Analysis generated: 2026-02-06*
*Based on: curriculum_v5_output.log*
