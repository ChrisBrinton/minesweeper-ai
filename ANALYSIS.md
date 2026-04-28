# Hybrid Training v5 — Failure Analysis

## TL;DR

1. **Training died** because the process had no signal handling or exception guards — most likely a terminal session close (SIGHUP) or MPS memory exhaustion silently killed it.
2. **Training degraded** because RL fine-tuning catastrophically destroys the pre-trained weights: 30% random exploration immediately wrecks a 25% win-rate policy, asymmetric rewards (+0.03 survival vs -1.0 death) teach the model to be timid, and the replay buffer overwrites the only good experiences once it fills.

---

## 1. Why Did Training Die at Episode 110,000?

### Evidence

- Last log entry: `Ep 110000/500000` at elapsed `10:46:50` (~06:08 AM Mar 10)
- No error, no traceback, no PATIENCE message, no COMPLETE message
- No Mac mini reboot since Mar 3
- Speed degrading: 3.9 eps/s (ep 2K) → 2.9 eps/s (ep 110K) — **26% slowdown**
- Replay buffer full at 100K since ep ~56K

### Root Cause Analysis

**Most likely: Terminal session closed (SIGHUP) or MPS silent crash**

The code has zero defensive infrastructure:

1. **No signal handling** — No `signal.signal(SIGTERM, ...)` or `SIGHUP` handler anywhere in `train_hybrid_v5.py`. If launched from a terminal (SSH or local) that closes, SIGHUP kills the process instantly with no output.

2. **No try/except around main loop** — The entire training loop (lines 537-626) runs bare. Any exception (MPS tensor error, numpy allocation failure, environment bug) kills the process with only a traceback to stdout — which is lost if the terminal is gone.

3. **No MPS memory management** — No calls to `torch.mps.empty_cache()` anywhere. Every episode creates:
   - New `MinesweeperEnvironment` object (line 547)
   - Multiple `torch.FloatTensor` allocations in `_nn_action()` (line 146-148)
   - Multiple tensor allocations in `train_step()` (lines 274-280)

   MPS on Apple Silicon has known memory fragmentation issues. Over 110K episodes (~10h46m), leaked MPS tensors could exhaust unified memory. The 26% speed degradation (3.9→2.9 eps/s) supports growing memory pressure.

4. **Evaluation creates 500 new environments every 2K episodes** — `evaluate_hybrid()` (line 402) creates 500 `MinesweeperEnvironment` objects per eval cycle. That's 55 eval cycles × 500 = 27,500 environment objects by ep 110K, in addition to 110K training environments.

5. **File handle kept open for entire run** — `log_path` is opened with `open(log_path, 'a')` at line 486 and held open for the entire training run. While this isn't likely the crash cause, it means the file handle doesn't get a proper close/flush on abnormal exit.

### Most Probable Scenario

The process was started directly in a terminal (the usage comment on line 13 shows `python3 train_hybrid_v5.py`, not `nohup` or `screen`). When the terminal session ended (user closed laptop lid, SSH timeout, Terminal.app quit), SIGHUP killed the process between the ep 110K log write and the next eval at ep 112K. The `log_file.flush()` at line 605 means the last successful eval was written, and the process died during the subsequent 2000-episode training block.

**Alternative**: MPS memory exhaustion caused a silent crash. The speed degradation pattern is consistent with this, and MPS is known to sometimes crash without raising a Python exception.

---

## 2. Why Is Hybrid Training Degrading From Its Baseline?

### The Devastating Numbers

| Metric | Pre-training Baseline | Best During Training | Typical During Training |
|--------|----------------------|---------------------|------------------------|
| Win Rate | **25.0%** | 14.8% (ep 86K) | 8-10% |
| Guess Survival Rate | **83.8%** | 72.7% (ep 86K) | 60-65% |
| Avg Guesses/Game | 11.8 | 5.8 | 5.0 |
| Avg Solver Moves | 136.9 | 101.7 | 85 |

The NN went from 83.8% guess survival to ~62% — it's making **dramatically worse guesses** than the pre-trained model.

### Root Cause 1: Catastrophic Forgetting from Epsilon Exploration (PRIMARY)

**The 30% epsilon destroys the pre-trained policy immediately.**

At `epsilon_start=0.3` (line 50), 30% of all guesses are random clicks on hidden cells. For expert minesweeper (16×30, 99 mines), a random guess on a hidden cell has roughly a `99/381 ≈ 26%` chance of hitting a mine. The pre-trained model's 83.8% survival rate means it had learned to avoid dangerous cells. Forcing 30% random guesses immediately tanks performance.

By episode 2000 (the first eval), win rate has already cratered from 25% to 6.2% and GSR from 83.8% to 66.4%. **The pre-trained knowledge is destroyed in the first ~500 episodes of actual training** (first 1000 episodes have insufficient buffer for training, but epsilon exploration still generates bad experiences).

The replay buffer then fills with these degraded experiences, and training reinforces the degraded policy.

### Root Cause 2: Massively Asymmetric Reward Structure

**`play_episode()` lines 182-249 — survival reward is ~100x smaller than death penalty.**

- **Guess survival reward** (line 184): `max(0.01, (cells_revealed_delta) / safe_cells)`. On expert, safe_cells = 381. A typical guess that leads to revealing 5-15 cells via solver yields reward **0.013 - 0.039**.
- **Guess death reward** (line 213): **-1.0**
- **Win reward** (line 210): **+1.0** (but very rare — only on the final guess of a won game)

This means:
- Surviving a guess: **+0.02 to +0.04** (typical)
- Dying on a guess: **-1.0**

The reward ratio is roughly **25:1 to 50:1 in favor of punishment**. The model learns that ALL guesses are dangerous and converges to a timid, non-discriminating policy. It can't learn *which* cells are safer because the positive signal is drowned out.

### Root Cause 3: Replay Buffer Overwrites Good Experiences

**Buffer fills at ep ~56K, then good pre-trained experiences are lost forever.**

The 100K replay buffer (line 55) fills around ep 56K (visible in log: `Buf: 100000` first appears). Before that, the buffer contains a mix of:
- Early experiences from the still-decent pre-trained model
- Increasingly degraded experiences from epsilon exploration

After ep 56K, every new experience evicts the oldest one. The good early experiences (from the 25% win-rate model) are systematically replaced by the degraded 8-10% win-rate experiences. There is no prioritized experience replay — good and bad experiences are treated equally.

### Root Cause 4: Gamma=0.99 Is Meaningless for Guess-to-Guess Transitions

**The DQN bootstraps Q-values between guess states, but there's no meaningful temporal structure.**

In `play_episode()`, only guess transitions are stored. Between guesses, the solver makes deterministic moves. The "next_state" of a guess isn't the immediate result — it's the state when the solver next gets stuck (potentially 10-50 solver moves later).

With `gamma=0.99` (line 49), the target Q-value is:
```
target_q = reward + 0.99 * Q(next_guess_state)
```

But the "next guess state" is essentially random from the current guess's perspective — the solver's deterministic moves create a complex, non-Markov chain between guess states. The bootstrapped Q-values are noise, not signal. The model would be better off with `gamma=0` (pure immediate reward) for this problem structure.

### Root Cause 5: Training Intensity vs Data Sparsity

**Lines 555-556: `train_iters = max(1, num_new_guesses // config['update_freq'])`**

With avg ~5 guesses/episode and `update_freq=4`, this is ~1-2 training steps per episode. But with only 5 new experiences and a batch size of 128, each training step samples mostly from old (increasingly bad) experiences. The model is being pulled toward the average of the replay buffer, which is dominated by degraded-policy experiences.

### Root Cause 6: Loss Never Converges

The loss hovers between 1.2-1.7 for the entire run — it never decreases. This is a clear sign that:
- The target Q-values are non-stationary (from the meaningless gamma bootstrapping)
- The reward signal is too noisy relative to its magnitude
- The model is not learning a stable value function

The hard target update every 1000 steps (`target_update_freq=1000`, line 53) with this noisy Q landscape means the target network is just as confused as the online network.

---

## 3. Specific Code-Level Problems

| Line(s) | Issue | Severity |
|---------|-------|----------|
| 50 | `epsilon_start: 0.3` — way too high for fine-tuning a pre-trained model | **Critical** |
| 184-185 | Survival reward `max(0.01, delta/381)` averages ~0.03, vs death=-1.0 | **Critical** |
| 49 | `gamma: 0.99` — meaningless for guess-to-guess transitions | **High** |
| 55 | `memory_size: 100000` with no prioritized replay — good experiences lost | **High** |
| 537-626 | No try/except, no signal handling — silent death on any error/signal | **High** |
| 547 | New environment object every episode — no MPS cache clearing | **Medium** |
| 302 | `target_q = rewards + 0.99 * next_q * (~dones)` — bootstrapping noise | **Medium** |
| 556 | `max(1, ng // 4)` training iters — trains on stale data | **Low** |

---

## 4. Fix Proposals (Ranked by Impact)

### Fix 1: Don't Use RL Fine-Tuning — Use Guided Self-Play Instead (HIGHEST IMPACT)

The fundamental problem is that DQN is the wrong algorithm for this task. The NN only needs to rank ~20-40 hidden cells by mine probability when the solver is stuck. This is a **ranking/regression problem**, not a sequential decision problem.

**Proposed approach:**
1. Play games with the hybrid agent (solver + current NN, epsilon=0)
2. On each guess, record the board state and which cell was chosen
3. If the guess survived, label it as positive. If it died, label as negative.
4. After many games, also record the ground truth: after a game ends, you know where ALL mines were. Use this to compute actual mine probabilities for each guess situation.
5. Train the NN with a **cross-entropy or MSE loss** to predict mine probability per cell, supervised by the post-hoc ground truth.

This avoids all the RL pathologies: no epsilon, no replay buffer, no bootstrapping, no catastrophic forgetting.

### Fix 2: If Sticking with RL — Fix the Reward Asymmetry

```python
# Replace the tiny survival reward with something meaningful
if pending_guess_state is not None and not is_guess:
    guess_reward = 0.5  # Fixed positive reward for surviving a guess
    # Scale slightly by progress
    progress = env.api.game_board.cells_revealed / (rows * cols - mines)
    guess_reward = 0.3 + 0.7 * progress  # 0.3 to 1.0

# Death penalty should match survival scale
guess_reward = -0.5  # instead of -1.0
```

This makes the reward ratio ~1:1 instead of ~30:1.

### Fix 3: Epsilon Must Start Near Zero for Pre-Trained Models

```python
'epsilon_start': 0.05,   # was 0.3 — the model ALREADY has a good policy
'epsilon_end': 0.01,     # was 0.05
'epsilon_decay_episodes': 50000,  # was 200000
```

Or even better, use Boltzmann exploration (softmax over Q-values) instead of epsilon-greedy, which preserves the model's ranking ability while still exploring.

### Fix 4: Use gamma=0 (or very small) for This Problem Structure

```python
'gamma': 0.0,  # was 0.99 — no meaningful temporal dependency between guesses
```

Each guess is essentially an independent decision. The outcome of guess N has no bearing on the optimal action for guess N+1 (the solver handles the deterministic parts in between). Treat each guess as a contextual bandit problem.

### Fix 5: Add Process Resilience

```python
import signal

def handle_signal(signum, frame):
    # Save checkpoint and exit cleanly
    torch.save({...}, os.path.join(save_dir, 'interrupted_checkpoint.pth'))
    log_file.write(f"\nINTERRUPTED by signal {signum} at ep {episode}\n")
    log_file.flush()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)

# Wrap main loop in try/except
try:
    for episode in range(1, max_episodes + 1):
        ...
except Exception as e:
    log_file.write(f"\nCRASHED at ep {episode}: {e}\n")
    traceback.print_exc(file=log_file)
    torch.save({...}, os.path.join(save_dir, 'crash_checkpoint.pth'))
    raise
```

### Fix 6: Add MPS Memory Management

```python
# Every N episodes, clear MPS cache
if episode % 100 == 0 and device.type == 'mps':
    torch.mps.empty_cache()
```

### Fix 7: Prioritized Experience Replay

Replace the uniform `ReplayBuffer` with a prioritized version that weights experiences by TD error magnitude. This keeps rare winning experiences and informative death experiences in the buffer longer, rather than being overwritten by the flood of mediocre mid-game transitions.

---

## 5. Recommended Next Approach

**Stop using DQN. Reframe as supervised learning on self-play data.**

```
Algorithm: Iterative Supervised Guess Improvement
1. Play 10K games with hybrid agent (solver + current NN, epsilon=0)
2. For each guess situation encountered:
   a. Record the 12-channel board state
   b. After game ends, retrieve true mine locations
   c. For each hidden cell at guess time, label it: mine=1, safe=0
3. Train NN to predict P(mine | state) for each cell using BCE loss
4. The NN's guess policy: click the hidden cell with lowest predicted P(mine)
5. Evaluate. If improved, save. Repeat from step 1.
```

This approach:
- Preserves the pre-trained model's knowledge (no random exploration)
- Uses ground truth labels (no reward engineering)
- Has a clear loss function with unambiguous gradients
- Can be mixed with the pre-trained imitation data to prevent forgetting
- Is 10x simpler to implement and debug

The pre-trained model already has 83.8% guess survival. A supervised approach that nudges mine probability estimates toward ground truth should be able to push that to 88-92%, which at ~6 guesses/game would yield **40-55% win rate** on expert.
