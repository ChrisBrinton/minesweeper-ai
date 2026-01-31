# Minesweeper AI Training System — Code Review

**Date:** 2025-01-20  
**Reviewer:** Claude (AI code review)  
**Scope:** Complete RL training pipeline review  
**Verdict:** ❌ **The model will not learn in its current form.** Multiple critical issues compound to make training nearly impossible.

---

## 1. Executive Summary

The training system has a well-organized code structure with curriculum learning, experience replay, dueling DQN, and action masking. However, **several fundamental design flaws prevent the model from learning**:

1. **The network is too infrequently updated** — training happens once per ~1000 steps with a single batch of 32, meaning the vast majority of experience is wasted.
2. **The state representation is poor** — Channel 0 mixes hidden cells (-3), flags (-2), mines (-1), and numbers (0-8) into a single channel with no normalization, making it very hard for convolutions to extract meaningful patterns.
3. **The action space is 3× too large** — including flag/unflag actions dramatically increases the action space while providing almost no learning signal (flagging gives +0.5 reward regardless of correctness during play).
4. **The CNN architecture is fixed-size** — it uses a fully-connected layer after conv layers, so it cannot transfer between board sizes in the curriculum. Moving from 5×5 to 7×7 requires a completely new network.
5. **Curriculum knowledge transfer is broken** — loading a state dict from a 5×5 model into a 7×7 model will crash with a size mismatch.
6. **Epsilon decays too fast** — with `epsilon_decay=0.995` and 1000 episodes per batch, epsilon drops from 0.9 to 0.006 in a single batch, long before the network has learned anything useful.
7. **No MPS (Apple Silicon) support** — the device detection only checks for CUDA, leaving M4 Mac training on CPU.

The net effect: the agent collects experience but barely learns from it, the state representation doesn't give it enough signal to learn spatial patterns, and curriculum transitions destroy whatever knowledge was acquired.

---

## 2. Critical Issues (Prevent Learning)

### C1. Network Update Frequency is Catastrophically Low

**Files:** `trainer.py` lines 72-73, 175-182  
**Impact:** The network barely trains at all.

```python
# trainer.py line 72
'update_freq': 1000,  # Update network every N steps (not episodes)
```

```python
# trainer.py lines 175-182
# Update network only after complete episodes, enough experiences, and enough steps
if (len(self.memory) >= self.config['min_memory_size'] and 
    self.steps_since_update >= self.config['update_freq']):
    loss = self._update_network()
    ...
    self.steps_since_update = 0  # Reset counter after update
```

**Problem:** The network updates **once** every 1000 steps, performing a **single** gradient step with batch_size=32. For a 5×5 board with 3 mines (22 safe cells), episodes are ~10-25 steps long. So the network trains once every ~50-100 episodes, seeing only 32 experiences out of 1000+ collected.

**Standard DQN practice:** Update every 1-4 steps, not every 1000. The original DQN paper (Mnih et al., 2015) updates every 4 steps.

**Fix:**
```python
'update_freq': 4,  # Update every 4 steps (standard DQN)
```
And move the update inside the step loop, not after the episode:
```python
# Inside the step loop in _train_episode:
if (len(self.memory) >= self.config['min_memory_size'] and 
    self.total_steps % self.config['update_freq'] == 0):
    loss = self._update_network()
```

---

### C2. State Representation is Inadequate for Learning

**Files:** `game_api.py` lines 154-175 (`get_board_array`)  
**Impact:** The neural network cannot extract meaningful spatial patterns.

```python
# game_api.py - get_board_array()
# Channel 0: Visible state (-3=hidden, -2=flag, -1=mine, 0-8=numbers)
# Channel 1: Is revealed (0 or 1)
# Channel 2: Is flagged (0 or 1)
```

**Problems:**

1. **Channel 0 is a mess.** It contains values from -3 to 8, mixing categorical states (hidden=-3, flag=-2) with ordinal values (numbers 0-8). A Conv2d layer with ReLU will treat -3 (hidden) identically to 0 after activation in early layers. The network has to learn that -3 means "unknown" while 3 means "3 adjacent mines" — these are fundamentally different concepts crammed into one channel.

2. **No normalization.** Values range from -3 to 8. Neural networks learn much better with normalized inputs (0-1 or -1 to 1).

3. **Channels 1 and 2 are redundant with Channel 0.** If a cell is revealed, Channel 0 shows its number; if flagged, Channel 0 shows -2. Channels 1 and 2 add almost no new information.

4. **Missing critical information:** The agent doesn't know the total mine count or remaining mines, which is essential for deductive reasoning.

**Fix:** Use a one-hot style multi-channel encoding:
```python
def get_board_array(self) -> np.ndarray:
    """
    Channels:
    0: Is hidden (1 if hidden, 0 otherwise)
    1: Is flagged (1 if flagged, 0 otherwise)  
    2: Is revealed (1 if revealed, 0 otherwise)
    3-11: Number channels (channel 3+n is 1 if revealed and adjacent_mines == n)
    """
    channels = []
    # Binary state channels
    hidden = np.zeros((self.rows, self.cols), dtype=np.float32)
    flagged = np.zeros((self.rows, self.cols), dtype=np.float32)
    revealed = np.zeros((self.rows, self.cols), dtype=np.float32)
    
    # Number channels (one-hot for 0-8)
    number_channels = [np.zeros((self.rows, self.cols), dtype=np.float32) for _ in range(9)]
    
    for row in range(self.rows):
        for col in range(self.cols):
            cell = self.game_board.get_cell(row, col)
            if cell.is_revealed():
                revealed[row, col] = 1.0
                number_channels[cell.adjacent_mines][row, col] = 1.0
            elif cell.is_flagged():
                flagged[row, col] = 1.0
            else:
                hidden[row, col] = 1.0
    
    return np.stack([hidden, flagged, revealed] + number_channels, axis=-1)  # 12 channels
```

---

### C3. Curriculum Knowledge Transfer is Broken (Size Mismatch)

**Files:** `models.py` lines 36-40 (MinesweeperNet), `curriculum.py` lines 289-312 (`_transfer_knowledge`)  
**Impact:** Transitions between curriculum stages will crash or silently fail.

```python
# models.py - MinesweeperNet
self.feature_size = self._calculate_feature_size()  # Depends on board_height × board_width
self.fc1 = nn.Linear(self.feature_size, 512)        # Fixed size!
```

The CNN uses `padding=1` with no pooling, so the spatial dimensions are preserved through all conv layers. For a 5×5 board: `feature_size = 128 * 5 * 5 = 3200`. For a 7×7 board: `feature_size = 128 * 7 * 7 = 6272`. The fully-connected layer `fc1` has a different shape for every board size.

When `_transfer_knowledge` copies a model from the `tiny` (5×5) stage to `small` (7×7), `load_state_dict()` will **raise a RuntimeError** because the tensor shapes don't match.

**Fix options:**

**Option A (Recommended): Use Global Average Pooling instead of flatten:**
```python
class MinesweeperNet(nn.Module):
    def __init__(self, board_height, board_width, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        # Global Average Pooling → fixed 128-dim vector regardless of board size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 256)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1).squeeze(-1)  # [B, 128]
        x = F.relu(self.fc1(x))
        return x
```

**Option B: Fully Convolutional Output.** Output Q-values per-cell using 1×1 conv, avoiding FC layers entirely. This is the cleanest approach for varying board sizes — see Section 5.

---

### C4. DQN Output Head Cannot Handle Variable Board Sizes

**Files:** `models.py` lines 85-91 (DQN.__init__)  
**Impact:** Same as C3 — the output layer `num_actions = rows * cols * 3` changes per stage.

```python
# models.py line 91
self.num_actions = num_actions or (board_height * board_width * 3)
self.advantage_head = nn.Linear(256, self.num_actions)  # Changes per board size!
```

Even if the backbone is fixed (via GAP), the output head changes dimension. A 5×5 board has 75 actions, a 7×7 board has 147 actions. Weights cannot be transferred.

**Fix:** Use a fully convolutional architecture that outputs per-cell action scores:
```python
class FCN_DQN(nn.Module):
    """Fully convolutional DQN — works for any board size"""
    def __init__(self, input_channels=12, num_action_types=1):
        super().__init__()
        # Shared conv backbone
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        
        # Per-cell value and advantage (dueling)
        self.value_conv = nn.Conv2d(64, 1, 1)  # [B, 1, H, W]
        self.advantage_conv = nn.Conv2d(64, num_action_types, 1)  # [B, A, H, W]
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        value = self.value_conv(x)  # [B, 1, H, W]
        advantage = self.advantage_conv(x)  # [B, A, H, W]
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q  # [B, A, H, W] — reshape to [B, H*W*A] for action selection
```

This architecture:
- Works for **any** board size with identical weights
- Is naturally spatial — conv filters learn local patterns that transfer
- Dramatically reduces parameter count
- Makes curriculum transitions seamless

---

### C5. Mines Are Not Placed Until First Reveal — State is Empty on Reset

**Files:** `board.py` line 127 (`_place_mines` called from `reveal_cell`), `environment.py` line 52 (`reset`)  
**Impact:** The initial observation is a board of all -3 (hidden) with no information, and the first action MUST be a reveal (anything else is useless). But the agent doesn't know this.

```python
# board.py line 127
# Place mines on first click
if not self.mines_placed:
    self._place_mines(row, col)
```

For the first move, the agent sees an entirely blank board. There's zero information to guide its decision. Any cell is equally good. But the agent will waste time trying to learn patterns from this empty state.

**Fix:** Auto-reveal a random cell on `reset()` to initialize the board:
```python
def reset(self) -> np.ndarray:
    self.api.reset_game()
    self.steps_taken = 0
    # Auto-reveal a random cell to place mines and give initial information
    row = random.randint(0, self.rows - 1)
    col = random.randint(0, self.cols - 1)
    self.api.take_action(row, col, Action.REVEAL)
    return self._get_observation()
```

---

## 3. Major Issues (Significantly Hurt Learning)

### M1. Action Space is 3× Too Large (Flag/Unflag Hurts Learning)

**Files:** `environment.py` line 40  
**Impact:** Action space explosion with near-zero learning signal for 2/3 of actions.

```python
self.action_space_size = rows * cols * 3  # reveal, flag, unflag
```

For a 9×9 beginner board: **243 actions** instead of 81. For 16×30 expert: **1440 actions**.

**Why flagging hurts:**
1. Flagging provides almost no reward signal during gameplay (fixed +0.5 regardless of correctness — `environment.py` line 197).
2. Flag correctness can't be verified until the game ends.
3. Unflagging gives -2.0 penalty, creating a ratchet that punishes exploration.
4. The agent can learn degenerate strategies like flag-cycling to avoid losing.
5. Flagging is **never required to win** Minesweeper — only revealing all safe cells matters.

**Fix:** Remove flag/unflag entirely for initial training:
```python
self.action_space_size = rows * cols  # reveal only
```
Add flagging back as an advanced feature only after the model can play at a basic level.

---

### M2. Epsilon Decay is Way Too Fast

**Files:** `curriculum.py` lines 282-300 (`_get_trainer_config`)  
**Impact:** Exploration dies before the agent learns anything useful.

```python
# For 5x5 boards:
'epsilon_decay': 0.99,
'epsilon_start': 0.5,  # phase_0
```

With 1000 episodes per batch and ~15 steps per episode, that's ~15,000 steps. Epsilon goes from 0.5 to `0.5 * 0.99^1000 = 0.00002` within the first batch. After just 460 episodes, epsilon is below 0.01.

But with `update_freq=1000`, the network has only trained ~15 times by that point! The agent has barely learned anything before it stops exploring.

**Fix:** Use step-based linear epsilon decay instead of exponential:
```python
# Linear decay over total training steps
epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (step / total_steps))
```

---

### M3. Target Network Updates Too Infrequently

**Files:** `trainer.py` lines 125-126  
**Impact:** Stale target Q-values cause unstable training.

```python
# trainer.py line 71
'target_update_freq': 100,  # Episodes, not steps
```

```python
# trainer.py lines 125-126
if episode % self.config['target_update_freq'] == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

This updates the target network every 100 **episodes**. But with `update_freq=1000` steps, the Q-network might have only trained once or twice between target updates. The interaction between these two frequencies is incoherent.

**Fix:** Update target network every N **training steps** (gradient updates), not episodes:
```python
# Use soft target updates every training step:
tau = 0.005
for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

---

### M4. No MPS Device Support (M4 Mac Mini Trains on CPU)

**Files:** `trainer.py` line 59  
**Impact:** Training is ~5-10× slower than necessary on Apple Silicon.

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Fix:**
```python
if torch.cuda.is_available():
    self.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    self.device = torch.device("mps")
else:
    self.device = torch.device("cpu")
```

**MPS caveats for PyTorch:**
- Some operations may not be supported on MPS — use `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable
- `torch.BoolTensor(...).to("mps")` can be problematic — use `torch.tensor(..., dtype=torch.bool, device=device)` instead
- BatchNorm can be slow on MPS for small batch sizes — consider GroupNorm or LayerNorm
- Profile to confirm MPS is actually faster for your model size (small models may not benefit)

---

### M5. Reward Shaping Creates Perverse Incentives

**Files:** `environment.py` lines 161-199  
**Impact:** The agent learns to avoid risk rather than play optimally.

**Problem 1: Step penalty accumulates and dominates.**
```python
'step_penalty': -0.1
```
On a 9×9 beginner board with 71 safe cells, even a perfect game requires 71 reveals (or fewer with cascades). That's -7.1 from step penalties alone. The agent learns "finish fast" rather than "play correctly."

**Problem 2: Cascade rewards create a lottery.**  
```python
'reveal_empty_cell': 8.0,   # Bonus for clicking empty cell
'reveal_multi_safe': 1.5,   # Per additional cascaded cell
```
Clicking an empty cell that cascades to reveal 20 cells gives 8.0 + 19×1.5 = 36.5 reward. Clicking a numbered cell gives 4.0. This teaches the agent to hunt for empty cells (low-information high-reward) rather than make logically safe moves, which is the core of Minesweeper skill.

**Problem 3: Win reward (150) vs loss penalty (-100) is asymmetric.**
Since the agent can win ~12% of beginner games by random play (first-click safety), and wins give +150 while losses give -100, the expected value of random play is positive: `0.12 × 150 + 0.88 × (-100) = -70`. Combined with per-step rewards from revealing cells before dying, random play may produce positive total rewards, meaning the agent has little pressure to improve.

**Fix:**
```python
reward_config = {
    'win': 1.0,           # Simple, clear signal
    'lose': -1.0,         # Balanced with win
    'reveal_safe': 0.01,  # Tiny positive — progress, not a goal
    'step_penalty': 0.0,  # Remove — let episode termination handle time pressure
    'invalid_action': -0.1,
}
```
Sparse rewards (win/loss only) work better for Minesweeper because they don't create shortcuts.

---

### M6. Replay Buffer is Too Small and Sampling is Uniform

**Files:** `trainer.py` line 74  
**Impact:** Important experiences (wins, losses, cascade reveals) are drowned out.

```python
'memory_size': 10000,  # Default; curriculum uses 20000-100000
```

For a 9×9 board with ~15 steps per episode, 10,000 experiences is only ~660 episodes. Given that wins are rare early in training, winning experiences are quickly overwritten.

**Fix:**
1. Increase buffer to 100,000+ minimum
2. Implement **Prioritized Experience Replay (PER)** — sample important transitions (high TD-error, wins, losses) more frequently
3. Keep a separate buffer for winning episodes to ensure they're always available

---

### M7. Single Gradient Step Per Update

**Files:** `trainer.py` line 195-212 (`_update_network`)  
**Impact:** Extremely sample-inefficient.

The `_update_network` method performs exactly one gradient step per call. With `update_freq=1000`, this means 1 gradient step per 1000 environment steps. Standard DQN does 1 gradient step per 4 environment steps — that's **250×** more training.

**Fix:** Either:
- Set `update_freq=4` and call `_update_network` per step (recommended)
- Or perform multiple gradient steps per update: `for _ in range(num_updates): self._update_network()`

---

## 4. Minor Issues (Nice to Have)

### m1. Dropout During Training (0.3) is High for RL

**File:** `models.py` line 37  
RL already has high variance from environment stochasticity. Dropout=0.3 in the backbone adds more noise to already noisy Q-value estimates. Use 0.0-0.1 max, or remove entirely.

### m2. MSE Loss Instead of Huber/SmoothL1

**File:** `trainer.py` line 100  
```python
self.criterion = nn.MSELoss()
```
MSE is sensitive to outlier Q-values (which are common early in training). `nn.SmoothL1Loss()` (Huber loss) is standard for DQN and more robust.

### m3. No Double DQN

**File:** `trainer.py` lines 204-206  
```python
next_q_values = self.target_network(next_states).max(1)[0]
```
This is standard DQN, which overestimates Q-values. Double DQN (use online network to select action, target network to evaluate) is a simple fix:
```python
# Double DQN
best_actions = self.q_network(next_states).argmax(1)
next_q_values = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
```

### m4. Action Mask Not Used in Target Q-Value Computation

**File:** `trainer.py` lines 204-206  
The next-state Q-values are computed without masking invalid actions. This means the target can include Q-values for impossible actions (like revealing an already-revealed cell), injecting noise into the training signal.

```python
# Fix: mask invalid actions in target computation too
next_masks = torch.BoolTensor(np.array([e.action_mask for e in batch])).to(self.device)
next_q = self.target_network(next_states)
next_q[~next_masks] = float('-inf')
next_q_values = next_q.max(1)[0]
# Handle case where all actions are invalid (terminal state)
next_q_values = torch.where(dones, torch.zeros_like(next_q_values), next_q_values)
```

### m5. `training_wins` Serialization Bug

**File:** `trainer.py` line 242  
```python
'training_wins': [bool(w) for w in self.training_wins],
```
`bool(None)` returns `False`, so incomplete games (None) are recorded as losses. Use:
```python
'training_wins': [w if w is None else bool(w) for w in self.training_wins],
```

### m6. Phase 0 Perfect Knowledge Has 4 Channels but Trainer Detects 3

The `PerfectKnowledgeMinesweeperEnvironment` returns 4 channels. The `DQNTrainer.__init__` does:
```python
sample_obs = env.reset()
input_channels = sample_obs.shape[-1] if len(sample_obs.shape) == 3 else 3
```
This correctly detects 4 channels for Phase 0. However, when transitioning FROM Phase 0 (4-channel) TO the `tiny` stage (3-channel), the model architecture changes and weights can't be transferred. The Phase 0 concept is sound but the transition is broken.

### m7. Learning Rate is Too High for DQN

**File:** `curriculum.py` line 276  
```python
'learning_rate': 0.002,  # 5x5 boards
```
For DQN with Adam, 0.002 is high. The original DQN paper used 0.00025 (with RMSprop). Start with 1e-4 or lower, especially since the network rarely trains (see C1).

### m8. Batch Normalization with Small Batches

**File:** `models.py` lines 32-34  
With `batch_size=32`, batch normalization statistics can be noisy. During evaluation (model.eval()), running statistics are used, which may not reflect training behavior well. Consider LayerNorm instead, which doesn't depend on batch size.

### m9. `_evaluate` Method Signature Mismatch

**File:** `trainer.py` line 219 vs `train_ai.py` line 305  
`_evaluate()` in `DQNTrainer` takes no `num_episodes` parameter, but `train_ai.py` calls `trainer._evaluate(num_episodes=eval_episodes)`. This will crash with a `TypeError`.

```python
# trainer.py
def _evaluate(self) -> Tuple[float, float]:  # No num_episodes parameter!
```

```python
# train_ai.py line 305
avg_score, win_rate = trainer._evaluate(num_episodes=eval_episodes)  # Will crash!
```

---

## 5. Recommended Fix Order

### Phase 1: Make it Train (Critical Fixes)
1. **C1: Fix update frequency** → Change to every 4 steps, move updates inside step loop
2. **C2: Fix state representation** → One-hot encoding with 12 channels
3. **M1: Remove flag/unflag** → Reveal-only action space
4. **M5: Fix rewards** → Sparse win/loss rewards
5. **M7: Train more per update** → 1 gradient step per 4 env steps
6. **M4: Add MPS support** → Apple Silicon GPU acceleration
7. **m2: Use Huber loss** → `nn.SmoothL1Loss()`
8. **m9: Fix _evaluate signature** → Add `num_episodes` parameter

### Phase 2: Make it Learn Well
9. **C3+C4: Fully convolutional architecture** → Board-size-independent CNN
10. **M2: Fix epsilon decay** → Linear schedule over total training steps
11. **M3: Soft target updates** → tau=0.005 every training step
12. **m3: Double DQN** → Simple fix, big improvement
13. **m4: Mask target Q-values** → Use action masks for next-state values
14. **m6: Fix Phase 0 transition** → Either remove Phase 0 or use same channel count

### Phase 3: Make it Master
15. **M6: Prioritized Experience Replay** → Sample important transitions more
16. **C5: Auto-reveal on reset** → Give agent initial information
17. **m1: Remove/reduce dropout** → 0.0-0.1
18. **m7: Lower learning rate** → 1e-4 with Adam
19. Add learning rate scheduling (decay as training progresses)

---

## 6. Proposed Training Plan for Local M4 Mac Mini

### Hardware Profile
- M4 Mac Mini: 10-core CPU, 10-core GPU, 16-32GB unified memory
- Use MPS backend for GPU acceleration
- Expected throughput: ~500-2000 episodes/minute for 5×5 boards (after fixes)

### Recommended Architecture

```python
class MinesweeperFCN(nn.Module):
    """Fully convolutional — works for any board size, ~50K parameters"""
    def __init__(self, input_channels=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )
        # Dueling: per-cell value and advantage
        self.value = nn.Conv2d(64, 1, 1)      # V(s) per cell
        self.advantage = nn.Conv2d(64, 1, 1)   # A(s,a) per cell (reveal only)
    
    def forward(self, x):
        # x: [B, C, H, W]
        features = self.net(x)
        v = self.value(features)       # [B, 1, H, W]
        a = self.advantage(features)   # [B, 1, H, W]
        q = v + a - a.mean(dim=(2, 3), keepdim=True)  # Dueling
        return q.squeeze(1)  # [B, H, W] — one Q-value per cell
```

### Hyperparameters

```python
config = {
    'learning_rate': 1e-4,
    'batch_size': 128,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_steps': 100_000,  # Linear decay over 100K steps
    'target_update_tau': 0.005,      # Soft update every training step
    'update_freq': 4,                # Train every 4 env steps
    'memory_size': 200_000,
    'min_memory_size': 5_000,
    'max_steps_per_episode': 200,
}
```

### Curriculum Plan

| Stage | Board | Mines | Target Win Rate | Est. Steps | Est. Time |
|-------|-------|-------|----------------|-----------|-----------|
| 1. Tiny | 5×5 | 3 | 60% | 200K | ~30 min |
| 2. Small | 7×7 | 7 | 45% | 500K | ~1.5 hr |
| 3. Mini | 8×8 | 9 | 40% | 800K | ~2.5 hr |
| 4. Beginner | 9×9 | 10 | 35% | 1.5M | ~5 hr |
| 5. Intermediate | 16×16 | 40 | 20% | 5M | ~15 hr |
| 6. Expert | 16×30 | 99 | 10% | 10M | ~30 hr |

**Total estimated time: ~55 hours** (2-3 days continuous on M4 Mac Mini)

**Key differences from current system:**
- **Drop Phase 0** (perfect knowledge) — it uses different channel counts and doesn't teach useful skills
- **Reveal-only action space** — add flagging only if needed for advanced play
- **FCN architecture** — same weights work for all board sizes
- **4× more training** per environment step
- **Auto-reveal on reset** — agent always starts with information
- **Sparse rewards** — win=+1, loss=-1, minimal shaping

### Training Script Outline

```python
# train_v2.py
env = MinesweeperEnv(5, 5, 3)  # Start small
model = MinesweeperFCN(input_channels=12).to("mps")
target_model = copy.deepcopy(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
buffer = PrioritizedReplayBuffer(200_000)

for step in range(total_steps):
    # Epsilon-greedy action
    epsilon = linear_decay(step, eps_start=1.0, eps_end=0.05, decay_steps=100_000)
    
    if random.random() < epsilon:
        action = random_valid_action(env)
    else:
        with torch.no_grad():
            q_values = model(state)  # [H, W]
            q_values[~valid_mask] = -inf
            action = q_values.argmax()
    
    next_state, reward, done, info = env.step(action)
    buffer.push(state, action, reward, next_state, done)
    
    # Train every 4 steps
    if step % 4 == 0 and len(buffer) > 5000:
        batch = buffer.sample(128)
        loss = compute_double_dqn_loss(model, target_model, batch, gamma=0.99)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        # Soft target update
        soft_update(target_model, model, tau=0.005)
    
    if done:
        state = env.reset()  # Auto-reveals first cell
        
        # Curriculum check every 1000 episodes
        if episodes % 1000 == 0:
            win_rate = evaluate(model, env, 200)
            if win_rate >= target and should_advance:
                env = next_curriculum_stage()
```

### Validation Milestones

Before proceeding to next stage, verify:
1. **Loss is decreasing** (not flat or oscillating wildly)
2. **Win rate on evaluation is above target** (averaged over 500+ games)
3. **Q-values are reasonable** (not diverging to ±infinity)
4. **Epsilon has decayed enough** that greedy policy is meaningful

### MPS-Specific Recommendations

```bash
# Set environment variable for MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run training
python train_v2.py --device mps
```

- Monitor GPU utilization with Activity Monitor → GPU History
- If MPS is slower than CPU for your model size (possible for very small models), fall back to CPU
- Use `torch.mps.synchronize()` before timing measurements
- Avoid frequent CPU↔MPS transfers — keep tensors on device
