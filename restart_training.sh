#!/bin/bash
# Minesweeper v4 training auto-restart script
# Called by cron monitor when training dies unexpectedly.
# Finds the latest checkpoint and warm-starts from it.

set -euo pipefail

REPO="$HOME/git_repos/minesweeper-ai"
LOG="$REPO/supervised_v4_output.log"
SAVE_DIR="$REPO/models_v4/supervised_v2/expert"

cd "$REPO"

# Safety: don't restart if already running
if pgrep -f "python.*train_supervised_v4" > /dev/null 2>&1; then
    echo "ALREADY_RUNNING"
    exit 0
fi

# Find best checkpoint to resume from
# Priority: best_model.pth (highest win rate), then latest checkpoint_iter*.pth
WARM_START=""
if [ -f "$SAVE_DIR/best_model.pth" ]; then
    WARM_START="$SAVE_DIR/best_model.pth"
elif ls "$SAVE_DIR"/checkpoint_iter*.pth 1>/dev/null 2>&1; then
    WARM_START=$(ls -t "$SAVE_DIR"/checkpoint_iter*.pth | head -1)
elif [ -f "$SAVE_DIR/interrupted_iter"*.pth ]; then
    WARM_START=$(ls -t "$SAVE_DIR"/interrupted_iter*.pth | head -1)
fi

if [ -z "$WARM_START" ]; then
    echo "NO_CHECKPOINT"
    exit 1
fi

# Figure out remaining iterations from log
COMPLETED=$(grep -c "^ITER.*EVAL:" "$LOG" 2>/dev/null || echo 0)
TOTAL=30
REMAINING=$((TOTAL - COMPLETED))
if [ "$REMAINING" -lt 1 ]; then
    REMAINING=1
fi
# Cap at 30
if [ "$REMAINING" -gt 30 ]; then
    REMAINING=30
fi

# Back up the log (append mode — keep history)
if [ -f "$LOG" ]; then
    cp "$LOG" "${LOG}.bak.$(date +%Y%m%d_%H%M%S)"
fi

# Launch training
nohup python3 -u train_supervised_v4.py \
    --device mps \
    --model "$WARM_START" \
    --num-iterations "$REMAINING" \
    > "$LOG" 2>&1 &

NEW_PID=$!
echo "RESTARTED:pid=$NEW_PID:model=$(basename $WARM_START):iters=$REMAINING:completed=$COMPLETED"
