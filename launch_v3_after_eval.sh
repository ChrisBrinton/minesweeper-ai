#!/bin/bash
# Wait for eval to finish, then launch supervised v3 training
cd /Users/christopherbrinton/git_repos/minesweeper-ai

echo "Waiting for eval_model.py to finish..."
while pgrep -f "eval_model.py" > /dev/null 2>&1; do
    sleep 30
done

echo "Eval finished. Results:"
cat eval_iter9_5000.log
echo ""
echo "Launching supervised v3 training..."
nohup python3 -u train_supervised_v3.py --device mps > supervised_v3_output.log 2>&1 &
V3_PID=$!
echo "Training launched (PID: $V3_PID)"

# Notify Bob
openclaw system event --text "Minesweeper: Eval complete, supervised v3 training launched (PID: $V3_PID). Check supervised_v3_output.log" --mode now
