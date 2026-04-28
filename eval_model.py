#!/usr/bin/env python3
"""Quick standalone eval script for minesweeper models."""
import os, sys, argparse, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ai.models_v3 import MinesweeperResNet
from src.ai.environment import MinesweeperEnvironment
from src.ai.algorithmic_solver import AlgorithmicSolver

EXPERT = {'rows': 16, 'cols': 30, 'mines': 99}

def evaluate(model, device, num_episodes, board_cfg=EXPERT):
    rows, cols, mines = board_cfg['rows'], board_cfg['cols'], board_cfg['mines']
    solver = AlgorithmicSolver(rows, cols, mines)
    
    wins = 0
    total_guesses = 0
    total_solver_moves = 0
    total_survivals = 0
    total_deaths = 0
    model.eval()
    
    for ep in range(num_episodes):
        env = MinesweeperEnvironment(rows=rows, cols=cols, mines=mines, use_v2=True, normalize_rewards=True)
        state = env.reset()
        done = False
        steps = 0
        pending_guess = False
        
        while not done and steps < 1000:
            hidden, flagged, revealed = solver._parse_state(state)
            safe, known_mines = solver._find_deterministic_moves(hidden, flagged, revealed)
            
            if safe:
                if pending_guess:
                    total_survivals += 1
                    pending_guess = False
                best = max(safe, key=lambda cell: solver._score_random_cell(cell[0], cell[1], revealed))
                action = best[0] * cols + best[1]
                total_solver_moves += 1
            else:
                if pending_guess:
                    total_survivals += 1
                
                action_mask = env.get_action_mask()
                for (r, c) in known_mines:
                    action_mask[r * cols + c] = False
                valid = np.where(action_mask)[0]
                if len(valid) == 0:
                    valid = np.where(env.get_action_mask())[0]
                    if len(valid) == 0:
                        break
                
                with torch.no_grad():
                    st = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(device)
                    logits = model(st).squeeze(0).reshape(-1)
                    # P(mine) mode: pick cell with lowest mine probability
                    probs = torch.sigmoid(logits)
                    masked_probs = torch.full_like(probs, float('inf'))
                    masked_probs[valid] = probs[valid]
                    action = masked_probs.argmin().item()
                
                total_guesses += 1
                pending_guess = True
            
            state, reward, done, info = env.step(action)
            steps += 1
        
        if pending_guess:
            if info.get('game_state') == 'won':
                total_survivals += 1
            else:
                total_deaths += 1
        
        if info.get('game_state') == 'won':
            wins += 1
        
        if (ep + 1) % 500 == 0 or ep == num_episodes - 1:
            wr = wins / (ep + 1) * 100
            gsr = total_survivals / max(1, total_survivals + total_deaths) * 100
            avg_g = total_guesses / max(1, ep + 1)
            avg_s = total_solver_moves / max(1, ep + 1)
            print(f"  [{ep+1}/{num_episodes}] Win={wr:.1f}% | GSR={gsr:.1f}% | Guesses={avg_g:.1f} | Solver={avg_s:.1f}")
        
        if (ep + 1) % 200 == 0:
            torch.mps.empty_cache() if device.type == 'mps' else None
    
    win_rate = wins / num_episodes
    gsr = total_survivals / max(1, total_survivals + total_deaths)
    return {
        'win_rate': win_rate,
        'wins': wins,
        'total': num_episodes,
        'gsr': gsr,
        'avg_guesses': total_guesses / num_episodes,
        'avg_solver': total_solver_moves / num_episodes,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to .pth checkpoint')
    parser.add_argument('--episodes', '-n', type=int, default=5000)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model = MinesweeperResNet().to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    print(f"Model: {args.checkpoint}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Board: {EXPERT['rows']}x{EXPERT['cols']}, {EXPERT['mines']} mines")
    print()
    
    start = time.time()
    results = evaluate(model, device, args.episodes)
    elapsed = time.time() - start
    
    print()
    print(f"═══════════════════════════════════════════")
    print(f"  Win Rate: {results['win_rate']*100:.1f}% ({results['wins']}/{results['total']})")
    print(f"  GSR:      {results['gsr']*100:.1f}%")
    print(f"  Avg Guesses: {results['avg_guesses']:.1f}")
    print(f"  Avg Solver:  {results['avg_solver']:.1f}")
    print(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"═══════════════════════════════════════════")

if __name__ == '__main__':
    main()
