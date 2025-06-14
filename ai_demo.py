#!/usr/bin/env python3
"""
Minesweeper AI Demo - Harness to run trained AI with visual GUI
Shows the trained AI model playing the game in real-time with the UI
"""

import sys
import os
import time
import torch
import numpy as np
import argparse
from threading import Thread, Event
from typing import Optional, Tuple

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui.gui import MinesweeperGUI
from ai.trainer import DQNTrainer, create_trainer
from ai.models import DQN
from ai.environment import MinesweeperEnvironment
from ai.model_storage import get_latest_checkpoint, find_latest_model_dir
from game.board import GameState, CellState


class AIPlayer:
    """AI Player that can interact with the GUI"""
    def __init__(self, model_path: str, difficulty: str = "beginner"):
        """
        Initialize AI player with trained model
        
        Args:
            model_path: Path to the trained model file
            difficulty: Game difficulty (beginner, intermediate, expert)
        """
        self.difficulty = difficulty.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Game configuration based on difficulty
        self.game_config = {
            'beginner': (9, 9, 10),
            'intermediate': (16, 16, 40),
            'expert': (16, 30, 99)
        }
        self.rows, self.cols, self.mines = self.game_config[self.difficulty]
        
        # Load trained model
        self.model, self.trainer = self._load_model(model_path)
        
        print(f"🤖 AI Player initialized for {difficulty} difficulty")
        print(f"   📊 Board: {self.rows}x{self.cols} with {self.mines} mines")
        print(f"   🧠 Model: {model_path}")
        print(f"   💻 Device: {self.device}")
        
    def _load_model(self, model_path: str) -> Tuple[DQN, DQNTrainer]:
        """Load the trained model"""
        try:
            # Load checkpoint (with weights_only=False for compatibility)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create environment and trainer with explicit parameters
            env = MinesweeperEnvironment(self.rows, self.cols, self.mines)
            trainer = DQNTrainer(env)
            
            # Load model state
            trainer.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            trainer.q_network.eval()  # Set to evaluation mode
              # Print model info
            if 'win_rate' in checkpoint:
                win_rate = checkpoint['win_rate']
                # Handle both decimal (0.947) and percentage (94.7) formats
                if win_rate > 1:
                    print(f"   🎯 Model win rate: {win_rate:.1f}%")
                else:
                    print(f"   🎯 Model win rate: {win_rate:.1%}")
            if 'total_episodes' in checkpoint:
                print(f"   📈 Trained episodes: {checkpoint['total_episodes']:,}")
            
            return trainer.q_network, trainer
            
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            raise
    
    def get_action(self, board_state: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Get AI action for current board state
        
        Args:
            board_state: Current board state as numpy array
            action_mask: Valid actions mask
            
        Returns:
            Action index (flattened coordinate)
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0).to(self.device)
            
            # Get Q-values
            q_values = self.model(state_tensor)
            
            # Apply action mask (set invalid actions to very low value)
            if action_mask is not None:
                masked_q_values = q_values.clone()
                masked_q_values[0][~action_mask] = float('-inf')
                action = masked_q_values.argmax().item()
            else:
                action = q_values.argmax().item()
            
            return action
    def action_to_coordinates(self, action: int) -> Tuple[int, int]:
        """Convert flattened action index to (row, col) coordinates"""
        # Action index includes action type, divide by 3 to get the cell position
        cell_position = action // 3
        row = cell_position // self.cols
        col = cell_position % self.cols
        return row, col


class AIDemo:
    """Main demo class that connects AI player with GUI"""
    
    def __init__(self, model_path: str, difficulty: str = "beginner", delay: float = 1.0, auto_restart: bool = True):
        """
        Initialize AI demo
        
        Args:
            model_path: Path to trained model
            difficulty: Game difficulty
            delay: Delay between AI moves (seconds)
            auto_restart: Whether to automatically restart games
        """
        self.model_path = model_path
        self.difficulty = difficulty
        self.delay = delay
        self.auto_restart = auto_restart
        
        # Initialize AI player
        self.ai_player = AIPlayer(model_path, difficulty)
        
        # Initialize GUI
        self.gui = MinesweeperGUI()
        
        # Demo state
        self.demo_running = False
        self.demo_thread: Optional[Thread] = None
        self.stop_event = Event()
        
        # Statistics
        self.games_played = 0
        self.games_won = 0
        self.total_moves = 0
        
        # Modify GUI for demo
        self._setup_demo_gui()
    
    def _setup_demo_gui(self):
        """Setup GUI modifications for AI demo"""
        # Change window title
        self.gui.root.title(f'Minesweeper AI Demo - {self.difficulty.title()} ({self.ai_player.device})')
        
        # Add AI control panel
        self._add_control_panel()
        
        # Start with correct difficulty
        self.gui._new_game(self.difficulty)
        
        # Override cell click handlers to prevent human input during demo
        self.original_cell_click = self.gui._on_cell_click
        self.original_cell_right_click = self.gui._on_cell_right_click
        
    def _add_control_panel(self):
        """Add AI control panel to GUI"""
        import tkinter as tk
        
        # Create control frame
        control_frame = tk.Frame(self.gui.root, bg='lightgray', relief='raised', bd=2)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # AI status label
        self.status_label = tk.Label(control_frame, text="🤖 AI Ready", bg='lightgray', font=('Arial', 10, 'bold'))
        self.status_label.pack(side='left', padx=5)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='lightgray')
        button_frame.pack(side='right', padx=5)
        
        self.start_button = tk.Button(button_frame, text="▶ Start AI", command=self.start_demo, bg='lightgreen')
        self.start_button.pack(side='left', padx=2)
        
        self.stop_button = tk.Button(button_frame, text="⏹ Stop AI", command=self.stop_demo, bg='lightcoral', state='disabled')
        self.stop_button.pack(side='left', padx=2)
        
        self.reset_button = tk.Button(button_frame, text="🔄 New Game", command=self.reset_game, bg='lightblue')
        self.reset_button.pack(side='left', padx=2)
        
        # Statistics frame
        stats_frame = tk.Frame(self.gui.root, bg='lightgray', relief='sunken', bd=2)
        stats_frame.pack(fill='x', padx=5, pady=2)
        self.stats_label = tk.Label(stats_frame, text="Games: 0 | Wins: 0 (0.0%) | Moves: 0", 
                                   bg='lightgray', font=('Arial', 9))
        self.stats_label.pack(padx=5, pady=2)
    
    def _update_status(self, message: str):
        """Update AI status display"""
        try:
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text=message)
                self.gui.root.update_idletasks()
        except Exception:
            pass  # GUI may be destroyed, ignore silently
    
    def _update_stats(self):
        """Update statistics display"""
        try:
            if hasattr(self, 'stats_label') and self.stats_label.winfo_exists():
                win_rate = (self.games_won / self.games_played * 100) if self.games_played > 0 else 0
                stats_text = f"Games: {self.games_played} | Wins: {self.games_won} ({win_rate:.1f}%) | Moves: {self.total_moves}"
                self.stats_label.config(text=stats_text)
                self.gui.root.update_idletasks()
        except Exception:
            pass  # GUI may be destroyed, ignore silently
    
    def start_demo(self):
        """Start AI demonstration"""
        if not self.demo_running:
            self.demo_running = True
            self.stop_event.clear()
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self._update_status("🤖 AI Playing...")
              # Start demo thread
            self.demo_thread = Thread(target=self._run_demo, daemon=True)
            self.demo_thread.start()
    
    def stop_demo(self):
        """Stop AI demonstration"""
        if self.demo_running:
            self.demo_running = False
            self.stop_event.set()
            
            # Update UI safely (GUI might be destroyed)
            try:
                if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                    self.start_button.config(state='normal')
                if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                    self.stop_button.config(state='disabled')
                self._update_status("🤖 AI Stopped")
            except Exception as e:
                print(f"⚠️  UI update error (GUI may be closed): {e}")
    
    def reset_game(self):
        """Reset game and statistics"""
        self.stop_demo()
        self.gui._new_game(self.difficulty)
        self.games_played = 0
        self.games_won = 0
        self.total_moves = 0
        self._update_stats()
        self._update_status("🤖 AI Ready")
    
    def _run_demo(self):
        """Main demo loop (runs in separate thread)"""
        try:
            while self.demo_running and not self.stop_event.is_set():
                # Play one game
                self._play_game()
                
                # Wait between games if auto-restart is enabled
                if self.auto_restart and self.demo_running:
                    time.sleep(self.delay * 2)  # Longer pause between games
                    if not self.stop_event.is_set():
                        self.gui.root.after(0, lambda: self.gui._new_game(self.difficulty))
                        time.sleep(0.5)  # Give GUI time to reset
                else:
                    break
                    
        except Exception as e:
            print(f"❌ Demo error: {e}")
            self.gui.root.after(0, lambda: self._update_status(f"❌ Error: {str(e)[:20]}..."))
        finally:
            if self.demo_running:
                self.gui.root.after(0, self.stop_demo)
    
    def _play_game(self):
        """Play one complete game"""
        game_moves = 0
        
        try:
            print(f"🎮 Starting new game - Initial state: {self.gui.game_board.game_state}")
            
            # Handle initial state (READY or PLAYING)
            while (self.demo_running and 
                   not self.stop_event.is_set() and 
                   self.gui.game_board.game_state in [GameState.READY, GameState.PLAYING]):
                
                # Get current board state for AI
                board_state = self._get_board_state()
                action_mask = self._get_action_mask()
                
                # Check if there are any valid moves
                if not action_mask.any():
                    print("⚠️  No valid moves available!")
                    break
                
                # Get AI action
                action = self.ai_player.get_action(board_state, action_mask)
                row, col = self.ai_player.action_to_coordinates(action)
                
                print(f"🤖 AI Move {game_moves + 1}: ({row}, {col})")
                  # Execute move on GUI thread
                print(f"   Executing move at ({row}, {col})")
                self.gui.root.after(0, lambda r=row, c=col: self._make_move(r, c))
                
                # Wait for move delay
                time.sleep(self.delay)
                game_moves += 1
                
                # Check if game state changed after the move
                current_state = self.gui.game_board.game_state
                print(f"   Game state after move: {current_state}")
                
                # Safety check to prevent infinite loops
                if game_moves > 1000:
                    print("⚠️  Game exceeded maximum moves, stopping")
                    break
            
            # Update statistics when game ends
            final_state = self.gui.game_board.game_state
            print(f"🏁 Game ended - Final state: {final_state}, Moves: {game_moves}")
            
            if final_state != GameState.PLAYING:
                self.games_played += 1
                self.total_moves += game_moves
                
                if final_state == GameState.WON:
                    self.games_won += 1
                    self.gui.root.after(0, lambda: self._update_status("🎉 AI Won!"))
                else:
                    self.gui.root.after(0, lambda: self._update_status("💥 AI Lost"))
                self.gui.root.after(0, self._update_stats)
                
        except Exception as e:
            print(f"❌ Error during game: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_board_state(self) -> np.ndarray:
        """Convert GUI board state to AI format"""
        board = self.gui.game_board
        
        # Create 3-channel state representation
        # Channel 0: Revealed cells (numbers + mines)
        # Channel 1: Hidden/flagged cells  
        # Channel 2: Flags
        
        state = np.zeros((3, board.rows, board.cols), dtype=np.float32)
        
        for row in range(board.rows):
            for col in range(board.cols):
                cell = board.board[row][col]  # Fixed: use board.board instead of board.grid
                
                if cell.state == CellState.REVEALED:
                    if cell.is_mine:
                        state[0, row, col] = -1  # Mine
                    else:
                        state[0, row, col] = cell.adjacent_mines / 8.0  # Normalize numbers
                elif cell.state == CellState.FLAGGED:
                    state[1, row, col] = 1  # Hidden/flagged
                    state[2, row, col] = 1  # Flag marker
                else:  # HIDDEN
                    state[1, row, col] = 1  # Hidden
        return state
    
    def _get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions"""
        board = self.gui.game_board
        
        # Create action mask for entire action space (rows * cols * 3)
        # The 3 represents the action types: 0=reveal, 1=flag, 2=unflag
        action_mask = np.zeros(board.rows * board.cols * 3, dtype=bool)
        
        for row in range(board.rows):
            for col in range(board.cols):
                cell = board.board[row][col]  # Fixed: use board.board instead of board.grid
                
                # For demo, we only use reveal action (type 0)
                # Calculate action indices for all three actions
                base_idx = (row * board.cols + col) * 3
                reveal_idx = base_idx + 0  # Reveal action
                
                # Can only click on hidden cells
                if cell.state == CellState.HIDDEN:
                    # Only allow reveal action for demo simplicity
                    action_mask[reveal_idx] = True
        
        return action_mask
    
    def _make_move(self, row: int, col: int):
        """Make a move on the GUI (must run on main thread)"""
        try:
            if (0 <= row < self.gui.game_board.rows and 
                0 <= col < self.gui.game_board.cols and
                self.gui.game_board.board[row][col].state == CellState.HIDDEN):  # Fixed: use board.board
                
                print(f"   🖱️ Making move at ({row}, {col})")
                # Use the original cell click handler
                self.original_cell_click(row, col)
                print(f"   ✓ Move completed - Game state: {self.gui.game_board.game_state}")
                
            else:
                print(f"   ❌ Invalid move at ({row}, {col}) - Cell state: {self.gui.game_board.board[row][col].state if 0 <= row < self.gui.game_board.rows and 0 <= col < self.gui.game_board.cols else 'out of bounds'}")
                
        except Exception as e:
            print(f"❌ Error making move ({row}, {col}): {e}")
    
    def run(self):
        """Run the demo"""
        print(f"🚀 Starting Minesweeper AI Demo")
        print(f"   🎮 Difficulty: {self.difficulty}")
        print(f"   ⏱️  Move delay: {self.delay}s")
        print(f"   🔄 Auto restart: {self.auto_restart}")
        print(f"   📊 Model path: {self.model_path}")
        print("\n🎯 Use the control panel to start/stop the AI")
        
        try:
            self.gui.run()
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        finally:
            self.stop_demo()


def find_best_model(difficulty: str = "beginner") -> Optional[str]:
    """Find the best trained model for given difficulty"""
    try:
        # Try to get latest checkpoint
        result = get_latest_checkpoint(difficulty)
        if result:
            checkpoint_dir, checkpoint_file = result
            return os.path.join(checkpoint_dir, checkpoint_file)
        
        # Try to find latest directory and look for best model
        latest_dir = find_latest_model_dir(difficulty)
        if latest_dir:
            best_model_path = os.path.join(latest_dir, "best_model_checkpoint.pth")
            if os.path.exists(best_model_path):
                return best_model_path
            
            final_model_path = os.path.join(latest_dir, "final_model.pth")
            if os.path.exists(final_model_path):
                return final_model_path
        
        print(f"❌ No trained models found for {difficulty} difficulty")
        return None
        
    except Exception as e:
        print(f"❌ Error finding model: {e}")
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Minesweeper AI Demo - Watch trained AI play with visual GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_demo.py --difficulty beginner --delay 0.5
  python ai_demo.py --model models/beginner/20250613_070751/best_model_checkpoint.pth
  python ai_demo.py --difficulty intermediate --no-auto-restart
        """
    )
    
    parser.add_argument('--model', 
                       type=str,
                       help='Path to trained model file (auto-detect if not provided)')
    
    parser.add_argument('--difficulty',
                       choices=['beginner', 'intermediate', 'expert'],
                       default='beginner',
                       help='Game difficulty (default: beginner)')
    
    parser.add_argument('--delay',
                       type=float,
                       default=1.0,
                       help='Delay between AI moves in seconds (default: 1.0)')
    
    parser.add_argument('--no-auto-restart',
                       action='store_true',
                       help='Disable automatic game restart after completion')
    
    args = parser.parse_args()
    
    # Find model if not specified
    model_path = args.model
    if not model_path:
        print(f"🔍 Searching for best {args.difficulty} model...")
        model_path = find_best_model(args.difficulty)
        if not model_path:
            print(f"❌ No trained model found for {args.difficulty}")
            print(f"💡 Train a model first: python train_ai.py --difficulty {args.difficulty}")
            return 1
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return 1
    
    # Create and run demo
    try:
        demo = AIDemo(
            model_path=model_path,
            difficulty=args.difficulty,
            delay=args.delay,
            auto_restart=not args.no_auto_restart
        )
        demo.run()
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
