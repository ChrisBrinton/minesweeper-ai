"""
Minesweeper GUI - Windows 3.1 Style Interface
Implements the classic minesweeper user interface using tkinter
"""

import tkinter as tk
from tkinter import messagebox, Menu, PhotoImage, font
import time
import os
from typing import List, Callable, Optional, Dict

from game import GameBoard, GameState, CellState
from .leaderboard import LeaderboardManager, show_leaderboard, congratulate_new_record


_PROFILE_UI = os.environ.get('MINESWEEPER_PROFILE_UI') == '1'


class DigitalDisplay(tk.Frame):
    """Digital display widget for mine counter and timer using 7-segment display images"""
    
    # Class variable to cache loaded digit images
    _digit_images = {}
    
    @classmethod
    def load_digit_images(cls):
        """Load all digit images if not already loaded"""
        if not cls._digit_images:
            digits_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'digits')
            
            # Load digits 0-9
            for i in range(10):
                image_path = os.path.join(digits_path, f"{i}_digit.png")
                if os.path.exists(image_path):
                    cls._digit_images[str(i)] = PhotoImage(file=image_path)
            
            # Load minus sign and blank digit
            minus_path = os.path.join(digits_path, "minus_digit.png")
            blank_path = os.path.join(digits_path, "blank_digit.png")
            
            if os.path.exists(minus_path):
                cls._digit_images['-'] = PhotoImage(file=minus_path)
            
            if os.path.exists(blank_path):
                cls._digit_images[' '] = PhotoImage(file=blank_path)
    
    def __init__(self, parent, width=3):
        super().__init__(parent, bg='black', relief='sunken', bd=3, padx=3, pady=3)
        self.width = width
        self.value = 0
        
        # Make sure digit images are loaded
        self.load_digit_images()
        
        # Create a frame to hold all digits
        digit_frame = tk.Frame(self, bg='black')
        digit_frame.pack(fill='both', expand=True)
        
        # Create labels for each digit position
        self.digit_labels = []
        for i in range(width):
            label = tk.Label(digit_frame, bg='black', bd=0, padx=0)
            label.pack(side='left')
            self.digit_labels.append(label)
        
        # Set initial value
        self.set_value(0)
    
    def _format_number(self, num: int) -> str:
        """Format number with leading zeros for digital display"""
        if num < 0:
            if abs(num) >= 10**(self.width-1):  # If too large to fit with minus sign
                return '-' + '9' * (self.width-1)
            return f"-{abs(num):0{self.width-1}d}"[:self.width]
        
        if num >= 10**self.width:  # If too large to fit
            return '9' * self.width
            
        return f"{num:0{self.width}d}"[:self.width]
    
    def set_value(self, value: int):
        """Update the display value using digit images"""
        self.value = value
        
        # Get formatted string representation
        formatted = self._format_number(value)
        
        # Update each digit label with the appropriate image
        for i, digit in enumerate(formatted):
            if i < len(self.digit_labels):
                if digit in self._digit_images:
                    self.digit_labels[i].config(image=self._digit_images[digit])
                else:
                    # Fallback if image not found
                    if digit == '-':
                        self.digit_labels[i].config(image=self._digit_images.get(' ', None), text='-', fg='red')
                    else:
                        self.digit_labels[i].config(image=self._digit_images.get(' ', None), text=digit, fg='red')


class SmileyButton(tk.Button):
    """Smiley face button that shows game state"""
    
    # Class variable to store loaded images
    _images: Dict[str, PhotoImage] = {}
    
    @classmethod
    def load_images(cls):
        """Load image files for smiley faces if not already loaded"""
        if cls._images:  # Images already loaded
            return
            
        # Define the base assets directory
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        
        # Define image filenames to look for
        image_files = {
            'ready': 'smiley_normal.png',
            'playing': 'smiley_normal.png',
            'won': 'smiley_win.png',
            'lost': 'smiley_dead.png',
            'pressed': 'smiley_pressed.png'
        }
        
        # Try to load each image
        for key, filename in image_files.items():
            try:
                image_path = os.path.join(assets_dir, filename)
                if os.path.exists(image_path):
                    cls._images[key] = PhotoImage(file=image_path)
                    print(f"Successfully loaded smiley image: {filename}")
                else:
                    print(f"Smiley image file not found: {image_path}")
                    cls._images[key] = None  # Mark as unavailable
            except Exception as e:
                print(f"Error loading smiley image {filename}: {e}")
                cls._images[key] = None
    
    # Fallback emojis if images are not available
    FACES = {
        GameState.READY: '😊',
        GameState.PLAYING: '😊', 
        GameState.WON: '😎',
        GameState.LOST: '😵'
    }
    
    def __init__(self, parent, command=None):
        super().__init__(
            parent,
            relief='raised',
            bd=2,
            command=command
        )
        self.current_state = GameState.READY
        
        # Load images if not already loaded
        self.load_images()
        
        # Set initial face - use a specific image key to ensure it works on startup
        if self._images.get('ready'):
            self.config(image=self._images['ready'], width=26, height=26)
        else:
            self.config(text=self.FACES.get(GameState.READY, '😊'))
        
        # Bind mouse events
        self.bind('<Button-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)
    
    def _on_press(self, event):
        """Handle mouse button press"""
        if self.current_state == GameState.PLAYING:
            self.set_pressed_face()
        
    def _on_release(self, event):
        """Handle mouse button release"""
        self.restore_face()
        # The actual command will be executed by the button's built-in handler
    
    def set_state(self, state: GameState):
        """Update smiley face based on game state"""
        if state != self.current_state:
            self.current_state = state
            
            # Use image if available, otherwise fallback to emoji
            state_key = state.value  # 'ready', 'playing', 'won', 'lost'
            if self._images.get(state_key):
                self.config(image=self._images[state_key], text='')
            else:
                self.config(text=self.FACES.get(state, '😊'), image='')
    
    def set_pressed_face(self):
        """Show worried face while mouse is pressed"""
        if self.current_state == GameState.PLAYING:
            if self._images.get('pressed'):
                self.config(image=self._images['pressed'], text='')
            else:
                self.config(text='😬', image='')
    
    def restore_face(self):
        """Restore normal face for current state"""
        state_key = self.current_state.value
        if self._images.get(state_key):
            self.config(image=self._images[state_key], text='')
        else:
            self.config(text=self.FACES.get(self.current_state, '😊'), image='')


class BoardCanvas(tk.Canvas):
    """Single-canvas renderer for the entire minesweeper grid.

    All cells are drawn as image items on one Canvas widget — no per-cell
    OS windows. Updates use dirty tracking so only cells whose visual state
    changed are repainted.
    """

    CELL_SIZE = 16  # PNG cell sprites are 16x16

    _images: Dict[str, PhotoImage] = {}

    @classmethod
    def load_images(cls):
        """Load cell sprites once. Must be called after a Tk root exists."""
        if cls._images:
            return

        assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'assets',
        )

        image_files = {
            'hidden': 'hidden_cell.png',
            'empty': 'empty_cell.png',
            'mine': 'mine_cell.png',
            'flag': 'flag_cell.png',
            'mine_red': 'mine_red_cell.png',
            'bad_flag': 'bad_flag_cell.png',
        }
        for i in range(1, 9):
            image_files[f'num_{i}'] = f'{i}_cell.png'

        loaded = 0
        for key, filename in image_files.items():
            path = os.path.join(assets_dir, filename)
            if os.path.exists(path):
                try:
                    cls._images[key] = PhotoImage(file=path)
                    loaded += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    cls._images[key] = None
            else:
                print(f"Image file not found: {path}")
                cls._images[key] = None

        print(f"Loaded {loaded}/{len(image_files)} cell images from {assets_dir}")

    def __init__(self, parent, rows: int, cols: int,
                 click_callback: Callable, right_click_callback: Callable):
        super().__init__(
            parent,
            width=cols * self.CELL_SIZE,
            height=rows * self.CELL_SIZE,
            highlightthickness=0,
            bd=0,
            bg='#c0c0c0',
        )
        self.rows = rows
        self.cols = cols
        self.click_callback = click_callback
        self.right_click_callback = right_click_callback

        self.load_images()

        # One image item per cell + last-drawn key for dirty tracking
        hidden_img = self._images.get('hidden')
        self._items: List[List[int]] = [[0] * cols for _ in range(rows)]
        self._keys: List[List[Optional[str]]] = [[None] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                x = c * self.CELL_SIZE
                y = r * self.CELL_SIZE
                self._items[r][c] = self.create_image(
                    x, y, image=hidden_img, anchor='nw'
                )
                self._keys[r][c] = 'hidden'

        self.bind('<Button-1>', self._on_left_click)
        self.bind('<Button-3>', self._on_right_click)

    def _xy_to_rc(self, x: int, y: int):
        c = x // self.CELL_SIZE
        r = y // self.CELL_SIZE
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None, None

    def _on_left_click(self, event):
        r, c = self._xy_to_rc(event.x, event.y)
        if r is not None:
            self.click_callback(r, c)

    def _on_right_click(self, event):
        r, c = self._xy_to_rc(event.x, event.y)
        if r is not None:
            self.right_click_callback(r, c)

    @staticmethod
    def _key_for_cell(cell, game_board) -> str:
        """Compute a sprite-key for a cell. Same key = no redraw needed."""
        if cell.state == CellState.REVEALED:
            if cell.is_mine:
                if (game_board.clicked_mine_pos and
                        game_board.clicked_mine_pos == (cell.row, cell.col)):
                    return 'mine_red'
                return 'mine'
            if cell.adjacent_mines > 0:
                return f'num_{cell.adjacent_mines}'
            return 'empty'
        if cell.state == CellState.FLAGGED:
            if (game_board.game_state == GameState.LOST and not cell.is_mine):
                return 'bad_flag'
            return 'flag'
        return 'hidden'

    def redraw(self, game_board) -> int:
        """Repaint cells whose visual state changed. Returns # cells repainted."""
        dirty = 0
        items = self._items
        keys = self._keys
        images = self._images
        for r in range(self.rows):
            row_cells = game_board.board[r]
            row_keys = keys[r]
            row_items = items[r]
            for c in range(self.cols):
                key = self._key_for_cell(row_cells[c], game_board)
                if key != row_keys[c]:
                    img = images.get(key)
                    if img is not None:
                        self.itemconfigure(row_items[c], image=img)
                    row_keys[c] = key
                    dirty += 1
        return dirty


class MinesweeperGUI:
    """Main GUI class for the minesweeper game"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Minesweeper')
        self.root.resizable(False, False)

        # Leaderboard system
        self.leaderboard_manager = LeaderboardManager()
          # Game components
        self.game_board: Optional[GameBoard] = None
        self.board_canvas: Optional[BoardCanvas] = None
        self.start_time: Optional[float] = None
        self.game_timer_id: Optional[str] = None
        self.current_difficulty: str = 'beginner'
        self.current_elapsed_time: int = 0  # Track current elapsed time for consistent leaderboard recording

        # GUI components
        self.mine_display: Optional[DigitalDisplay] = None
        self.timer_display: Optional[DigitalDisplay] = None
        self.smiley_button: Optional[SmileyButton] = None
        self.game_frame: Optional[tk.Frame] = None
        self._setup_gui()
        
        # Start with the last played difficulty
        last_difficulty = self.leaderboard_manager.get_last_difficulty()
        self._new_game(last_difficulty)
    
    def _setup_gui(self):
        """Setup the main GUI components"""
        # Main container
        main_frame = tk.Frame(self.root, bg='lightgray', relief='raised', bd=3)
        main_frame.pack(padx=5, pady=5)
        
        # OPTIMIZATION: Load all images once during startup
        DigitalDisplay.load_digit_images()
        SmileyButton.load_images()
        BoardCanvas.load_images()
        
        # Top panel with displays and smiley
        top_frame = tk.Frame(main_frame, bg='lightgray')
        top_frame.pack(fill='x', padx=5, pady=5)
        
        # Mine counter
        self.mine_display = DigitalDisplay(top_frame)
        self.mine_display.pack(side='left')
        
        # Smiley button (centered)
        smiley_frame = tk.Frame(top_frame, bg='lightgray')
        smiley_frame.pack(side='left', expand=True)
        
        self.smiley_button = SmileyButton(smiley_frame, command=self._restart_game)
        self.smiley_button.pack()
        
        # Timer
        self.timer_display = DigitalDisplay(top_frame)
        self.timer_display.pack(side='right')
        
        # Game board frame
        self.game_frame = tk.Frame(main_frame, bg='lightgray')
        self.game_frame.pack(padx=5, pady=5)
        
        # Menu bar
        self._setup_menu()
    
    def _setup_menu(self):
        """Setup the menu bar"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
          # Game menu
        game_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self._restart_game)
        game_menu.add_separator()
        game_menu.add_command(label="Beginner", command=lambda: self._new_game('beginner'))
        game_menu.add_command(label="Intermediate", command=lambda: self._new_game('intermediate'))
        game_menu.add_command(label="Expert", command=lambda: self._new_game('expert'))
        game_menu.add_separator()
        game_menu.add_command(label="Best Times...", command=self._show_leaderboard)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="How to Play", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _new_game(self, difficulty: str):
        """Start a new game with specified difficulty"""
        # Stop current timer
        if self.game_timer_id:
            self.root.after_cancel(self.game_timer_id)
            self.game_timer_id = None
        
        # Save current difficulty
        self.current_difficulty = difficulty
        self.leaderboard_manager.set_last_difficulty(difficulty)
        
        # Create new game board
        rows, cols, mines = GameBoard.DIFFICULTIES[difficulty]
        self.game_board = GameBoard(rows, cols, mines)
        
        # Reset displays
        self.mine_display.set_value(mines)
        self.timer_display.set_value(0)
          # Update smiley face state
        self.smiley_button.set_state(GameState.READY)
        self.start_time = None
        self.current_elapsed_time = 0  # Reset elapsed time tracking
        
        # Reuse the existing canvas if board size hasn't changed
        target_size = (rows, cols)
        current_size = (
            (self.board_canvas.rows, self.board_canvas.cols)
            if self.board_canvas else (0, 0)
        )
        if current_size != target_size:
            self._recreate_board_canvas(rows, cols)
        else:
            # Same size: just repaint to hidden via dirty tracking
            self._update_display()

        # Single layout update at the end instead of multiple updates
        self.root.update_idletasks()
    
    def _restart_game(self):
        """Restart the current game"""
        if self.game_board:
            difficulty = self._get_current_difficulty()
            self._new_game(difficulty)
    def _get_current_difficulty(self) -> str:
        """Get the current difficulty level"""
        return self.current_difficulty
    
    def _on_cell_click(self, row: int, col: int):
        """Handle left click on a cell"""
        if not self.game_board:
            return
        
        # Check if this is the first click
        is_first_click = self.game_board.game_state == GameState.READY
        
        # Reveal cell
        self.game_board.reveal_cell(row, col)
        
        # Start timer on first click (after game state changes to PLAYING)
        if is_first_click and self.game_board.game_state == GameState.PLAYING:
            self.start_time = time.time()
            self._update_timer()
        
        # Update smiley face
        self.smiley_button.set_state(self.game_board.game_state)
        
        # Update displays
        self._update_display()
        
        # Handle game end
        if self.game_board.game_state in [GameState.WON, GameState.LOST]:
            self._end_game()
    
    def _on_cell_right_click(self, row: int, col: int):
        """Handle right click on a cell"""
        if not self.game_board:
            return
        
        # Can't flag if game hasn't started or is over
        if self.game_board.game_state in [GameState.READY, GameState.WON, GameState.LOST]:
            return
        
        # Toggle flag
        self.game_board.toggle_flag(row, col)
        
        # Update displays
        self._update_display()
    
    def _update_display(self):
        """Update all visual elements"""
        if not self.game_board or not self.board_canvas:
            return

        # Update mine counter
        remaining_mines = self.game_board.get_remaining_mines()
        self.mine_display.set_value(remaining_mines)

        # Update smiley face
        self.smiley_button.set_state(self.game_board.game_state)

        # Repaint only cells whose visual state changed
        if _PROFILE_UI:
            t0 = time.perf_counter()
            dirty = self.board_canvas.redraw(self.game_board)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[ui] redraw {dirty:>3} cell(s) in {elapsed_ms:.2f} ms")
        else:
            self.board_canvas.redraw(self.game_board)
    
    def _update_timer(self):
        """Update the game timer"""
        if (self.start_time and 
            self.game_board and 
            self.game_board.game_state == GameState.PLAYING):
            
            elapsed = int(time.time() - self.start_time)
            self.current_elapsed_time = elapsed  # Store the current elapsed time
            self.timer_display.set_value(min(elapsed, 999))  # Cap at 999
            
            # Schedule next update
            self.game_timer_id = self.root.after(1000, self._update_timer)
    
    def _end_game(self):
        """Handle game end"""
        # Stop timer
        if self.game_timer_id:
            self.root.after_cancel(self.game_timer_id)
            self.game_timer_id = None
        
        # Check for new leaderboard entry if player won
        if (self.game_board and 
            self.game_board.game_state == GameState.WON and 
            self.start_time):
            
            # Use the current elapsed time that was displayed on the timer
            # instead of recalculating to avoid timing discrepancies
            elapsed_time = self.current_elapsed_time
            
            # Check if it's a top 10 time
            if self.leaderboard_manager.is_top_10_time(self.current_difficulty, elapsed_time):
                # Add to leaderboard
                made_top_10 = self.leaderboard_manager.add_score(
                    self.current_difficulty, 
                    elapsed_time
                )
                
                if made_top_10:
                    # Find the rank of this score
                    leaderboard = self.leaderboard_manager.get_leaderboard(self.current_difficulty)
                    rank = next((i + 1 for i, entry in enumerate(leaderboard) 
                               if entry.time_seconds == elapsed_time), 1)
                    
                    # Show congratulations after a short delay
                    self.root.after(1000, lambda: congratulate_new_record(
                        self.root, self.leaderboard_manager, 
                        self.current_difficulty, elapsed_time, rank
                    ))
                    
                    # Show leaderboard after congratulations
                    self.root.after(3000, lambda: show_leaderboard(
                        self.root, self.leaderboard_manager, self.current_difficulty
                    ))
    
    def _show_help(self):
        """Show help dialog"""
        help_text = """How to Play Minesweeper:

🎯 Objective: Find all mines without detonating any

🖱️ Controls:
• Left click: Reveal a cell
• Right click: Flag/unflag a cell

📊 Numbers show how many mines are adjacent to that cell

💡 Tips:
• Use numbers to deduce mine locations
• Flag suspected mines
• Empty cells auto-reveal adjacent cells
• First click is always safe

🏆 Win by revealing all non-mine cells!"""
        
        messagebox.showinfo("How to Play", help_text)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """Minesweeper
Classic Windows 3.1 Style

Created with Python and tkinter
Faithful recreation of the original game

© 2025"""
        
        messagebox.showinfo("About Minesweeper", about_text)
    
    def _show_leaderboard(self):
        """Show leaderboard dialog"""
        show_leaderboard(self.root, self.leaderboard_manager, self.current_difficulty)
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
    
    def _recreate_board_canvas(self, rows: int, cols: int):
        """Create a fresh BoardCanvas (only when board size changes)."""
        for widget in self.game_frame.winfo_children():
            widget.destroy()

        self.board_canvas = BoardCanvas(
            self.game_frame, rows, cols,
            self._on_cell_click, self._on_cell_right_click,
        )
        self.board_canvas.pack()

        # Initial paint
        self._update_display()
