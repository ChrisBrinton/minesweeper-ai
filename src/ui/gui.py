"""
Minesweeper GUI - Windows 3.1 Style Interface
Implements the classic minesweeper user interface using tkinter
"""

import tkinter as tk
from tkinter import messagebox, Menu, PhotoImage, font
import sys
import threading
import time
import os
from typing import List, Callable, Optional, Dict

from PIL import Image, ImageTk

from game import GameBoard, GameState, CellState
from .leaderboard import LeaderboardManager, show_leaderboard, congratulate_new_record
from .history import GameHistoryManager, GameRecord, show_stats
from .settings import SettingsManager, show_settings


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

        self._heatmap_ids: List[int] = []

        # Highlight overlay for the AI Suggest button — drawn on top of cell
        # sprites, hidden by default. tag_raise keeps it above all images.
        self._highlight_id = self.create_rectangle(
            0, 0, self.CELL_SIZE, self.CELL_SIZE,
            outline='#ffd700', width=2, state='hidden',
        )

    def highlight_cell(self, row: int, col: int):
        """Draw the AI suggestion outline at (row, col)."""
        x0 = col * self.CELL_SIZE + 1
        y0 = row * self.CELL_SIZE + 1
        x1 = (col + 1) * self.CELL_SIZE - 1
        y1 = (row + 1) * self.CELL_SIZE - 1
        self.coords(self._highlight_id, x0, y0, x1, y1)
        self.itemconfigure(self._highlight_id, state='normal')
        self.tag_raise(self._highlight_id)

    def clear_highlight(self):
        self.itemconfigure(self._highlight_id, state='hidden')

    # ── Heatmap overlay ──────────────────────────────────────────────────

    _HEATMAP_STEPS = 20

    @classmethod
    def _build_heatmap_palette(cls):
        """Pre-render overlay images: red gradient, green for safe, lavender for best guess."""
        if hasattr(cls, '_heatmap_images'):
            return
        cls._heatmap_images: List[ImageTk.PhotoImage] = []
        sz = cls.CELL_SIZE
        for i in range(cls._HEATMAP_STEPS + 1):
            alpha = int(180 * i / cls._HEATMAP_STEPS)
            img = Image.new('RGBA', (sz, sz), (255, 40, 40, alpha))
            cls._heatmap_images.append(ImageTk.PhotoImage(img))
        cls._heatmap_green = ImageTk.PhotoImage(
            Image.new('RGBA', (sz, sz), (40, 200, 40, 160)))
        cls._heatmap_lavender = ImageTk.PhotoImage(
            Image.new('RGBA', (sz, sz), (180, 130, 255, 160)))
        cls._heatmap_warning = ImageTk.PhotoImage(
            Image.new('RGBA', (sz, sz), (255, 200, 0, 140)))

    def show_heatmap(self, probabilities):
        """Draw colored overlays on hidden cells based on P(mine).

        Green: solver-determined safe cells (click with certainty).
        Lavender: model's best guess(es) among uncertain cells.
        Red gradient: mine risk for everything else.
        """
        self._build_heatmap_palette()
        self.clear_heatmap()
        import numpy as np
        steps = self._HEATMAP_STEPS
        palette = self._heatmap_images
        green = self._heatmap_green
        lavender = self._heatmap_lavender
        ids = self._heatmap_ids

        # Best-guess threshold: lowest P(mine) among uncertain cells
        # (not solver-determined 0.0 or 1.0).
        best_p = None
        for r in range(self.rows):
            for c in range(self.cols):
                p = probabilities[r, c]
                if not np.isnan(p) and 0.0 < p < 1.0:
                    if best_p is None or p < best_p:
                        best_p = p

        for r in range(self.rows):
            for c in range(self.cols):
                p = probabilities[r, c]
                if np.isnan(p):
                    continue
                x = c * self.CELL_SIZE
                y = r * self.CELL_SIZE
                if p == 0.0:
                    item = self.create_image(x, y, image=green, anchor='nw')
                elif best_p is not None and 0.0 < p < 1.0 and abs(p - best_p) < 1e-6:
                    item = self.create_image(x, y, image=lavender, anchor='nw')
                else:
                    idx = max(0, min(steps, int(round(p * steps))))
                    if idx == 0:
                        continue
                    item = self.create_image(x, y, image=palette[idx], anchor='nw')
                ids.append(item)
        self.tag_raise(self._highlight_id)

    def show_comparison_heatmap(self, probabilities, divergence):
        """Draw heatmap with divergence warnings.

        Uses the standard red/green/lavender overlay for probabilities,
        but highlights cells where model and constraint engine disagree
        by >0.15 with a yellow warning overlay.
        """
        self.show_heatmap(probabilities)
        import numpy as np
        self._build_heatmap_palette()
        warning = self._heatmap_warning
        ids = self._heatmap_ids
        for r in range(self.rows):
            for c in range(self.cols):
                d = divergence[r, c]
                if not np.isnan(d) and d > 0.15:
                    x = c * self.CELL_SIZE
                    y = r * self.CELL_SIZE
                    item = self.create_image(x, y, image=warning, anchor='nw')
                    ids.append(item)
        self.tag_raise(self._highlight_id)

    def clear_heatmap(self):
        for item in self._heatmap_ids:
            self.delete(item)
        self._heatmap_ids.clear()

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
        self.history_manager = GameHistoryManager()
        self.settings_manager = SettingsManager()
        # Serializes the off-thread export workers so concurrent finalizes
        # don't race on the .npz file
        self._export_lock = threading.Lock()
          # Game components
        self.game_board: Optional[GameBoard] = None
        self.board_canvas: Optional[BoardCanvas] = None
        self.start_time: Optional[float] = None
        self.game_timer_id: Optional[str] = None
        self.current_difficulty: str = 'beginner'
        self.current_elapsed_time: int = 0  # Track current elapsed time for consistent leaderboard recording
        self.current_record: Optional[GameRecord] = None

        # Persist any in-progress record on window close
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # GUI components
        self.mine_display: Optional[DigitalDisplay] = None
        self.timer_display: Optional[DigitalDisplay] = None
        self.smiley_button: Optional[SmileyButton] = None
        self.game_frame: Optional[tk.Frame] = None
        self.live_cpm_label: Optional[tk.Label] = None
        self.live_fpm_label: Optional[tk.Label] = None
        self.live_progress_label: Optional[tk.Label] = None
        self.suggest_button: Optional[tk.Button] = None
        self.autoplay_button: Optional[tk.Button] = None
        self.heatmap_button: Optional[tk.Checkbutton] = None
        # Lazy MinesweeperInference; constructed on first Suggest click
        self._inference = None
        # Auto-play loop state — Tk after-id of the next pending step (or None)
        self._autoplay_after_id: Optional[str] = None
        # Per-step timing: highlight visible, then move executes
        self._autoplay_show_ms = 100
        self._autoplay_pause_ms = 150
        self._setup_gui()
        
        # Start with the last played difficulty
        last_difficulty = self.leaderboard_manager.get_last_difficulty()
        self._new_game(last_difficulty)

        # Preload the model in the background so the first Suggest /
        # Auto-play click doesn't pay the ~500ms CUDA warmup cost.
        self._maybe_preload_model()

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

        # Live stats + AI buttons.  Two-row layout so everything fits
        # within the board width:
        #   Row 1: Cells/min  |  Progress   |  [Auto-play  ]
        #   Row 2: Flags/min  |  Safe: N    |  [Suggest move]
        #                                      [Heatmap     ]
        live_frame = tk.Frame(main_frame, bg='lightgray')
        live_frame.pack(fill='x', padx=5, pady=(0, 5))

        stats_frame = tk.Frame(live_frame, bg='lightgray')
        stats_frame.pack(side='left', fill='both', expand=True)

        stats_row1 = tk.Frame(stats_frame, bg='lightgray')
        stats_row1.pack(fill='x')
        stats_row2 = tk.Frame(stats_frame, bg='lightgray')
        stats_row2.pack(fill='x')

        self.live_cpm_label = tk.Label(
            stats_row1, text='Cells/min: —', width=14, anchor='w',
            font=('Arial', 9), bg='lightgray', fg='black',
        )
        self.live_cpm_label.pack(side='left', padx=(8, 0))
        self.live_progress_label = tk.Label(
            stats_row1, text='Progress: —', width=13, anchor='w',
            font=('Arial', 9), bg='lightgray', fg='black',
        )
        self.live_progress_label.pack(side='left', padx=8)

        self.live_fpm_label = tk.Label(
            stats_row2, text='Flags/min: —', width=14, anchor='w',
            font=('Arial', 9), bg='lightgray', fg='black',
        )
        self.live_fpm_label.pack(side='left', padx=(8, 0))
        self.safe_count_label = tk.Label(
            stats_row2, text='Safe: —', width=9, anchor='w',
            font=('Arial', 9), bg='lightgray', fg='#0a7d0a',
        )
        if self.settings_manager.safe_count_enabled:
            self.safe_count_label.pack(side='left', padx=8)

        # AI action buttons: vertical stack on the right
        self._ai_button_stack = tk.Frame(live_frame, bg='lightgray')
        self.autoplay_var = tk.IntVar(value=0)
        self.autoplay_button = tk.Checkbutton(
            self._ai_button_stack, text='Auto-play',
            font=('Arial', 9), state='disabled',
            variable=self.autoplay_var,
            indicatoron=False,
            selectcolor='#fff3a8',
            width=14,
            command=self._on_autoplay_toggle,
        )
        self.autoplay_button.pack(fill='x')
        self.suggest_button = tk.Button(
            self._ai_button_stack, text='Suggest move',
            font=('Arial', 9), state='disabled',
            command=self._on_suggest_click,
        )
        self.suggest_button.pack(fill='x')
        self.confidence_label = tk.Label(
            self._ai_button_stack, text='',
            font=('Arial', 8), bg='lightgray', fg='#666',
            anchor='center',
        )
        self.confidence_label.pack(fill='x')
        self.heatmap_var = tk.IntVar(value=0)
        self.heatmap_button = tk.Checkbutton(
            self._ai_button_stack, text='Heatmap',
            font=('Arial', 9), state='disabled',
            variable=self.heatmap_var,
            indicatoron=False,
            selectcolor='#ffcccc',
            width=14,
            command=self._on_heatmap_toggle,
        )
        self.heatmap_button.pack(fill='x')

        if self.settings_manager.ai_enabled:
            self._show_ai_buttons()

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
        game_menu.add_command(label="Statistics...", command=self._show_stats)
        game_menu.add_command(label="Settings...", command=self._show_settings)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self._on_close)
        
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
        # Stop any in-flight auto-play loop and flip the toggle off
        self._stop_autoplay()

        # Finalize any in-progress record as abandoned
        self._abandon_current_record()

        # Save current difficulty
        self.current_difficulty = difficulty
        self.leaderboard_manager.set_last_difficulty(difficulty)

        # Create new game board
        rows, cols, mines = GameBoard.DIFFICULTIES[difficulty]
        self.game_board = GameBoard(rows, cols, mines)
        self.current_record = GameRecord(difficulty, rows, cols, mines)
        
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
            # And clear any stale AI overlays
            if self.board_canvas is not None:
                self.board_canvas.clear_highlight()
                self.board_canvas.clear_heatmap()
        self.heatmap_var.set(0)
        self.confidence_label.config(text='')

        self._reset_live_stats()
        self._update_action_buttons()

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

        # Ignore clicks on cells that won't change anything (already revealed/flagged
        # before game start, or any click after game over) so we don't pollute the
        # move log with no-ops.
        cell = self.game_board.get_cell(row, col)
        if (self.game_board.game_state in (GameState.WON, GameState.LOST)
                or cell is None
                or cell.state != CellState.HIDDEN):
            return

        # Any click clears a stale AI suggestion highlight
        if self.board_canvas is not None:
            self.board_canvas.clear_highlight()

        # Check if this is the first click
        is_first_click = self.game_board.game_state == GameState.READY

        if self.current_record is not None:
            self.current_record.append_move(row, col, 'reveal')

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
        self._update_live_stats()
        self._update_action_buttons()
        self._refresh_heatmap()

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

        cell = self.game_board.get_cell(row, col)
        if cell is None or cell.state == CellState.REVEALED:
            return

        # Any click clears a stale AI suggestion highlight
        if self.board_canvas is not None:
            self.board_canvas.clear_highlight()

        # Determine the action that will be taken (HIDDEN→flag, FLAGGED→unflag)
        action = 'flag' if cell.state == CellState.HIDDEN else 'unflag'
        if self.current_record is not None:
            self.current_record.append_move(row, col, action)

        # Toggle flag
        self.game_board.toggle_flag(row, col)

        # Update displays
        self._update_display()
        self._update_live_stats()
        self._refresh_heatmap()
    
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
            self._update_live_stats()

            # Schedule next update
            self.game_timer_id = self.root.after(1000, self._update_timer)

    def _reset_live_stats(self):
        """Reset live-stats labels to placeholder for a fresh game."""
        if self.live_cpm_label is not None:
            self.live_cpm_label.config(
                text='Cells/min: —', fg='black', font=('Arial', 9))
        if self.live_fpm_label is not None:
            self.live_fpm_label.config(
                text='Flags/min: —', fg='black', font=('Arial', 9))
        if self.live_progress_label is not None:
            self.live_progress_label.config(
                text='Progress: —', fg='black', font=('Arial', 9))
        if self.safe_count_label is not None:
            self.safe_count_label.config(text='Safe: —')

    def _update_live_stats(self):
        """Refresh the live cells/min, flags/min, and progress% labels.

        Cells/min and flags/min are highlighted (bold + dark green) when the
        current rate exceeds the historical best for this difficulty. If
        AI assistance was used in this game, the rate values are blanked
        out (and progress is still shown — it's board state, not skill).
        """
        if self.live_cpm_label is None or not self.game_board:
            return

        # Progress % is meaningful any time after first reveal
        total_safe = self.game_board.rows * self.game_board.cols - self.game_board.total_mines
        if total_safe > 0:
            cells_for_progress = self.game_board.cells_revealed
            if self.game_board.game_state == GameState.LOST and cells_for_progress > 0:
                cells_for_progress -= 1
            pct = 100.0 * cells_for_progress / total_safe
            self.live_progress_label.config(text=f'Progress: {pct:.0f}%')

        self._update_safe_count()

        ai_used = bool(self.current_record and self.current_record.ai_used)
        if ai_used:
            # Rates are meaningless once AI is helping — show neutral values
            self.live_cpm_label.config(
                text='Cells/min: — (AI)', fg='#888', font=('Arial', 9))
            self.live_fpm_label.config(
                text='Flags/min: — (AI)', fg='#888', font=('Arial', 9))
            return

        # Rates only make sense once the timer has started
        elapsed = self.current_elapsed_time
        if elapsed <= 0 or self.game_board.game_state != GameState.PLAYING:
            return
        minutes = elapsed / 60.0

        cells_revealed = self.game_board.cells_revealed
        cpm = cells_revealed / minutes
        fpm = self._count_correct_player_flags() / minutes

        best = self.history_manager.best_rates_for(self.current_difficulty)
        cpm_record = best['cells_per_minute'] > 0 and cpm > best['cells_per_minute']
        fpm_record = best['flags_per_minute'] > 0 and fpm > best['flags_per_minute']

        record_color = '#0a7d0a'  # dark green
        self.live_cpm_label.config(
            text=f'Cells/min: {cpm:.0f}',
            fg=record_color if cpm_record else 'black',
            font=('Arial', 9, 'bold') if cpm_record else ('Arial', 9),
        )
        self.live_fpm_label.config(
            text=f'Flags/min: {fpm:.1f}',
            fg=record_color if fpm_record else 'black',
            font=('Arial', 9, 'bold') if fpm_record else ('Arial', 9),
        )
    
    def _update_safe_count(self):
        """Update the solver safe-cell count label (if enabled)."""
        if self.safe_count_label is None or not self.settings_manager.safe_count_enabled:
            return
        if not self.game_board or self.game_board.game_state != GameState.PLAYING:
            self.safe_count_label.config(text='Safe: —')
            return
        from src.ai.inference import _state_from_board
        from src.ai.algorithmic_solver import AlgorithmicSolver
        state = _state_from_board(self.game_board)
        solver = AlgorithmicSolver(
            self.game_board.rows, self.game_board.cols, self.game_board.total_mines)
        hidden, flagged, revealed = solver._parse_state(state)
        safe, _ = solver._find_deterministic_moves(hidden, flagged, revealed)
        self.safe_count_label.config(text=f'Safe: {len(safe)}')

    def _end_game(self):
        """Handle game end"""
        # Stop timer
        if self.game_timer_id:
            self.root.after_cancel(self.game_timer_id)
            self.game_timer_id = None

        # Capture the AI-tainted flag before _finalize_record nulls out
        # current_record — we need it for the leaderboard check below
        ai_used = bool(self.current_record and self.current_record.ai_used)

        # Finalize and persist the game record
        self._finalize_record(
            'won' if self.game_board.game_state == GameState.WON else 'lost'
        )

        # Check for new leaderboard entry if player won (and didn't use AI)
        if (self.game_board and
            self.game_board.game_state == GameState.WON and
            self.start_time and
            not ai_used):
            
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
    
    def _collect_mine_positions(self) -> List[tuple]:
        """Return list of (row, col) for all mines on the current board.
        Empty list if mines haven't been placed yet (game never started)."""
        if not self.game_board or not self.game_board.mines_placed:
            return []
        positions = []
        for r in range(self.game_board.rows):
            for c in range(self.game_board.cols):
                if self.game_board.board[r][c].is_mine:
                    positions.append((r, c))
        return positions

    def _count_correct_player_flags(self) -> int:
        """Count cells the player has flagged (per move log) that are mines.

        Uses the in-progress record's move log so we don't count the win-time
        auto-flagging that GameBoard does when the player wins.
        """
        if self.current_record is None or not self.game_board:
            return 0
        flagged = set()
        for m in self.current_record.moves:
            if m['a'] == 'flag':
                flagged.add((m['r'], m['c']))
            elif m['a'] == 'unflag':
                flagged.discard((m['r'], m['c']))
        if not flagged:
            return 0
        return sum(
            1 for (r, c) in flagged
            if 0 <= r < self.game_board.rows and 0 <= c < self.game_board.cols
            and self.game_board.board[r][c].is_mine
        )

    def _finalize_record(self, result: str):
        """Finalize the in-progress record with a terminal result and persist it."""
        if self.current_record is None:
            return
        cells_revealed = self.game_board.cells_revealed if self.game_board else 0
        if result == 'lost' and cells_revealed > 0:
            cells_revealed -= 1  # don't count the clicked mine
        self.current_record.finalize(
            result=result,
            elapsed_seconds=self.current_elapsed_time,
            mine_positions=self._collect_mine_positions(),
            cells_revealed=cells_revealed,
            correct_flags=self._count_correct_player_flags(),
        )
        # Flush pending UI repaints (win/lose smiley, last-cell reveal,
        # mine layout reveal on loss, win-time auto-flagging) so they hit the
        # screen BEFORE any disk I/O — otherwise the user perceives a freeze
        # between their click and the result render.
        if result in ('won', 'lost'):
            self.root.update_idletasks()
        self.history_manager.append(self.current_record)
        # Auto-export this game's samples if the user opted in
        if result in ('won', 'lost'):
            self._auto_export_record(self.current_record)
        self.current_record = None

    def _abandon_current_record(self):
        """If a game is in progress (any moves made), record it as abandoned."""
        if self.current_record is None:
            return
        if not self.current_record.moves:
            # Never started — don't pollute history
            self.current_record = None
            return
        self._finalize_record('abandoned')

    def _show_stats(self):
        """Show the statistics dialog."""
        show_stats(self.root, self.history_manager, self.current_difficulty)

    def _show_settings(self):
        """Show the Settings dialog."""
        def _on_settings_changed():
            # Drop any cached inference so a changed model_path takes effect
            # on the next click / preload.
            self._inference = None
            if self.settings_manager.ai_enabled:
                self._show_ai_buttons()
                self._maybe_preload_model()
            else:
                # Tear everything down — no buttons, no model in memory
                self._stop_autoplay()
                self._hide_ai_buttons()
            if self.settings_manager.safe_count_enabled:
                self.safe_count_label.pack(side='left', padx=8)
                self._update_safe_count()
            else:
                self.safe_count_label.pack_forget()
        show_settings(self.root, self.settings_manager,
                      on_changed=_on_settings_changed)

    def _show_ai_buttons(self):
        """Pack the AI button stack on the right side of the live-stats area."""
        self._ai_button_stack.pack(side='right', padx=8)
        self._update_action_buttons()

    def _hide_ai_buttons(self):
        """Remove the AI button stack from the live-stats area."""
        self._ai_button_stack.pack_forget()

    def _maybe_preload_model(self):
        """If AI is enabled, kick off model load on a daemon thread so the
        first Suggest/Auto-play click is fast. No-op if already loaded or
        disabled. Errors are logged; user gets a nice message when they
        actually click."""
        if not self.settings_manager.ai_enabled:
            return
        inf = self._ensure_inference()
        if inf is None or inf.is_loaded():
            return

        def _load():
            try:
                inf.load()
                print('[ai] model preloaded')
            except FileNotFoundError as e:
                print(f'[ai] preload skipped: {e}')
            except Exception as e:
                print(f'[ai] preload failed: {e}')

        threading.Thread(target=_load, daemon=True,
                         name='minesweeper-ai-preload').start()

    def _update_action_buttons(self):
        """Enable Suggest and Auto-play only while a game is in progress.

        If autoplay was running and the game just ended, also flip the
        toggle off so the button doesn't sit visually 'on' while disabled.
        """
        if not self.game_board:
            return
        state = ('normal' if self.game_board.game_state == GameState.PLAYING
                 else 'disabled')
        if self.suggest_button is not None:
            self.suggest_button.config(state=state)
        if self.autoplay_button is not None:
            self.autoplay_button.config(state=state)
            if state == 'disabled' and self.autoplay_var.get():
                self.autoplay_var.set(0)
                self._cancel_autoplay()
        if self.heatmap_button is not None:
            self.heatmap_button.config(state=state)
            if state == 'disabled' and self.heatmap_var.get():
                self.heatmap_var.set(0)
                if self.board_canvas is not None:
                    self.board_canvas.clear_heatmap()

    def _mark_ai_used(self):
        """Flag the current game as AI-assisted. Idempotent."""
        if self.current_record is not None and not self.current_record.ai_used:
            self.current_record.ai_used = True
            # Refresh live stats so the rate values blank out immediately
            self._update_live_stats()

    def _ensure_inference(self):
        """Lazy-construct (and return) the MinesweeperInference instance.

        Returns the instance, or None if the inference module itself can't
        be imported. Does NOT load the model — callers do that and handle
        FileNotFoundError so they can show appropriate UI.
        """
        if self._inference is not None:
            return self._inference
        try:
            # Lazy import — avoids torch at GUI startup
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from src.ai.inference import MinesweeperInference
        except ImportError as e:
            messagebox.showerror(
                'AI inference', f'Failed to load inference module:\n{e}')
            return None
        self._inference = MinesweeperInference(self.settings_manager.model_path)
        return self._inference

    def _on_suggest_click(self):
        """Run inference and highlight the suggested cell."""
        if not self.game_board or self.game_board.game_state != GameState.PLAYING:
            return

        inf = self._ensure_inference()
        if inf is None:
            return

        # Mark this game AI-tainted before we even fetch the suggestion —
        # using the button is the act that disqualifies stats, regardless
        # of whether the user accepts the highlighted cell.
        self._mark_ai_used()

        # Show busy cursor while loading + running
        self.root.config(cursor='watch')
        self.suggest_button.config(state='disabled')
        self.root.update_idletasks()
        try:
            try:
                suggestion = inf.suggest_move(self.game_board)
            except FileNotFoundError as e:
                messagebox.showerror(
                    'AI model not found',
                    f'{e}\n\nSet the model path in Game → Settings.')
                self._inference = None
                return
            except Exception as e:
                messagebox.showerror('AI suggestion failed', str(e))
                return
        finally:
            self.root.config(cursor='')
            self._update_action_buttons()

        if suggestion is None:
            return
        self.board_canvas.highlight_cell(suggestion['row'], suggestion['col'])

        mp = suggestion.get('mine_probability')
        source = suggestion.get('source', '')
        if source == 'solver':
            self.confidence_label.config(text='Source: solver (safe)', fg='#0a7d0a')
        elif mp is not None:
            if mp < 0.15:
                color = '#0a7d0a'
            elif mp < 0.35:
                color = '#b8860b'
            else:
                color = '#cc0000'
            self.confidence_label.config(text=f'P(mine): {mp:.0%}', fg=color)
        else:
            self.confidence_label.config(text='')

    def _on_heatmap_toggle(self):
        if self.heatmap_var.get():
            self._mark_ai_used()
            self._refresh_heatmap()
        else:
            if self.board_canvas is not None:
                self.board_canvas.clear_heatmap()

    def _refresh_heatmap(self):
        """Recompute and redraw the heatmap overlay (if the toggle is on)."""
        if not self.heatmap_var.get():
            return
        if not self.game_board or self.game_board.game_state != GameState.PLAYING:
            return

        source = self.settings.heatmap_source

        if source == 'constraint':
            inf = self._ensure_inference()
            if inf is None:
                return
            try:
                prob_data = inf.get_constraint_probabilities(self.game_board)
            except Exception as e:
                messagebox.showerror('Heatmap failed', str(e))
                self.heatmap_var.set(0)
                return
            if prob_data is None:
                return
            self.board_canvas.show_heatmap(prob_data['probabilities'])

        elif source == 'both':
            inf = self._ensure_inference()
            if inf is None:
                return
            try:
                prob_data = inf.get_comparison_probabilities(self.game_board)
            except FileNotFoundError as e:
                messagebox.showerror(
                    'AI model not found',
                    f'{e}\n\nSet the model path in Game → Settings.')
                self._inference = None
                self.heatmap_var.set(0)
                return
            except Exception as e:
                messagebox.showerror('Heatmap failed', str(e))
                self.heatmap_var.set(0)
                return
            if prob_data is None:
                return
            self.board_canvas.show_comparison_heatmap(
                prob_data['probabilities'], prob_data['divergence'])

        else:
            inf = self._ensure_inference()
            if inf is None:
                return
            try:
                prob_data = inf.get_mine_probabilities(self.game_board)
            except FileNotFoundError as e:
                messagebox.showerror(
                    'AI model not found',
                    f'{e}\n\nSet the model path in Game → Settings.')
                self._inference = None
                self.heatmap_var.set(0)
                return
            except Exception as e:
                messagebox.showerror('Heatmap failed', str(e))
                self.heatmap_var.set(0)
                return
            if prob_data is None:
                return
            self.board_canvas.show_heatmap(prob_data['probabilities'])

    def _on_autoplay_toggle(self):
        """User clicked Auto-play. Tk has already toggled autoplay_var, so
        the new state tells us whether we're starting or stopping."""
        if self.autoplay_var.get():
            if not self._start_autoplay():
                # Failed to start — roll the toggle back so the visual
                # matches reality
                self.autoplay_var.set(0)
        else:
            self._stop_autoplay()

    def _start_autoplay(self) -> bool:
        """Begin the auto-play loop. Returns True iff we actually started."""
        if not self.game_board or self.game_board.game_state != GameState.PLAYING:
            return False

        inf = self._ensure_inference()
        if inf is None:
            return False

        # Same as Suggest: turning autoplay on disqualifies this game's stats
        self._mark_ai_used()
        # First-time model load can take ~500ms (CUDA warmup); do it
        # eagerly with a busy cursor so the user isn't staring at nothing.
        if not inf.is_loaded():
            self.root.config(cursor='watch')
            self.root.update_idletasks()
            try:
                inf.load()
            except FileNotFoundError as e:
                messagebox.showerror(
                    'AI model not found',
                    f'{e}\n\nSet the model path in Game → Settings.')
                self._inference = None
                return False
            except Exception as e:
                messagebox.showerror('AI model load failed', str(e))
                return False
            finally:
                self.root.config(cursor='')

        self._autoplay_step()
        return True

    def _stop_autoplay(self):
        """Stop the auto-play loop. Untoggles the button visual."""
        self._cancel_autoplay()
        if self.autoplay_var.get():
            self.autoplay_var.set(0)

    def _cancel_autoplay(self):
        """Cancel any pending after-callback. Doesn't touch the toggle."""
        if self._autoplay_after_id is not None:
            try:
                self.root.after_cancel(self._autoplay_after_id)
            except Exception:
                pass
            self._autoplay_after_id = None

    def _autoplay_step(self):
        """Get a suggestion, highlight it, schedule the click after a delay."""
        self._autoplay_after_id = None
        if not self.game_board or self.game_board.game_state != GameState.PLAYING:
            return
        if self._inference is None or not self._inference.is_loaded():
            return  # press-handler should have loaded it; bail safely
        try:
            suggestion = self._inference.suggest_move(self.game_board)
        except Exception as e:
            print(f"[autoplay] suggest failed: {e}")
            return
        if suggestion is None:
            return
        self.board_canvas.highlight_cell(suggestion['row'], suggestion['col'])
        self.root.update_idletasks()
        # Schedule the actual click after a brief flash of the highlight,
        # so the user can see what's about to be played
        self._autoplay_after_id = self.root.after(
            self._autoplay_show_ms,
            lambda s=suggestion: self._autoplay_finish_step(s),
        )

    def _autoplay_finish_step(self, suggestion):
        """Apply the move and schedule the next step (if game still playing)."""
        self._autoplay_after_id = None
        if not self.game_board or self.game_board.game_state != GameState.PLAYING:
            return
        self._on_cell_click(suggestion['row'], suggestion['col'])
        if self.game_board.game_state == GameState.PLAYING:
            self._autoplay_after_id = self.root.after(
                self._autoplay_pause_ms, self._autoplay_step)

    def _auto_export_record(self, record: GameRecord):
        """If export is enabled, append this record's guess samples to the
        configured .npz file on a background daemon thread so disk I/O
        doesn't block the GUI. The export lock serializes concurrent
        finalizes. Errors are logged and silently swallowed so a broken
        export never blocks gameplay."""
        if not self.settings_manager.export_enabled:
            return
        path = self.settings_manager.export_path
        if not path:
            return

        # Snapshot the record on the main thread; the worker only sees an
        # immutable dict so there's no shared mutable state.
        record_dict = record.to_dict()
        export_lock = self._export_lock

        def _worker():
            with export_lock:
                try:
                    # Lazy import — pulls in numpy/AI modules; don't load
                    # at GUI startup.
                    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))))
                    if repo_root not in sys.path:
                        sys.path.insert(0, repo_root)
                    import export_training_data as exporter
                    added = exporter.append_record_to_file(record_dict, path)
                    if added:
                        print(f"[export] +{added} samples -> {path}")
                except Exception as e:
                    print(f"[export] failed: {e}")

        threading.Thread(target=_worker, daemon=True,
                         name='minesweeper-export').start()

    def _on_close(self):
        """Called when the user closes the window (X button or Game→Exit)."""
        self._cancel_autoplay()
        self._abandon_current_record()
        # Wait briefly for any in-flight background export to land so we
        # don't lose a finished game's contribution to the .npz. Daemon
        # threads die abruptly when the process exits, but the export uses
        # an atomic .tmp.npz->rename so the file itself can't end up
        # corrupted — we'd just lose the most recent game's samples.
        if self._export_lock.acquire(timeout=2.0):
            self._export_lock.release()
        self.root.destroy()

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
