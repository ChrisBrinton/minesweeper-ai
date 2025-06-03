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


class CellButton(tk.Frame):
    """Individual cell button on the minesweeper grid"""
    # Colors for different numbers
    NUMBER_COLORS = {
        1: 'blue',
        2: 'green', 
        3: 'red',
        4: 'purple',
        5: 'maroon',
        6: 'turquoise',
        7: 'black',
        8: 'gray'
    }
    
    # Class variable to store loaded images
    _images: Dict[str, PhotoImage] = {}
    
    @classmethod
    def load_images(cls):
        """Load image files for cells if not already loaded"""
        if cls._images:  # Images already loaded
            return
            
        # Define the base assets directory
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        print(f"Loading images from: {assets_dir}")
          
        # Define image filenames to look for
        image_files = {
            'hidden': 'hidden_cell.png',
            'empty': 'empty_cell.png',
            'mine': 'mine_cell.png',
            'flag': 'flag_cell.png',
            'mine_red': 'mine_red_cell.png',  # For clicked mine
        }
        
        # Add numbered cells
        for i in range(1, 9):
            image_files[f'num_{i}'] = f'{i}_cell.png'
          
        # Try to load each image
        loaded_count = 0
        for key, filename in image_files.items():
            try:
                image_path = os.path.join(assets_dir, filename)
                if os.path.exists(image_path):
                    cls._images[key] = PhotoImage(file=image_path)
                    print(f"Successfully loaded image: {filename}")
                    loaded_count += 1
                else:
                    print(f"Image file not found: {image_path}")
                    cls._images[key] = None  # Mark as unavailable
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                cls._images[key] = None
                
        print(f"Successfully loaded {loaded_count} of {len(image_files)} images")
    
    def __init__(self, parent, row: int, col: int, click_callback: Callable, 
                right_click_callback: Callable):
        super().__init__(
            parent,
            width=16,  # Standard size for minesweeper cells
            height=16,  # Standard size for minesweeper cells
            relief='raised',
            bd=1,  # Thin border between cells
            bg='#c0c0c0'  # Standard Windows gray
        )
        
        # Make frame non-expandable to maintain exact size
        self.pack_propagate(False)
        self.grid_propagate(False)
        
        # Load images if not already loaded
        self.load_images()
        
        # Create label for content (image or text)
        self.label = tk.Label(
            self,
            bg='#c0c0c0',
            bd=0,
            highlightthickness=0,
            padx=0, 
            pady=0
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        
        self.row = row
        self.col = col
        self.click_callback = click_callback
        self.right_click_callback = right_click_callback
        
        # Initialize with hidden cell appearance
        if self._images.get('hidden'):
            self.label.config(image=self._images['hidden'], text='', bg='#c0c0c0')
        else:
            # Fallback to default appearance if image not available
            self.label.config(text='', bg='#c0c0c0')
        
        # Bind click events
        for widget in [self, self.label]:
            widget.bind('<Button-1>', self._on_left_click)
            widget.bind('<Button-3>', self._on_right_click)
    
    def _on_left_click(self, event):
        """Handle left mouse click"""
        self.click_callback(self.row, self.col)
        return "break"
    
    def _on_right_click(self, event):
        """Handle right mouse click"""
        self.right_click_callback(self.row, self.col)
        return "break"
    
    def _on_release(self, event):
        """Handle mouse button release"""
        # Relief will be updated by update_display method
        pass
    
    def update_display(self, cell, game_board=None):
        """Update button display based on cell state"""
        if cell.state == CellState.REVEALED:
            # Configure frame for revealed state
            self.config(relief='sunken', bd=1)
            
            if cell.is_mine:
                # Check if this is the clicked mine (should be red) or just a regular mine
                is_clicked_mine = (game_board and 
                                 game_board.clicked_mine_pos and 
                                 game_board.clicked_mine_pos == (self.row, self.col))
                
                if is_clicked_mine:
                    # Use red mine image for the clicked mine
                    if self._images.get('mine_red'):
                        self.label.config(image=self._images['mine_red'], text='')
                    else:
                        self.label.config(text='💣', image='')
                    self.config(bg='red')
                    self.label.config(bg='red')
                else:
                    # Use regular mine image for other mines
                    if self._images.get('mine'):
                        self.label.config(image=self._images['mine'], text='')
                    else:
                        self.label.config(text='💣', image='')
                    self.config(bg='#c0c0c0')
                    self.label.config(bg='#c0c0c0')
            elif cell.adjacent_mines > 0:
                # Use numbered cell image if available, otherwise fallback to text
                img_key = f'num_{cell.adjacent_mines}'
                if self._images.get(img_key):
                    self.label.config(image=self._images[img_key], text='')
                else:
                    self.label.config(
                        text=str(cell.adjacent_mines),
                        fg=self.NUMBER_COLORS.get(cell.adjacent_mines, 'black'),
                        image=''
                    )
                self.config(bg='#c0c0c0')
                self.label.config(bg='#c0c0c0')
            else:
                # Use empty cell image if available, otherwise just clear the text
                if self._images.get('empty'):
                    self.label.config(image=self._images['empty'], text='')
                else:
                    self.label.config(text='', image='')
                self.config(bg='#c0c0c0')
                self.label.config(bg='#c0c0c0')
        elif cell.state == CellState.FLAGGED:
            # Configure frame for flagged state
            self.config(relief='raised', bd=1, bg='#c0c0c0')
            
            # Use flag image if available, otherwise fallback to emoji
            if self._images.get('flag'):
                self.label.config(image=self._images['flag'], text='', bg='#c0c0c0')
            else:
                self.label.config(text='🚩', image='', bg='#c0c0c0')
        else:  # HIDDEN
            # Configure frame for hidden state
            self.config(relief='raised', bd=1, bg='#c0c0c0')
            
            # Use hidden cell image if available, otherwise just clear the text
            if self._images.get('hidden'):
                self.label.config(image=self._images['hidden'], text='', bg='#c0c0c0')
            else:
                self.label.config(text='', image='', bg='#c0c0c0')


class MinesweeperGUI:
    """Main GUI class for the minesweeper game"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Minesweeper')
        self.root.resizable(False, False)
        
        # Game components
        self.game_board: Optional[GameBoard] = None
        self.cell_buttons: List[List[CellButton]] = []
        self.start_time: Optional[float] = None
        self.game_timer_id: Optional[str] = None
        
        # GUI components
        self.mine_display: Optional[DigitalDisplay] = None
        self.timer_display: Optional[DigitalDisplay] = None
        self.smiley_button: Optional[SmileyButton] = None
        self.game_frame: Optional[tk.Frame] = None
        self._setup_gui()
        self._new_game('beginner')
    
    def _setup_gui(self):
        """Setup the main GUI components"""
        # Main container
        main_frame = tk.Frame(self.root, bg='lightgray', relief='raised', bd=3)
        main_frame.pack(padx=5, pady=5)
        
        # OPTIMIZATION: Load all images once during startup
        DigitalDisplay.load_digit_images()
        SmileyButton.load_images()
        CellButton.load_images()
        
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
        
        # Create new game board
        rows, cols, mines = GameBoard.DIFFICULTIES[difficulty]
        self.game_board = GameBoard(rows, cols, mines)
        
        # Reset displays
        self.mine_display.set_value(mines)
        self.timer_display.set_value(0)
        
        # Update smiley face state
        self.smiley_button.set_state(GameState.READY)
        self.start_time = None
        
        # OPTIMIZATION: Check if we can reuse existing buttons
        current_size = (len(self.cell_buttons), len(self.cell_buttons[0])) if self.cell_buttons else (0, 0)
        target_size = (rows, cols)
        
        if current_size != target_size:
            # Size changed - need to recreate buttons
            self._recreate_buttons(rows, cols)
        else:
            # Same size - just reset existing buttons (much faster!)
            self._reset_existing_buttons()
        
        # OPTIMIZATION: Single layout update at the end instead of multiple updates
        self.root.update_idletasks()
    
    def _restart_game(self):
        """Restart the current game"""
        if self.game_board:
            difficulty = self._get_current_difficulty()
            self._new_game(difficulty)
    
    def _get_current_difficulty(self) -> str:
        """Get the current difficulty level"""
        if not self.game_board:
            return 'beginner'
        
        size = (self.game_board.rows, self.game_board.cols, self.game_board.total_mines)
        for diff, params in GameBoard.DIFFICULTIES.items():
            if params == size:
                return diff
        return 'beginner'
    
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
        if not self.game_board:
            return
        
        # Update mine counter
        remaining_mines = self.game_board.get_remaining_mines()
        self.mine_display.set_value(remaining_mines)
        
        # Update smiley face
        self.smiley_button.set_state(self.game_board.game_state)
        
        # Update cell buttons
        for row in range(self.game_board.rows):
            for col in range(self.game_board.cols):
                cell = self.game_board.get_cell(row, col)
                self.cell_buttons[row][col].update_display(cell, self.game_board)
    
    def _update_timer(self):
        """Update the game timer"""
        if (self.start_time and 
            self.game_board and 
            self.game_board.game_state == GameState.PLAYING):
            
            elapsed = int(time.time() - self.start_time)
            self.timer_display.set_value(min(elapsed, 999))  # Cap at 999
            
            # Schedule next update
            self.game_timer_id = self.root.after(1000, self._update_timer)
    
    def _end_game(self):
        """Handle game end"""
        # Stop timer
        if self.game_timer_id:
            self.root.after_cancel(self.game_timer_id)
            self.game_timer_id = None
        
        # No popup needed - the game state is shown through the smiley face
        # and the revealed board state
    
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
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
    
    def _recreate_buttons(self, rows: int, cols: int):
        """Create new buttons when board size changes (OPTIMIZATION: Only when needed)"""
        # Clear existing buttons
        for widget in self.game_frame.winfo_children():
            widget.destroy()
        
        # Create new cell buttons
        self.cell_buttons = []
        for row in range(rows):
            button_row = []
            for col in range(cols):
                button = CellButton(
                    self.game_frame,
                    row, col, self._on_cell_click,
                    self._on_cell_right_click
                )
                button.grid(row=row, column=col, padx=0, pady=0)
                button_row.append(button)
            self.cell_buttons.append(button_row)
        
        # Initialize display for all cells
        self._update_display()
    
    def _reset_existing_buttons(self):
        """Reset existing buttons to hidden state (OPTIMIZATION: Reuse existing widgets)"""
        # Just update display without recreating widgets - much faster!
        self._update_display()
    
    def _create_buttons(self):
        """Create initial button grid (used only during GUI setup)"""
        # This method is kept for the initial GUI setup
        rows, cols, mines = GameBoard.DIFFICULTIES['beginner']
        self._recreate_buttons(rows, cols)
