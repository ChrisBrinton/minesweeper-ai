"""
Minesweeper GUI - Windows 3.1 Style Interface
Implements the classic minesweeper user interface using tkinter
"""

import tkinter as tk
from tkinter import messagebox, Menu
import time
from typing import List, Callable, Optional

from game import GameBoard, GameState, CellState


class DigitalDisplay(tk.Frame):
    """Digital display widget for mine counter and timer"""
    
    def __init__(self, parent, width=3):
        super().__init__(parent, bg='black', relief='sunken', bd=2)
        self.width = width
        self.value = 0
        
        self.label = tk.Label(
            self, 
            text=self._format_number(0),
            font=('Courier', 16, 'bold'),
            fg='red',
            bg='black',
            width=self.width
        )
        self.label.pack(padx=2, pady=2)
    
    def _format_number(self, num: int) -> str:
        """Format number with leading zeros for digital display"""
        if num < 0:
            return f"-{abs(num):0{self.width-1}d}"[:self.width]
        return f"{num:0{self.width}d}"[:self.width]
    
    def set_value(self, value: int):
        """Update the display value"""
        self.value = value
        self.label.config(text=self._format_number(value))


class SmileyButton(tk.Button):
    """Smiley face button that shows game state"""
    
    FACES = {
        GameState.READY: '😊',
        GameState.PLAYING: '😊', 
        GameState.WON: '😎',
        GameState.LOST: '😵'
    }
    
    def __init__(self, parent, command=None):
        super().__init__(
            parent,
            text=self.FACES[GameState.READY],
            font=('Arial', 16),
            width=3,
            height=1,
            relief='raised',
            bd=2,
            command=command
        )
        self.current_state = GameState.READY
    
    def set_state(self, state: GameState):
        """Update smiley face based on game state"""
        if state != self.current_state:
            self.current_state = state
            self.config(text=self.FACES.get(state, '😊'))
    
    def set_pressed_face(self):
        """Show worried face while mouse is pressed"""
        if self.current_state == GameState.PLAYING:
            self.config(text='😬')
    
    def restore_face(self):
        """Restore normal face for current state"""
        self.config(text=self.FACES.get(self.current_state, '😊'))


class CellButton(tk.Button):
    """Individual cell button on the minesweeper grid"""
      # Colors for different numbers
    NUMBER_COLORS = {
        1: 'blue',
        2: 'green', 
        3: 'red',
        4: 'purple',
        5: 'maroon',
        6: 'turquoise',
        7: 'black',        8: 'gray'
    }
    
    def __init__(self, parent, row: int, col: int, click_callback: Callable, 
                 right_click_callback: Callable):
        super().__init__(
            parent,
            width=2,
            height=1,
            font=('Arial', 9, 'bold'),
            relief='raised',
            bd=2,
            bg='lightgray',
            command=lambda: click_callback(row, col)  # Use command for left click
        )
        
        self.row = row
        self.col = col
        self.click_callback = click_callback
        self.right_click_callback = right_click_callback
        
        # Bind mouse events
        self.bind('<Button-3>', self._on_right_click)  # Right click only        self.bind('<ButtonPress-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)
    
    def _on_right_click(self, event):
        """Handle right mouse click"""
        self.right_click_callback(self.row, self.col)
        return "break"  # Prevent event propagation
    
    def _on_press(self, event):
        """Handle mouse button press"""
        if self['relief'] == 'raised':
            self.config(relief='sunken')
    
    def _on_release(self, event):
        """Handle mouse button release"""
        # Relief will be updated by update_display method
        pass
    
    def update_display(self, cell):
        """Update button display based on cell state"""
        if cell.state == CellState.REVEALED:
            self.config(relief='sunken', bg='lightgray')
            if cell.is_mine:
                self.config(text='💣', bg='red')
            elif cell.adjacent_mines > 0:
                self.config(
                    text=str(cell.adjacent_mines),
                    fg=self.NUMBER_COLORS.get(cell.adjacent_mines, 'black')
                )
            else:
                self.config(text='')
        elif cell.state == CellState.FLAGGED:
            self.config(text='🚩', relief='raised', bg='lightgray')
        else:  # HIDDEN
            self.config(text='', relief='raised', bg='lightgray')


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
        self.smiley_button.set_state(GameState.READY)
        self.start_time = None
        
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
                    row, col,
                    self._on_cell_click,
                    self._on_cell_right_click
                )
                button.grid(row=row, column=col, padx=1, pady=1)
                button_row.append(button)
            self.cell_buttons.append(button_row)
          # Update window size
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
                self.cell_buttons[row][col].update_display(cell)
    
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
        
        # Show appropriate message
        if self.game_board.game_state == GameState.WON:
            elapsed = int(time.time() - self.start_time) if self.start_time else 0
            messagebox.showinfo(
                "Congratulations!", 
                f"You won!\nTime: {elapsed} seconds"
            )
        elif self.game_board.game_state == GameState.LOST:
            messagebox.showinfo("Game Over", "You hit a mine!")
    
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
