"""
Minesweeper Leaderboard System
Manages high scores and persistent game preferences
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tkinter import messagebox, Toplevel, Label, Button, Listbox, Scrollbar, Frame
import tkinter as tk
from tkinter import ttk

class LeaderboardEntry:
    """Represents a single leaderboard entry"""
    
    def __init__(self, time_seconds: int, date: str = None, player_name: str = "Player"):
        self.time_seconds = time_seconds
        self.date = date or datetime.now().strftime("%Y-%m-%d %H:%M")
        self.player_name = player_name
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "time_seconds": self.time_seconds,
            "date": self.date,
            "player_name": self.player_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LeaderboardEntry':
        """Create from dictionary"""
        return cls(
            time_seconds=data["time_seconds"],
            date=data.get("date", ""),
            player_name=data.get("player_name", "Player")
        )
    
    def format_time(self) -> str:
        """Format time as MM:SS"""
        minutes = self.time_seconds // 60
        seconds = self.time_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"


class LeaderboardManager:
    """Manages leaderboard data and persistence"""
    
    def __init__(self, data_file: str = None):
        # Default data file location
        if data_file is None:
            data_dir = os.path.join(os.path.expanduser("~"), ".minesweeper")
            os.makedirs(data_dir, exist_ok=True)
            data_file = os.path.join(data_dir, "leaderboard.json")
        
        self.data_file = data_file
        self.data = self._load_data()
    
    def _get_default_data(self) -> Dict:
        """Get default data structure"""
        return {
            "leaderboards": {
                "beginner": [],
                "intermediate": [],
                "expert": []
            },
            "preferences": {
                "last_difficulty": "beginner",
                "player_name": "Player"
            }
        }
    
    def _load_data(self) -> Dict:
        """Load data from file or create default"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Ensure all required keys exist
                default_data = self._get_default_data()
                for key in default_data:
                    if key not in data:
                        data[key] = default_data[key]
                        
                # Ensure all difficulties exist in leaderboards
                for difficulty in default_data["leaderboards"]:
                    if difficulty not in data["leaderboards"]:
                        data["leaderboards"][difficulty] = []
                
                return data
            else:
                return self._get_default_data()
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading leaderboard data: {e}")
            return self._get_default_data()
    
    def _save_data(self):
        """Save data to file"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except IOError as e:
            print(f"Error saving leaderboard data: {e}")
    
    def add_score(self, difficulty: str, time_seconds: int, player_name: str = None) -> bool:
        """
        Add a new score to the leaderboard
        Returns True if it made the top 10, False otherwise
        """
        if difficulty not in self.data["leaderboards"]:
            return False
        
        if player_name is None:
            player_name = self.data["preferences"]["player_name"]
        
        # Create new entry
        entry = LeaderboardEntry(time_seconds, player_name=player_name)
        
        # Get current leaderboard for this difficulty
        leaderboard = self.data["leaderboards"][difficulty]
        
        # Convert existing entries from dict format if needed
        entries = []
        for item in leaderboard:
            if isinstance(item, dict):
                entries.append(LeaderboardEntry.from_dict(item))
            else:
                entries.append(item)
        
        # Add new entry
        entries.append(entry)
        
        # Sort by time (ascending - faster times are better)
        entries.sort(key=lambda x: x.time_seconds)
        
        # Keep only top 10
        entries = entries[:10]
        
        # Check if the new entry made it into top 10
        made_top_10 = any(e.time_seconds == time_seconds and e.date == entry.date for e in entries)
        
        # Convert back to dict format and save
        self.data["leaderboards"][difficulty] = [e.to_dict() for e in entries]
        self._save_data()
        
        return made_top_10
    
    def get_leaderboard(self, difficulty: str) -> List[LeaderboardEntry]:
        """Get leaderboard for a specific difficulty"""
        if difficulty not in self.data["leaderboards"]:
            return []
        
        entries = []
        for item in self.data["leaderboards"][difficulty]:
            if isinstance(item, dict):
                entries.append(LeaderboardEntry.from_dict(item))
            else:
                entries.append(item)
        
        return entries
    
    def get_last_difficulty(self) -> str:
        """Get the last played difficulty"""
        return self.data["preferences"].get("last_difficulty", "beginner")
    
    def set_last_difficulty(self, difficulty: str):
        """Set the last played difficulty"""
        self.data["preferences"]["last_difficulty"] = difficulty
        self._save_data()
    
    def get_player_name(self) -> str:
        """Get the current player name"""
        return self.data["preferences"].get("player_name", "Player")
    
    def set_player_name(self, name: str):
        """Set the player name"""
        self.data["preferences"]["player_name"] = name
        self._save_data()
    
    def is_top_10_time(self, difficulty: str, time_seconds: int) -> bool:
        """Check if a time would make it into the top 10"""
        leaderboard = self.get_leaderboard(difficulty)
        
        if len(leaderboard) < 10:
            return True
        
        # Check if time is better than the worst (10th) time
        return time_seconds < leaderboard[-1].time_seconds


class LeaderboardDialog:
    """Dialog window to display leaderboard"""
    
    def __init__(self, parent, leaderboard_manager: LeaderboardManager, difficulty: str = None):
        self.leaderboard_manager = leaderboard_manager
        self.current_difficulty = difficulty or leaderboard_manager.get_last_difficulty()
        
        # Create dialog window
        self.dialog = Toplevel(parent)
        self.dialog.title("Leaderboard")
        self.dialog.geometry("400x500")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        self._create_widgets()
        self._update_display()
    
    def _create_widgets(self):
        """Create dialog widgets"""
        # Title
        title_label = Label(self.dialog, text="🏆 Best Times", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Difficulty selection
        difficulty_frame = Frame(self.dialog)
        difficulty_frame.pack(pady=5)
        
        Label(difficulty_frame, text="Difficulty:", font=("Arial", 10)).pack(side="left", padx=5)
        
        self.difficulty_var = tk.StringVar(value=self.current_difficulty)
        self.difficulty_combo = ttk.Combobox(
            difficulty_frame, 
            textvariable=self.difficulty_var,
            values=["beginner", "intermediate", "expert"],
            state="readonly",
            width=15
        )
        self.difficulty_combo.pack(side="left", padx=5)
        self.difficulty_combo.bind("<<ComboboxSelected>>", self._on_difficulty_changed)
        
        # Leaderboard display
        list_frame = Frame(self.dialog)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Headers
        header_frame = Frame(list_frame)
        header_frame.pack(fill="x", pady=(0, 5))
        
        Label(header_frame, text="Rank", font=("Arial", 10, "bold"), width=6).pack(side="left")
        Label(header_frame, text="Time", font=("Arial", 10, "bold"), width=8).pack(side="left")
        Label(header_frame, text="Date", font=("Arial", 10, "bold"), width=12).pack(side="left")
        Label(header_frame, text="Player", font=("Arial", 10, "bold")).pack(side="left", fill="x", expand=True)
        
        # Listbox with scrollbar
        listbox_frame = Frame(list_frame)
        listbox_frame.pack(fill="both", expand=True)
        
        self.listbox = Listbox(listbox_frame, font=("Courier", 10), height=15)
        scrollbar = Scrollbar(listbox_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        Button(self.dialog, text="Close", command=self.dialog.destroy, font=("Arial", 10)).pack(pady=10)
    
    def _on_difficulty_changed(self, event=None):
        """Handle difficulty selection change"""
        self.current_difficulty = self.difficulty_var.get()
        self._update_display()
    
    def _update_display(self):
        """Update the leaderboard display"""
        self.listbox.delete(0, tk.END)
        
        leaderboard = self.leaderboard_manager.get_leaderboard(self.current_difficulty)
        
        if not leaderboard:
            self.listbox.insert(tk.END, "                 No times recorded yet")
            return
        
        for i, entry in enumerate(leaderboard, 1):
            # Format: "  1.   01:23   2024-12-07   Player"
            rank = f"{i:2d}."
            time_str = entry.format_time()
            date_str = entry.date.split()[0] if ' ' in entry.date else entry.date[:10]
            player_str = entry.player_name[:15]  # Truncate if too long
            
            line = f"  {rank:<4} {time_str:<8} {date_str:<12} {player_str}"
            self.listbox.insert(tk.END, line)


def show_leaderboard(parent, leaderboard_manager: LeaderboardManager, difficulty: str = None):
    """Show leaderboard dialog"""
    LeaderboardDialog(parent, leaderboard_manager, difficulty)


def congratulate_new_record(parent, leaderboard_manager: LeaderboardManager, 
                          difficulty: str, time_seconds: int, rank: int):
    """Show congratulations for a new record"""
    time_str = f"{time_seconds // 60:02d}:{time_seconds % 60:02d}"
    
    if rank == 1:
        title = "🎉 NEW RECORD!"
        message = f"Congratulations! You set a new record for {difficulty.title()} difficulty!\n\nTime: {time_str}"
    else:
        title = "🏆 Top 10!"
        message = f"Great job! You made it to #{rank} on the {difficulty.title()} leaderboard!\n\nTime: {time_str}"
    
    messagebox.showinfo(title, message)
