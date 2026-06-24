"""
Game history persistence and statistics dialog.

Every completed (or abandoned) game is appended to ~/.minesweeper/history.json.
A record stores enough information to fully replay the game — mine positions
plus the click sequence — so it can be converted to supervised training data
later.
"""

import json
import os
import secrets
import tkinter as tk
from datetime import datetime
from tkinter import Toplevel, Label, Button, Frame
from tkinter import ttk
from typing import Dict, List, Optional, Tuple


HISTORY_VERSION = 1
DIFFICULTIES = ('beginner', 'intermediate', 'expert')


def _default_history_path() -> str:
    data_dir = os.path.join(os.path.expanduser('~'), '.minesweeper')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'history.json')


class GameRecord:
    """In-progress or completed game record. Mutable while a game is live."""

    __slots__ = (
        'id', 'started_at', 'ended_at', 'difficulty', 'rows', 'cols',
        'mines', 'result', 'elapsed_seconds', 'first_click',
        'mine_positions', 'moves', 'cells_revealed', 'correct_flags',
        'ai_used',
    )

    def __init__(self, difficulty: str, rows: int, cols: int, mines: int):
        self.id = datetime.now().strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(3)
        self.started_at = datetime.now().isoformat(timespec='seconds')
        self.ended_at: Optional[str] = None
        self.difficulty = difficulty
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.result: Optional[str] = None  # 'won' | 'lost' | 'abandoned'
        self.elapsed_seconds: int = 0
        self.first_click: Optional[Tuple[int, int]] = None
        self.mine_positions: List[Tuple[int, int]] = []
        self.moves: List[Dict] = []
        self.cells_revealed: int = 0
        self.correct_flags: int = 0
        # True if Suggest or Auto-play was used during this game. Tainted
        # games are still recorded but excluded from personal stats.
        self.ai_used: bool = False

    def append_move(self, row: int, col: int, action: str):
        """Append a click. action ∈ {'reveal', 'flag', 'unflag'}."""
        if self.first_click is None and action == 'reveal':
            self.first_click = (row, col)
        self.moves.append({'r': row, 'c': col, 'a': action})

    def finalize(self, result: str, elapsed_seconds: int,
                 mine_positions: List[Tuple[int, int]],
                 cells_revealed: int, correct_flags: int):
        self.result = result
        self.elapsed_seconds = elapsed_seconds
        self.mine_positions = mine_positions
        self.cells_revealed = cells_revealed
        self.correct_flags = correct_flags
        self.ended_at = datetime.now().isoformat(timespec='seconds')

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'difficulty': self.difficulty,
            'rows': self.rows,
            'cols': self.cols,
            'mines': self.mines,
            'result': self.result,
            'elapsed_seconds': self.elapsed_seconds,
            'first_click': list(self.first_click) if self.first_click else None,
            'mine_positions': [list(p) for p in self.mine_positions],
            'moves': self.moves,
            'cells_revealed': self.cells_revealed,
            'correct_flags': self.correct_flags,
            'ai_used': self.ai_used,
        }


def derive_record_metrics(record: Dict) -> Tuple[int, int]:
    """Return (cells_revealed, correct_flags) for a record dict.

    If both fields are stored, returns them directly. Otherwise derives them
    from `moves` + `mine_positions`:
      - correct_flags: count of player flag actions on cells whose final
        flagged state lands on a real mine
      - cells_revealed: replays the click sequence against a reconstructed
        board with the recorded mine layout
    """
    if 'cells_revealed' in record and 'correct_flags' in record:
        return int(record['cells_revealed']), int(record['correct_flags'])

    mine_positions = {tuple(p) for p in record.get('mine_positions', [])}
    moves = record.get('moves', [])

    # Derive correct_flags from final player-flag state in the move log
    flagged_by_player = set()
    for move in moves:
        action = move.get('a')
        pos = (move.get('r'), move.get('c'))
        if action == 'flag':
            flagged_by_player.add(pos)
        elif action == 'unflag':
            flagged_by_player.discard(pos)
    correct_flags = sum(1 for p in flagged_by_player if p in mine_positions)

    # Derive cells_revealed by replaying through GameBoard
    cells_revealed = 0
    if mine_positions:
        try:
            # Lazy import to avoid pulling game logic into module load
            import sys
            import os as _os
            _src = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _src not in sys.path:
                sys.path.insert(0, _src)
            from game import GameBoard, GameState
            rows = record.get('rows')
            cols = record.get('cols')
            mines_count = record.get('mines', len(mine_positions))
            if rows and cols:
                board = GameBoard(rows, cols, mines_count)
                for (r, c) in mine_positions:
                    board.board[r][c].place_mine()
                board._calculate_adjacent_mines()
                board.mines_placed = True
                board.game_state = GameState.PLAYING
                for move in moves:
                    a = move.get('a')
                    r, c = move.get('r'), move.get('c')
                    if a == 'reveal':
                        board.reveal_cell(r, c)
                    elif a in ('flag', 'unflag'):
                        board.toggle_flag(r, c)
                cells_revealed = board.cells_revealed
                if board.game_state == GameState.LOST:
                    cells_revealed -= 1  # don't count the clicked mine
        except Exception as e:
            print(f"derive_record_metrics replay failed: {e}")
            cells_revealed = 0

    return cells_revealed, correct_flags


class GameHistoryManager:
    """Append-only persistence layer for completed games."""

    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file or _default_history_path()
        self._records: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if not os.path.exists(self.data_file):
            return []
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading game history: {e}")
            return []
        if isinstance(data, dict) and 'games' in data:
            return data['games']
        if isinstance(data, list):
            return data
        return []

    def _save(self):
        try:
            payload = {'version': HISTORY_VERSION, 'games': self._records}
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            tmp = self.data_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.data_file)
        except IOError as e:
            print(f"Error saving game history: {e}")

    def append(self, record: GameRecord):
        if record.result is None:
            return  # never append unfinalized records
        self._records.append(record.to_dict())
        self._save()

    def all_records(self) -> List[Dict]:
        return list(self._records)

    def stats_for(self, difficulty: str, last_n: Optional[int] = None) -> Dict:
        """Aggregate stats for one difficulty.

        AI-assisted games (records with ai_used=True) are excluded — they
        don't count toward personal records.

        If *last_n* is set, only the most recent *last_n* non-AI games of
        the given difficulty are considered.
        """
        games = [r for r in self._records
                 if r.get('difficulty') == difficulty
                 and not r.get('ai_used', False)]
        if last_n is not None:
            games = games[-last_n:]
        played = len(games)
        won = sum(1 for r in games if r.get('result') == 'won')
        lost = sum(1 for r in games if r.get('result') == 'lost')
        abandoned = sum(1 for r in games if r.get('result') == 'abandoned')

        win_records = [r for r in games if r.get('result') == 'won']
        win_times = [r['elapsed_seconds'] for r in win_records]
        avg_win = (sum(win_times) / len(win_times)) if win_times else None
        best_win = min(win_times) if win_times else None

        # Streaks operate on completed games only (won/lost), in chronological order
        completed = [r for r in games if r.get('result') in ('won', 'lost')]
        completed.sort(key=lambda r: r.get('ended_at') or '')
        cur_streak = 0
        longest_streak = 0
        run = 0
        for r in completed:
            if r['result'] == 'won':
                run += 1
                longest_streak = max(longest_streak, run)
            else:
                run = 0
        # Current win streak counts from the most-recent end backwards
        for r in reversed(completed):
            if r['result'] == 'won':
                cur_streak += 1
            else:
                break

        # Best rates over winning games — used for live "record pace" highlight
        best_cpm = 0.0
        best_fpm = 0.0
        for r in win_records:
            elapsed = r.get('elapsed_seconds', 0) or 0
            if elapsed <= 0:
                continue
            cells, flags = derive_record_metrics(r)
            minutes = elapsed / 60.0
            best_cpm = max(best_cpm, cells / minutes)
            best_fpm = max(best_fpm, flags / minutes)

        return {
            'played': played,
            'won': won,
            'lost': lost,
            'abandoned': abandoned,
            'win_rate': (won / (won + lost)) if (won + lost) else 0.0,
            'avg_win_seconds': avg_win,
            'best_win_seconds': best_win,
            'current_streak': cur_streak,
            'longest_streak': longest_streak,
            'best_cells_per_minute': best_cpm if win_records else None,
            'best_flags_per_minute': best_fpm if win_records else None,
        }

    def peak_rolling_win_rate(self, difficulty: str, window: int = 100
                              ) -> Optional[float]:
        """Highest win rate over any sliding window of *window* games.

        Uses the same won/(won+lost) formula as stats_for, so abandoned
        games don't count against the rate.  Returns None when fewer than
        *window* games have been played.
        """
        games = [r for r in self._records
                 if r.get('difficulty') == difficulty
                 and not r.get('ai_used', False)]
        if len(games) < window:
            return None

        best = 0.0
        for start in range(len(games) - window + 1):
            chunk = games[start:start + window]
            won = sum(1 for r in chunk if r.get('result') == 'won')
            lost = sum(1 for r in chunk if r.get('result') == 'lost')
            if won + lost:
                best = max(best, won / (won + lost))
        return best

    def best_rates_for(self, difficulty: str) -> Dict[str, float]:
        """Return best cells/min and flags/min over winning games.

        Returns 0.0 for either metric when there are no winning games yet.
        """
        s = self.stats_for(difficulty)
        return {
            'cells_per_minute': s.get('best_cells_per_minute') or 0.0,
            'flags_per_minute': s.get('best_flags_per_minute') or 0.0,
        }

    def overall_stats(self, last_n: Optional[int] = None) -> Dict:
        """Aggregate across all difficulties. Excludes AI-assisted games.

        If *last_n* is set, only the most recent *last_n* non-AI games
        (across all difficulties) are considered.
        """
        clean = [r for r in self._records if not r.get('ai_used', False)]
        if last_n is not None:
            clean = clean[-last_n:]
        played = len(clean)
        total_seconds = sum(r.get('elapsed_seconds', 0) for r in clean)
        return {
            'played': played,
            'total_seconds': total_seconds,
        }


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return '—'
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    return f"{s // 60}:{s % 60:02d}"


def _format_total(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m}m"


LAST_N = 100


class StatsDialog:
    """Per-difficulty statistics dialog with All-Time and Last-100 columns."""

    LABELS = (
        ('Games played', 'played'),
        ('Won', 'won'),
        ('Lost', 'lost'),
        ('Abandoned', 'abandoned'),
        ('Win rate', '_win_rate_pct'),
        ('Peak win rate', '_peak_wr_pct'),
        ('Average win time', '_avg_win_str'),
        ('Best win time', '_best_win_str'),
        ('Current win streak', 'current_streak'),
        ('Longest win streak', 'longest_streak'),
        ('Best cells/min (wins)', '_best_cpm_str'),
        ('Best flags/min (wins)', '_best_fpm_str'),
    )

    def __init__(self, parent, history: GameHistoryManager,
                 difficulty: Optional[str] = None):
        self.history = history
        self.current_difficulty = difficulty or 'beginner'

        self.dialog = Toplevel(parent)
        self.dialog.title('Statistics')
        self.dialog.geometry('560x600')
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.geometry(
            '+%d+%d' % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self._build()
        self._refresh()

    def _build(self):
        Label(self.dialog, text='📊 Statistics', font=('Arial', 16, 'bold')).pack(pady=10)

        diff_frame = Frame(self.dialog)
        diff_frame.pack(pady=5)
        Label(diff_frame, text='Difficulty:', font=('Arial', 10)).pack(side='left', padx=5)
        self.difficulty_var = tk.StringVar(value=self.current_difficulty)
        combo = ttk.Combobox(
            diff_frame,
            textvariable=self.difficulty_var,
            values=list(DIFFICULTIES),
            state='readonly',
            width=15,
        )
        combo.pack(side='left', padx=5)
        combo.bind('<<ComboboxSelected>>', self._on_difficulty_changed)

        body = Frame(self.dialog, padx=20, pady=10)
        body.pack(fill='both', expand=True)

        # Column headers
        Label(body, text='', width=22).grid(row=0, column=0)
        Label(body, text='All Time', font=('Arial', 10, 'bold'),
              width=12, anchor='center').grid(row=0, column=1, pady=(0, 4))
        Label(body, text=f'Last {LAST_N}', font=('Arial', 10, 'bold'),
              width=12, anchor='center').grid(row=0, column=2, pady=(0, 4))

        self.all_labels: Dict[str, Label] = {}
        self.recent_labels: Dict[str, Label] = {}
        for i, (label_text, key) in enumerate(self.LABELS, start=1):
            Label(body, text=label_text + ':', font=('Arial', 10),
                  anchor='w', width=22).grid(row=i, column=0, sticky='w', pady=2)
            v_all = Label(body, text='—', font=('Courier', 10), anchor='center', width=12)
            v_all.grid(row=i, column=1, pady=2)
            self.all_labels[key] = v_all
            v_recent = Label(body, text='—', font=('Courier', 10), anchor='center', width=12)
            v_recent.grid(row=i, column=2, pady=2)
            self.recent_labels[key] = v_recent

        # Streak histogram
        sep1 = ttk.Separator(self.dialog, orient='horizontal')
        sep1.pack(fill='x', padx=20, pady=(10, 5))
        Label(self.dialog, text=f'Last {LAST_N} games',
              font=('Arial', 9, 'bold'), fg='#555').pack(pady=(2, 0))
        self.canvas = tk.Canvas(
            self.dialog, width=510, height=70,
            bg='#f8f8f8', highlightthickness=1, highlightbackground='#ddd',
        )
        self.canvas.pack(padx=25, pady=(2, 0))
        legend = Frame(self.dialog)
        legend.pack(pady=(1, 0))
        for color, text in (('#22c55e', 'Win'), ('#ef4444', 'Loss'),
                            ('#9ca3af', 'Abandoned')):
            Label(legend, text='█', fg=color,
                  font=('Arial', 8)).pack(side='left')
            Label(legend, text=text, font=('Arial', 8),
                  fg='#777').pack(side='left', padx=(0, 6))

        # Overall footer
        sep2 = ttk.Separator(self.dialog, orient='horizontal')
        sep2.pack(fill='x', padx=20, pady=(6, 5))
        self.overall_label = Label(self.dialog, text='', font=('Arial', 9), fg='#555')
        self.overall_label.pack(pady=(2, 10))

    def _on_difficulty_changed(self, _event=None):
        self.current_difficulty = self.difficulty_var.get()
        self._refresh()

    @staticmethod
    def _derived_display(s: Dict) -> Dict:
        """Add human-readable display keys to a raw stats dict."""
        s['_win_rate_pct'] = (
            f"{s['win_rate']*100:.1f}%" if (s['won'] + s['lost']) else '—'
        )
        s.setdefault('_peak_wr_pct', '—')
        s['_avg_win_str'] = _format_seconds(s['avg_win_seconds'])
        s['_best_win_str'] = _format_seconds(s['best_win_seconds'])
        s['_best_cpm_str'] = (
            f"{s['best_cells_per_minute']:.0f}"
            if s.get('best_cells_per_minute') else '—'
        )
        s['_best_fpm_str'] = (
            f"{s['best_flags_per_minute']:.1f}"
            if s.get('best_flags_per_minute') else '—'
        )
        return s

    # Green shades (light → dark) indexed by streak depth
    _WIN_COLORS = ('#86efac', '#4ade80', '#22c55e', '#16a34a', '#15803d')
    # Red shades (light → dark) indexed by streak depth
    _LOSS_COLORS = ('#fca5a5', '#f87171', '#ef4444', '#dc2626', '#b91c1c')
    _ABANDONED_COLOR = '#9ca3af'

    def _draw_histogram(self):
        self.canvas.delete('all')
        games = [r for r in self.history._records
                 if r.get('difficulty') == self.current_difficulty
                 and not r.get('ai_used', False)]
        games = games[-LAST_N:]
        if not games:
            self.canvas.create_text(
                255, 35, text='No games yet', fill='#aaa',
                font=('Arial', 10, 'italic'))
            return

        cw = int(self.canvas['width'])
        ch = int(self.canvas['height'])
        top_pad = 6
        usable_h = ch - top_pad
        n = len(games)
        bar_w = max((cw / n) - 1, 1)
        step = cw / n

        # First pass: compute streak depth for each game and find the max
        streaks: List[int] = []
        streak = 0
        prev_result = None
        for g in games:
            result = g.get('result')
            if result == prev_result and result in ('won', 'lost'):
                streak += 1
            else:
                streak = 1
            prev_result = result
            streaks.append(streak)

        max_streak = max(streaks)
        base_h = max(round(usable_h * 0.15), 2)
        incr = (usable_h - base_h) / max(max_streak - 1, 1)

        # Second pass: draw bars
        for i, (g, sk) in enumerate(zip(games, streaks)):
            result = g.get('result')

            if result == 'abandoned':
                color = self._ABANDONED_COLOR
                bar_h = base_h
            elif result == 'won':
                bar_h = round(base_h + (sk - 1) * incr)
                color = self._WIN_COLORS[min(sk - 1,
                                             len(self._WIN_COLORS) - 1)]
            elif result == 'lost':
                bar_h = round(base_h + (sk - 1) * incr)
                color = self._LOSS_COLORS[min(sk - 1,
                                              len(self._LOSS_COLORS) - 1)]
            else:
                continue

            x1 = i * step
            x2 = x1 + bar_w
            self.canvas.create_rectangle(
                x1, ch - bar_h, x2, ch, fill=color, outline='')

    def _refresh(self):
        s_all = self._derived_display(
            self.history.stats_for(self.current_difficulty))
        s_recent = self._derived_display(
            self.history.stats_for(self.current_difficulty, last_n=LAST_N))

        peak = self.history.peak_rolling_win_rate(
            self.current_difficulty, window=LAST_N)
        s_recent['_peak_wr_pct'] = (
            f"{peak*100:.1f}%" if peak is not None else '—'
        )

        for _, key in self.LABELS:
            self.all_labels[key].config(text=str(s_all.get(key, '—')))
            self.recent_labels[key].config(text=str(s_recent.get(key, '—')))

        self._draw_histogram()

        overall = self.history.overall_stats()
        self.overall_label.config(
            text=f"All games: {overall['played']} played · "
                 f"{_format_total(overall['total_seconds'])} of play time"
        )


def show_stats(parent, history: GameHistoryManager,
               difficulty: Optional[str] = None):
    StatsDialog(parent, history, difficulty)
