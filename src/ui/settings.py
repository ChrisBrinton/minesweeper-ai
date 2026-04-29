"""
Application settings persistence and dialog.

Settings are stored at ~/.minesweeper/settings.json. Currently:

    export_training_data : bool   off by default
    export_path          : str    where to write the training-data .npz
                                  default: <repo_root>/training_data.npz
"""

import json
import os
import tkinter as tk
from tkinter import (
    Toplevel, Label, Button, Frame, Entry, Checkbutton, filedialog, messagebox,
)
from typing import Callable, Dict, Optional


def _settings_path() -> str:
    data_dir = os.path.join(os.path.expanduser('~'), '.minesweeper')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'settings.json')


def _default_export_path() -> str:
    """Default to <repo_root>/training_data.npz.

    __file__ = <repo>/src/ui/settings.py → repo root is two parents up.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(repo_root, 'training_data.npz')


class SettingsManager:
    """Tiny JSON-backed key-value store for app settings."""

    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file or _settings_path()
        self.data = self._load()

    def _defaults(self) -> Dict:
        return {
            'export_training_data': False,
            'export_path': _default_export_path(),
        }

    def _load(self) -> Dict:
        merged = self._defaults()
        if not os.path.exists(self.data_file):
            return merged
        try:
            with open(self.data_file, 'r') as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                merged.update(saved)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading settings: {e}")
        return merged

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            tmp = self.data_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.data_file)
        except IOError as e:
            print(f"Error saving settings: {e}")

    @property
    def export_enabled(self) -> bool:
        return bool(self.data.get('export_training_data'))

    @property
    def export_path(self) -> str:
        return self.data.get('export_path') or _default_export_path()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] = v
        self.save()


class SettingsDialog:
    def __init__(self, parent, settings: SettingsManager,
                 on_changed: Optional[Callable] = None):
        self.settings = settings
        self.on_changed = on_changed

        self.dialog = Toplevel(parent)
        self.dialog.title('Settings')
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.geometry(
            '+%d+%d' % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50)
        )

        self._build()

    def _build(self):
        Label(self.dialog, text='Settings',
              font=('Arial', 14, 'bold')).pack(pady=(15, 10), padx=20)

        body = Frame(self.dialog, padx=20, pady=5)
        body.pack(fill='both', expand=True)

        # Export toggle
        self.export_var = tk.BooleanVar(value=self.settings.export_enabled)
        Checkbutton(
            body, text='Export training data after each completed game',
            variable=self.export_var, font=('Arial', 10),
        ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))

        # Export path entry + browse
        Label(body, text='Export file:', font=('Arial', 10)).grid(
            row=1, column=0, sticky='w', padx=(0, 5))
        self.path_var = tk.StringVar(value=self.settings.export_path)
        Entry(body, textvariable=self.path_var, width=42).grid(
            row=1, column=1, sticky='ew')
        Button(body, text='Browse...', command=self._browse).grid(
            row=1, column=2, padx=(5, 0))

        Label(
            body,
            text='Each finalized game appends its guess samples to this file.\n'
                 'Use export_training_data.py to rebuild from full history.',
            font=('Arial', 8), fg='#666', justify='left',
        ).grid(row=2, column=0, columnspan=3, sticky='w', pady=(5, 0))

        # OK / Cancel
        btns = Frame(self.dialog, padx=15, pady=15)
        btns.pack()
        Button(btns, text='OK', command=self._ok, width=10).pack(side='left', padx=5)
        Button(btns, text='Cancel', command=self.dialog.destroy, width=10).pack(
            side='left', padx=5)

    def _browse(self):
        cur = self.path_var.get() or _default_export_path()
        chosen = filedialog.asksaveasfilename(
            title='Choose training data file',
            initialdir=os.path.dirname(cur) or os.getcwd(),
            initialfile=os.path.basename(cur) or 'training_data.npz',
            defaultextension='.npz',
            filetypes=[('NumPy archive', '*.npz'), ('All files', '*.*')],
        )
        if chosen:
            self.path_var.set(chosen)

    def _ok(self):
        path = self.path_var.get().strip()
        if self.export_var.get() and not path:
            messagebox.showerror(
                'Settings', 'Please choose an export file path.')
            return
        self.settings.update(
            export_training_data=self.export_var.get(),
            export_path=path,
        )
        if self.on_changed:
            self.on_changed()
        self.dialog.destroy()


def show_settings(parent, settings: SettingsManager,
                  on_changed: Optional[Callable] = None):
    SettingsDialog(parent, settings, on_changed)
