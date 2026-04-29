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


def _repo_root() -> str:
    """__file__ = <repo>/src/ui/settings.py → repo root is two parents up."""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _default_export_path() -> str:
    return os.path.join(_repo_root(), 'training_data.npz')


def _default_model_path() -> str:
    return os.path.join(_repo_root(), 'best_model.pth')


class SettingsManager:
    """Tiny JSON-backed key-value store for app settings."""

    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file or _settings_path()
        self.data = self._load()

    def _defaults(self) -> Dict:
        return {
            'ai_functions_enabled': True,
            'export_training_data': False,
            'export_path': _default_export_path(),
            'model_path': _default_model_path(),
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
    def ai_enabled(self) -> bool:
        return bool(self.data.get('ai_functions_enabled', True))

    @property
    def export_enabled(self) -> bool:
        return bool(self.data.get('export_training_data'))

    @property
    def export_path(self) -> str:
        return self.data.get('export_path') or _default_export_path()

    @property
    def model_path(self) -> str:
        return self.data.get('model_path') or _default_model_path()

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

        # AI functions toggle (controls button visibility + model preload)
        self.ai_var = tk.BooleanVar(value=self.settings.ai_enabled)
        Checkbutton(
            body, text='Enable AI functions (Suggest + Auto-play buttons, model preload)',
            variable=self.ai_var, font=('Arial', 10),
        ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 10))

        # Export toggle
        self.export_var = tk.BooleanVar(value=self.settings.export_enabled)
        Checkbutton(
            body, text='Export training data after each completed game',
            variable=self.export_var, font=('Arial', 10),
        ).grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 10))

        # Export path entry + browse
        Label(body, text='Export file:', font=('Arial', 10)).grid(
            row=2, column=0, sticky='w', padx=(0, 5))
        self.path_var = tk.StringVar(value=self.settings.export_path)
        Entry(body, textvariable=self.path_var, width=42).grid(
            row=2, column=1, sticky='ew')
        Button(body, text='Browse...', command=self._browse).grid(
            row=2, column=2, padx=(5, 0))

        Label(
            body,
            text='Each finalized game appends its guess samples to this file.\n'
                 'Use export_training_data.py to rebuild from full history.',
            font=('Arial', 8), fg='#666', justify='left',
        ).grid(row=3, column=0, columnspan=3, sticky='w', pady=(5, 10))

        # AI model path entry + browse
        Label(body, text='AI model:', font=('Arial', 10)).grid(
            row=4, column=0, sticky='w', padx=(0, 5))
        self.model_var = tk.StringVar(value=self.settings.model_path)
        Entry(body, textvariable=self.model_var, width=42).grid(
            row=4, column=1, sticky='ew')
        Button(body, text='Browse...', command=self._browse_model).grid(
            row=4, column=2, padx=(5, 0))

        Label(
            body,
            text='Used by the Suggest-move button. Leave at default if you copied\n'
                 'best_model.pth into the project root.',
            font=('Arial', 8), fg='#666', justify='left',
        ).grid(row=5, column=0, columnspan=3, sticky='w', pady=(5, 0))

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

    def _browse_model(self):
        cur = self.model_var.get() or _default_model_path()
        chosen = filedialog.askopenfilename(
            title='Choose AI model checkpoint',
            initialdir=os.path.dirname(cur) or os.getcwd(),
            initialfile=os.path.basename(cur) or 'best_model.pth',
            filetypes=[('PyTorch checkpoint', '*.pth *.pt'), ('All files', '*.*')],
        )
        if chosen:
            self.model_var.set(chosen)

    def _ok(self):
        path = self.path_var.get().strip()
        if self.export_var.get() and not path:
            messagebox.showerror(
                'Settings', 'Please choose an export file path.')
            return
        self.settings.update(
            ai_functions_enabled=self.ai_var.get(),
            export_training_data=self.export_var.get(),
            export_path=path,
            model_path=self.model_var.get().strip(),
        )
        if self.on_changed:
            self.on_changed()
        self.dialog.destroy()


def show_settings(parent, settings: SettingsManager,
                  on_changed: Optional[Callable] = None):
    SettingsDialog(parent, settings, on_changed)
