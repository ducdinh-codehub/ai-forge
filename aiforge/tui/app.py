"""AIForge terminal application."""
from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from ..spec import Workspace
from .colors import DARK, LIGHT
from .home import HomeScreen


class AIForgeApp(App):
    TITLE = "AIForge"
    SUB_TITLE = "no-code neural network studio"
    CSS_PATH = "styles.tcss"
    BINDINGS = [Binding("ctrl+t", "toggle_theme", "Light/dark")]

    def __init__(self, workspace: Path):
        super().__init__()
        self.ws = Workspace(workspace)

    def on_mount(self) -> None:
        self.sub_title = f"workspace: {self.ws.root}"
        saved = self.ws.get_setting("theme", DARK)
        self.theme = saved if saved in self.available_themes else DARK
        # Also catches themes picked from the command palette (Ctrl+P → Theme).
        self.theme_changed_signal.subscribe(self, lambda theme: self.ws.set_setting("theme", theme.name))
        self.push_screen(HomeScreen())

    def action_toggle_theme(self) -> None:
        self.theme = LIGHT if self.current_theme.dark else DARK
        self.notify(f"{'Dark' if self.current_theme.dark else 'Light'} theme. Press Ctrl+T to switch back.",
                    timeout=3)
