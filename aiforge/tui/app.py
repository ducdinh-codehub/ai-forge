"""AIForge terminal application."""
from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from ..spec import Workspace
from .home import HomeScreen

DARK_THEME = "textual-dark"
LIGHT_THEME = "textual-light"


class AIForgeApp(App):
    TITLE = "AIForge"
    SUB_TITLE = "no-code neural network studio"
    CSS_PATH = "styles.tcss"
    BINDINGS = [Binding("ctrl+t", "toggle_theme", "Dark/Light")]

    def __init__(self, workspace: Path):
        super().__init__()
        self.ws = Workspace(workspace)

    def on_mount(self) -> None:
        self.sub_title = f"workspace: {self.ws.root}"
        saved = self.ws.get_setting("theme")
        if saved in self.available_themes:
            self.theme = saved
        # Remember the choice whether it came from the button or the command palette.
        self.theme_changed_signal.subscribe(self, lambda t: self.ws.set_setting("theme", t.name))
        self.push_screen(HomeScreen())

    def action_toggle_theme(self) -> None:
        self.theme = LIGHT_THEME if self.current_theme.dark else DARK_THEME
