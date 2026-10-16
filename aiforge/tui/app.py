"""AIForge terminal application."""
from __future__ import annotations

from pathlib import Path

from textual.app import App

from ..spec import Workspace
from .home import HomeScreen


class AIForgeApp(App):
    TITLE = "AIForge"
    SUB_TITLE = "no-code neural network studio"
    CSS_PATH = "styles.tcss"

    def __init__(self, workspace: Path):
        super().__init__()
        self.ws = Workspace(workspace)

    def on_mount(self) -> None:
        self.sub_title = f"workspace: {self.ws.root}"
        self.push_screen(HomeScreen())
