"""Ask, once and politely, for a star on GitHub.

The app can't star the repository itself (that needs the user's GitHub account), so it opens the
repository page and the user clicks ⭐ Star there. The question comes after the first model that
trained successfully - when AIForge has just been useful - and never during the guided tour.
"Maybe later" asks again a few trainings later; starring or "Don't ask again" ends it for good.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from .. import feedback
from .modals import Dialog

ASK_AGAIN_AFTER = 5  # successful trainings after "Maybe later"


def count_training(ws) -> None:
    """Called after every training run that saved a model."""
    ws.set_setting("trained_runs", ws.get_setting("trained_runs", 0) + 1)


def open_repo(app) -> None:
    app.ws.set_setting("star_answered", True)  # they found the button: no need to ask
    app.open_url(feedback.REPO_URL)
    app.notify("Opened AIForge on GitHub in your browser - click [b]⭐ Star[/b] at the top right. Thank you!",
               timeout=10)


def offer(screen) -> None:
    app, ws = screen.app, screen.app.ws
    if getattr(app, "tour", None) or ws.get_setting("star_answered"):
        return
    runs = ws.get_setting("trained_runs", 0)
    if runs < ws.get_setting("star_ask_at", 1):
        return

    # Saved before asking: "Maybe later" (or Esc) needs nothing more, and the home screen resuming
    # under the closing dialog can't ask a second time.
    ws.set_setting("star_ask_at", runs + ASK_AGAIN_AFTER)

    def done(answer):
        if answer == "star":
            open_repo(app)
        elif answer == "never":
            ws.set_setting("star_answered", True)
    app.push_screen(StarDialog(), done)


class StarDialog(Dialog):
    """Returns "star", "later" or "never" (Esc counts as later)."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⭐ Enjoying AIForge?", classes="title")
            yield Static("You just trained a model! If AIForge is useful to you, a star on GitHub helps "
                         "other people find it and keeps the project going.\n\n"
                         "[dim]This opens the project page in your browser - click ⭐ Star at the top "
                         "right. AIForge never sees your GitHub account.[/]")
            yield Horizontal(Button("Don't ask again", id="never"), Button("Maybe later", id="cancel"),
                             Button("⭐ Star on GitHub", id="ok", variant="primary"), classes="buttons")

    def on_mount(self) -> None:
        self.query_one("#ok", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "never":
            self.dismiss("never")

    def action_cancel(self) -> None:
        self.dismiss("later")

    def submit(self) -> None:
        self.dismiss("star")
