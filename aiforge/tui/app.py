"""AIForge terminal application."""
from __future__ import annotations

from pathlib import Path

from textual.app import App, SystemCommand
from textual.binding import Binding

from .. import feedback
from ..spec import Workspace
from .home import HomeScreen

DARK_THEME = "textual-dark"
LIGHT_THEME = "textual-light"


class AIForgeApp(App):
    TITLE = "AIForge"
    SUB_TITLE = "no-code neural network studio"
    CSS_PATH = "styles.tcss"
    BINDINGS = [Binding("ctrl+t", "toggle_theme", "Dark/Light"),
                Binding("ctrl+o", "report_issue", "Report issue")]

    def __init__(self, workspace: Path):
        super().__init__()
        self.ws = Workspace(workspace)
        self.errors = feedback.ErrorLog()  # attached to bug reports
        self.last_crash: str | None = None
        self.tour = None  # the guided first run, while it is on

    def on_mount(self) -> None:
        self.sub_title = f"workspace: {self.ws.root}"
        saved = self.ws.get_setting("theme")
        if saved in self.available_themes:
            self.theme = saved
        # Remember the choice whether it came from the button or the command palette.
        self.theme_changed_signal.subscribe(self, lambda t: self.ws.set_setting("theme", t.name))
        self.push_screen(HomeScreen())
        self.last_crash = feedback.take_crash(self.ws.root)
        if self.last_crash:
            self.notify("AIForge closed because of an error last time. Press [b]Ctrl+O[/b] to report it - "
                        "the error details are filled in for you.", severity="warning", timeout=20)

    def action_tour(self) -> None:
        from .tour import start

        start(self)

    def action_agent_tour(self) -> None:
        """Walk through a first reinforcement-learning agent. On an agent's Train screen it starts
        there; anywhere else it starts on the home screen."""
        from .tour import AGENT_STEPS, start

        screen = self.screen
        if type(screen).__name__ == "TrainScreen" and getattr(screen, "rl", False):
            return start(self, AGENT_STEPS, first=next(i for i, s in enumerate(AGENT_STEPS)
                                                       if s.screen == "TrainScreen"))
        if type(screen).__name__ != "HomeScreen":
            self.notify("The agent tour starts on the home screen: press Esc to go back there.", timeout=8)
        start(self, AGENT_STEPS)

    def action_wizard(self) -> None:
        home = next((s for s in self.screen_stack if isinstance(s, HomeScreen)), None)
        if home is None or self.screen is not home:
            self.notify("The goal wizard starts on the home screen: press Esc to go back there.", timeout=8)
            return
        home.action_wizard()

    def action_toggle_theme(self) -> None:
        self.theme = LIGHT_THEME if self.current_theme.dark else DARK_THEME

    def action_star(self) -> None:
        from .star import open_repo

        open_repo(self)

    # -- reporting problems ------------------------------------------------------

    def notify(self, message, *, title="", severity="information", timeout=None, markup=True) -> None:
        if severity == "error":
            self.errors.add(f"{title}: {message}" if title else message)
        super().notify(message, title=title, severity=severity, timeout=timeout, markup=markup)

    def _handle_exception(self, error: Exception) -> None:
        feedback.save_crash(self.ws.root, error)  # offered for a report at the next start
        super()._handle_exception(error)

    def get_system_commands(self, screen):
        yield from super().get_system_commands(screen)
        yield SystemCommand("Report a problem or idea", "Open a GitHub issue with the details filled in",
                            self.action_report_issue)
        yield SystemCommand("Describe what you want (goal wizard)", "Say what the model should do and point at "
                            "your data: AIForge picks the model and settings", self.action_wizard)
        yield SystemCommand("Take the guided tour", "Build, train and use a first model, step by step",
                            self.action_tour)
        yield SystemCommand("Take the agent tour", "Train a reinforcement-learning agent, watch it play and "
                            "test it, step by step", self.action_agent_tour)
        yield SystemCommand("Star AIForge on GitHub", "Open the project page so you can give it a ⭐ - thank you!",
                            self.action_star)

    def _current_spec(self):
        """The model the user is looking at, so a report can include its structure."""
        for screen in reversed(self.screen_stack):
            if isinstance(screen, HomeScreen):
                name = screen.selected
                return self.ws.load_spec(name) if name else None
            if getattr(screen, "spec", None) is not None:
                return screen.spec
        return None

    def action_report_issue(self) -> None:
        from .report import ReportDialog

        if isinstance(self.screen, ReportDialog):
            return
        try:
            spec = self._current_spec()
            model_text = feedback.model_summary(spec) if spec else ""
        except Exception:  # noqa: BLE001 - a broken model must not stop the user reporting it
            spec, model_text = None, ""

        def done(path):
            if path:
                self.last_crash = None
                self.notify(f"Opened GitHub in your browser - check the issue and click Submit. Didn't open? "
                            f"The report is on your clipboard and saved in {feedback.scrub(str(path))}; "
                            f"paste it at {feedback.NEW_ISSUE}", timeout=20)
        self.push_screen(ReportDialog(spec.name if spec else None, model_text, self.errors.markdown(),
                                      self.last_crash or ""), done)
