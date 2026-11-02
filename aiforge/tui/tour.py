"""The guided first run: the 5-minute quick start, one highlighted step at a time.

A coach bar at the bottom of the screen says what to do next and the widget to use is outlined.
Each step finishes by itself when the user has done it (a dialog opened, data loaded, training
done...), so the tour never gets in the way of using the app normally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Static

TARGET = "tour-target"  # CSS class that outlines the widget to use


def show(screen, coach) -> None:
    """Put the bar at the bottom of the screen, above the footer's key hints. Docked widgets
    overlap, so with a footer the bar joins the normal layout (the content shrinks a little)."""
    footer = next(iter(screen.query(Footer)), None)
    if footer is not None:
        screen.mount(coach, before=footer)
    else:  # dialogs: no footer, dock it
        coach.add_class("docked")
        screen.mount(coach)


@dataclass
class Step:
    screen: str  # class name of the screen the step happens on
    target: str | None  # the widget to outline
    text: str
    done: Callable  # (app) -> bool, checked on any screen
    enter: Callable | None = None  # (screen) -> None, once, e.g. to prefill a field


def _on(name: str) -> Callable:
    return lambda app: type(app.screen).__name__ == name


def _model_trained(app) -> bool:
    s = app.screen
    return type(s).__name__ == "TrainScreen" and s.trainer is None and bool(app.ws.load_history(s.model_name))


def _fit(app) -> bool:
    from ..trainer import data_mismatch

    s = app.screen
    return type(s).__name__ == "TrainScreen" and s.info is not None and not data_mismatch(s.spec, s.info)


def _answered(app) -> bool:
    s = app.screen
    return type(s).__name__ == "PredictScreen" and "→" in str(s.query_one("#result").render())


def _name_it(dialog) -> None:
    dialog.query_one("#name").value = dialog.ws.unique_name("spiral-classifier")


def _spirals(screen) -> None:
    screen.query_one("#source").value = "synthetic"
    screen.query_one("#kind").value = "spirals"
    screen.query_one("#epochs").value = "40"
    screen.query_one("#lr").value = "0.005"


STEPS = [
    Step("HomeScreen", "#new", "[b]Step 1 of 5 · Create.[/b] Press [reverse] n [/reverse] (or click ✚ New) "
         "to make your first model.", _on("NewModelDialog")),
    Step("NewModelDialog", "#ok", "Keep [b]Tabular data → MLP classifier[/b]: a small network for tables of "
         "numbers. Click [b]Create & open builder[/b].", _on("BuilderScreen"), _name_it),
    Step("BuilderScreen", "#stack", "[b]Step 2 of 5 · Look.[/b] This is your network: every layer, its output "
         "shape and its number of parameters. Press [reverse] Ctrl+S [/reverse] to save it, then "
         "[reverse] Esc [/reverse] to go back.", _on("HomeScreen")),
    Step("HomeScreen", "#train", "[b]Step 3 of 5 · Train.[/b] Press [reverse] t [/reverse] (▶ Train) to "
         "teach it with examples.", _on("TrainScreen")),
    Step("TrainScreen", "#load", "The built-in [b]spirals[/b] toy data is picked: three spiral arms of "
         "points to tell apart. Click [b]Load data[/b].", lambda app: getattr(app.screen, "info", None) is not None,
         _spirals),
    Step("TrainScreen", "#autofit", "The model's input and output don't match this data yet. Click "
         "[b]Auto-fit model to data[/b].", _fit),
    Step("TrainScreen", "#start", "Click [b]▶ Start[/b] and watch the charts: loss going [b]down[/b] and "
         "accuracy going [b]up[/b] means it is learning.", _model_trained),
    Step("TrainScreen", None, "[b]Trained![/b] AIForge saved the best epoch. Press [reverse] Esc [/reverse] "
         "to go back.", _on("HomeScreen")),
    Step("HomeScreen", "#predict", "[b]Step 4 of 5 · Use.[/b] Press [reverse] p [/reverse] (◉ Use) to ask "
         "your model something.", _on("PredictScreen")),
    Step("PredictScreen", "#raw", "Type a point, [b]0.1, 0.45[/b], and press [reverse] Enter [/reverse]: "
         "which spiral arm is it on?", _answered),
    Step("PredictScreen", "#explain", "[b]Step 5 of 5 · 🎉 Done![/b] You built, trained and used a neural "
         "network. Try [b]🔍 Explain[/b] to see why it answered that; back home, [reverse] x [/reverse] "
         "exports it and [reverse] g [/reverse] has the lessons.", lambda app: False),
]


class Coach(Horizontal):
    """The bar at the bottom of the screen: what to do next, and a way out."""

    DEFAULT_CSS = """
    Coach { height: auto; padding: 0 1; background: $boost; border-top: thick $warning; }
    Coach.docked { dock: bottom; }
    Coach #coach-text { width: 1fr; padding: 1 1; }
    Coach Button { margin: 1 0 0 1; min-width: 12; }
    """

    def __init__(self, text: str, buttons: list[tuple[str, str]]):
        super().__init__(classes="coach")  # a class, not an id: an old bar may still be on its way out
        self.text, self.choices = text, buttons

    def compose(self) -> ComposeResult:
        yield Static(self.text, id="coach-text")
        for label, button_id in self.choices:
            yield Button(label, id=button_id, variant="default" if button_id in ("tour-end", "tour-no") else "warning")

    def say(self, text: str, end_label: str) -> None:
        self.text = text
        self.query_one("#coach-text", Static).update(text)
        self.query_one("#tour-end", Button).label = end_label

    @on(Button.Pressed)
    async def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        app = self.app
        if event.button.id == "tour-start":
            await self.remove()
            start(app)
        elif event.button.id == "tour-no":
            self.remove()
        elif event.button.id == "tour-end" and getattr(app, "tour", None):
            app.tour.end()


class Tour:
    def __init__(self, app):
        self.app, self.i = app, 0
        self.coach: Coach | None = None
        self.coach_screen = None  # where the bar was mounted (it attaches a moment later)
        self.target = None
        self.entered = -1
        self.timer = app.set_interval(0.2, self.tick)
        self.tick()

    @property
    def step(self) -> Step:
        return STEPS[self.i]

    def _done(self) -> bool:
        try:
            return self.step.done(self.app)
        except NoMatches:  # the screen is still being built
            return False

    def tick(self) -> None:
        while self.i < len(STEPS) - 1 and self._done():
            self._clear_target()
            self.i += 1
        screen = self.app.screen
        if type(screen).__name__ != self.step.screen:
            self._detach()  # e.g. a file browser opened on top: wait until we are back
            return
        if self.entered != self.i and self.step.enter:
            try:
                self.step.enter(screen)
            except NoMatches:
                return  # widgets not mounted yet: try again on the next tick
        self.entered = self.i
        end = "Finish" if self.i == len(STEPS) - 1 else "Skip tour"
        if self.coach is None or self.coach_screen is not screen:
            self._detach()
            self.coach, self.coach_screen = Coach(self.step.text, [(end, "tour-end")]), screen
            show(screen, self.coach)
        elif self.coach.text != self.step.text and self.coach.is_mounted:
            self.coach.say(self.step.text, end)
        self._point_at(screen)

    def _point_at(self, screen) -> None:
        try:
            target = screen.query_one(self.step.target) if self.step.target else None
        except NoMatches:
            target = None
        if target is not self.target:
            self._clear_target()
            if target is not None:
                target.add_class(TARGET)
            self.target = target

    def _clear_target(self) -> None:
        if self.target is not None:
            self.target.remove_class(TARGET)
            self.target = None

    def _detach(self) -> None:
        self._clear_target()
        if self.coach is not None and self.coach.is_attached:
            self.coach.remove()
        self.coach = self.coach_screen = None

    def end(self) -> None:
        self.timer.stop()
        self._detach()
        self.app.tour = None


def start(app) -> Tour:
    if getattr(app, "tour", None):
        app.tour.end()
    app.ws.set_setting("tour_offered", True)
    app.tour = Tour(app)
    return app.tour


def offer(screen) -> None:
    """On the very first launch (no models yet), offer the tour once."""
    ws = screen.app.ws
    if ws.list_models() or ws.get_setting("tour_offered"):
        return
    ws.set_setting("tour_offered", True)
    show(screen, Coach("[b]New here?[/b] Build, train and use your first model in 5 minutes, one "
                       "highlighted step at a time. (Restart it any time with [reverse] F2 [/reverse].)",
                       [("▶ Start the tour", "tour-start"), ("No thanks", "tour-no")]))
