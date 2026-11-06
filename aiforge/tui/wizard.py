"""The goal wizard: say what you want in plain words, point at your data, get a ready model."""
from __future__ import annotations

from rich.markup import escape
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static

from .. import data as datamod
from ..spec import safe_name
from ..wizard import Plan, PlanError, plan
from .modals import Dialog, browse_into

EXAMPLES = ("predict house prices", "sort photos of cats and dogs", "is a review positive or negative?",
            "find groups of customers", "spot unusual transactions", "write poems like these",
            "recognise handwritten digits", "balance a pole on a cart")


class GoalDialog(Dialog):
    """Returns {"plan": Plan, "name": str, "open": "train" | "builder"} or None."""

    DEFAULT_CSS = """
    GoalDialog > Vertical { width: 100; }
    /* Once there is a plan the dialog takes the screen's height and the plan scrolls in what's left,
       so the buttons stay visible even in a small terminal. */
    GoalDialog > Vertical.planned { height: 90%; }
    GoalDialog > Vertical.planned .hint { display: none; }
    GoalDialog #plan-box { height: 1fr; margin-top: 1; border: round $primary-darken-2; padding: 0 1; }
    GoalDialog #plan-box.empty { display: none; }
    GoalDialog #make-plan { margin-top: 1; width: 100%; }
    GoalDialog #plan-row { height: auto; margin-top: 1; }
    GoalDialog #plan-row Label { margin: 1 1 0 0; width: auto; }
    GoalDialog #plan-row Select { width: 1fr; margin-right: 2; }
    GoalDialog #plan-row Input { width: 1fr; }
    """

    def __init__(self, ws, goal: str = "", path: str = ""):
        super().__init__()
        self.ws, self._goal, self._path = ws, goal, path
        self.plan: Plan | None = None
        self._planned_for: tuple[str, str] | None = None
        self._hub: dict | None = None  # the famous dataset picked in the hub, while it is the data
        self._suggested: str | None = None  # the hub dataset that fits the goal best

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("✨ Tell AIForge what you want", classes="title")
            yield Static("Say what the model should do and point at your data. AIForge picks the kind of "
                         "model, the label column and the settings, and tells you why.\n"
                         f"[dim]e.g. {' · '.join(EXAMPLES[:4])}[/]", classes="hint")
            yield Label("① What should the model do?")
            yield Input(self._goal, placeholder="e.g. sort photos of cats and dogs", id="goal")
            yield Label("② Choose your data  [dim]your own table, image folder or .txt file, or a famous dataset[/]")
            with Horizontal(classes="path-row"):
                yield Input(self._path, placeholder="~/data/houses.csv  (or: mnist)", id="data")
                yield Button("📂 Browse", id="browse")
                yield Button("🗄 Famous dataset", id="hub")
            yield Button("🔮 Make a plan", id="make-plan", variant="primary")
            with VerticalScroll(id="plan-box", classes="empty"):
                yield Static("", id="plan")
            with Horizontal(id="plan-row"):
                yield Label("Label column", id="label-title")
                yield Select([("-", "-")], allow_blank=False, id="label")
                yield Label("Name")
                yield Input("", id="name")
            yield Static("", id="error", classes="error")
            yield Horizontal(Button("Cancel", id="cancel"), Button("✎ Open in builder", id="builder", disabled=True),
                             Button("▶ Create & train", id="ok", variant="success", disabled=True),
                             classes="buttons")

    def on_mount(self) -> None:
        self.query_one("#plan-row").display = False
        self.query_one("#goal", Input).focus()

    def _inputs(self) -> tuple[str, str]:
        return self.query_one("#goal", Input).value.strip(), self.query_one("#data", Input).value.strip()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "browse":
            browse_into(self, self.query_one("#data", Input), title="Choose your data", mode="any",
                        extensions=datamod.TABLE_EXT | datamod.TEXT_EXT,
                        hint="Pick a table (.csv, .xlsx, .parquet), a .txt file to imitate, or a folder of "
                             "images (one sub-folder per class, e.g. cats/ and dogs/).")
        elif bid == "hub":
            self._open_hub()
        elif bid == "make-plan":
            self._make_plan()
        elif bid == "builder":
            self._finish("builder")

    def _open_hub(self) -> None:
        from .datasets import DatasetScreen

        if self._suggested is None and (goal := self._inputs()[0]):
            from ..wizard import suggest_datasets

            self._suggested = next(iter(suggest_datasets(goal)), None)
        self.app.push_screen(DatasetScreen("✓ Use this dataset", focus=self._suggested), self._use_dataset)

    def _use_dataset(self, cfg: dict | None) -> None:
        """A famous dataset from the hub: its file (or name, for the built-in image sets) becomes the
        data, and its known label column is used."""
        if not cfg:
            return
        self._hub = cfg
        self.query_one("#data", Input).value = cfg.get("path") or cfg["source"]
        if self._inputs()[0]:
            self._make_plan()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.prevent_default()  # instead of Dialog's own handler, which would submit
        if event.input.id == "name" and self.plan:
            return self._finish("train")
        self._make_plan()

    @on(Input.Changed, "#goal")
    @on(Input.Changed, "#data")
    def _inputs_changed(self) -> None:
        stale = self.plan is not None and self._inputs() != self._planned_for
        self.query_one("#ok", Button).disabled = self.query_one("#builder", Button).disabled = \
            self.plan is None or stale
        if stale:
            self.error("The goal or data changed: make a new plan.")

    @on(Input.Changed, "#goal")
    def _goal_changed(self) -> None:
        self._suggested = None  # suggest again for the new goal

    @on(Select.Changed, "#label")
    def _label_changed(self, event: Select.Changed) -> None:
        if self.plan and event.value not in (Select.BLANK, self.plan.label):
            self._make_plan(label=event.value)

    def _make_plan(self, label: str | None = None) -> None:
        goal, path = self._inputs()
        hub = self._hub if self._hub and path in (self._hub.get("path"), self._hub["source"]) else None
        if label is None and hub:
            label = hub.get("target")
        self.error("")
        self.query_one("#make-plan", Button).disabled = True
        self.query_one("#plan", Static).update("[dim]Looking at your data…[/]")
        self.query_one("#plan-box").remove_class("empty")
        self._plan_worker(goal, path, label)

    @work(thread=True, exclusive=True)
    def _plan_worker(self, goal: str, path: str, label: str | None) -> None:
        try:
            result = plan(goal, path or None, label)
        except PlanError as e:
            self.app.call_from_thread(self._failed, str(e), e.suggestions, e.needs_data)
            return
        except Exception as e:  # noqa: BLE001 - shown to the person, the dialog stays usable
            self.app.call_from_thread(self._failed, f"Something went wrong reading the data: {e}")
            return
        self.app.call_from_thread(self._show, result, (goal, path))

    def _failed(self, msg: str, suggestions: list[str] | None = None, needs_data: bool = False) -> None:
        if suggestions:
            self._suggested = suggestions[0]
        self.query_one("#make-plan", Button).disabled = False
        self.query_one("#plan-box").add_class("empty")
        if needs_data:  # a next step, not a mistake
            self.error(f"[yellow]💡 {escape(msg)}[/]")
            self.query_one("#hub" if suggestions else "#browse", Button).focus()
        else:
            self.error(escape(msg))

    def _show(self, result: Plan, planned_for: tuple[str, str]) -> None:
        keep_name = self.plan is not None and self.query_one("#name", Input).value.strip()
        self.plan, self._planned_for = result, planned_for
        self.query_one("#make-plan", Button).disabled = False
        self.query_one("#make-plan", Button).label = "🔮 New plan"
        self.query_one(Vertical).add_class("planned")
        self.query_one("#plan", Static).update(result.summary())
        self.query_one("#plan-row").display = True
        has_label = bool(result.columns and result.label)
        self.query_one("#label-title").display = self.query_one("#label", Select).display = has_label
        if result.columns and result.label:
            sel = self.query_one("#label", Select)
            with sel.prevent(Select.Changed):
                sel.set_options([(c, c) for c in result.columns])
                sel.value = result.label
        if not keep_name:
            self.query_one("#name", Input).value = self.ws.unique_name(result.name)
        self.query_one("#ok", Button).disabled = self.query_one("#builder", Button).disabled = False
        self.query_one("#ok", Button).focus()

    def submit(self) -> None:
        if self.plan:
            self._finish("train")
        else:
            self._make_plan()

    def _finish(self, open_in: str) -> None:
        if not self.plan or self._inputs() != self._planned_for:
            return self.error("Make a plan first.")
        name = safe_name(self.query_one("#name", Input).value)
        if not name:
            return self.error("Please enter a name (letters, numbers, - _ .)")
        if self.ws.exists(name):
            return self.error(f"A model named '{name}' already exists")
        self.dismiss({"plan": self.plan, "name": name, "open": open_in})
