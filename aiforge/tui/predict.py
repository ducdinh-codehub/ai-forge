"""Use a model: single predictions and batch prediction to CSV."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from ..predict import input_hint, predict_batch, predict_one

BAR = 30


class PredictScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, name: str):
        super().__init__()
        self.model_name = name

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="predict"):
            yield Static("", id="predict-title", classes="section-title")
            yield Static("", id="predict-warn")
            yield Label("", id="hint")
            with Horizontal(classes="input-row"):
                yield Input(id="raw", placeholder="type an input and press Enter")
                yield Button("Predict", id="go", variant="primary")
            yield Static("", id="result")
            yield Static("Batch prediction", classes="section-title")
            yield Static("A CSV (one input per row) or a folder of images → a CSV with a prediction column.",
                         classes="hint")
            with Horizontal(classes="input-row"):
                yield Input(id="batch-src", placeholder="~/data/new_rows.csv  or  ~/photos/")
                yield Input(id="batch-dst", placeholder="output .csv (optional)")
                yield Button("Run batch", id="batch")
            yield RichLog(id="predict-log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        ws = self.app.ws
        self.spec, self.model, note = ws.load_model(self.model_name)
        self.model.eval()
        self.query_one("#predict-title", Static).update(
            f"Use [b]{self.model_name}[/b] · {self.spec.task}"
            + (f" · {len(self.spec.class_names)} classes" if self.spec.class_names else ""))
        if not ws.has_weights(self.model_name):
            self.query_one("#predict-warn", Static).update(
                "[yellow]⚠ This model has not been trained - predictions are random.[/]")
        if not self.spec.data.get("preprocess"):
            self.query_one("#predict-warn", Static).update(
                "[yellow]⚠ No preprocessing info saved (model wasn't trained here) - "
                "inputs are used as-is.[/]")
        self.query_one("#hint", Label).update(f"Input: {input_hint(self.spec)}")
        self.query_one("#raw", Input).focus()

    @on(Input.Submitted, "#raw")
    @on(Button.Pressed, "#go")
    def _predict(self) -> None:
        raw = self.query_one("#raw", Input).value
        if not raw.strip():
            return
        box = self.query_one("#result", Static)
        try:
            res = predict_one(self.spec, self.model, raw)
        except Exception as e:  # noqa: BLE001
            box.update(f"[red]{e}[/]")
            return
        if self.spec.task == "classification":
            lines = [f"[b]→ {res[0][0]}[/b]  ({res[0][1]:.1%} confident)", ""]
            for label, p in res:
                bar = "█" * round(p * BAR)
                lines.append(f"  {label[:20]:<20} [cyan]{bar:<{BAR}}[/] {p:6.1%}")
            box.update("\n".join(lines))
        else:
            box.update("\n".join(f"[b]{label}: {v:.6g}[/b]" for label, v in res))

    @on(Button.Pressed, "#batch")
    def _batch(self) -> None:
        src = self.query_one("#batch-src", Input).value.strip()
        if not src:
            self.notify("Enter a CSV file or image folder.", severity="warning")
            return
        self.query_one("#predict-log", RichLog).write(f"Predicting {src}…")
        self._run_batch(src, self.query_one("#batch-dst", Input).value.strip() or None)

    @work(thread=True, exclusive=True)
    def _run_batch(self, src: str, dst: str | None) -> None:
        log = self.query_one("#predict-log", RichLog)
        try:
            out, n = predict_batch(self.spec, self.model, src, dst)
            self.app.call_from_thread(log.write, f"[green]✓ {n:,} predictions written to {out}[/]")
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(log.write, f"[red]Batch failed: {e}[/]")
