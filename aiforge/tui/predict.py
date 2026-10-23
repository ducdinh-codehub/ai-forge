"""Use a model: single predictions and batch prediction to CSV."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from rich.markup import escape
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
                yield Button("📂 Browse", id="browse-raw")
                yield Button("Predict", id="go", variant="primary")
            with Horizontal(id="gen-row"):
                yield Label("Length [dim](tokens)[/]")
                yield Input("200", id="gen-length", type="integer")
                yield Label("Temperature [dim](0 = safest, 1.5 = wild)[/]")
                yield Input("0.8", id="gen-temp", type="number")
            yield Static("", id="result")
            with Vertical(id="batch-section"):
                yield Static("Batch prediction", classes="section-title")
                yield Static("A CSV (one input per row) or a folder of images → a CSV with a prediction column.",
                             classes="hint")
                with Horizontal(classes="input-row"):
                    yield Input(id="batch-src", placeholder="~/data/new_rows.csv  or  ~/photos/")
                    yield Button("📂 Browse", id="browse-batch")
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
        self.is_image = self.spec.data.get("preprocess", {}).get("type") == "image" or \
            len(self.spec.input_shape) == 3
        self.query_one("#browse-raw").display = self.is_image
        self.is_lm = self.spec.task == "language_model"
        self.query_one("#gen-row").display = self.is_lm
        self.query_one("#batch-section").display = not self.is_lm
        if self.is_lm:
            self.query_one("#raw", Input).placeholder = "e.g. Twinkle, twinkle  (Enter = write)"
            self.query_one("#go", Button).label = "✎ Write"
        self.query_one("#raw", Input).focus()

    @on(Input.Submitted, "#raw")
    @on(Button.Pressed, "#go")
    def _predict(self) -> None:
        raw = self.query_one("#raw", Input).value
        if self.is_lm:
            return self._write(raw)
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

    # -- language models -------------------------------------------------------

    def _write(self, prompt: str) -> None:
        try:
            length = int(self.query_one("#gen-length", Input).value or 200)
            temp = float(self.query_one("#gen-temp", Input).value or 0.8)
            if not 1 <= length <= 5000 or temp < 0:
                raise ValueError
        except ValueError:
            self.query_one("#result", Static).update(
                "[red]Length must be 1-5000 tokens and temperature 0 or more.[/]")
            return
        self.query_one("#go", Button).disabled = True
        self.query_one("#result", Static).update("[yellow]Writing…[/]")
        self._generate(prompt, length, temp)

    @work(thread=True, exclusive=True, group="generate")
    def _generate(self, prompt: str, length: int, temp: float) -> None:
        from ..textgen import generate

        try:
            guesses = predict_one(self.spec, self.model, prompt)
            text = generate(self.model, self.spec.data["preprocess"], prompt, length, temp)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._written, prompt, None, [], str(e))
            return
        self.app.call_from_thread(self._written, prompt, text, guesses, None)

    def _written(self, prompt: str, text: str | None, guesses, error: str | None) -> None:
        self.query_one("#go", Button).disabled = False
        box = self.query_one("#result", Static)
        if error:
            box.update(f"[red]{escape(error)}[/]")
            return
        lines = [f"[b]{escape(prompt)}[/b][cyan]{escape(text)}[/]", "",
                 "[dim]Bold = your prompt, blue = written by the model. Each run differs (it samples); "
                 "temperature 0 always gives the same text.[/]", "", "[b]Most likely next token[/b]"]
        for token, p in guesses:
            bar = "█" * round(p * BAR)
            lines.append(f"  {escape(token[:20]):<20} [cyan]{bar:<{BAR}}[/] {p:6.1%}")
        box.update("\n".join(lines))
        self.query_one("#predict-log", RichLog).write(
            f"[dim]temperature {self.query_one('#gen-temp', Input).value}:[/] "
            f"[b]{escape(prompt)}[/b]{escape(text)}")

    @on(Button.Pressed, "#browse-raw")
    def _browse_raw(self) -> None:
        from .browser import FileBrowser
        from ..data import IMAGE_EXT

        def done(path):
            if path:
                self.query_one("#raw", Input).value = path
                self._predict()
        self.app.push_screen(FileBrowser("Choose an image to classify", mode="file",
                                         extensions=IMAGE_EXT,
                                         start=self.query_one("#raw", Input).value.strip() or None), done)

    @on(Button.Pressed, "#browse-batch")
    def _browse_batch(self) -> None:
        from .browser import CSV_EXT, FileBrowser

        def done(path):
            if path:
                self.query_one("#batch-src", Input).value = path
        self.app.push_screen(FileBrowser(
            "Choose data to predict", mode="dir" if self.is_image else "file",
            extensions=None if self.is_image else CSV_EXT,
            start=self.query_one("#batch-src", Input).value.strip() or None,
            hint="Pick a folder of images." if self.is_image else
                 "Pick a CSV file with one input per row."), done)

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
