"""Use a model: single predictions and batch prediction to CSV."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from rich.markup import escape
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Select, Static

from ..predict import describe, input_hint, predict_batch, predict_one
from ..spec import TASK_LABELS, is_reinforcement, is_unsupervised
from .evaluate import _names_word

BAR = 30


class PredictScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back"), Binding("ctrl+e", "explain", "Explain"),
                # Agents: play an episode yourself (ignored for other models)
                Binding("up", "human('up')", "You: up", show=False),
                Binding("down", "human('down')", "You: down", show=False),
                Binding("left", "human('left')", "You: left", show=False),
                Binding("right", "human('right')", "You: right", show=False)]
    SPEEDS = {"slow": 0.6, "normal": 0.25, "fast": 0.06}

    def __init__(self, name: str):
        super().__init__()
        self.model_name = name
        self.agent = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="predict"):
            yield Static("", id="predict-title", classes="section-title")
            yield Static("", id="predict-warn")
            yield Label("", id="hint")
            with Vertical(id="play-section"):  # agents: watch it play (or play yourself)
                with Horizontal(classes="input-row"):
                    yield Button("▶ Play", id="play", variant="success")
                    yield Button("⏭ Step", id="step")
                    yield Button("↻ New episode", id="new-episode")
                    yield Select([(k, k) for k in self.SPEEDS], value="normal", allow_blank=False, id="speed")
                with Horizontal(id="play-row"):
                    yield Static("", id="board")
                    yield Static("", id="scores")
                yield Static("", id="play-status")
            with Horizontal(classes="input-row", id="single-row"):
                yield Input(id="raw", placeholder="type an input and press Enter")
                yield Button("📂 Browse", id="browse-raw")
                yield Button("Predict", id="go", variant="primary")
                yield Button("🔍 Explain", id="explain", tooltip="Why this answer? (Ctrl+E)")
            with Horizontal(id="gen-row"):
                yield Label("Length [dim](tokens)[/]")
                yield Input("200", id="gen-length", type="integer")
                yield Label("Temperature [dim](0 = safest, 1.5 = wild)[/]")
                yield Input("0.8", id="gen-temp", type="number")
            yield Static("", id="result")
            yield Static("", id="explanation")
            with Vertical(id="batch-section"):
                yield Static("Batch prediction", classes="section-title")
                yield Static("A CSV (one input per row) or a folder of images → a CSV with a prediction column.",
                             classes="hint", id="batch-hint")
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
            f"Use [b]{self.model_name}[/b] · {TASK_LABELS.get(self.spec.task, self.spec.task)}"
            + (f" · {len(self.spec.class_names)} {_names_word(self.spec.task)}"
               if self.spec.class_names else ""))
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
        self.unsupervised = is_unsupervised(self.spec.task)
        self.query_one("#explain").display = not self.is_lm and not self.unsupervised
        if self.unsupervised:
            self.query_one("#go", Button).label = "Check"
            what = "its group" if self.spec.task == "clustering" else "its rebuild error"
            self.query_one("#batch-hint", Static).update(
                f"A CSV (one input per row) or a folder of images → a CSV with {what} and how unusual "
                "each one is (sort it by unusual_percentile to find the oddities).")
        if self.is_lm:
            self.query_one("#raw", Input).placeholder = "e.g. Twinkle, twinkle  (Enter = write)"
            self.query_one("#go", Button).label = "✎ Write"
        self.query_one("#play-section").display = False
        if is_reinforcement(self.spec.task):
            self._setup_agent()
            return
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
            if self.unsupervised:
                return box.update(self._describe(describe(self.spec, self.model, raw)))
            res = predict_one(self.spec, self.model, raw)
        except Exception as e:  # noqa: BLE001
            box.update(f"[red]{escape(str(e))}[/]")
            return
        if self.spec.task == "classification":
            lines = [f"[b]→ {res[0][0]}[/b]  ({res[0][1]:.1%} confident)", ""]
            for label, p in res:
                bar = "█" * round(p * BAR)
                lines.append(f"  {label[:20]:<20} [cyan]{bar:<{BAR}}[/] {p:6.1%}")
            box.update("\n".join(lines))
        else:
            box.update("\n".join(f"[b]{label}: {v:.6g}[/b]" for label, v in res))

    # -- agents ------------------------------------------------------------------

    def _setup_agent(self) -> None:
        from .. import rl

        self.agent = True
        for wid in ("#single-row", "#gen-row", "#batch-section", "#result", "#explanation", "#predict-log"):
            self.query_one(wid).display = False
        self.query_one("#play-section").display = True
        self.env = rl.make_env(self.spec.data.get("preprocess") or self.spec.data.get("env") or {})
        keys = "↑ ↓ ← →" if self.env.name == "gridworld" else "← →"
        self.query_one("#hint", Label).update(f"Watch the agent play, or press {keys} to play yourself.")
        if self.app.ws.has_weights(self.model_name) and self.spec.data.get("preprocess"):  # else: a warning
            self.query_one("#predict-warn", Static).update(f"[dim]{self.env.describe()}[/]")
        self.timer = self.set_interval(self.SPEEDS["normal"], self._tick, pause=True)
        self.playing = False
        self._new_episode()
        self.query_one("#play", Button).focus()

    def _new_episode(self) -> None:
        self.obs, self.total, self.steps, self.done, self.who = self.env.reset(), 0.0, 0, False, set()
        self._draw("A new episode: press ▶ Play or ⏭ Step.")

    def _draw(self, status: str) -> None:
        from rich.markup import escape

        from .. import rl

        self.query_one("#board", Static).update(escape(self.env.render(path=self.done)))
        q = rl.q_values(self.model, self.obs)
        lo, hi = float(q.min()), float(q.max())
        lines = ["[b]How the agent rates each action now[/b]  [dim](higher = better; it picks →)[/]"]
        for i in q.argsort(descending=True).tolist():
            bar = "█" * max(1, round((float(q[i]) - lo) / ((hi - lo) or 1) * 20))
            mark = "[b]→[/b]" if float(q[i]) == hi else " "
            lines.append(f"{mark} {self.env.arrows[i]} {self.env.actions[i]:<11}[cyan]{bar:<20}[/] {float(q[i]):7.3g}")
        self.query_one("#scores", Static).update("\n".join(lines))
        self.query_one("#play-status", Static).update(
            f"step {self.steps} · total reward [b]{self.total:.3g}[/b]   {status}")

    def _act(self, action: int, who: str) -> None:
        if self.done:
            self._new_episode()
        s = self.env.step(action)
        self.obs, self.total, self.steps = s.obs, self.total + s.reward, self.steps + 1
        self.who.add(who)
        self.done = s.done
        status = f"{'🤖 agent' if who == 'agent' else '🙂 you'}: {self.env.actions[action]} ({s.reward:+.3g})"
        if s.done:
            self._pause()
            player = "The agent" if self.who == {"agent"} else "You" if self.who == {"you"} else "Together you"
            status = f"🏁 {player} {s.event} · reward {self.total:.3g} in {self.steps} steps. ↻ for another try."
        self._draw(status)

    def _tick(self) -> None:
        from .. import rl

        self._act(int(rl.q_values(self.model, self.obs).argmax()), "agent")

    def _pause(self) -> None:
        self.timer.pause()
        self.playing = False
        self.query_one("#play", Button).label = "▶ Play"

    @on(Button.Pressed, "#play")
    def _play(self) -> None:
        if self.playing:
            return self._pause()
        if self.done:
            self._new_episode()
        self.timer.resume()
        self.playing = True
        self.query_one("#play", Button).label = "⏸ Pause"

    @on(Button.Pressed, "#step")
    def _step(self) -> None:
        self._tick()

    @on(Button.Pressed, "#new-episode")
    def _restart(self) -> None:
        self._pause()
        self._new_episode()

    @on(Select.Changed, "#speed")
    def _speed(self, event: Select.Changed) -> None:
        if not self.agent:
            return
        self.timer.stop()
        self.timer = self.set_interval(self.SPEEDS[str(event.value)], self._tick, pause=not self.playing)

    def action_human(self, key: str) -> None:
        if not self.agent:
            return
        if key not in self.env.actions and not (self.env.name == "cartpole" and key in ("left", "right")):
            return
        self._pause()
        self._act(self.env.actions.index(key) if key in self.env.actions else ["left", "right"].index(key), "you")

    def _describe(self, d: dict) -> str:
        """Group or rebuild error, how unusual the input is, and its compressed code."""
        if self.spec.task == "clustering":
            label, dist = d["pairs"][0]
            lines = [f"[b]→ {label}[/b]  (distance {dist:.4g} to its centre)", d["verdict"], ""]
            far = max(v for _, v in d["pairs"]) or 1
            for name, v in d["pairs"][:8]:
                bar = "█" * max(1, round((1 - v / far) * BAR))
                lines.append(f"  {name:<10} [cyan]{bar:<{BAR}}[/] distance {v:.4g}")
            lines.append("[dim]  longer bar = closer to that group's centre[/]")
            return "\n".join(lines)
        name = "component" if self.spec.task == "pca" else "code"
        code = " · ".join(f"{v:.3g}" for v in d["code"][:8]) + (" …" if len(d["code"]) > 8 else "")
        return "\n".join([f"[b]rebuild error {d['score']:.4g}[/b]", d["verdict"], "",
                          f"{name}{'s' if len(d['code']) > 1 else ''}: {code}",
                          "[dim]what the model squeezed this input into[/]"])

    def action_explain(self) -> None:
        self._explain()

    @on(Button.Pressed, "#explain")
    def _explain(self) -> None:
        raw = self.query_one("#raw", Input).value
        if self.is_lm or self.unsupervised or not raw.strip():
            return
        self._predict()  # show the answer that is being explained
        box = self.query_one("#explanation", Static)
        box.display = True
        box.update("[dim]Working out why…[/]")
        self._run_explain(raw)

    @work(thread=True, exclusive=True, group="explain")
    def _run_explain(self, raw: str) -> None:
        from ..explain import explain, render

        try:
            text = render(explain(self.spec, self.model, raw))
        except Exception as e:  # noqa: BLE001 - shown to the user
            text = f"[red]Can't explain this: {e}[/]"
        self.app.call_from_thread(self.query_one("#explanation", Static).update, text)

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
