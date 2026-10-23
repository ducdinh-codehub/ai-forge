"""Evaluate a trained model: pick the data and checks, read the report, save it."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown, ProgressBar, Select, Static, Switch

from .. import data as datamod
from .. import evaluate as ev
from ..trainer import pick_device

INTRO = """# Evaluate a model

Find out how good the model **really** is, on examples it never trained on.

1. Choose the data: the **validation split** kept aside during training (or the official
   test set for MNIST/CIFAR), or **a different file / folder** with the true answers.
2. Click **▶ Evaluate**.

**What you get**

- **Classification:** accuracy, balanced accuracy, precision / recall / F1 per class,
  ROC-AUC, PR-AUC, the confusion matrix, log loss and calibration
- **Regression:** MAE, RMSE, MSE, R², MAPE, median and largest error
- **Language models:** perplexity, next-token accuracy, the hardest tokens and samples of
  what it writes
- A **baseline** (always guessing the most common class / the average) to compare with
- **95% confidence intervals**, so you know how much to trust the numbers
- **Error analysis:** the model's worst mistakes, to look for patterns
- **Slices:** results per group for tables with category columns (fairness check)
- **Robustness:** how much a little noise in the inputs hurts
- **Learning curves:** over- or underfitting, from the training history
- **Compare with another model**, with a significance test (McNemar / paired bootstrap)
"""


class EvaluateScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back"),
                Binding("ctrl+r", "run", "Evaluate", priority=True)]

    def __init__(self, name: str):
        super().__init__()
        self.model_name = name
        self.report: dict | None = None

    @property
    def ws(self):
        return self.app.ws

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="eval-main"):
            with VerticalScroll(id="eval-config"):
                yield Static("① Data", classes="section-title")
                yield Select([("Validation / test data from training", "val"),
                              ("A different file / folder", "file")], value="val", allow_blank=False,
                             id="eval-source")
                with Vertical(id="grp-eval-file", classes="group"):
                    yield Label("", id="eval-path-label")
                    with Horizontal(classes="path-row"):
                        yield Input(id="eval-path")
                        yield Button("📂 Browse", id="eval-browse")
                    with Vertical(id="grp-eval-target", classes="group"):
                        yield Label("Target column  [dim](the true answers)[/]")
                        yield Select([], prompt="last column", id="eval-target")
                    with Vertical(id="grp-eval-text", classes="group"):
                        yield Label("Text column")
                        yield Select([], prompt="first other column", id="eval-text")
                yield Static("② Checks", classes="section-title")
                with Horizontal(classes="switch-row", id="row-robust"):
                    yield Switch(True, id="eval-robust")
                    yield Label("Robustness test (adds noise)")
                yield Label("Compare with another model")
                yield Select([], prompt="none", id="eval-other")
                yield Button("▶ Evaluate", id="eval-go", variant="success")
                yield Button("💾 Save report", id="eval-save", disabled=True)
                yield Static("", id="eval-history", classes="hint")
            with Vertical(id="eval-result"):
                yield Static("", id="eval-title", classes="section-title")
                yield Static("", id="eval-status")
                yield ProgressBar(id="eval-progress", show_eta=False)
                with VerticalScroll(id="eval-body"):
                    yield Markdown(INTRO, id="eval-report")
        yield Footer()

    def on_mount(self) -> None:
        self.spec = self.ws.load_spec(self.model_name)
        pre = self.spec.data.get("preprocess", {})
        self.kind = pre.get("type")
        self.query_one("#eval-title", Static).update(
            f"Evaluate [b]{self.model_name}[/b] · {self.spec.task}"
            + (f" · {len(self.spec.class_names)} classes" if self.spec.class_names else ""))
        warn = ""
        if not self.ws.has_weights(self.model_name):
            warn = "[yellow]⚠ This model isn't trained yet: expect random results.[/]"
        elif not pre:
            warn = "[yellow]⚠ Not trained in AIForge: evaluate on a file in the format it expects.[/]"
        self.query_one("#eval-status", Static).update(warn)
        self.query_one("#eval-progress").display = False
        if not self.spec.data.get("dataset_cfg"):
            self.query_one("#eval-source", Select).value = "file"
        self.query_one("#row-robust").display = self.spec.input_dtype == "float"
        others = [(r["name"], r["name"]) for r in self.ws.list_models()
                  if r["trained"] and ev.compatible(self.spec, self.ws.load_spec(r["name"]))]
        sel = self.query_one("#eval-other", Select)
        sel.set_options(others)
        sel.disabled = not others
        if not others:
            sel.prompt = "none (no compatible trained model)"
        self._show_history()
        self._update_fields()

    def _show_history(self) -> None:
        import json

        try:
            hist = json.loads((self.ws.model_dir(self.model_name) / "evaluations.json").read_text())
        except (OSError, ValueError):
            hist = []
        lines = []
        for h in hist[-5:][::-1]:
            score = (f"acc {h['accuracy']:.1%} · F1 {h['macro_f1']:.3f}" if "accuracy" in h else
                     f"perplexity {h['perplexity']:.2f} · next-token acc {h['token_accuracy']:.1%}"
                     if "perplexity" in h else
                     f"MAE {h['mae']:.4g} · R² {h['r2']:.3f}" if h.get("r2") is not None else f"MAE {h['mae']:.4g}")
            lines.append(f"{h['date']}  {score}")
        self.query_one("#eval-history", Static).update(
            ("Earlier evaluations\n" + "\n".join(lines)) if lines else "")

    # -- data fields -----------------------------------------------------------

    @on(Select.Changed, "#eval-source")
    def _update_fields(self) -> None:
        file = self.query_one("#eval-source", Select).value == "file"
        self.query_one("#grp-eval-file").display = file
        image = self.kind == "image" or (self.kind is None and len(self.spec.input_shape) == 3)
        lm = self.spec.task == "language_model"
        self.query_one("#eval-path-label", Label).update(
            "Folder  [dim](one sub-folder per class)[/]" if image else
            "Text file  [dim](.txt the model hasn't seen)[/]" if lm else "CSV file  [dim](with the true answers)[/]")
        self.query_one("#eval-path", Input).placeholder = \
            "~/data/test_photos/" if image else "~/data/other_book.txt" if lm else "~/data/test.csv"
        self.query_one("#grp-eval-target").display = not image and not lm
        self.query_one("#grp-eval-text").display = self.kind == "text"
        self._refresh_columns()

    @on(Input.Changed, "#eval-path")
    def _refresh_columns(self) -> None:
        cols = datamod.csv_columns(self.query_one("#eval-path", Input).value)
        for sel_id in ("#eval-target", "#eval-text"):
            sel = self.query_one(sel_id, Select)
            current = sel.value
            sel.set_options([(c, c) for c in cols])
            if current in cols:
                sel.value = current

    @on(Button.Pressed, "#eval-browse")
    def _browse(self) -> None:
        from .browser import CSV_EXT, FileBrowser

        current = self.query_one("#eval-path", Input).value.strip() or None
        if self.kind == "image" or (self.kind is None and len(self.spec.input_shape) == 3):
            dlg = FileBrowser("Choose a test image folder", mode="dir", start=current,
                              hint="A folder with one sub-folder per class, named like the model's classes.")
        elif self.spec.task == "language_model":
            dlg = FileBrowser("Choose a test text file", mode="file", extensions=datamod.TEXT_EXT, start=current,
                              hint="A plain-text file in the same style (and language) as the training text.")
        else:
            dlg = FileBrowser("Choose a test CSV file", mode="file", extensions=CSV_EXT, start=current,
                              hint="Same columns as the training data, including the true answers.")

        def done(path):
            if path:
                self.query_one("#eval-path", Input).value = path
                self._refresh_columns()
        self.app.push_screen(dlg, done)

    # -- running -----------------------------------------------------------------

    def _value(self, sel_id: str):
        v = self.query_one(sel_id, Select).value
        return None if v in (Select.NULL, Select.BLANK) else v

    def action_run(self) -> None:
        self._run_pressed()

    @on(Button.Pressed, "#eval-go")
    def _run_pressed(self) -> None:
        opts = {"source": self.query_one("#eval-source", Select).value,
                "path": self.query_one("#eval-path", Input).value.strip(),
                "target": self._value("#eval-target"), "text_column": self._value("#eval-text"),
                "robust": self.query_one("#eval-robust", Switch).value, "other": self._value("#eval-other")}
        if opts["source"] == "file" and not opts["path"]:
            self.notify("Choose a file or folder to evaluate on.", severity="error")
            return
        self.query_one("#eval-go", Button).disabled = True
        self.query_one("#eval-save", Button).disabled = True
        bar = self.query_one("#eval-progress", ProgressBar)
        bar.display = True
        bar.update(total=None, progress=0)
        self.query_one("#eval-status", Static).update("[yellow]Loading data…[/]")
        self._evaluate(opts)

    @work(thread=True, exclusive=True, group="eval")
    def _evaluate(self, opts: dict) -> None:
        def progress(stage, done, total):
            self.app.call_from_thread(self._progress, stage, done, total)

        try:
            spec, model, _ = self.ws.load_model(self.model_name)
            ds, ref, desc, orig = ev.load_eval_data(spec, self.ws.data_dir, opts["source"], opts["path"],
                                                    opts["target"], opts["text_column"])
            other = None
            if opts["other"]:
                s2, m2, _ = self.ws.load_model(opts["other"])
                other = (opts["other"], s2, m2)
            rep = ev.evaluate_model(spec, model, ds, ref, desc, orig, pick_device("auto"),
                                    self.ws.load_history(self.model_name), robustness=opts["robust"],
                                    other=other, progress=progress)
            ev.remember(rep, self.ws.model_dir(self.model_name))
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._done, None, str(e))
            return
        self.app.call_from_thread(self._done, rep, None)

    def _progress(self, stage: str, done: int, total: int) -> None:
        self.query_one("#eval-progress", ProgressBar).update(total=total, progress=done)
        self.query_one("#eval-status", Static).update(f"[yellow]{stage}…[/] {done:,} / {total:,}")

    def _done(self, rep: dict | None, error: str | None) -> None:
        self.query_one("#eval-go", Button).disabled = False
        self.query_one("#eval-progress").display = False
        if error:
            self.query_one("#eval-status", Static).update(f"[red]Couldn't evaluate: {error}[/]")
            return
        self.report = rep
        self.query_one("#eval-save", Button).disabled = False
        self.query_one("#eval-status", Static).update(f"[green]✓[/] Evaluated on {rep['data']}")
        self.query_one("#eval-report", Markdown).update(ev.report_markdown(rep))
        self.query_one("#eval-body", VerticalScroll).scroll_home(animate=False)
        self._show_history()

    @on(Button.Pressed, "#eval-save")
    def _save(self) -> None:
        if self.report:
            path = ev.save_report(self.report, self.ws.export_dir)
            self.notify(f"Saved {path} (+ .json)", timeout=10)
