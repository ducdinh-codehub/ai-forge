"""Train or fine-tune a model: choose data, hyper-parameters, watch it learn."""
from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Button, Footer, Header, Input, Label, Markdown, OptionList, ProgressBar,
                             RichLog, Select, SelectionList, Static, Switch)
from textual.widgets.option_list import Option

from .. import data as datamod
from ..spec import safe_name
from ..textgen import TOKENIZERS
from .chart import LineChart
from .resize import DEFAULT_FRACTION, MAX_FRACTION, MIN_FRACTION, ResizeHandle
from .sysmon import SystemPanel
from ..trainer import (TrainConfig, Trainer, available_devices, data_mismatch, fit_spec_to_data,
                       pick_device, reset_head)


class TrainScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back"), Binding("ctrl+r", "start", "Start", priority=True),
                Binding("f1,ctrl+g", "toggle_guide", "Guide", priority=True),
                Binding("less_than_sign", "resize_guide(0.05)", "Wider guide", show=False),
                Binding("greater_than_sign", "resize_guide(-0.05)", "Narrower guide", show=False)]

    def __init__(self, name: str, finetune: bool, preset: dict | None = None):
        """preset: data settings to load right away (e.g. picked in the dataset hub)."""
        super().__init__()
        self.model_name = name
        self.finetune = finetune
        self.preset = preset
        self.hub_key = preset.get("hub") if preset else None
        self.spec = None  # working copy, set on mount
        self.info = None
        self.train_ds = self.val_ds = None
        self.trainer: Trainer | None = None
        self.loss_hist: list[float] = []
        self.val_hist: list[float] = []
        self.metric_hist: list[float] = []
        self.train_metric_hist: list[float] = []

    @property
    def ws(self):
        return self.app.ws

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="train-main"):
            with VerticalScroll(id="train-config"):
                yield Static("① Data", classes="section-title")
                yield Select([(v, k) for k, v in datamod.SOURCES.items()], value="synthetic",
                             allow_blank=False, id="source")
                yield Button("🗄 Get a famous dataset…", id="hub")
                with Vertical(id="grp-path", classes="group"):
                    yield Label("Path", id="path-label")
                    with Horizontal(classes="path-row"):
                        yield Input(placeholder="~/data/table.csv", id="path")
                        yield Button("📂 Browse", id="browse-data")
                with Vertical(id="grp-csv", classes="group"):
                    yield Label("Target column  [dim](what to predict)[/]")
                    yield Select([], prompt="last column", id="target")
                with Vertical(id="grp-text", classes="group"):
                    yield Label("Text column")
                    yield Select([], prompt="first other column", id="text_column")
                with Vertical(id="grp-task", classes="group"):
                    yield Label("Task")
                    yield Select([("auto-detect", "auto"), ("classification", "classification"),
                                  ("regression", "regression")], value="auto", allow_blank=False, id="task")
                with Vertical(id="grp-synth", classes="group"):
                    yield Label("Toy dataset")
                    yield Select([(k, k) for k in datamod.SYNTHETIC], value="spirals", allow_blank=False,
                                 id="kind")
                with Vertical(id="grp-corpus", classes="group"):
                    yield Button("📜 Use the sample text (nursery rhymes)", id="sample-text")
                    yield Label("Split the text into")
                    yield Select([(v, k) for k, v in TOKENIZERS.items()], value="char", allow_blank=False,
                                 id="tokenizer")
                with Horizontal(classes="row"):
                    with Vertical():
                        yield Label("Validation split")
                        yield Input("0.2", id="val_split", type="number")
                    with Vertical():
                        yield Label("Max samples [dim](0=all)[/]")
                        yield Input("0", id="max_samples", type="integer")
                yield Button("Load data", id="load", variant="primary")

                yield Static("② Training settings", classes="section-title")
                with Horizontal(classes="row"):
                    with Vertical():
                        yield Label("Epochs")
                        yield Input("20" if not self.finetune else "10", id="epochs", type="integer")
                    with Vertical():
                        yield Label("Batch size")
                        yield Input("64", id="batch_size", type="integer")
                with Horizontal(classes="row"):
                    with Vertical():
                        yield Label("Learning rate")
                        yield Input("0.001" if not self.finetune else "0.0005", id="lr", type="number")
                    with Vertical():
                        yield Label("Weight decay")
                        yield Input("0", id="weight_decay", type="number")
                with Horizontal(classes="row"):
                    with Vertical():
                        yield Label("Optimizer")
                        yield Select([(o, o) for o in ("Adam", "AdamW", "SGD")], value="Adam",
                                     allow_blank=False, id="optimizer")
                    with Vertical():
                        yield Label("LR schedule")
                        yield Select([("constant", "none"), ("cosine", "cosine"), ("step ÷10", "step")],
                                     value="none", allow_blank=False, id="scheduler")
                with Horizontal(classes="row"):
                    with Vertical():
                        yield Label("Early stop [dim](patience)[/]")
                        yield Input("0", id="early_stop", type="integer")
                    with Vertical():
                        yield Label("Device")
                        yield Select([(d, d) for d in available_devices()], value="auto",
                                     allow_blank=False, id="device")
                if not self.finetune:
                    with Horizontal(classes="switch-row"):
                        yield Switch(True, id="resume")
                        yield Label("Continue from saved weights (if any)")
                else:
                    yield Static("③ Fine-tuning", classes="section-title")
                    yield Label("Freeze layers  [dim](frozen = not updated)[/]")
                    yield SelectionList(id="freeze")
                    with Horizontal(classes="toolbar"):
                        yield Button("Freeze all but head", id="freeze-body")
                        yield Button("Unfreeze all", id="freeze-none")
                    with Horizontal(classes="switch-row"):
                        yield Switch(False, id="reset_head")
                        yield Label("Re-initialise the output layer")
                    yield Label("Save result as")
                    yield Input(id="save_as")
            with Vertical(id="train-status"):
                yield Static("", id="model-line", classes="section-title")
                yield SystemPanel(id="sysmon")
                yield Static("[dim]Load a dataset to begin.[/]", id="data-info")
                with Horizontal(id="fit-row"):
                    # Button first: it stays visible when the guide drawer covers the right side.
                    yield Button("Auto-fit model to data", id="autofit", variant="warning")
                    yield Button("🔧 Fix model…", id="fixmodel", variant="warning")
                    yield Static("", id="mismatch")
                with Horizontal(classes="toolbar"):
                    yield Button("▶ Start", id="start", variant="success")
                    yield Button("■ Stop", id="stop", variant="error", disabled=True)
                    yield Button("📖 Guide", id="guide-btn")
                    yield Static("", id="epoch-label")
                yield ProgressBar(id="progress", show_eta=True)
                with Horizontal(id="charts"):
                    yield LineChart("Loss", id="chart-loss", fmt="{:.3f}")
                    yield LineChart("Accuracy", id="chart-metric", fmt="{:.0%}")
                yield RichLog(id="log", markup=True, wrap=True)
            # Slide-over panel (its own layer, so it covers the charts instead of squeezing them).
            with Vertical(id="guide-drawer"):
                yield ResizeHandle(id="drawer-handle")
                with Horizontal(id="drawer-head"):
                    yield Static("📖 Guide", id="drawer-title")
                    yield Button("◀", id="drawer-wider", tooltip="Wider  ( < )")
                    yield Button("▶", id="drawer-narrower", tooltip="Narrower  ( > )")
                    yield Button("Full guide ⤢", id="drawer-full")
                    yield Button("✕", id="drawer-close")
                yield OptionList(id="drawer-topics")
                with VerticalScroll(id="drawer-body"):
                    yield Markdown(id="drawer-text")
        yield Footer()

    def on_mount(self) -> None:
        self.spec = self.ws.load_spec(self.model_name)
        mode = "Fine-tune" if self.finetune else "Train"
        weights = "trained weights" if self.ws.has_weights(self.model_name) else "no weights yet"
        self.query_one("#model-line", Static).update(
            f"{mode} [b]{self.model_name}[/b] · {self.spec.total_params():,} params · {weights}")
        self.query_one("#fit-row").display = False
        self.query_one("#guide-drawer").display = False
        self._set_guide_width(self.ws.get_setting("guide_width", DEFAULT_FRACTION), save=False)
        self._monitor = None
        self._sample_system()
        self.set_interval(1.0, self._sample_system)
        self._apply_previous_dataset()
        self._update_source_fields()
        if self.preset:
            self._use_dataset(self.preset)
        if self.finetune:
            self.query_one("#save_as", Input).value = self.ws.unique_name(self.model_name + "-ft")
            self._fill_freeze_list()
        self.query_one("#log", RichLog).write(
            "[dim]Tip: start with the defaults. If validation loss rises while train loss falls, the "
            "model is overfitting - add Dropout, lower epochs or use early stopping.[/]")

    # -- system monitor --------------------------------------------------------

    @work(thread=True, exclusive=True, group="sysmon")
    def _sample_system(self) -> None:
        from ..sysinfo import Monitor, specs

        panel = self.query_one("#sysmon", SystemPanel)
        if self._monitor is None:
            self.app.call_from_thread(panel.set_specs, specs())
            self._monitor = Monitor()
            return  # the first CPU reading needs a time interval to be meaningful
        sample = self._monitor.sample()
        self.app.call_from_thread(panel.update_sample, sample)

    @on(Select.Changed, "#device")
    def _device_changed(self, event: Select.Changed) -> None:
        self.query_one("#sysmon", SystemPanel).set_device_choice(str(event.value))

    def _apply_previous_dataset(self) -> None:
        prev = self.spec.data.get("dataset_cfg")
        if not prev and self.spec.task == "language_model":
            # A new language model starts on the bundled sample text, ready to train.
            prev = {"source": "text_corpus", "path": str(datamod.EXAMPLE_CORPUS),
                    "tokenizer": self.spec.data.get("tokenizer", "char")}
        if not prev:
            return
        try:
            self.query_one("#source", Select).value = prev["source"]
            if prev.get("path"):
                self.query_one("#path", Input).value = prev["path"]
            if prev.get("kind"):
                self.query_one("#kind", Select).value = prev["kind"]
            if prev.get("tokenizer"):
                self.query_one("#tokenizer", Select).value = prev["tokenizer"]
        except Exception:  # noqa: BLE001
            pass

    @on(Button.Pressed, "#hub")
    def _open_hub(self) -> None:
        from .datasets import DatasetScreen

        self.app.push_screen(DatasetScreen("✓ Use this dataset", focus=self.hub_key), self._use_dataset)

    def _use_dataset(self, cfg: dict | None) -> None:
        """Fill the data fields from a hub dataset and load it."""
        if not cfg:
            return
        self.hub_key = cfg.get("hub")
        self.query_one("#source", Select).value = cfg["source"]
        self.query_one("#path", Input).value = cfg.get("path") or ""
        self._update_source_fields()  # fills the column lists from the new CSV
        for sel_id, key in (("#target", "target"), ("#text_column", "text_column")):
            if cfg.get(key):
                self.query_one(sel_id, Select).value = cfg[key]
        self.query_one("#task", Select).value = cfg.get("task") or "auto"
        if cfg.get("tokenizer"):
            self.query_one("#tokenizer", Select).value = cfg["tokenizer"]
        self._load_pressed()

    def _fill_freeze_list(self) -> None:
        sl = self.query_one("#freeze", SelectionList)
        sl.clear_options()
        for info in self.spec.analyze():
            if info.params:
                sl.add_option((f"{info.index + 1:>2}. {info.spec.type:<18} {info.params:>11,}",
                               info.spec.id, info.spec.id in self.spec.frozen))

    # -- data source fields --------------------------------------------------

    @on(Select.Changed, "#source")
    def _source_changed(self) -> None:
        self._update_source_fields()

    def _update_source_fields(self) -> None:
        src = self.query_one("#source", Select).value
        self.query_one("#grp-path").display = src in datamod.FILE_SOURCES
        self.query_one("#grp-csv").display = src in ("csv", "csv_text")
        self.query_one("#grp-text").display = src == "csv_text"
        self.query_one("#grp-task").display = src == "csv"
        self.query_one("#grp-synth").display = src == "synthetic"
        self.query_one("#grp-corpus").display = src == "text_corpus"
        label = {"image_folder": "Folder  [dim](one sub-folder per class)[/]",
                 "text_corpus": "Text file  [dim](.txt - the model learns to write like it)[/]"}.get(src, "CSV file")
        self.query_one("#path-label", Label).update(label)
        self.query_one("#path", Input).placeholder = \
            "~/data/book.txt" if src == "text_corpus" else "~/data/table.csv"
        self._refresh_columns()

    @on(Button.Pressed, "#sample-text")
    def _use_sample_text(self) -> None:
        self.query_one("#path", Input).value = str(datamod.EXAMPLE_CORPUS)
        self._load_pressed()

    @on(Input.Changed, "#path")
    def _refresh_columns(self) -> None:
        src = self.query_one("#source", Select).value
        if src not in ("csv", "csv_text"):
            return
        cols = datamod.csv_columns(self.query_one("#path", Input).value)
        for sel_id in ("#target", "#text_column"):
            sel = self.query_one(sel_id, Select)
            current = sel.value
            sel.set_options([(c, c) for c in cols])
            if current in cols:
                sel.value = current

    @on(Button.Pressed, "#browse-data")
    def _browse_data(self) -> None:
        from .browser import CSV_EXT, FileBrowser

        src = self.query_one("#source", Select).value
        current = self.query_one("#path", Input).value.strip()
        if src == "image_folder":
            dlg = FileBrowser("Choose an image folder", mode="dir", start=current or None,
                              hint="Pick the folder that contains one sub-folder per class "
                                   "(e.g. cat/, dog/). The preview shows how many images each class has.")
        elif src == "text_corpus":
            dlg = FileBrowser("Choose a text file", mode="file", extensions=datamod.TEXT_EXT,
                              start=current or None,
                              hint="Pick a plain-text file: a book, poems, song lyrics, code... More text "
                                   "(at least ~1 MB) gives a better model.")
        else:
            dlg = FileBrowser("Choose a CSV file", mode="file", extensions=CSV_EXT, start=current or None,
                              hint="Pick a .csv file. The preview shows its columns and first rows.")

        def done(path):
            if path:
                self.query_one("#path", Input).value = path
                self._refresh_columns()
        self.app.push_screen(dlg, done)

    def _data_cfg(self) -> dict:
        def val(sel_id):
            v = self.query_one(sel_id, Select).value
            return None if v in (Select.NULL, Select.BLANK, "auto") else v
        return {
            "source": self.query_one("#source", Select).value,
            "path": self.query_one("#path", Input).value.strip(),
            "target": val("#target"),
            "text_column": val("#text_column"),
            "task": val("#task"),
            "kind": self.query_one("#kind", Select).value,
            "tokenizer": self.query_one("#tokenizer", Select).value,
            "val_split": float(self.query_one("#val_split", Input).value or 0.2),
        }

    # -- loading -------------------------------------------------------------

    @on(Button.Pressed, "#load")
    def _load_pressed(self) -> None:
        try:
            cfg = self._data_cfg()
        except ValueError:
            self.notify("Validation split must be a number like 0.2", severity="error")
            return
        if cfg["source"] in datamod.FILE_SOURCES and not cfg["path"]:
            self.notify("Enter a path first.", severity="error")
            return
        self.query_one("#data-info", Static).update("[yellow]Loading data…[/] "
                                                     "(built-in datasets download on first use)")
        self.query_one("#load", Button).disabled = True
        self._load_data(cfg)

    @work(thread=True, exclusive=True, group="data")
    def _load_data(self, cfg: dict) -> None:
        try:
            tr, va, info = datamod.load(cfg, self.ws.data_dir, self.spec)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._data_failed, str(e))
            return
        self.app.call_from_thread(self._data_loaded, cfg, tr, va, info)

    def _data_failed(self, msg: str) -> None:
        self.query_one("#load", Button).disabled = False
        self.query_one("#data-info", Static).update(f"[red]Couldn't load data: {msg}[/]")

    def _data_loaded(self, cfg, tr, va, info) -> None:
        self.query_one("#load", Button).disabled = False
        n = int(self.query_one("#max_samples", Input).value or 0)
        self.train_ds, self.val_ds, self.info = datamod.limit(tr, n), va, info
        self.data_cfg = cfg
        self._refresh_drawer("settings")
        classes = (f"{len(info.class_names)} classes: {', '.join(info.class_names[:8])}"
                   f"{'…' if len(info.class_names) > 8 else ''}") if info.class_names else "regression target"
        if info.task == "language_model":
            classes = f"predicts the next of {info.n_outputs:,} {info.preprocess['tokenizer']} tokens"
        used = f" (using {len(self.train_ds):,})" if n else ""
        self.query_one("#data-info", Static).update(
            f"[green]✓[/] {info.description}\n"
            f"   input {tuple(info.input_shape)} {info.input_dtype} · {info.task} · {classes}\n"
            f"   {len(tr):,} train{used} / {len(va):,} validation samples")
        self._check_mismatch()
        tokens = getattr(tr, "tokens", None)
        if info.task == "language_model" and tokens is not None and len(tokens) < 100_000:
            self.query_one("#log", RichLog).write(
                f"[yellow]💡 This text is short ({len(tokens):,} tokens).[/] The model will learn its shape - "
                "line breaks, rhymes, common words - but not real sentences, and it starts memorizing "
                "after a few epochs (AIForge keeps the best one). For real text, get [b]Tiny Shakespeare[/b] "
                "(1 MB) from 🗄 Get a famous dataset: a few epochs already write recognisable words.")
        if self.finetune and self.spec.class_names and info.class_names != self.spec.class_names:
            self.query_one("#reset_head", Switch).value = True

    def _check_mismatch(self) -> None:
        probs = data_mismatch(self.spec, self.info) if self.info else []
        problems = self.spec.validate() if not probs else []
        row = self.query_one("#fit-row")
        row.display = bool(probs or problems)
        self.query_one("#autofit", Button).display = bool(probs)
        self.query_one("#fixmodel", Button).display = bool(problems) and not probs
        self.query_one("#mismatch", Static).update(
            "[yellow]⚠ " + "\n⚠ ".join(probs or problems) + "[/]")

    @on(Button.Pressed, "#autofit")
    def _autofit(self) -> None:
        changes = fit_spec_to_data(self.spec, self.info)
        for c in changes:
            self.query_one("#log", RichLog).write(f"[cyan]auto-fit:[/] {c}")
        problems = self.spec.validate()
        if problems:
            self._refresh_after_change()
            self._open_fix_dialog()
            return
        else:
            self.notify(f"Model adjusted ({len(changes)} change{'s' if len(changes) != 1 else ''}), "
                        f"now {self.spec.total_params():,} params. Saved when training finishes.")
        self.query_one("#model-line", Static).update(
            f"{'Fine-tune' if self.finetune else 'Train'} [b]{self.model_name}[/b] · "
            f"{self.spec.total_params():,} params")
        if self.finetune:
            self._fill_freeze_list()
        self._check_mismatch()
        if self.query_one("#guide-drawer").display:
            self._fill_topics()

    # -- fixing a model that doesn't fit the data ----------------------------------

    def _refresh_after_change(self) -> None:
        self.query_one("#model-line", Static).update(
            f"{'Fine-tune' if self.finetune else 'Train'} [b]{self.model_name}[/b] · "
            f"{self.spec.total_params() if not self.spec.validate() else '?'} params")
        if self.finetune:
            self._fill_freeze_list()
        self._check_mismatch()
        if self.query_one("#guide-drawer").display:
            self._fill_topics()

    @on(Button.Pressed, "#fixmodel")
    def _open_fix_dialog(self) -> None:
        from ..fixes import explain, suggest
        from .fixdialog import FixModelDialog

        if self.info is None:
            self.notify("Load a dataset first.", severity="warning")
            return
        self.app.push_screen(FixModelDialog(explain(self.spec, self.info), suggest(self.spec, self.info)),
                             self._apply_fix)

    def _apply_fix(self, fix) -> None:
        if fix is None:
            return
        log = self.query_one("#log", RichLog)
        if fix.kind == "builder":
            self._open_builder()
            return
        old_ids = {l.id for l in self.spec.layers}
        fix.spec.frozen = [f for f in self.spec.frozen if f in {l.id for l in fix.spec.layers}]
        self.spec = fix.spec
        if fix.kind == "template":
            self._fresh_weights = True  # nothing from the old layers can be reused
        added = [l.type for l in self.spec.layers if l.id not in old_ids]
        log.write(f"[cyan]fix:[/] {fix.title} → {self.spec.total_params():,} params"
                  + (f" (new: {', '.join(added)})" if added and fix.kind == 'auto' else ""))
        self.notify(f"✓ {fix.title}. The model now fits your data ({self.spec.total_params():,} params) "
                    "and is saved when training finishes.", timeout=8)
        self._refresh_after_change()

    def _open_builder(self) -> None:
        from .builder import BuilderScreen

        stored = self.ws.load_spec(self.model_name)
        builder = BuilderScreen(self.spec.clone())
        builder.saved_state = stored.to_dict()  # so the data-fitted changes count as unsaved
        self._builder = builder
        self.app.push_screen(builder)

    def on_screen_resume(self) -> None:
        builder = getattr(self, "_builder", None)
        # The resume event from closing the fix dialog can arrive after the builder was pushed.
        if builder is None or builder in self.app.screen_stack:
            return
        self._builder = None
        if builder.original_name and builder.original_name != self.model_name:
            self.model_name = builder.original_name  # renamed in the builder
        self.spec = self.ws.load_spec(self.model_name)
        problems = self.spec.validate() or (data_mismatch(self.spec, self.info) if self.info else [])
        if problems:
            self.notify(f"The saved model still doesn't fit: {problems[0]}", severity="warning", timeout=8)
        else:
            self.notify("✓ The model from the builder fits your data.")
        self._refresh_after_change()

    @on(Button.Pressed, "#freeze-body")
    def _freeze_body(self) -> None:
        sl = self.query_one("#freeze", SelectionList)
        sl.select_all()
        head = next((l.id for l in reversed(self.spec.layers) if l.type == "Linear"), None)
        if head:
            sl.deselect(head)

    @on(Button.Pressed, "#freeze-none")
    def _freeze_none(self) -> None:
        self.query_one("#freeze", SelectionList).deselect_all()

    # -- training ------------------------------------------------------------

    def _train_cfg(self) -> TrainConfig:
        q = lambda i: self.query_one(f"#{i}", Input).value.strip()  # noqa: E731
        s = lambda i: self.query_one(f"#{i}", Select).value  # noqa: E731
        cfg = TrainConfig(epochs=int(q("epochs")), batch_size=int(q("batch_size")), lr=float(q("lr")),
                          optimizer=s("optimizer"), weight_decay=float(q("weight_decay") or 0),
                          scheduler=s("scheduler"), early_stop=int(q("early_stop") or 0),
                          device=s("device"))
        if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.lr <= 0:
            raise ValueError
        return cfg

    def action_start(self) -> None:
        self._start()

    @on(Button.Pressed, "#start")
    def _start(self) -> None:
        if self.trainer:
            return
        if not self.info:
            self.notify("Load a dataset first.", severity="warning")
            return
        if probs := data_mismatch(self.spec, self.info):
            self.notify(probs[0] + " - press 'Auto-fit model to data'.", severity="error", timeout=8)
            return
        if probs := self.spec.validate():
            self.notify(probs[0], severity="error", timeout=8)
            return
        try:
            cfg = self._train_cfg()
        except ValueError:
            self.notify("Check training settings: epochs/batch must be positive integers, lr > 0.",
                        severity="error")
            return
        save_as = self.model_name
        if self.finetune:
            save_as = safe_name(self.query_one("#save_as", Input).value) or self.model_name
            if save_as != self.model_name and self.ws.exists(save_as):
                self.notify(f"'{save_as}' already exists - pick another name.", severity="error")
                return
            self.spec.frozen = list(self.query_one("#freeze", SelectionList).selected)
        else:
            self.spec.frozen = []

        use_saved = (self.finetune or self.query_one("#resume", Switch).value) and \
            not getattr(self, "_fresh_weights", False)
        if use_saved:
            _, model, note = self.ws.load_model(self.model_name, self.spec)
        else:
            model, note = self.spec.build(), "fresh random weights"
        if self.finetune and self.query_one("#reset_head", Switch).value:
            reset_head(self.spec, model)
            note += ", output layer re-initialised"
        self.spec.apply_freeze(model)

        log = self.query_one("#log", RichLog)
        log.write(f"[b]── {'Fine-tuning' if self.finetune else 'Training'} → '{save_as}' "
                  f"({note})[/b]")
        self.loss_hist, self.val_hist, self.metric_hist, self.train_metric_hist = [], [], [], []
        chart = self.query_one("#chart-metric", LineChart)
        chart.title, chart.fmt = {"classification": ("Accuracy", "{:.0%}"),
                                  "language_model": ("Perplexity", "{:.3g}")}.get(
            self.spec.task, ("Mean abs. error", "{:.3g}"))
        self.query_one("#chart-loss", LineChart).set_data([], [])
        chart.set_data([], [])
        self.query_one("#start", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#progress", ProgressBar).update(total=cfg.epochs, progress=0)
        self._batch_size, self._device = cfg.batch_size, pick_device(cfg.device).type
        call = self.app.call_from_thread
        self.trainer = Trainer(
            self.spec, model, self.train_ds, self.val_ds, cfg,
            on_log=lambda m: call(log.write, m),
            on_batch=lambda e, s, n, l: call(self._on_batch, e, s, n, l, cfg.epochs),
            on_epoch=lambda r: call(self._on_epoch, r),
            target_std=self.info.preprocess.get("target_std", 1.0), preprocess=self.info.preprocess)
        self._run(save_as, model)

    def _on_batch(self, epoch, step, steps, loss, epochs) -> None:
        self.query_one("#sysmon", SystemPanel).training_progress(
            ((epoch - 1) * steps + step) * self._batch_size, self._device)
        self.query_one("#progress", ProgressBar).update(progress=epoch - 1 + step / steps)
        self.query_one("#epoch-label", Static).update(f"  epoch {epoch}/{epochs} · batch {step}/{steps} · "
                                                       f"loss {loss:.4f}")

    def _on_epoch(self, r: dict) -> None:
        self.loss_hist.append(r["train_loss"])
        parts = [f"epoch {r['epoch']:>3}", f"loss {r['train_loss']:.4f}"]
        if "train_acc" in r:
            self.train_metric_hist.append(r["train_acc"])
            parts.append(f"acc {r['train_acc']:.1%}")
        if "train_ppl" in r:
            self.train_metric_hist.append(r["train_ppl"])
            parts.append(f"ppl {r['train_ppl']:.2f}")
        if "val_loss" in r:
            self.val_hist.append(r["val_loss"])
            parts.append(f"│ val loss {r['val_loss']:.4f}")
        if "val_acc" in r:
            self.metric_hist.append(r["val_acc"])
            parts.append(f"val acc [b]{r['val_acc']:.1%}[/b]")
        elif "val_mae" in r:
            self.metric_hist.append(r["val_mae"])
            parts.append(f"val MAE [b]{r['val_mae']:.4g}[/b]")
        elif "val_ppl" in r:
            self.metric_hist.append(r["val_ppl"])
            parts.append(f"val ppl [b]{r['val_ppl']:.2f}[/b]")
        self._refresh_drawer("charts")
        self.query_one("#chart-loss", LineChart).set_data(self.loss_hist, self.val_hist)
        self.query_one("#chart-metric", LineChart).set_data(self.train_metric_hist, self.metric_hist)
        parts.append(f"[dim]{r['seconds']}s[/]")
        log = self.query_one("#log", RichLog)
        log.write("  ".join(parts))
        if r.get("sample"):
            from rich.markup import escape

            log.write(f"[dim]  ✎ sample:[/] [italic]{escape(r['sample'].strip())}[/]")

    @work(thread=True, exclusive=True, group="train")
    def _run(self, save_as: str, model) -> None:
        try:
            run = self.trainer.run()
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._finished, None, save_as, model, str(e))
            return
        self.app.call_from_thread(self._finished, run, save_as, model, None)

    def _finished(self, run, save_as, model, error) -> None:
        self.trainer = None
        self.query_one("#sysmon", SystemPanel).training_stopped()
        self.query_one("#start", Button).disabled = False
        self.query_one("#stop", Button).disabled = True
        log = self.query_one("#log", RichLog)
        if error:
            log.write(f"[red]Training failed: {error}[/]")
            self.notify(f"Training failed: {error}", severity="error", timeout=10)
            return
        if not run["epochs"]:
            log.write("[yellow]No epoch completed - nothing saved.[/]")
            return
        self.spec.data["preprocess"] = self.info.preprocess
        cfg = {k: v for k, v in self.data_cfg.items() if v not in (None, "")}
        if cfg.get("path"):
            cfg["path"] = str(Path(cfg["path"]).expanduser())
        self.spec.data["dataset_cfg"] = cfg
        self._fresh_weights = False
        run["kind"] = "fine-tune" if self.finetune else "train"
        run["dataset"] = self.info.description
        if save_as != self.model_name:
            run["parent"] = self.model_name
        self.spec.name = save_as
        self.ws.save_spec(self.spec)
        self.ws.save_weights(save_as, model)
        self.ws.append_history(save_as, run)
        last = run["epochs"][-1]
        metric = (f", val acc {last['val_acc']:.1%}" if "val_acc" in last else
                  f", val MAE {last['val_mae']:.4g}" if "val_mae" in last else
                  f", val perplexity {last['val_ppl']:.2f}" if "val_ppl" in last else "")
        msg = f"Saved '{save_as}' (best weights, {len(run['epochs'])} epochs{metric})"
        log.write(f"[green]✓ {msg}[/]")
        self.notify(msg)
        if self.finetune and save_as != self.model_name:
            # Further runs on this screen continue from the new model.
            self.model_name = save_as
            self.query_one("#save_as", Input).value = self.ws.unique_name(save_as)

    @on(Button.Pressed, "#stop")
    def _stop(self) -> None:
        if self.trainer:
            self.trainer.stop()
            self.query_one("#stop", Button).disabled = True
            self.query_one("#log", RichLog).write("[yellow]Stopping after the current batch…[/]")

    # -- guide drawer ------------------------------------------------------------

    def action_toggle_guide(self) -> None:
        drawer = self.query_one("#guide-drawer")
        if drawer.display:
            drawer.display = False
            return
        self._fill_topics()
        drawer.display = True
        self.query_one("#drawer-topics", OptionList).focus()

    def _set_guide_width(self, fraction: float, save: bool = True) -> None:
        """Width as a fraction of the screen, so it adapts when the terminal is resized."""
        try:
            fraction = float(fraction)
        except (TypeError, ValueError):
            fraction = DEFAULT_FRACTION
        fraction = round(min(MAX_FRACTION, max(MIN_FRACTION, fraction)), 3)
        self._guide_width = fraction
        self.query_one("#guide-drawer").styles.width = f"{fraction * 100:.1f}%"
        self.query_one("#drawer-title", Static).update(f"📖 Guide  [dim]{fraction:.0%} of the screen[/]")
        self.query_one("#drawer-wider", Button).disabled = fraction >= MAX_FRACTION
        self.query_one("#drawer-narrower", Button).disabled = fraction <= MIN_FRACTION
        if save:
            self.ws.set_setting("guide_width", fraction)

    def action_resize_guide(self, delta: float) -> None:
        if self.query_one("#guide-drawer").display:
            self._set_guide_width(self._guide_width + delta)

    @on(Button.Pressed, "#drawer-wider")
    def _wider(self) -> None:
        self.action_resize_guide(0.05)

    @on(Button.Pressed, "#drawer-narrower")
    def _narrower(self) -> None:
        self.action_resize_guide(-0.05)

    @on(ResizeHandle.Resized)
    def _dragged(self, event: ResizeHandle.Resized) -> None:
        self._set_guide_width(event.fraction, save=event.done)

    @on(ResizeHandle.Reset)
    def _reset_width(self) -> None:
        self._set_guide_width(DEFAULT_FRACTION)

    @on(Button.Pressed, "#guide-btn")
    def _guide_btn(self) -> None:
        self.action_toggle_guide()

    @on(Button.Pressed, "#drawer-close")
    def _drawer_close(self) -> None:
        self.query_one("#guide-drawer").display = False

    @on(Button.Pressed, "#drawer-full")
    def _drawer_full(self) -> None:
        from .guide import GuideScreen

        topic = getattr(self, "_topic", "overview")
        self.app.push_screen(GuideScreen(self.spec, topic if topic.startswith(("overview", "layer:"))
                                         else "overview"))

    def _fill_topics(self) -> None:
        """(Re)build the topic list - the model may have changed (auto-fit)."""
        topics = self.query_one("#drawer-topics", OptionList)
        keep = getattr(self, "_topic", "charts" if self.loss_hist else "overview")
        topics.clear_options()
        opts = [Option("[b]Training[/b]", disabled=True),
                Option(" 📈 Reading the charts (live)", id="charts"),
                Option(" ⚙  Your training settings", id="settings"),
                Option(f"[b]Model: {self.spec.name}[/b]", disabled=True),
                Option(" 🧭 Overview & review", id="overview")]
        opts += [Option(f" {i + 1:>2}. {l.type}", id=f"layer:{i}") for i, l in enumerate(self.spec.layers)]
        topics.add_options(opts)
        ids = [o.id for o in opts]
        topics.highlighted = ids.index(keep) if keep in ids else ids.index("overview")
        self._show_topic(ids[topics.highlighted])

    @on(OptionList.OptionHighlighted, "#drawer-topics")
    def _topic_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id:
            self._show_topic(event.option.id)

    def _show_topic(self, topic: str) -> None:
        from ..guide import charts_markdown, layer_markdown, model_markdown, training_settings_markdown

        self._topic = topic
        if topic == "charts":
            text = charts_markdown(self.loss_hist, self.val_hist, self.metric_hist, self.spec.task,
                                   self.spec.vocab_size() if self.spec.task == "language_model"
                                   else len(self.spec.class_names))
        elif topic == "settings":
            text = training_settings_markdown(self.spec, self._raw_settings(),
                                              len(self.train_ds) if self.train_ds is not None else None,
                                              self.finetune, self._frozen_params())
        elif topic == "overview":
            text = model_markdown(self.spec)
        else:
            text = layer_markdown(self.spec, int(topic.split(":")[1]))
        self.query_one("#drawer-text", Markdown).update(text)

    def _raw_settings(self) -> dict:
        vals = {i: self.query_one(f"#{i}", Input).value for i in
                ("epochs", "batch_size", "lr", "weight_decay", "early_stop", "val_split")}
        vals.update({i: str(self.query_one(f"#{i}", Select).value) for i in ("optimizer", "scheduler", "device")})
        return vals

    def _frozen_params(self) -> int:
        if not self.finetune:
            return 0
        frozen = set(self.query_one("#freeze", SelectionList).selected)
        return sum(i.params for i in self.spec.analyze() if i.spec.id in frozen)

    def _refresh_drawer(self, *topics: str) -> None:
        """Re-render the open drawer if it currently shows one of ``topics``."""
        if self.query_one("#guide-drawer").display and getattr(self, "_topic", None) in topics:
            self._show_topic(self._topic)

    @on(Input.Changed)
    @on(Select.Changed)
    @on(SelectionList.SelectedChanged)
    def _settings_changed(self) -> None:
        self._refresh_drawer("settings")

    def action_back(self) -> None:
        drawer = self.query_one("#guide-drawer")
        if drawer.display:
            drawer.display = False
            return
        if self.trainer:
            self.notify("Training is running - press Stop first.", severity="warning")
            return
        self.app.pop_screen()
