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
from .chart import LineChart
from .sysmon import SystemPanel
from ..trainer import (TrainConfig, Trainer, available_devices, data_mismatch, fit_spec_to_data,
                       pick_device, reset_head)


class TrainScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back"), Binding("ctrl+r", "start", "Start", priority=True),
                Binding("f1,ctrl+g", "toggle_guide", "Guide", priority=True)]

    def __init__(self, name: str, finetune: bool):
        super().__init__()
        self.model_name = name
        self.finetune = finetune
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
                with Horizontal(id="drawer-head"):
                    yield Static("📖 Guide", id="drawer-title")
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
        self._monitor = None
        self._sample_system()
        self.set_interval(1.0, self._sample_system)
        self._apply_previous_dataset()
        self._update_source_fields()
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
        if not prev:
            return
        try:
            self.query_one("#source", Select).value = prev["source"]
            if prev.get("path"):
                self.query_one("#path", Input).value = prev["path"]
            if prev.get("kind"):
                self.query_one("#kind", Select).value = prev["kind"]
        except Exception:  # noqa: BLE001
            pass

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
        self.query_one("#grp-path").display = src in ("csv", "csv_text", "image_folder")
        self.query_one("#grp-csv").display = src in ("csv", "csv_text")
        self.query_one("#grp-text").display = src == "csv_text"
        self.query_one("#grp-task").display = src == "csv"
        self.query_one("#grp-synth").display = src == "synthetic"
        label = {"image_folder": "Folder  [dim](one sub-folder per class)[/]"}.get(src, "CSV file")
        self.query_one("#path-label", Label).update(label)
        self._refresh_columns()

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
        if cfg["source"] in ("csv", "csv_text", "image_folder") and not cfg["path"]:
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
        used = f" (using {len(self.train_ds):,})" if n else ""
        self.query_one("#data-info", Static).update(
            f"[green]✓[/] {info.description}\n"
            f"   input {tuple(info.input_shape)} {info.input_dtype} · {info.task} · {classes}\n"
            f"   {len(tr):,} train{used} / {len(va):,} validation samples")
        self._check_mismatch()
        if self.finetune and self.spec.class_names and info.class_names != self.spec.class_names:
            self.query_one("#reset_head", Switch).value = True

    def _check_mismatch(self) -> None:
        probs = data_mismatch(self.spec, self.info) if self.info else []
        problems = self.spec.validate() if not probs else []
        row = self.query_one("#fit-row")
        row.display = bool(probs or problems)
        self.query_one("#autofit", Button).display = bool(probs)
        self.query_one("#mismatch", Static).update(
            "[yellow]⚠ " + "\n⚠ ".join(probs or problems) + "[/]")

    @on(Button.Pressed, "#autofit")
    def _autofit(self) -> None:
        changes = fit_spec_to_data(self.spec, self.info)
        for c in changes:
            self.query_one("#log", RichLog).write(f"[cyan]auto-fit:[/] {c}")
        problems = self.spec.validate()
        if problems:
            self.notify("Model still has problems: " + problems[0] + " - open the builder to fix.",
                        severity="error", timeout=10)
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

        use_saved = self.finetune or self.query_one("#resume", Switch).value
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
        is_cls = self.spec.task == "classification"
        chart = self.query_one("#chart-metric", LineChart)
        chart.title, chart.fmt = ("Accuracy", "{:.0%}") if is_cls else ("Mean abs. error", "{:.3g}")
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
            target_std=self.info.preprocess.get("target_std", 1.0))
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
        if "val_loss" in r:
            self.val_hist.append(r["val_loss"])
            parts.append(f"│ val loss {r['val_loss']:.4f}")
        if "val_acc" in r:
            self.metric_hist.append(r["val_acc"])
            parts.append(f"val acc [b]{r['val_acc']:.1%}[/b]")
        elif "val_mae" in r:
            self.metric_hist.append(r["val_mae"])
            parts.append(f"val MAE [b]{r['val_mae']:.4g}[/b]")
        self._refresh_drawer("charts")
        self.query_one("#chart-loss", LineChart).set_data(self.loss_hist, self.val_hist)
        self.query_one("#chart-metric", LineChart).set_data(self.train_metric_hist, self.metric_hist)
        parts.append(f"[dim]{r['seconds']}s[/]")
        self.query_one("#log", RichLog).write("  ".join(parts))

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
                  f", val MAE {last['val_mae']:.4g}" if "val_mae" in last else "")
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
                                   len(self.spec.class_names))
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
