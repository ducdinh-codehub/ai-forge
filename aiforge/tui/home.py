"""Model library: the starting screen."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Static

from .. import model_io
from ..spec import safe_name
from .banner import Banner
from .modals import ConfirmDialog, ImportDialog, NewModelDialog, TextDialog

WELCOME = """[b]Welcome to AIForge![/b]
Build, train and share neural networks - no code needed.

[b $accent]How it works[/]
 [reverse] n [/reverse] [b]Create[/b]     pick a template, then add or remove layers
 [reverse] t [/reverse] [b]Train[/b]      teach the model with your data
 [reverse] f [/reverse] [b]Fine-tune[/b]  adapt a trained model to new data
 [reverse] p [/reverse] [b]Use[/b]        get predictions from the model
 [reverse] v [/reverse] [b]Evaluate[/b]   measure how good it really is (metrics, mistakes…)
 [reverse] x [/reverse] [b]Export[/b]     save it as .pt, .safetensors, ONNX, …
 [reverse] g [/reverse] [b]Guide[/b]      learn ML basics, model types and every layer
 [reverse] s [/reverse] [b]Datasets[/b]   download famous, trusted datasets to train on

Already have a model? Press [reverse] i [/reverse] to import it.
Found a bug or have an idea? Press [reverse] Ctrl+O [/reverse] to tell us on GitHub.

[dim]New here? Press [b]g[/b] to read the beginner lessons, or press [b]n[/b], keep the
default template, then press [b]t[/b] and train on the built-in "spirals" toy data.[/dim]"""


class HomeScreen(Screen):
    BINDINGS = [
        Binding("n", "new", "New"),
        Binding("i", "import", "Import"),
        Binding("e", "edit", "Edit structure"),
        Binding("t", "train", "Train"),
        Binding("f", "finetune", "Fine-tune"),
        Binding("p", "predict", "Use"),
        Binding("v", "evaluate", "Evaluate"),
        Binding("x", "export", "Export"),
        Binding("g,question_mark", "guide", "Guide"),
        Binding("s", "datasets", "Datasets"),
        Binding("w", "load_weights", "Load weights", show=False),
        Binding("c", "duplicate", "Duplicate", show=False),
        Binding("r", "rename", "Rename", show=False),
        Binding("delete,d", "delete", "Delete", show=False),
        Binding("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Banner(id="banner")
        with Horizontal(id="home-body"):
            with Vertical(id="library"):
                yield Static("Model library", classes="section-title")
                yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
                yield Static("No models yet.\n\nPress [reverse] n [/reverse] to create your first model "
                             "or [reverse] i [/reverse] to import one.", id="empty-hint")
                with Horizontal(classes="toolbar"):
                    yield Button("✚ New", id="new", variant="primary")
                    yield Button("⇪ Import", id="import")
                    yield Button("🗄 Datasets", id="datasets")
                    yield Button("Load weights", id="load_weights")
                    yield Button("Duplicate", id="duplicate")
                    yield Button("Rename", id="rename")
                    yield Button("Delete", id="delete", variant="error")
            with Vertical(id="details-pane"):
                with Horizontal(classes="toolbar"):
                    yield Button("✎ Edit", id="edit")
                    yield Button("▶ Train", id="train", variant="success")
                    yield Button("⟳ Fine-tune", id="finetune", variant="warning")
                    yield Button("◉ Use", id="predict")
                    yield Button("📊 Evaluate", id="evaluate")
                    yield Button("⇩ Export", id="export")
                    yield Button("📖 Guide", id="guide")
                    yield Button(self._theme_label(), id="toggle_theme")
                with VerticalScroll(id="details"):
                    yield Static(WELCOME, id="details-text")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#models", DataTable)
        self.app.theme_changed_signal.subscribe(self, lambda _: self._sync_theme_button())
        t.add_columns("Name", "Task", "Layers", "Params", "Weights", "Epochs", "Modified")
        self.refresh_models()

    def _theme_label(self) -> str:
        # The button names the theme it switches *to*.
        return "☀ Light" if self.app.current_theme.dark else "☾ Dark"

    def _sync_theme_button(self) -> None:
        self.query_one("#toggle_theme", Button).label = self._theme_label()

    def action_toggle_theme(self) -> None:
        self.app.action_toggle_theme()

    def on_resize(self, event) -> None:
        # Give the model list the room on small terminals.
        self.query_one("#banner").display = event.size.height >= 32 and event.size.width >= 60

    def on_screen_resume(self) -> None:
        self.refresh_models(keep=self.selected)

    @property
    def ws(self):
        return self.app.ws

    @property
    def selected(self) -> str | None:
        t = self.query_one("#models", DataTable)
        if t.row_count == 0:
            return None
        return t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value

    def refresh_models(self, keep: str | None = None) -> None:
        t = self.query_one("#models", DataTable)
        t.clear()
        rows = self.ws.list_models()
        for r in rows:
            t.add_row(r["name"], r["task"], str(r["layers"]), f"{r['params']:,}",
                      "[green]trained[/]" if r["trained"] else "[dim]none[/]",
                      str(r["epochs"] or "-"), r["modified"], key=r["name"])
        self.query_one("#empty-hint").display = not rows
        t.display = bool(rows)
        names = [r["name"] for r in rows]
        if keep in names:
            t.move_cursor(row=names.index(keep))
        self.show_details()

    @on(DataTable.RowHighlighted, "#models")
    def _highlight(self) -> None:
        self.show_details()

    @on(DataTable.RowSelected, "#models")
    def _select(self) -> None:
        self.action_edit()

    def show_details(self) -> None:
        box = self.query_one("#details-text", Static)
        name = self.selected
        if not name:
            box.update(WELCOME)
            return
        spec = self.ws.load_spec(name)
        text = [f"[b $accent]{name}[/]", "", model_io.summary_text(spec)]
        hist = self.ws.load_history(name)
        if hist:
            text += ["", "[b]Training runs[/b]"]
            for i, run in enumerate(hist[-8:], max(1, len(hist) - 7)):
                last = run["epochs"][-1] if run.get("epochs") else {}
                metric = (f"val acc {last['val_acc']:.1%}" if "val_acc" in last else
                          f"val MAE {last['val_mae']:.4g}" if "val_mae" in last else
                          f"val perplexity {last['val_ppl']:.2f}" if "val_ppl" in last else "")
                kind = run.get("kind", "train")
                text.append(f" {i}. {run['started']}  {kind:<9} {len(run.get('epochs', []))} ep  "
                            f"loss {last.get('val_loss', last.get('train_loss', 0)):.4f}  {metric}"
                            f"  [dim]{run.get('dataset', '')}[/]")
        elif not self.ws.has_weights(name):
            text += ["", "[yellow]Not trained yet - press [b]t[/b] to train.[/]"]
        if evals := self._evaluations(name):
            h = evals[-1]
            score = (f"accuracy {h['accuracy']:.1%} · macro F1 {h['macro_f1']:.3f}" if "accuracy" in h
                     else f"perplexity {h['perplexity']:.2f} · next-token accuracy {h['token_accuracy']:.1%}"
                     if "perplexity" in h
                     else f"MAE {h['mae']:.4g}" + (f" · R² {h['r2']:.3f}" if h.get("r2") is not None else ""))
            text += ["", "[b]Last evaluation[/b]", f" {h['date']}  {score}  [dim]{h['data']}[/]"]
        box.update("\n".join(text))

    def _evaluations(self, name: str) -> list[dict]:
        import json

        try:
            return json.loads((self.ws.model_dir(name) / "evaluations.json").read_text())
        except (OSError, ValueError):
            return []

    # -- actions -----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = getattr(self, f"action_{event.button.id}", None)
        if action:
            action()

    def _need_model(self) -> str | None:
        name = self.selected
        if not name:
            self.notify("Create or import a model first.", severity="warning")
        return name

    def action_new(self) -> None:
        def done(spec):
            if spec:
                from .builder import BuilderScreen

                self.app.push_screen(BuilderScreen(spec, is_new=True))
        self.app.push_screen(NewModelDialog(self.ws), done)

    def action_edit(self) -> None:
        if name := self._need_model():
            from .builder import BuilderScreen

            self.app.push_screen(BuilderScreen(self.ws.load_spec(name)))

    def action_train(self) -> None:
        if name := self._need_model():
            from .train import TrainScreen

            self.app.push_screen(TrainScreen(name, finetune=False))

    def action_finetune(self) -> None:
        if name := self._need_model():
            if not self.ws.has_weights(name):
                self.notify("This model has no trained weights yet - fine-tuning starts from random "
                            "weights. Train it (or load weights) first for real transfer.",
                            severity="warning", timeout=8)
            from .train import TrainScreen

            self.app.push_screen(TrainScreen(name, finetune=True))

    def action_predict(self) -> None:
        if name := self._need_model():
            from .predict import PredictScreen

            self.app.push_screen(PredictScreen(name))

    def action_evaluate(self) -> None:
        if name := self._need_model():
            from .evaluate import EvaluateScreen

            self.app.push_screen(EvaluateScreen(name))

    def action_export(self) -> None:
        if name := self._need_model():
            from .export import ExportScreen

            self.app.push_screen(ExportScreen(name))

    def action_guide(self) -> None:
        from .guide import GuideScreen

        name = self.selected  # the lessons and encyclopedia work without a model
        self.app.push_screen(GuideScreen(self.ws.load_spec(name) if name else None))

    def action_datasets(self) -> None:
        from .datasets import DatasetScreen

        name = self.selected

        def done(cfg):
            if cfg and name:
                from .train import TrainScreen

                self.app.push_screen(TrainScreen(name, finetune=False, preset=cfg))
        self.app.push_screen(DatasetScreen(f"▶ Train '{name}' on it" if name else None), done)

    def action_import(self) -> None:
        def done(res):
            if res:
                self._do_import(res)
        self.app.push_screen(ImportDialog(), done)

    @work(thread=True, exclusive=True)
    def _do_import(self, res: dict) -> None:
        try:
            name, msg = model_io.import_path(self.ws, res["path"], res["name"], res["shape"])
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.notify, f"Import failed: {e}", severity="error", timeout=10)
            return
        self.app.call_from_thread(self.notify, msg, timeout=8)
        self.app.call_from_thread(self.refresh_models, name)

    def action_load_weights(self) -> None:
        if not (name := self._need_model()):
            return

        def done(path):
            if path:
                try:
                    self.notify(model_io.load_weights_into(self.ws, name, path))
                except Exception as e:  # noqa: BLE001
                    self.notify(f"Couldn't load weights: {e}", severity="error", timeout=10)
                self.refresh_models(name)
        self.app.push_screen(TextDialog(
            f"Load weights into '{name}'", "Weights file (.pt / .pth / .safetensors)",
            hint="Tensors are matched by layer position and shape; anything that doesn't fit is skipped.",
            browse={"title": "Choose a weights file", "mode": "file",
                    "extensions": {".pt", ".pth", ".bin", ".safetensors"}}),
            done)

    def _name_check(self, v: str) -> str | None:
        if not safe_name(v):
            return "Enter a valid name"
        if self.ws.exists(safe_name(v)):
            return "That name is taken"
        return None

    def action_duplicate(self) -> None:
        if not (name := self._need_model()):
            return

        def done(new):
            if new:
                self.ws.duplicate(name, safe_name(new))
                self.refresh_models(safe_name(new))
        self.app.push_screen(TextDialog("Duplicate model", "New name", self.ws.unique_name(name + "-copy"),
                                        validate=self._name_check), done)

    def action_rename(self) -> None:
        if not (name := self._need_model()):
            return

        def done(new):
            if new:
                self.ws.rename(name, safe_name(new))
                self.refresh_models(safe_name(new))
        self.app.push_screen(TextDialog("Rename model", "New name", name, validate=self._name_check), done)

    def action_delete(self) -> None:
        if not (name := self._need_model()):
            return

        def done(ok):
            if ok:
                self.ws.delete(name)
                self.notify(f"Deleted '{name}'")
                self.refresh_models()
        self.app.push_screen(ConfirmDialog("Delete model?", f"'{name}' and its weights and history will be "
                                           "removed permanently.", "Delete"), done)
