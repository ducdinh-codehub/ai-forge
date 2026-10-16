"""Export weights / structure / full model."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, SelectionList, Static

from .. import model_io


class ExportScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, name: str):
        super().__init__()
        self.model_name = name

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="export"):
            yield Static(f"Export [b]{self.model_name}[/b]", classes="section-title")
            yield Static("", id="export-warn")
            yield Label("Formats  [dim](space to toggle)[/]")
            default = {"bundle", "weights_pt", "spec_json"}
            yield SelectionList(*[(f"{label:<40} [dim]{desc}[/]", key, key in default)
                                  for key, (label, desc) in model_io.EXPORT_FORMATS.items()],
                                id="formats")
            yield Label("Output folder")
            with Horizontal(classes="input-row"):
                yield Input(str(self.app.ws.export_dir / self.model_name), id="out")
                yield Button("⇩ Export", id="go", variant="primary")
            yield RichLog(id="export-log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        if not self.app.ws.has_weights(self.model_name):
            self.query_one("#export-warn", Static).update(
                "[yellow]⚠ Not trained yet - exported weights will be random. "
                "Structure exports are still useful.[/]")
        problems = self.app.ws.load_spec(self.model_name).validate()
        if any("Layer" in p for p in problems):
            self.query_one("#export-warn", Static).update(f"[red]✗ {problems[0]} - fix it in the builder.[/]")
            self.query_one("#go", Button).disabled = True

    @on(Button.Pressed, "#go")
    def _go(self) -> None:
        fmts = list(self.query_one("#formats", SelectionList).selected)
        if not fmts:
            self.notify("Choose at least one format.", severity="warning")
            return
        self.query_one("#go", Button).disabled = True
        self._export(fmts, self.query_one("#out", Input).value.strip())

    @work(thread=True, exclusive=True)
    def _export(self, fmts: list[str], out: str) -> None:
        log = self.query_one("#export-log", RichLog)
        write = lambda m: self.app.call_from_thread(log.write, m)  # noqa: E731
        ok = 0
        for f in fmts:
            label = model_io.EXPORT_FORMATS[f][0]
            try:
                path = model_io.export(self.app.ws, self.model_name, f, out or None)
                size = path.stat().st_size
                write(f"[green]✓[/] {label:<40} → {path}  [dim]{size / 1024:,.1f} KB[/]")
                ok += 1
            except Exception as e:  # noqa: BLE001
                write(f"[red]✗ {label}: {str(e).splitlines()[0][:200]}[/]")
        write(f"[b]Done: {ok}/{len(fmts)} exported.[/b]")
        self.app.call_from_thread(setattr, self.query_one("#go", Button), "disabled", False)
