"""Serve the selected model as a web API + demo page while this dialog is open."""
from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Static

from .modals import Dialog


class ServeDialog(Dialog):
    DEFAULT_CSS = """
    ServeDialog #port { width: 14; }
    ServeDialog #status { height: auto; margin-top: 1; }
    """

    def __init__(self, ws, name: str):
        super().__init__()
        self.ws, self.model_name = ws, name
        self.server = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"🌐 Serve '{self.model_name}'", classes="title")
            yield Static("Start a small web server for this model: a demo page you can open in the "
                         "browser, and a JSON API other programs can call. Only this computer can reach "
                         "it. It stops when you close this window.", classes="hint")
            yield Label("Port")
            yield Input("8000", id="port", type="integer")
            yield Static("", id="status")
            yield Static("", id="error", classes="error")
            yield Horizontal(Button("Close", id="cancel"), Button("Open in browser", id="open", disabled=True),
                             Button("▶ Start", id="ok", variant="success"), classes="buttons")

    def on_mount(self) -> None:
        from ..serve import missing_dependencies

        if problem := missing_dependencies():
            self.error(problem)
            self.query_one("#ok", Button).disabled = True

    def submit(self) -> None:
        if self.server:
            self._stop()
            return
        try:
            port = int(self.query_one("#port", Input).value)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            self.error("The port is a number from 1 to 65535, e.g. 8000")
            return
        self.error("")
        self.query_one("#ok", Button).disabled = True
        self.query_one("#status", Static).update("[yellow]Starting…[/]")
        self._start(port)

    @work(thread=True, exclusive=True)
    def _start(self, port: int) -> None:
        from ..serve import Server, check_servable

        try:
            check_servable(self.ws, self.model_name)
            server = Server(self.ws, self.model_name, port=port)
            server.start()
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._failed, str(e))
            return
        self.app.call_from_thread(self._started, server)

    def _started(self, server) -> None:
        self.server = server
        self.query_one("#status", Static).update(
            f"[green]● Running[/] at [b]{server.url}[/b]\n"
            f"  API docs: {server.url}/docs · try it from a terminal:\n"
            f"  [dim]curl {server.url}/info[/]")
        self.query_one("#port", Input).disabled = True
        self.query_one("#open", Button).disabled = False
        ok = self.query_one("#ok", Button)
        ok.label, ok.variant, ok.disabled = "■ Stop", "error", False

    def _failed(self, msg: str) -> None:
        self.query_one("#status", Static).update("")
        self.error(msg)
        self.query_one("#ok", Button).disabled = False

    def _stop(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None
        self.query_one("#status", Static).update("[dim]Stopped.[/]")
        self.query_one("#port", Input).disabled = False
        self.query_one("#open", Button).disabled = True
        ok = self.query_one("#ok", Button)
        ok.label, ok.variant = "▶ Start", "success"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open" and self.server:
            self.app.open_url(self.server.url)

    def dismiss(self, result=None):
        if self.server:
            self.server.stop()
            self.server = None
        return super().dismiss(result)
