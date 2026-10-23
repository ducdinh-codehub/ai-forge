"""🐞 Report a problem or idea: fill in a GitHub issue and open it in the browser."""
from __future__ import annotations

import time

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static, Switch, TextArea

from .. import feedback
from .modals import Dialog


class ReportDialog(Dialog):
    """Dismisses with the saved report's path once GitHub was opened, or None."""

    DEFAULT_CSS = """
    ReportDialog > Vertical { width: 84; height: 90%; max-height: 50; }
    ReportDialog #report-form { height: 1fr; }
    ReportDialog TextArea { height: 7; }
    ReportDialog .switch-row { height: auto; }
    ReportDialog .switch-row Label { margin: 1 0 0 1; }
    ReportDialog #report-preview-box { height: 9; border: round $panel; margin-top: 1; }
    """

    def __init__(self, model_name: str | None = None, model_text: str = "", errors: str = "",
                 crash: str = "", kind: str = "bug"):
        super().__init__()
        self.model_name, self.model_text = model_name, model_text
        self.errors, self.crash, self.kind = errors, crash, kind
        self.system = feedback.system_info()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🐞 Report a problem or idea", classes="title")
            # The form scrolls; the error and buttons below it stay visible in small terminals.
            with VerticalScroll(id="report-form"):
                yield Static("This opens GitHub's [b]New issue[/b] page with everything below filled "
                             "in. Check it there and click [b]Submit[/b] (a free GitHub account is "
                             "needed). Nothing is sent before that.", classes="hint")
                yield Select([(label, k) for k, (label, _) in feedback.KINDS.items()], value=self.kind,
                             allow_blank=False, id="report-kind")
                yield Label("Title  [dim](one line)[/]")
                yield Input(placeholder="e.g. Training stops with 'nan' loss on my CSV", id="report-title")
                yield Label("", id="report-prompt")
                yield TextArea(id="report-text")
                if self.crash:
                    with Horizontal(classes="switch-row"):
                        yield Switch(True, id="inc-crash")
                        yield Label("Include the crash details from last time [dim](error traceback)[/]")
                if self.errors:
                    with Horizontal(classes="switch-row"):
                        yield Switch(True, id="inc-errors")
                        yield Label("Include the recent error messages")
                if self.model_text:
                    with Horizontal(classes="switch-row"):
                        yield Switch(False, id="inc-model")
                        yield Label(f"Include the layers of '{escape(self.model_name or '')}' [dim](structure "
                                    "only - no data or weights)[/]")
                with Horizontal(classes="switch-row"):
                    yield Switch(True, id="inc-system")
                    yield Label("Include versions and hardware [dim](AIForge, Python, PyTorch, OS)[/]")
                with VerticalScroll(id="report-preview-box"):
                    yield Static("", id="report-preview")
            yield Static("", id="error", classes="error")
            yield Horizontal(Button("Cancel", id="cancel"), Button("📋 Copy text", id="copy"),
                             Button("Open on GitHub ↗", id="ok", variant="primary"), classes="buttons")

    def on_mount(self) -> None:
        self._kind_changed()
        self.query_one("#report-title", Input).focus()

    def _switch(self, sid: str) -> bool:
        found = self.query(f"#{sid}")
        return bool(found) and found.first(Switch).value

    def report(self) -> feedback.Report:
        return feedback.Report(
            kind=self.query_one("#report-kind", Select).value,
            title=self.query_one("#report-title", Input).value,
            description=self.query_one("#report-text", TextArea).text,
            system=self.system if self._switch("inc-system") else "",
            errors=self.errors if self._switch("inc-errors") else "",
            model=self.model_text if self._switch("inc-model") else "",
            crash=self.crash if self._switch("inc-crash") else "")

    @on(Select.Changed, "#report-kind")
    def _kind_changed(self) -> None:
        kind = self.query_one("#report-kind", Select).value
        self.query_one("#report-prompt", Label).update(feedback.PROMPTS[kind])
        self._preview()

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Switch.Changed)
    def _preview(self) -> None:
        r = self.report()
        self.query_one("#report-preview", Static).update(
            f"[dim]Preview - exactly what goes into the issue[/]\n[b]{escape(r.full_title)}[/b]\n\n"
            f"{escape(r.body())}")

    def _save(self, r: feedback.Report):
        """Keep a copy on disk: the clipboard doesn't work in every terminal."""
        folder = self.app.ws.export_dir / "issue-reports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md"
        path.write_text(f"# {r.full_title}\n\n{r.body()}\n", encoding="utf-8")
        return path

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy":
            r = self.report()
            self.app.copy_to_clipboard(f"{r.full_title}\n\n{r.body()}")
            self.notify(f"Copied. Paste it at {feedback.NEW_ISSUE}", timeout=8)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.prevent_default()  # skip Dialog's handler: Enter in the title moves on, it doesn't send
        self.query_one("#report-text", TextArea).focus()

    def submit(self) -> None:
        r = self.report()
        if not r.title.strip():
            self.error("Please give it a short title.")
            self.query_one("#report-title", Input).focus()
            return
        if r.kind == "bug" and not r.description.strip() and not r.crash:
            self.error("Please describe what happened - it's the most useful part of the report.")
            self.query_one("#report-text", TextArea).focus()
            return
        url, _ = feedback.issue_url(r)
        path = self._save(r)
        self.app.copy_to_clipboard(f"{r.full_title}\n\n{r.body()}")
        self.app.open_url(url)
        self.dismiss(path)
