"""Reusable dialogs."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ..spec import TEMPLATES, from_template, safe_name


class Dialog(ModalScreen):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    Dialog { align: center middle; }
    Dialog > Vertical {
        width: 76; height: auto; max-height: 90%;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    Dialog .title { text-style: bold; color: $accent; margin-bottom: 1; }
    Dialog .hint { color: $text-muted; margin-bottom: 1; }
    Dialog .error { color: $error; height: auto; }
    Dialog Label { margin-top: 1; }
    Dialog .buttons { height: auto; margin-top: 1; align-horizontal: right; }
    Dialog .buttons Button { margin-left: 1; }
    """

    def action_cancel(self) -> None:
        self.dismiss(None)

    def buttons(self, ok: str = "OK", variant: str = "primary") -> Horizontal:
        return Horizontal(Button("Cancel", id="cancel"), Button(ok, id="ok", variant=variant),
                          classes="buttons")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "ok":
            self.submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submit()

    def submit(self) -> None:  # pragma: no cover - overridden
        self.dismiss(True)

    def error(self, msg: str) -> None:
        self.query_one("#error", Static).update(msg)


class ConfirmDialog(Dialog):
    def __init__(self, title: str, message: str, ok: str = "Yes", variant: str = "error"):
        super().__init__()
        self._title, self._message, self._ok, self._variant = title, message, ok, variant

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes="title")
            yield Static(self._message)
            yield self.buttons(self._ok, self._variant)

    def submit(self) -> None:
        self.dismiss(True)


class SaveChangesDialog(Dialog):
    """Returns True (save), False (discard) or None (stay)."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Unsaved changes", classes="title")
            yield Static("Save your changes to this model before leaving?")
            yield Horizontal(Button("Stay", id="cancel"), Button("Discard", id="discard", variant="error"),
                             Button("Save", id="ok", variant="primary"), classes="buttons")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "discard":
            self.dismiss(False)
        else:
            super().on_button_pressed(event)

    def submit(self) -> None:
        self.dismiss(True)


class TextDialog(Dialog):
    """Ask for a single value."""

    def __init__(self, title: str, label: str, value: str = "", hint: str = "",
                 validate=None):
        super().__init__()
        self._title, self._label, self._value, self._hint = title, label, value, hint
        self._validate = validate

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes="title")
            if self._hint:
                yield Static(self._hint, classes="hint")
            yield Label(self._label)
            yield Input(self._value, id="value")
            yield Static("", id="error", classes="error")
            yield self.buttons()

    def submit(self) -> None:
        v = self.query_one("#value", Input).value.strip()
        err = self._validate(v) if self._validate else None
        if err:
            self.error(err)
        else:
            self.dismiss(v)


class NewModelDialog(Dialog):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("✚ Create a new model", classes="title")
            yield Static("Pick a starting point - you can change every layer afterwards.", classes="hint")
            yield Label("Model name")
            yield Input(self.ws.unique_name("my-model"), id="name")
            yield Label("Start from")
            yield Select([(t, t) for t in TEMPLATES], value="MLP classifier (tabular)",
                         allow_blank=False, id="template")
            yield Static("", id="preview", classes="hint")
            yield Static("", id="error", classes="error")
            yield self.buttons("Create & open builder")

    def on_mount(self) -> None:
        self._preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._preview()

    def _preview(self) -> None:
        spec = from_template(self.query_one("#template", Select).value, "x")
        layers = " → ".join(l.type for l in spec.layers) or "(empty - add layers yourself)"
        self.query_one("#preview", Static).update(
            f"Input {tuple(spec.input_shape)} → {layers}\n{spec.total_params():,} parameters")

    def submit(self) -> None:
        name = safe_name(self.query_one("#name", Input).value)
        if not name:
            return self.error("Please enter a name (letters, numbers, - _ .)")
        if self.ws.exists(name):
            return self.error(f"A model named '{name}' already exists")
        self.dismiss(from_template(self.query_one("#template", Select).value, name))


class ImportDialog(Dialog):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⇪ Import an existing model", classes="title")
            yield Static("Accepted: AIForge bundle (.zip) or model folder, structure (.json), "
                         "AIForge .safetensors, or a PyTorch nn.Sequential saved with torch.save (.pt/.pth).\n"
                         "[b]Only import .pt files you trust[/b] - they can run code when loaded.",
                         classes="hint")
            yield Label("File or folder path")
            yield Input(placeholder="~/Downloads/model.aiforge.zip", id="path")
            yield Label("Name (optional)")
            yield Input(placeholder="keep original", id="name")
            yield Label("Input shape (only needed for .pt files that start with a conv layer)")
            yield Input(placeholder="e.g. 3,32,32", id="shape")
            yield Static("", id="error", classes="error")
            yield self.buttons("Import")

    def submit(self) -> None:
        path = self.query_one("#path", Input).value.strip()
        if not path:
            return self.error("Enter a path")
        shape_txt = self.query_one("#shape", Input).value.strip()
        shape = None
        if shape_txt:
            try:
                shape = [int(v) for v in shape_txt.replace("x", ",").split(",") if v.strip()]
            except ValueError:
                return self.error("Input shape must look like 3,32,32")
        self.dismiss({"path": path, "name": self.query_one("#name", Input).value.strip() or None,
                      "shape": shape})
