"""Reusable dialogs."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ..spec import CLASSIC_TASKS, LEARNING_STYLES, MANUAL, TEMPLATES, from_template, safe_name


def browse_into(screen, target: Input, **kwargs) -> None:
    """Open the file browser and put the chosen path into ``target``."""
    from .browser import FileBrowser

    current = target.value.strip()
    kwargs.setdefault("start", current or None)

    def done(path):
        if path:
            target.value = path
            target.focus()
    screen.app.push_screen(FileBrowser(**kwargs), done)


class Dialog(ModalScreen):
    """Base dialog. Note: Textual calls ``on_button_pressed`` of *every* class in the MRO, so
    subclasses handle only their own buttons and must not call ``super()`` for it."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    Dialog { align: center middle; }
    Dialog > Vertical {
        width: 76; height: auto; max-height: 90%; overflow-y: auto;  /* scrolls in a small terminal */
        border: thick $primary; background: $surface; padding: 1 2;
    }
    Dialog .title { text-style: bold; color: $accent; margin-bottom: 1; }
    Dialog .hint { color: $text-muted; margin-bottom: 1; }
    Dialog .error { color: $error; height: auto; }
    Dialog Label { margin-top: 1; }
    Dialog .buttons { height: auto; margin-top: 1; align-horizontal: right; }
    Dialog .buttons Button { margin-left: 1; }
    Dialog .path-row { height: auto; }
    Dialog .path-row Input { width: 1fr; }
    Dialog .path-row Button { margin-left: 1; }
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

    def submit(self) -> None:
        self.dismiss(True)


class TextDialog(Dialog):
    """Ask for a single value."""

    def __init__(self, title: str, label: str, value: str = "", hint: str = "",
                 validate=None, browse: dict | None = None):
        """browse: FileBrowser kwargs; adds a Browse button next to the input."""
        super().__init__()
        self._title, self._label, self._value, self._hint = title, label, value, hint
        self._validate = validate
        self._browse = browse

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes="title")
            if self._hint:
                yield Static(self._hint, classes="hint")
            yield Label(self._label)
            if self._browse is not None:
                with Horizontal(classes="path-row"):
                    yield Input(self._value, id="value")
                    yield Button("📂 Browse", id="browse")
            else:
                yield Input(self._value, id="value")
            yield Static("", id="error", classes="error")
            yield self.buttons()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse":
            browse_into(self, self.query_one("#value", Input), **self._browse)

    def submit(self) -> None:
        v = self.query_one("#value", Input).value.strip()
        err = self._validate(v) if self._validate else None
        if err:
            self.error(err)
        else:
            self.dismiss(v)


STYLE_CARDS = {  # learning style -> (card label, hint under the cards)
    "supervised": ("🎯 Supervised\n[dim]learns from labels[/]",
                   "Learns from examples with the right answers (labels): classify or predict."),
    "unsupervised": ("🔍 Unsupervised\n[dim]no answers needed[/]",
                     "Finds structure by itself: groups, oddities and a map of your data. "
                     "Labels are optional, only to check the result."),
    "reinforcement": ("🎮 Reinforcement\n[dim]learns from rewards[/]",
                      "An agent learns to act in a world by trial and error: rewards for good moves, "
                      "penalties for bad ones. No dataset needed. First time? Ctrl+P → agent tour."),
}


class NewModelDialog(Dialog):
    DEFAULT_CSS = """
    NewModelDialog #wizard { width: 100%; height: 1; border: none; }
    NewModelDialog #styles { height: auto; margin-top: 1; }
    NewModelDialog .style-card { width: 1fr; height: 4; min-width: 10; margin-right: 1; }
    NewModelDialog .style-card:last-of-type { margin-right: 0; }
    """

    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.style = "supervised"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("✚ Create a new model", classes="title")
            yield Button("✨ Not sure? Just describe what you want", id="wizard", variant="success")
            yield Static("Or pick a starting point - you can change every layer afterwards.", classes="hint")
            yield Label("Model name")
            yield Input(self.ws.unique_name("my-model"), id="name")
            yield Label("How will it learn?  [dim](Guide → lesson 3 explains the difference)[/]")
            with Horizontal(id="styles"):
                for key, (label, _) in STYLE_CARDS.items():
                    yield Button(label, id=f"style-{key}", classes="style-card",
                                 variant="primary" if key == self.style else "default",
                                 disabled=key not in LEARNING_STYLES)
            yield Static("", id="style-hint", classes="hint")
            yield Label("Model type")
            types = LEARNING_STYLES[self.style]
            yield Select([(t, t) for t in types], value=next(iter(types)), allow_blank=False, id="type")
            yield Label("Start from", id="template-label")
            yield Select([], allow_blank=True, id="template")
            yield Static("", id="preview", classes="hint")
            yield Static("", id="error", classes="error")
            yield self.buttons("Create & open builder")

    def on_mount(self) -> None:
        self._fill_types()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "wizard":
            self.dismiss("wizard")
            return
        key = (event.button.id or "").removeprefix("style-")
        if key in LEARNING_STYLES and key != self.style:
            self.style = key
            for k in STYLE_CARDS:
                self.query_one(f"#style-{k}", Button).variant = "primary" if k == key else "default"
            self._fill_types()

    def _fill_types(self) -> None:
        types = LEARNING_STYLES[self.style]
        sel = self.query_one("#type", Select)
        sel.set_options([(t, t) for t in types])
        sel.value = next(iter(types))
        self.query_one("#style-hint", Static).update(STYLE_CARDS[self.style][1])
        self._fill_templates()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "type":
            self._fill_templates()
        else:
            self._preview()

    def _kind(self) -> str:
        types = LEARNING_STYLES[self.style]
        value = self.query_one("#type", Select).value
        return value if value in types else next(iter(types))

    def _fill_templates(self) -> None:
        kind = self._kind()
        names = LEARNING_STYLES[self.style][kind]
        template = self.query_one("#template", Select)
        template.set_options([(t, t) for t in names])
        template.value = names[0]
        manual = kind == MANUAL
        template.display = not manual
        self.query_one("#template-label", Label).display = not manual
        self._preview()

    def _template(self) -> str:
        value = self.query_one("#template", Select).value
        return value if value in TEMPLATES else LEARNING_STYLES[self.style][self._kind()][0]

    def _preview(self) -> None:
        spec = from_template(self._template(), "x")
        layers = " → ".join(l.type for l in spec.layers) or "(empty - add layers yourself)"
        size = ("a classic algorithm: fitted in seconds, no parameters to train"
                if spec.task in CLASSIC_TASKS else f"{spec.total_params():,} parameters")
        self.query_one("#preview", Static).update(f"Input {tuple(spec.input_shape)} → {layers}\n{size}")

    def submit(self) -> None:
        name = safe_name(self.query_one("#name", Input).value)
        if not name:
            return self.error("Please enter a name (letters, numbers, - _ .)")
        if self.ws.exists(name):
            return self.error(f"A model named '{name}' already exists")
        self.dismiss(from_template(self._template(), name))


class ImportDialog(Dialog):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("⇪ Import an existing model", classes="title")
            yield Static("Accepted: AIForge bundle (.zip) or model folder, structure (.json), "
                         "AIForge .safetensors, or a PyTorch nn.Sequential saved with torch.save (.pt/.pth).\n"
                         "[b]Only import .pt files you trust[/b] - they can run code when loaded.",
                         classes="hint")
            yield Label("File or folder path")
            with Horizontal(classes="path-row"):
                yield Input(placeholder="~/Downloads/model.aiforge.zip", id="path")
                yield Button("📂 Browse", id="browse")
            yield Label("Name (optional)")
            yield Input(placeholder="keep original", id="name")
            yield Label("Input shape (only needed for .pt files that start with a conv layer)")
            yield Input(placeholder="e.g. 3,32,32", id="shape")
            yield Static("", id="error", classes="error")
            yield self.buttons("Import")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "browse":
            from .browser import MODEL_EXT

            browse_into(self, self.query_one("#path", Input), title="Choose a model to import",
                        mode="any", extensions=MODEL_EXT,
                        hint="Pick an AIForge .zip bundle / model folder, a .json structure, "
                             "a .safetensors file or a PyTorch .pt/.pth file.")

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
