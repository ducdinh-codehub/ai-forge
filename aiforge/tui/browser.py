"""File / folder browser dialog with a live preview of what's selected."""
from __future__ import annotations

import csv
import itertools
import json
import time
import zipfile
from pathlib import Path
from typing import Iterable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label, OptionList, Static, Switch
from textual.widgets.option_list import Option

from .. import EXAMPLES_DIR
from ..data import IMAGE_EXT, TEXT_EXT

# Ready-made filters for the places that ask for a path.
CSV_EXT = {".csv", ".tsv", ".txt"}
MODEL_EXT = {".zip", ".json", ".safetensors", ".pt", ".pth", ".bin"}
WEIGHT_EXT = {".pt", ".pth", ".bin", ".safetensors"}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return str(n)


class FilteredTree(DirectoryTree):
    """DirectoryTree that hides dot-files and files with unwanted extensions."""

    def __init__(self, path, extensions: set[str] | None, dirs_only: bool, **kw):
        super().__init__(path, **kw)
        self.extensions = extensions
        self.dirs_only = dirs_only
        self.show_hidden = False
        self.show_all = False

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        out = []
        for p in paths:
            if not self.show_hidden and p.name.startswith("."):
                continue
            try:
                is_dir = p.is_dir()
            except OSError:
                continue
            if is_dir:
                out.append(p)
            elif self.show_all:
                out.append(p)
            elif not self.dirs_only and (self.extensions is None or p.suffix.lower() in self.extensions):
                out.append(p)
        return out


class FileBrowser(ModalScreen[str | None]):
    """Pick a file or folder. Returns the chosen path as a string, or None."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"),
                Binding("backspace", "up", "Parent folder", show=False),
                Binding("ctrl+h", "toggle_hidden", "Hidden files", show=False)]
    DEFAULT_CSS = """
    FileBrowser { align: center middle; }
    FileBrowser > Vertical {
        width: 95%; height: 92%; border: thick $primary; background: $surface; padding: 0 1;
    }
    FileBrowser .title { text-style: bold; color: $accent; margin-top: 1; height: auto; }
    FileBrowser .hint { color: $text-muted; height: auto; }
    FileBrowser #fb-main { height: 1fr; margin-top: 1; }
    FileBrowser #fb-places { width: 22; border: round $panel; }
    FileBrowser #fb-tree { width: 1fr; border: round $panel; }
    FileBrowser #fb-preview-box { width: 46; border: round $panel; padding: 0 1; }
    FileBrowser #fb-path-row { height: auto; margin-top: 1; }
    FileBrowser #fb-path { width: 1fr; }
    FileBrowser #fb-bottom { height: auto; margin-bottom: 1; }
    FileBrowser #fb-bottom Label { margin: 1 2 0 1; }
    FileBrowser #fb-status { width: 1fr; padding-top: 1; height: auto; }
    FileBrowser #fb-bottom Button { margin-left: 1; }
    """

    def __init__(self, title: str = "Choose a file", mode: str = "file",
                 extensions: set[str] | None = None, start: str | Path | None = None,
                 hint: str = "", extra_places: list[tuple[str, Path]] | None = None):
        """mode: "file" (pick a file), "dir" (pick a folder) or "any" (file or folder)."""
        super().__init__()
        self.title_text, self.mode, self.extensions, self.hint = title, mode, extensions, hint
        self.extra_places = extra_places or []
        self.selected: Path | None = None
        start_path = Path(str(start or "")).expanduser() if start else Path.cwd()
        if start_path.is_file():
            self.selected = start_path
            start_path = start_path.parent
        while not start_path.is_dir() and start_path != start_path.parent:
            start_path = start_path.parent
        self.root = start_path.resolve() if start_path.is_dir() else Path.home()

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        what = {"file": "a file", "dir": "a folder", "any": "a file or folder"}[self.mode]
        exts = ", ".join(sorted(self.extensions)) if self.extensions and self.mode != "dir" else ""
        with Vertical():
            yield Static(f"📂 {self.title_text}", classes="title")
            yield Static(self.hint or f"Choose {what}" + (f" ({exts})" if exts else "") +
                         ". ↑↓ move · Enter open/choose · Backspace parent folder", classes="hint")
            with Horizontal(id="fb-main"):
                yield OptionList(*self._places(), id="fb-places")
                yield FilteredTree(self.root, self.extensions, self.mode == "dir", id="fb-tree")
                with VerticalScroll(id="fb-preview-box"):
                    yield Static("", id="fb-preview")
            with Horizontal(id="fb-path-row"):
                yield Button("⬆ Up", id="fb-up")
                yield Input(str(self.selected or self.root), id="fb-path",
                            placeholder="type a path and press Enter")
            with Horizontal(id="fb-bottom"):
                yield Switch(False, id="fb-hidden")
                yield Label("hidden")
                yield Switch(False, id="fb-all")
                yield Label("all file types")
                yield Static("", id="fb-status")
                yield Button("Cancel", id="fb-cancel")
                yield Button("Select", id="fb-ok", variant="primary")

    def _places(self) -> list[Option]:
        home = Path.home()
        places = [("🏠 Home", home), ("🖥  Desktop", home / "Desktop"),
                  ("⬇  Downloads", home / "Downloads"), ("📄 Documents", home / "Documents"),
                  ("📦 Examples", EXAMPLES_DIR), ("📁 Current folder", Path.cwd())]
        try:
            places.append(("🧠 Workspace", self.app.ws.root))
        except Exception:  # noqa: BLE001 - no app workspace (e.g. tests)
            pass
        places += self.extra_places + [("💽 Computer", Path(Path.cwd().anchor))]
        self._place_paths = {}
        opts = []
        for i, (label, p) in enumerate(places):
            if p.is_dir():
                self._place_paths[f"place{i}"] = p
                opts.append(Option(label, id=f"place{i}"))
        return opts

    def on_mount(self) -> None:
        self.query_one("#fb-tree", FilteredTree).focus()
        self._set_selected(self.selected or self.root)

    # -- navigation ----------------------------------------------------------

    def go(self, folder: Path) -> None:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            return
        self.root = folder
        tree = self.query_one("#fb-tree", FilteredTree)
        tree.path = folder
        self._set_selected(folder)
        tree.focus()

    @on(OptionList.OptionSelected, "#fb-places")
    def _place(self, event: OptionList.OptionSelected) -> None:
        self.go(self._place_paths[event.option.id])

    def action_up(self) -> None:
        if isinstance(self.focused, Input):
            return
        self.go(self.root.parent)

    @on(Button.Pressed, "#fb-up")
    def _up(self) -> None:
        self.go(self.root.parent)

    @on(DirectoryTree.NodeHighlighted, "#fb-tree")
    def _highlight(self, event: DirectoryTree.NodeHighlighted) -> None:
        if event.node.data is not None:
            self._set_selected(Path(event.node.data.path))

    @on(DirectoryTree.FileSelected, "#fb-tree")
    def _file_chosen(self, event: DirectoryTree.FileSelected) -> None:
        self._set_selected(Path(event.path))
        if self.mode in ("file", "any"):
            self._accept()

    @on(DirectoryTree.DirectorySelected, "#fb-tree")
    def _dir_chosen(self, event: DirectoryTree.DirectorySelected) -> None:
        self._set_selected(Path(event.path))

    @on(Input.Submitted, "#fb-path")
    def _typed(self, event: Input.Submitted) -> None:
        p = Path(event.value.strip().strip("'\"")).expanduser()
        if p.is_dir() and self.mode == "file":
            self.go(p)
        elif p.exists():
            self._set_selected(p)
            self._accept()
        else:
            self._status(f"[red]Not found: {p}[/]")

    @on(Switch.Changed, "#fb-hidden")
    def _hidden(self, event: Switch.Changed) -> None:
        tree = self.query_one("#fb-tree", FilteredTree)
        tree.show_hidden = event.value
        tree.reload()

    @on(Switch.Changed, "#fb-all")
    def _all(self, event: Switch.Changed) -> None:
        tree = self.query_one("#fb-tree", FilteredTree)
        tree.show_all = event.value
        tree.reload()

    def action_toggle_hidden(self) -> None:
        sw = self.query_one("#fb-hidden", Switch)
        sw.value = not sw.value

    # -- choosing ------------------------------------------------------------

    def _acceptable(self, p: Path | None) -> str | None:
        """None if p can be returned, else the reason it can't."""
        if p is None or not p.exists():
            return "Nothing selected"
        if self.mode == "dir" and not p.is_dir():
            return "Please choose a folder"
        if self.mode == "file" and p.is_dir():
            return "Please choose a file (Enter opens folders)"
        return None

    def _set_selected(self, p: Path) -> None:
        self.selected = p
        inp = self.query_one("#fb-path", Input)
        if inp.value != str(p):
            inp.value = str(p)
        self.query_one("#fb-preview", Static).update(preview(p))
        reason = self._acceptable(p)
        self.query_one("#fb-ok", Button).disabled = reason is not None
        self._status("" if reason is None else f"[dim]{reason}[/]")

    def _status(self, msg: str) -> None:
        self.query_one("#fb-status", Static).update(msg)

    def _accept(self) -> None:
        reason = self._acceptable(self.selected)
        if reason:
            self._status(f"[yellow]{reason}[/]")
            return
        self.dismiss(str(self.selected))

    @on(Button.Pressed, "#fb-ok")
    def _ok(self) -> None:
        self._accept()

    @on(Button.Pressed, "#fb-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def preview(p: Path) -> Text | str:
    try:
        if p.is_dir():
            return _preview_dir(p)
        suffix = p.suffix.lower()
        head = f"[b]{p.name}[/b]\n[dim]{_human(p.stat().st_size)} · modified " \
               f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}[/]\n\n"
        if suffix in TEXT_EXT and (suffix != ".txt" or not _looks_tabular(p)):
            return head + _preview_text(p)
        if suffix in CSV_EXT:
            return head + _preview_csv(p)
        if suffix in IMAGE_EXT:
            return _preview_image(p, head)
        if suffix == ".zip":
            return head + _preview_zip(p)
        if suffix == ".json":
            return head + _preview_json(p)
        if suffix in WEIGHT_EXT:
            return head + _preview_weights(p)
        return head + "[dim]No preview for this file type.[/]"
    except PermissionError:
        return "[red]Permission denied[/]"
    except Exception as e:  # noqa: BLE001 - preview must never crash the dialog
        return f"[red]Can't preview: {str(e)[:120]}[/]"


def _preview_dir(p: Path) -> str:
    try:
        entries = [e for e in itertools.islice(p.iterdir(), 5000) if not e.name.startswith(".")]
    except PermissionError:
        return f"[b]📁 {p.name or p}[/b]\n\n[red]Permission denied[/]"
    dirs = sorted((e for e in entries if e.is_dir()), key=lambda e: e.name.lower())
    files = [e for e in entries if e.is_file()]
    lines = [f"[b]📁 {p.name or p}[/b]", f"[dim]{len(dirs)} folders · {len(files)} files[/]", ""]
    # Does this look like an image-classification dataset (one sub-folder per class)?
    classes = []
    for d in dirs[:200]:
        try:
            n = sum(1 for f in itertools.islice(d.iterdir(), 5000) if f.suffix.lower() in IMAGE_EXT)
        except PermissionError:
            continue
        if n:
            classes.append((d.name, n))
    if len(classes) >= 2:
        total = sum(n for _, n in classes)
        lines += [f"[green]✓ Image dataset: {len(classes)} classes, {total:,} images[/]", ""]
        top = max(n for _, n in classes)
        for name, n in classes[:12]:
            bar = "█" * max(1, round(n / top * 14))
            lines.append(f" {name[:14]:<14} [cyan]{bar}[/] {n:,}")
        if len(classes) > 12:
            lines.append(f" [dim]… {len(classes) - 12} more classes[/]")
        if min(n for _, n in classes) < 20:
            lines += ["", "[yellow]Some classes have < 20 images - consider the "
                          "Transfer learning template.[/]"]
        return "\n".join(lines)
    imgs = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    csvs = [f for f in files if f.suffix.lower() in CSV_EXT]
    if imgs:
        lines.append(f"🖼  {len(imgs)} images (usable for batch prediction)")
    if csvs:
        lines.append(f"📄 {len(csvs)} CSV files")
    if (p / "spec.json").exists():
        lines.append("[green]✓ AIForge model folder[/]")
    lines.append("")
    for e in (dirs + sorted(files, key=lambda e: e.name.lower()))[:18]:
        lines.append(f" {'📁' if e.is_dir() else '·'} {e.name[:38]}")
    if len(entries) > 18:
        lines.append(f" [dim]… {len(entries) - 18} more[/]")
    return "\n".join(lines)


def _preview_csv(p: Path) -> str:
    with p.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = []
        n = 0
        for row in csv.reader(f, dialect):
            if n < 7:
                rows.append(row)
            n += 1
    if not rows:
        return "[yellow]Empty file[/]"
    header, data = rows[0], rows[1:]
    lines = [f"[green]{n - 1:,} rows · {len(header)} columns[/]", "", "[b]Columns[/b]"]
    for i, h in enumerate(header[:14]):
        vals = [r[i] for r in data if i < len(r) and r[i].strip()]
        numeric = vals and all(_is_num(v) for v in vals)
        kind = "[cyan]number[/]" if numeric else "[magenta]text[/]"
        lines.append(f" {h[:22]:<22} {kind}")
    if len(header) > 14:
        lines.append(f" [dim]… {len(header) - 14} more[/]")
    lines += ["", f"[dim]Last column '{header[-1]}' is the default target.[/]", "", "[b]First rows[/b]"]
    for r in data[:5]:
        lines.append(" " + ", ".join(v[:10] for v in r[:6])[:42])
    return "\n".join(lines)


def _looks_tabular(p: Path) -> bool:
    """A .txt can be a table (tab/comma separated) or plain prose: every first line having the
    same, non-zero number of separators means a table."""
    with p.open(encoding="utf-8", errors="replace") as f:
        lines = [l for l in (f.readline() for _ in range(8)) if l.strip()]
    return any(len({l.count(sep) for l in lines}) == 1 and lines[0].count(sep) > 0
               for sep in ("\t", ",", ";", "|")) if len(lines) >= 2 else False


def _preview_text(p: Path) -> str:
    from rich.markup import escape

    with p.open(encoding="utf-8", errors="replace") as f:
        sample = f.read(1200)
    size = p.stat().st_size
    words = len(sample.split()) * max(1, size // max(1, len(sample.encode())))
    lines = [f"[green]Plain text · about {words:,} words[/]", "",
             "[dim]For a language model: it learns to write text like this.[/]", "", "[b]Beginning[/b]"]
    lines += [" " + escape(l[:42]) for l in sample.splitlines()[:16]]
    return "\n".join(lines)


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _preview_image(p: Path, head: str) -> Text:
    from PIL import Image

    out = Text.from_markup(head)
    with Image.open(p) as im:
        out.append_text(Text.from_markup(f"[green]{im.width}×{im.height} px · {im.mode}[/]\n\n"))
        w = 40
        h = max(2, round(im.height / im.width * w / 2) * 2)
        h = min(h, 40)
        small = im.convert("RGB").resize((w, h))
    px = small.load()
    # Half-block thumbnail: each character shows two pixels (top = fg, bottom = bg).
    for y in range(0, h, 2):
        for x in range(w):
            t, b = px[x, y], px[x, y + 1]
            out.append("▀", style=f"rgb({t[0]},{t[1]},{t[2]}) on rgb({b[0]},{b[1]},{b[2]})")
        out.append("\n")
    return out


def _preview_zip(p: Path) -> str:
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        lines = [f"{len(names)} entries"]
        if "spec.json" in names:
            spec = json.loads(z.read("spec.json"))
            lines = ["[green]✓ AIForge model bundle[/]", ""] + _spec_lines(spec) + \
                    ["", "weights: " + ("[green]yes[/]" if "weights.pt" in names else "[yellow]no[/]")]
        else:
            lines += [f" · {n}" for n in names[:15]]
    return "\n".join(lines)


def _spec_lines(spec: dict) -> list[str]:
    layers = spec.get("layers", [])
    lines = [f"model:  {spec.get('name')}", f"task:   {spec.get('task')}",
             f"input:  {tuple(spec.get('input_shape', []))}", f"layers: {len(layers)}"]
    lines += [f"  {i + 1:>2}. {l.get('type')}" for i, l in enumerate(layers[:10])]
    if len(layers) > 10:
        lines.append(f"  … {len(layers) - 10} more")
    if spec.get("class_names"):
        lines.append(f"classes: {', '.join(spec['class_names'][:6])}")
    return lines


def _preview_json(p: Path) -> str:
    if p.stat().st_size > 5_000_000:
        return "[dim]Large JSON file[/]"
    d = json.loads(p.read_text())
    if isinstance(d, dict) and "layers" in d and "input_shape" in d:
        return "\n".join(["[green]✓ AIForge model structure[/]", ""] + _spec_lines(d))
    keys = list(d)[:15] if isinstance(d, dict) else []
    return "JSON " + (f"object with keys:\n" + "\n".join(f" · {k}" for k in keys) if keys
                      else type(d).__name__)


def _preview_weights(p: Path) -> str:
    if p.suffix.lower() == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(p), "pt") as f:
            keys = list(f.keys())
            meta = f.metadata() or {}
        lines = [f"[green]safetensors · {len(keys)} tensors[/]"]
        if "aiforge_spec" in meta:
            lines += ["[green]✓ includes AIForge structure[/]", ""] + _spec_lines(json.loads(meta["aiforge_spec"]))
        else:
            lines += [""] + [f" · {k}" for k in keys[:12]]
        return "\n".join(lines)
    return ("PyTorch file (weights or a saved model).\n\n"
            "[yellow]Only open .pt files you trust - they can run code when loaded.[/]")
