"""Dialog offering verified ways to make a model fit the loaded data."""
from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from ..fixes import Fix
from ..spec import ModelSpec

ICONS = {"auto": "🔧", "template": "📦", "builder": "✎"}


def _chain(spec: ModelSpec, inserted: tuple[int, ...]) -> str:
    rows = ["| # | Layer | Output | Params |", "|---|---|---|---|",
            f"| | Input | `{'×'.join(map(str, spec.input_shape))}` | |"]
    for i in spec.analyze():
        new = i.index in inserted
        name = f"**➕ {i.spec.type}** *(new)*" if new else i.spec.type
        rows.append(f"| {i.index + 1} | {name} | `{'×'.join(map(str, i.out_shape))}` | {i.params:,} |")
    return "\n".join(rows)


class FixModelDialog(ModalScreen[Fix | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    FixModelDialog { align: center middle; }
    FixModelDialog > Vertical {
        width: 92%; height: 90%; border: thick $warning; background: $surface; padding: 0 2;
    }
    FixModelDialog #fx-title { text-style: bold; color: $warning; margin-top: 1; height: auto; }
    FixModelDialog #fx-why { height: auto; max-height: 12; margin-top: 1; }
    FixModelDialog #fx-main { height: 1fr; margin-top: 1; }
    FixModelDialog #fx-options { width: 46; height: 100%; }
    FixModelDialog #fx-preview { width: 1fr; height: 100%; border: round $panel; padding: 0 1; margin-left: 1; }
    FixModelDialog #fx-buttons { height: auto; margin: 1 0; align-horizontal: right; }
    FixModelDialog #fx-buttons Button { margin-left: 1; }
    FixModelDialog #fx-keys { width: 1fr; padding-top: 1; }
    """

    def __init__(self, explanation: str, fixes: list[Fix]):
        super().__init__()
        self.explanation, self.fixes = explanation, fixes

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("🔧 The model doesn't fit this data yet", id="fx-title")
            with VerticalScroll(id="fx-why"):
                yield Markdown(self.explanation)
            with Horizontal(id="fx-main"):
                yield OptionList(*[self._option(i, f) for i, f in enumerate(self.fixes)], id="fx-options")
                with VerticalScroll(id="fx-preview"):
                    yield Markdown(id="fx-preview-text")
            with Horizontal(id="fx-buttons"):
                yield Static("[dim]↑↓ choose · Enter apply · Esc cancel[/]", id="fx-keys")
                yield Button("Cancel", id="fx-cancel")
                yield Button("Apply", id="fx-apply", variant="warning")

    def _option(self, i: int, f: Fix) -> Option:
        t = Text()
        t.append(f"{ICONS[f.kind]} {f.title}", style="bold")
        if i == 0 and f.kind != "builder":
            t.append("  ★ suggested", style="green")
        if f.spec is not None:
            before = f"{f.params_before:,}" if f.params_before else "—"
            t.append(f"\n   params {before} → {f.params_after:,}", style="dim")
        else:
            t.append("\n   do it by hand", style="dim")
        return Option(t, id=str(i))

    def on_mount(self) -> None:
        opts = self.query_one("#fx-options", OptionList)
        opts.highlighted = 0
        opts.focus()
        self._preview(0)

    @on(OptionList.OptionHighlighted, "#fx-options")
    def _highlight(self, event: OptionList.OptionHighlighted) -> None:
        self._preview(int(event.option.id))

    def _preview(self, i: int) -> None:
        f = self.fixes[i]
        md = [f"## {f.title}", "", f.description]
        if f.spec is not None:
            md += ["", "### The model after this fix", "", _chain(f.spec, f.inserted), "",
                   f"✅ **Checked: fits your data** · {f.spec.total_params():,} parameters"]
        self.query_one("#fx-preview-text", Markdown).update("\n".join(md))
        self.query_one("#fx-apply", Button).label = "Open builder" if f.kind == "builder" else "Apply"

    @on(OptionList.OptionSelected, "#fx-options")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.fixes[int(event.option.id)])

    @on(Button.Pressed, "#fx-apply")
    def _apply(self) -> None:
        idx = self.query_one("#fx-options", OptionList).highlighted
        self.dismiss(self.fixes[idx or 0])

    @on(Button.Pressed, "#fx-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)
