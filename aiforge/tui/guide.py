"""Full-screen model guide: walk-through of the model, each of its layers, and an encyclopedia."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from ..guide import encyclopedia_markdown, layer_markdown, model_markdown
from ..layers import by_category
from ..spec import ModelSpec


class GuideScreen(Screen):
    BINDINGS = [Binding("escape,q", "app.pop_screen", "Back"),
                Binding("left,h", "focus_nav", "Topics", show=False),
                Binding("right,l", "focus_text", "Read", show=False)]

    def __init__(self, spec: ModelSpec, focus: str = "overview"):
        super().__init__()
        self.spec = spec
        self.start = focus  # "overview", "layer:<index>" or "type:<LayerType>"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="guide-nav-pane"):
                yield Static("📖 Guide", classes="section-title")
                yield OptionList(*self._options(), id="guide-nav")
            with VerticalScroll(id="guide-body"):
                yield Markdown(id="guide-text")
        yield Footer()

    def _options(self) -> list:
        opts: list = [Option(f"[b]This model: {self.spec.name}[/b]", disabled=True),
                      Option(" Overview & review", id="overview")]
        for i, layer in enumerate(self.spec.layers):
            opts.append(Option(f" {i + 1:>2}. {layer.type}", id=f"layer:{i}"))
        opts += [None, Option("[b]Layer encyclopedia[/b]", disabled=True)]
        for cat, items in by_category().items():
            if items:
                opts.append(Option(f"[dim]{cat}[/dim]", disabled=True))
                opts += [Option(f"   {lt.name}", id=f"type:{lt.name}") for lt in items]
        return opts

    def on_mount(self) -> None:
        nav = self.query_one("#guide-nav", OptionList)
        for idx in range(nav.option_count):
            if nav.get_option_at_index(idx).id == self.start:
                nav.highlighted = idx
                break
        else:
            nav.highlighted = 1
        nav.focus()

    @on(OptionList.OptionHighlighted, "#guide-nav")
    def _show(self, event: OptionList.OptionHighlighted) -> None:
        key = event.option.id or ""
        if key == "overview":
            text = model_markdown(self.spec)
        elif key.startswith("layer:"):
            text = layer_markdown(self.spec, int(key.split(":")[1]))
        elif key.startswith("type:"):
            text = encyclopedia_markdown(key.split(":")[1])
        else:
            return
        self.query_one("#guide-text", Markdown).update(text)
        self.query_one("#guide-body", VerticalScroll).scroll_home(animate=False)

    @on(OptionList.OptionSelected, "#guide-nav")
    def _selected(self) -> None:
        self.action_focus_text()

    def action_focus_nav(self) -> None:
        self.query_one("#guide-nav", OptionList).focus()

    def action_focus_text(self) -> None:
        self.query_one("#guide-body", VerticalScroll).focus()
