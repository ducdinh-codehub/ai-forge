"""Visual model builder: layer palette | layer stack with live shapes & params | properties."""
from __future__ import annotations

import copy

import torch
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, Markdown, OptionList, Select, Static,
                             Switch, TabbedContent, TabPane)
from textual.widgets.option_list import Option

from ..guide import layer_card_markdown, model_card_markdown
from ..layers import REGISTRY, by_category, coerce
from ..spec import LayerSpec, ModelSpec, remap_state, safe_name
from .modals import SaveChangesDialog

BAR = 8

# Input dimensions offered in Model settings: (label, name of each dimension, default shape)
RANKS = {
    1: ("1D · values (tables, text)", ["Values"], [10]),
    2: ("2D · grid (length×features)", ["Length", "Features"], [28, 28]),
    3: ("3D · image (channels×H×W)", ["Channels", "Height", "Width"], [3, 32, 32]),
}


def reshape_input(shape: list[int], rank: int, dtype: str) -> list[int]:
    """A sensible input shape after switching to ``rank`` dimensions, reusing what fits."""
    shape = list(shape)
    if rank == len(shape):
        return shape
    if rank == 1:  # flatten: 1×28×28 -> 784, like a Flatten layer would
        return [128] if dtype == "long" else [max(1, int(torch.tensor(shape).prod()))]
    if rank == 2:
        return shape[1:] if len(shape) == 3 else list(RANKS[2][2])
    if rank == 3 and len(shape) == 2:  # add a channel: 28×28 -> 1×28×28
        return [1, *shape]
    return list(RANKS[rank][2])


def _shape(s) -> str:
    return "×".join(str(d) for d in s) if s else "-"


def _cfg_summary(layer: LayerSpec) -> str:
    lt = REGISTRY[layer.type]
    parts = []
    for p in lt.params:
        v = layer.config.get(p.name)
        if p.kind == "bool":
            if v != p.default:
                parts.append(f"{p.name}={'on' if v else 'off'}")
        else:
            parts.append(f"{v}" if len(lt.params) == 1 else f"{_short(p.name)}={v}")
    return " ".join(parts)


def _short(name: str) -> str:
    return {"out_features": "out", "out_channels": "ch", "kernel_size": "k", "stride": "s",
            "padding": "pad", "num_embeddings": "vocab", "embedding_dim": "dim",
            "hidden_size": "hidden", "num_layers": "layers", "negative_slope": "slope",
            "num_heads": "heads", "ff_dim": "ff", "patch_size": "patch", "embed_dim": "dim"}.get(name, name)


class BuilderScreen(Screen):
    BINDINGS = [
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("escape", "back", "Back"),
        Binding("a", "focus_palette", "Add layer"),
        Binding("delete,backspace", "remove", "Remove"),
        Binding("shift+up,K", "move(-1)", "Move up"),
        Binding("shift+down,J", "move(1)", "Move down"),
        Binding("ctrl+d", "duplicate_layer", "Duplicate"),
        Binding("ctrl+z", "undo", "Undo"),
        Binding("question_mark,f1", "guide", "Guide"),
        Binding("ctrl+e", "toggle_explain", "Explain tab"),
    ]

    def __init__(self, spec: ModelSpec, is_new: bool = False):
        super().__init__()
        self.spec = spec
        self.original_name = None if is_new else spec.name
        self.saved_state = None if is_new else copy.deepcopy(spec.to_dict())
        self.undo_stack: list[dict] = []
        self._props_for: str | None = None  # id of what the properties panel shows

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="palette-pane"):
                yield Static("Layers  [dim](Enter = add)[/]", classes="section-title")
                options: list[Option | None] = []
                for cat, items in by_category().items():
                    if not items:
                        continue
                    options.append(Option(f"[b]{cat}[/b]", disabled=True))
                    options += [Option(f" {lt.name}", id=lt.name) for lt in items]
                yield OptionList(*options, id="palette")
                yield Static("", id="palette-help")
            with Vertical(id="stack-pane"):
                yield Static("", id="stack-title", classes="section-title")
                yield DataTable(id="stack", cursor_type="row", zebra_stripes=True)
                yield Static("", id="totals")
                with Horizontal(classes="toolbar"):
                    yield Button("↑", id="up")
                    yield Button("↓", id="down")
                    yield Button("Duplicate", id="dup")
                    yield Button("Remove", id="remove", variant="error")
                    yield Button("Undo", id="undo")
                    yield Static(classes="spacer")
                    yield Button("Save", id="save", variant="primary")
            with TabbedContent(id="side-tabs", initial="tab-settings"):
                with TabPane("Settings", id="tab-settings"):
                    yield VerticalScroll(id="props")
                with TabPane("Explain", id="tab-explain"):
                    with VerticalScroll(id="explain-scroll"):
                        yield Markdown(id="explain")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#stack", DataTable)
        t.add_column("#", key="n", width=3)
        t.add_column("Layer", key="layer", width=16)
        t.add_column("Settings", key="cfg", width=20)
        t.add_column("Output shape", key="out", width=13)
        t.add_column("Params", key="params", width=11)
        t.add_column("Share", key="share", width=BAR + 6)
        t.add_column("How it's counted", key="formula")
        self.refresh_stack()
        self.query_one("#palette", OptionList).highlighted = 1
        t.focus()

    # -- stack rendering ---------------------------------------------------

    def refresh_stack(self, cursor: int | None = None) -> None:
        t = self.query_one("#stack", DataTable)
        infos = self.spec.analyze()
        total = sum(i.params for i in infos) or 1
        rows = [("·", "[b]Input[/b]", f"{self.spec.input_dtype}", _shape(self.spec.input_shape),
                 "", "", "[dim]select to edit model settings[/]")]
        for i in infos:
            frozen = " ❄" if i.spec.id in self.spec.frozen else ""
            if i.error:
                out, note = "[red]✗ error[/]", f"[red]{i.error}[/]"
            else:
                out, note = _shape(i.out_shape), f"[dim]{i.formula}[/]"
            share = i.params / total
            bar = "█" * round(share * BAR)
            rows.append((str(i.index + 1), f"{i.spec.type}{frozen}", _cfg_summary(i.spec), out,
                         f"{i.params:,}" if i.params else "[dim]0[/]",
                         f"{share:>4.0%} [cyan]{bar}[/]" if i.params else "", note))
        keep = t.cursor_row if cursor is None else cursor
        if t.row_count == len(rows):  # update in place so focus/scroll stay put
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    t.update_cell_at((r, c), val)
        else:
            t.clear()
            for r, row in enumerate(rows):
                t.add_row(*row, key=str(r))
        t.move_cursor(row=max(0, min(keep, len(rows) - 1)))
        self._update_totals(infos)
        self.update_explain()

    def _update_totals(self, infos) -> None:
        total = sum(i.params for i in infos)
        frozen = sum(i.params for i in infos if i.spec.id in self.spec.frozen)
        problems = self.spec.validate()
        status = "[green]✓ ready to train[/]" if not problems else f"[red]✗ {problems[0]}[/]"
        out = self.spec.output_shape() if not any(i.error for i in infos) else None
        self.query_one("#totals", Static).update(
            f"[b]Total parameters: {total:,}[/b]   trainable {total - frozen:,}   "
            f"size {total * 4 / 1024 ** 2:.2f} MB (fp32)   output {_shape(out)}\n{status}")
        dirty = "  [yellow]● unsaved[/]" if self.dirty else ""
        self.query_one("#stack-title", Static).update(
            f"Model [b]{self.spec.name}[/b] · {self.spec.task} · {len(self.spec.layers)} layers{dirty}")

    @property
    def dirty(self) -> bool:
        return self.saved_state != self.spec.to_dict()

    @property
    def cursor_layer(self) -> int | None:
        """Index into spec.layers of the highlighted row (None for the Input row)."""
        r = self.query_one("#stack", DataTable).cursor_row
        return r - 1 if r >= 1 else None

    # -- properties panel --------------------------------------------------

    @on(DataTable.RowHighlighted, "#stack")
    def _row_changed(self, event: DataTable.RowHighlighted) -> None:
        idx = self.cursor_layer
        target = "__model__" if idx is None else self.spec.layers[idx].id
        if target != self._props_for:
            self.show_props()
            self.update_explain()

    def update_explain(self) -> None:
        """Refresh the Explain tab for the highlighted row (only while that tab is visible)."""
        if self.query_one("#side-tabs", TabbedContent).active != "tab-explain":
            return
        idx = self.cursor_layer
        if idx is None or idx >= len(self.spec.layers):
            text = model_card_markdown(self.spec)
        else:
            text = layer_card_markdown(self.spec, idx)
        self.query_one("#explain", Markdown).update(text)

    @on(TabbedContent.TabActivated, "#side-tabs")
    def _tab_changed(self) -> None:
        self.update_explain()

    def action_toggle_explain(self) -> None:
        tabs = self.query_one("#side-tabs", TabbedContent)
        tabs.active = "tab-settings" if tabs.active == "tab-explain" else "tab-explain"

    def action_guide(self) -> None:
        from .guide import GuideScreen

        idx = self.cursor_layer
        focus = "overview" if idx is None or idx >= len(self.spec.layers) else f"layer:{idx}"
        self.app.push_screen(GuideScreen(self.spec, focus))

    def show_props(self) -> None:
        """Rebuild the properties panel (queued so rebuilds never overlap)."""
        self.call_later(self._rebuild_props)

    async def _rebuild_props(self) -> None:
        panel = self.query_one("#props", VerticalScroll)
        await panel.remove_children()
        idx = self.cursor_layer
        if idx is None or idx >= len(self.spec.layers):
            self._props_for = "__model__"
            await panel.mount_all(self._model_form())
            return
        layer = self.spec.layers[idx]
        self._props_for = layer.id
        lt = REGISTRY[layer.type]
        info = self.spec.analyze()[idx]
        widgets = [Static(f"[b $accent]{idx + 1}. {lt.name}[/]", classes="section-title"),
                   Static(lt.description, classes="hint")]
        widgets.append(Static(
            f"in  {_shape(info.in_shape)}\nout {_shape(info.out_shape) if not info.error else '✗'}\n"
            f"params {info.params:,}" + (f"  = {info.formula}" if info.formula else ""), id="layer-io"))
        if info.error:
            widgets.append(Static(f"[red]{info.error}[/]", id="layer-error"))
        if not lt.params:
            widgets.append(Static("[dim]This layer has no settings.[/]"))
        for p in lt.params:
            val = layer.config.get(p.name, p.default)
            widgets.append(Label(f"{p.name}  [dim]{p.help}[/]"))
            if p.kind == "bool":
                widgets.append(Switch(bool(val), id=f"p-{p.name}"))
            elif p.kind == "choice":
                widgets.append(Select([(c, c) for c in p.choices], value=val, allow_blank=False,
                                      id=f"p-{p.name}"))
            else:
                widgets.append(Input(str(val), id=f"p-{p.name}",
                                     type="integer" if p.kind == "int" else "number"))
        widgets.append(Static("", id="prop-error", classes="error"))
        await panel.mount_all(widgets)

    def _model_form(self) -> list:
        s = self.spec
        return [
            Static("[b $accent]Model settings[/]", classes="section-title"),
            Static("These describe what goes in and what comes out. Training's "
                   "'Auto-fit' can set them from your dataset.", classes="hint"),
            Label("Name"), Input(s.name, id="m-name"),
            Label("Input dimensions"),
            Select(self._rank_options(), value=len(s.input_shape), allow_blank=False, id="m-rank"),
            Horizontal(*[Vertical(Label(name), Input(str(d), id=f"m-dim-{i}", type="integer"))
                         for i, (name, d) in enumerate(zip(self._dim_names(), s.input_shape))],
                       id="m-dims"),
            Static(f"[dim]Input shape: {_shape(s.input_shape)}[/]", id="m-shape-text"),
            Label("Input type"),
            Select([("float - numbers / images", "float"), ("long - token ids (text)", "long")],
                   value=s.input_dtype, allow_blank=False, id="m-dtype"),
            Label("Task"),
            Select([("classification", "classification"), ("regression", "regression")],
                   value=s.task, allow_blank=False, id="m-task"),
            Label("Class names  [dim]comma-separated, optional[/]"),
            Input(", ".join(s.class_names), id="m-classes"),
            Static("", id="prop-error", classes="error"),
        ]

    def _rank_options(self) -> list:
        opts = [(label, r) for r, (label, _, _) in RANKS.items()]
        n = len(self.spec.input_shape)
        return opts if n in RANKS else opts + [(f"{n}D · custom", n)]

    def _dim_names(self) -> list[str]:
        n = len(self.spec.input_shape)
        if n == 1 and self.spec.input_dtype == "long":
            return ["Sequence length (tokens)"]
        return RANKS[n][1] if n in RANKS else [f"Dim {i + 1}" for i in range(n)]

    def _prop_error(self, msg: str) -> None:
        for w in self.query("#prop-error"):
            w.update(msg)

    def _apply(self, widget_id: str, raw) -> None:
        """Apply a change from the properties panel."""
        if widget_id.startswith("m-"):
            return self._apply_model(widget_id, raw)
        idx = self.cursor_layer
        if idx is None:
            return
        layer = self.spec.layers[idx]
        p = next(p for p in REGISTRY[layer.type].params if p.name == widget_id[2:])
        try:
            val = coerce(p, raw)
        except ValueError as e:
            return self._prop_error(str(e))
        if layer.config.get(p.name) == val:
            return self._prop_error("")
        self._push_undo()
        layer.config[p.name] = val
        self._prop_error("")
        self.refresh_stack()
        self._refresh_layer_io(idx)

    def _refresh_layer_io(self, idx: int) -> None:
        info = self.spec.analyze()[idx]
        for w in self.query("#layer-io"):
            w.update(f"in  {_shape(info.in_shape)}\nout {_shape(info.out_shape) if not info.error else '✗'}\n"
                     f"params {info.params:,}" + (f"  = {info.formula}" if info.formula else ""))
        if info.error:
            self._prop_error(info.error)

    def _apply_model(self, wid: str, raw) -> None:
        s = self.spec
        try:
            if wid == "m-name":
                name = safe_name(raw)
                if not name:
                    raise ValueError("Name can't be empty")
                new = {"name": name}
            elif wid == "m-rank":
                rank = int(raw)
                new = {"input_shape": reshape_input(s.input_shape, rank, s.input_dtype)}
                if rank == 3:
                    new["input_dtype"] = "float"  # images are numbers, never token ids
            elif wid.startswith("m-dim-"):
                i, d = int(wid.rsplit("-", 1)[1]), int(raw)
                if d <= 0:
                    raise ValueError("Each dimension must be a positive whole number")
                new = {"input_shape": [d if j == i else v for j, v in enumerate(s.input_shape)]}
            elif wid == "m-dtype":
                new = {"input_dtype": raw}
            elif wid == "m-task":
                new = {"task": raw}
            elif wid == "m-classes":
                new = {"class_names": [c.strip() for c in str(raw).split(",") if c.strip()]}
            else:
                return
        except ValueError as e:
            return self._prop_error(str(e) if "invalid literal" not in str(e) else
                                    "Each dimension must be a whole number")
        if all(getattr(s, k) == v for k, v in new.items()):
            return self._prop_error("")
        self._push_undo()
        for k, v in new.items():
            setattr(s, k, v)
        self._prop_error("")
        self.refresh_stack()
        if wid in ("m-rank", "m-dtype"):
            self.show_props()  # different boxes for the new number of dimensions
        else:
            for w in self.query("#m-shape-text"):
                w.update(f"[dim]Input shape: {_shape(s.input_shape)}[/]")

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        if event.input.id and event.input.id[:2] in ("p-", "m-"):
            if event.input.id == "m-name":
                self._apply(event.input.id, event.value)
            elif event.value.strip():
                self._apply(event.input.id, event.value)

    @on(Switch.Changed)
    def _switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id:
            self._apply(event.switch.id, event.value)

    @on(Select.Changed)
    def _select_changed(self, event: Select.Changed) -> None:
        if event.select.id and event.value is not Select.NULL:
            self._apply(event.select.id, event.value)

    # -- palette -----------------------------------------------------------

    @on(OptionList.OptionHighlighted, "#palette")
    def _palette_help(self, event: OptionList.OptionHighlighted) -> None:
        lt = REGISTRY.get(event.option.id or "")
        if lt:
            self.query_one("#palette-help", Static).update(f"[b]{lt.name}[/b]\n{lt.description}")

    @on(OptionList.OptionSelected, "#palette")
    def _palette_add(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.add_layer(event.option.id)

    def add_layer(self, type_name: str) -> None:
        self._push_undo()
        idx = self.cursor_layer
        pos = 0 if idx is None else idx + 1  # after the highlighted row (Input row = top)
        layer = LayerSpec.new(type_name)
        # Smart default: if this is going to be the head, match the number of classes.
        if type_name == "Linear" and pos == len(self.spec.layers) and self.spec.class_names:
            layer.config["out_features"] = len(self.spec.class_names)
        self.spec.layers.insert(pos, layer)
        self.refresh_stack(cursor=pos + 1)
        self.show_props()
        info = self.spec.analyze()[pos]
        if info.error:
            self.notify(f"{type_name}: {info.error}", severity="warning")

    # -- actions -----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "up":
            self.action_move(-1)
        elif bid == "down":
            self.action_move(1)
        elif bid == "dup":
            self.action_duplicate_layer()
        elif bid == "remove":
            self.action_remove()
        elif bid == "undo":
            self.action_undo()
        elif bid == "save":
            self.action_save()

    def _push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.spec.to_dict()))
        del self.undo_stack[:-100]

    def action_undo(self) -> None:
        if not self.undo_stack:
            self.notify("Nothing to undo")
            return
        self.spec = ModelSpec.from_dict(self.undo_stack.pop())
        self.refresh_stack()
        self.show_props()

    def action_focus_palette(self) -> None:
        self.query_one("#palette", OptionList).focus()

    def _focus_in_props(self) -> bool:
        f = self.focused
        return f is not None and (isinstance(f, Input) or f.id == "palette")

    def action_remove(self) -> None:
        if self._focus_in_props():
            return
        idx = self.cursor_layer
        if idx is None:
            return
        self._push_undo()
        removed = self.spec.layers.pop(idx)
        self.spec.frozen = [f for f in self.spec.frozen if f != removed.id]
        self.refresh_stack(cursor=idx if idx < len(self.spec.layers) + 1 else idx)
        self.show_props()

    def action_move(self, delta: int) -> None:
        if isinstance(self.focused, Input):
            return
        idx = self.cursor_layer
        if idx is None or not (0 <= idx + delta < len(self.spec.layers)):
            return
        self._push_undo()
        L = self.spec.layers
        L[idx], L[idx + delta] = L[idx + delta], L[idx]
        self.refresh_stack(cursor=idx + delta + 1)

    def action_duplicate_layer(self) -> None:
        idx = self.cursor_layer
        if idx is None:
            return
        self._push_undo()
        src = self.spec.layers[idx]
        self.spec.layers.insert(idx + 1, LayerSpec(src.type, copy.deepcopy(src.config)))
        self.refresh_stack(cursor=idx + 2)
        self.show_props()

    def action_save(self) -> bool:
        ws = self.app.ws
        s = self.spec
        if s.name != self.original_name and ws.exists(s.name):
            self.notify(f"A model named '{s.name}' already exists - choose another name.", severity="error")
            return False
        errors = [i for i in s.analyze() if i.error]
        msg = []
        if self.original_name and s.name != self.original_name:
            ws.rename(self.original_name, s.name)
            msg.append(f"renamed to '{s.name}'")
        if self.original_name and ws.has_weights(s.name) and not errors:
            old = ModelSpec.from_dict(self.saved_state)
            if old.to_dict()["layers"] != s.to_dict()["layers"] or old.input_shape != s.input_shape:
                state = torch.load(ws.model_dir(s.name) / "weights.pt", map_location="cpu", weights_only=True)
                model = s.build()
                kept, total = remap_state(old, state, s, model)
                ws.save_weights(s.name, model)
                msg.append(f"kept trained weights for {kept}/{total} layers")
        ws.save_spec(s)
        self.original_name = s.name
        self.saved_state = copy.deepcopy(s.to_dict())
        self.refresh_stack()
        note = f"Saved '{s.name}'" + (f" - {', '.join(msg)}" if msg else "")
        if errors:
            note += " (has errors - fix them before training)"
        self.notify(note, severity="warning" if errors else "information")
        return True

    def action_back(self) -> None:
        if not self.dirty:
            self.app.pop_screen()
            return

        def done(save):
            if save is None:
                return
            if save and not self.action_save():
                return
            self.app.pop_screen()

        self.app.push_screen(SaveChangesDialog(), done)
