"""Dataset hub screen: download well-known public datasets, grouped by model type."""
from __future__ import annotations

import time

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Markdown, OptionList, ProgressBar, Static
from textual.widgets.option_list import Option

from .. import hub
from .modals import ConfirmDialog


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


class DatasetScreen(Screen):
    """Dismisses with the Train-screen data settings of the chosen dataset, or None."""

    BINDINGS = [Binding("escape,q", "back", "Back"),
                Binding("enter", "primary", "Download / use", show=False),
                Binding("left,h", "focus_list", "List", show=False),
                Binding("right,l", "focus_text", "Read", show=False)]

    def __init__(self, use_label: str | None = None, focus: str | None = None):
        """use_label: caption of the button that returns the dataset (hidden when None)."""
        super().__init__()
        self.use_label = use_label
        self.start = focus
        self.downloading: str | None = None
        self._cancel = False

    @property
    def ws(self):
        return self.app.ws

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="hub-nav-pane"):
                yield Static("🗄 Datasets", classes="section-title")
                yield Static("Trusted public datasets, ready to train on.", classes="hint")
                yield OptionList(id="hub-list")
            with Vertical(id="hub-detail-pane"):
                with VerticalScroll(id="hub-body"):
                    yield Markdown(id="hub-text")
                yield Static("", id="hub-status")
                yield ProgressBar(id="hub-progress", show_eta=True)
                with Horizontal(classes="toolbar"):
                    yield Button("⇩ Download", id="download", variant="primary")
                    yield Button(self.use_label or "", id="use", variant="success")
                    yield Button("■ Cancel", id="cancel-dl", variant="error")
                    yield Button("🗑 Delete files", id="remove")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#hub-progress").display = False
        self._fill_list()
        lst = self.query_one("#hub-list", OptionList)
        ids = [lst.get_option_at_index(i).id for i in range(lst.option_count)]
        lst.highlighted = ids.index(self.start) if self.start in hub.BY_KEY else 1
        lst.focus()

    def _fill_list(self) -> None:
        lst = self.query_one("#hub-list", OptionList)
        keep = lst.highlighted
        opts: list = []
        for group in hub.GROUPS:
            opts += [None] if opts else []
            opts.append(Option(f"[b]{group}[/b]", disabled=True))
            for d in hub.CATALOG:
                if d.group == group:
                    mark = "[green]✓[/]" if hub.status(d, self.ws.data_dir) else " "
                    opts.append(Option(f" {mark} {d.name}  [dim]{d.size}[/]", id=d.key))
        lst.clear_options()
        lst.add_options(opts)
        if keep is not None:
            lst.highlighted = keep

    @property
    def current(self) -> hub.HubDataset | None:
        lst = self.query_one("#hub-list", OptionList)
        if lst.highlighted is None:
            return None
        return hub.BY_KEY.get(lst.get_option_at_index(lst.highlighted).id or "")

    @on(OptionList.OptionHighlighted, "#hub-list")
    def _highlighted(self) -> None:
        self._show()
        self.query_one("#hub-body", VerticalScroll).scroll_home(animate=False)

    @on(OptionList.OptionSelected, "#hub-list")
    def _selected(self) -> None:
        self.action_primary()

    def _show(self) -> None:
        d = self.current
        if d is None:
            return
        rec = hub.status(d, self.ws.data_dir)
        md = [f"# {d.name}", "", d.summary, "",
              f"- **Samples:** {d.samples}",
              f"- **Task:** {d.task}",
              f"- **Download size:** {d.size}",
              f"- **Published by:** {d.source}",
              f"- **License:** {d.license}",
              f"- **Homepage:** {d.url}", ""]
        if d.notes:
            md += [f"💡 {d.notes}", ""]
        md += ["**Good templates for it**", ""] + [f"- {t}" for t in d.templates] + [""]
        if rec:
            where = hub.train_cfg(d, self.ws.data_dir)["path"] or str(self.ws.data_dir / d.key)
            md += [f"✅ **Downloaded** {rec['downloaded']} · {_fmt_bytes(rec.get('bytes', 0))} on disk",
                   "", f"`{where}`"]
        else:
            md += ["Not downloaded yet. Press **⇩ Download** (or Enter)."]
        self.query_one("#hub-text", Markdown).update("\n".join(md))
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        d = self.current
        busy = self.downloading is not None
        ready = bool(d and hub.status(d, self.ws.data_dir))
        self.query_one("#download", Button).display = not ready and not busy
        self.query_one("#use", Button).display = ready and not busy and bool(self.use_label)
        self.query_one("#remove", Button).display = ready and not busy
        self.query_one("#cancel-dl", Button).display = busy
        if not busy:
            self.query_one("#hub-status", Static).update("")

    # -- actions -------------------------------------------------------------

    def action_primary(self) -> None:
        d = self.current
        if d is None or self.downloading:
            return
        if not hub.status(d, self.ws.data_dir):
            self._start_download()
        elif self.use_label:
            self._use()

    def action_focus_list(self) -> None:
        self.query_one("#hub-list", OptionList).focus()

    def action_focus_text(self) -> None:
        self.query_one("#hub-body", VerticalScroll).focus()

    def action_back(self) -> None:
        if self.downloading:
            self._cancel = True
            self.notify("Download cancelled.", severity="warning")
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "download":
            self._start_download()
        elif bid == "use":
            self._use()
        elif bid == "cancel-dl":
            self._cancel = True
            self.query_one("#hub-status", Static).update("[yellow]Cancelling…[/]")
        elif bid == "remove":
            self._remove()

    def _use(self) -> None:
        if d := self.current:
            self.dismiss(hub.train_cfg(d, self.ws.data_dir))

    def _remove(self) -> None:
        d = self.current
        if d is None:
            return

        def done(ok):
            if ok:
                hub.remove(d, self.ws.data_dir)
                self.notify(f"Deleted the {d.name} files.")
                self._fill_list()
                self._show()
        self.app.push_screen(ConfirmDialog("Delete dataset files?", f"The downloaded {d.name} files are "
                                           "removed from the workspace. You can download them again "
                                           "any time.", "Delete"), done)

    def _start_download(self) -> None:
        d = self.current
        if d is None or self.downloading:
            return
        self.downloading, self._cancel = d.key, False
        self._refresh_buttons()
        bar = self.query_one("#hub-progress", ProgressBar)
        bar.display = True
        bar.update(total=None, progress=0)
        self.query_one("#hub-status", Static).update(f"[yellow]Connecting to {d.url}…[/]")
        self._download(d)

    @work(thread=True, exclusive=True, group="hub")
    def _download(self, d: hub.HubDataset) -> None:
        last = [0.0]

        def progress(stage: str, done: int, total: int | None) -> None:
            now = time.monotonic()
            if now - last[0] < 0.2 and not (total and done >= total):
                return  # don't flood the UI thread
            last[0] = now
            self.app.call_from_thread(self._progress, stage, done, total)

        try:
            rec = hub.download(d, self.ws.data_dir, progress, lambda: self._cancel)
        except hub.Cancelled:
            hub.remove(d, self.ws.data_dir)
            self.app.call_from_thread(self._finished, d, None, "cancelled")
        except Exception as e:  # noqa: BLE001
            hub.remove(d, self.ws.data_dir)
            self.app.call_from_thread(self._finished, d, None, str(e))
        else:
            self.app.call_from_thread(self._finished, d, rec, None)

    def _progress(self, stage: str, done: int, total: int | None) -> None:
        if not self.is_attached:
            return
        bar = self.query_one("#hub-progress", ProgressBar)
        bytes_stage = stage.startswith("Downloading")
        bar.update(total=total, progress=done)
        amount = (f"{_fmt_bytes(done)}" + (f" / {_fmt_bytes(total)}" if total else "")) if bytes_stage \
            else f"{done:,}" + (f" / {total:,}" if total else "")
        self.query_one("#hub-status", Static).update(f"[yellow]{stage}…[/] {amount}")

    def _finished(self, d: hub.HubDataset, rec: dict | None, error: str | None) -> None:
        self.downloading = None
        if not self.is_attached:
            return
        self.query_one("#hub-progress").display = False
        if error == "cancelled":
            self.notify(f"{d.name}: download cancelled.", severity="warning")
        elif error:
            self.notify(f"Couldn't download {d.name}: {error}", severity="error", timeout=12)
        else:
            self.notify(f"✓ {d.name} is ready ({_fmt_bytes(rec.get('bytes', 0))}).")
        self._fill_list()
        self._show()
        if error and error != "cancelled":
            self.query_one("#hub-status", Static).update(f"[red]Download failed: {error}[/]")
