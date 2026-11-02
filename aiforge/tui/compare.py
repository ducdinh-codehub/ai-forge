"""Compare training runs side by side: their curves on one chart and a summary table."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Select, SelectionList, Static

from ..compare import METRICS, collect_runs, common_metrics, table
from .chart import PALETTE, LineChart

MAX_RUNS = len(PALETTE)


class CompareScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, focus: str | None = None):
        """focus: the model whose runs are selected at first."""
        super().__init__()
        self.focus_model = focus
        self.runs = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="cmp-body"):
            with Vertical(id="cmp-runs"):
                yield Static("Training runs", classes="section-title")
                yield Static("[dim]Tick up to 8 runs - of one model, or of different models on the "
                             "same data.[/]", id="cmp-hint")
                yield SelectionList(id="runs")
            with Vertical(id="cmp-main"):
                with Horizontal(id="cmp-metric-row"):
                    yield Label("Compare")
                    yield Select([], id="metric", allow_blank=True, prompt="pick runs first")
                yield LineChart("", id="cmp-chart")
                yield DataTable(id="cmp-table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self.runs = collect_runs(self.app.ws)
        runs = self.query_one("#runs", SelectionList)
        mine = [i for i, r in enumerate(self.runs) if r.model == self.focus_model][-4:]
        if not mine:  # nothing for this model: start with the latest runs overall
            mine = sorted(range(len(self.runs)), key=lambda i: self.runs[i].started)[-2:]
        for i, r in enumerate(self.runs):
            runs.add_option((f"{r.label[:13]:<13} {r.started[5:16]} {len(r.epochs):>3}ep", i, i in mine))
        if not self.runs:
            self.query_one("#cmp-hint", Static).update("[yellow]No training runs yet - train a model first.[/]")
        self._refresh()

    @on(SelectionList.SelectedChanged, "#runs")
    def _runs_changed(self) -> None:
        self._refresh()

    @on(Select.Changed, "#metric")
    def _metric_changed(self) -> None:
        self._draw()

    def _selected(self):
        picked = [self.runs[i] for i in sorted(self.query_one("#runs", SelectionList).selected)]
        if len(picked) > MAX_RUNS:
            self.notify(f"Showing the first {MAX_RUNS} runs - untick some to see others.", severity="warning")
        return picked[:MAX_RUNS]

    def _refresh(self) -> None:
        """New selection: offer the metrics every picked run has, keep the current one if possible."""
        metrics = common_metrics(self._selected())
        sel = self.query_one("#metric", Select)
        current = sel.value
        with self.prevent(Select.Changed):
            sel.set_options([(METRICS[m][0], m) for m in metrics])
            if metrics:
                sel.value = current if current in metrics else metrics[0]
        self._draw()

    def _draw(self) -> None:
        runs, metric = self._selected(), self.query_one("#metric", Select).value
        chart, grid = self.query_one("#cmp-chart", LineChart), self.query_one("#cmp-table", DataTable)
        grid.clear(columns=True)
        if not runs or metric in (Select.BLANK, Select.NULL, None):
            chart.title = "Tick runs that share a metric (e.g. two classifiers)" if runs else "Tick some runs"
            chart.set_series([])
            return
        name, fmt, goal = METRICS[metric]
        chart.title, chart.fmt = f"{name} ({'lower' if goal == 'min' else 'higher'} is better)", fmt
        chart.set_series([(r.label, r.series(metric)) for r in runs])
        header, rows, best = table(runs, metric)
        grid.add_columns(*header)
        for i, row in enumerate(rows):
            colour = PALETTE[i % len(PALETTE)]
            row = [f"[{colour}]━[/] {row[0]}", *row[1:]]
            if i == best:
                row[4] = f"[b green]★ {row[4]}[/]"
            grid.add_row(*row)
