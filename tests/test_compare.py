"""Comparing training runs: the right runs, metrics and best run, in the CLI and on screen."""
import asyncio

import pytest

from aiforge.cli import main
from aiforge.compare import collect_runs, common_metrics, table
from aiforge.spec import Workspace


@pytest.fixture
def trained(tmp_path):
    """Two classifiers on the same data: one run of 'small', two runs (lr 0.01, 0.001) of 'big'."""
    ws_dir = tmp_path / "ws"
    run = lambda *a: main(["-w", str(ws_dir), *map(str, a)])  # noqa: E731
    assert run("new", "small", "-t", "mlp classifier") == 0
    assert run("new", "big", "-t", "mlp classifier") == 0
    assert run("train", "small", "-d", "spirals", "-e", 3, "--lr", 0.01, "--device", "cpu") == 0
    assert run("train", "big", "-d", "spirals", "-e", 5, "--lr", 0.01, "--device", "cpu") == 0
    assert run("train", "big", "-e", 2, "--lr", 0.001, "--device", "cpu") == 0
    run.ws = Workspace(ws_dir)
    return run


def test_runs_metrics_and_best(trained):
    runs = collect_runs(trained.ws)
    assert sorted(r.label for r in runs) == ["big #1", "big #2", "small #1"]
    assert common_metrics(runs)[:2] == ["val_loss", "train_loss"] and "val_acc" in common_metrics(runs)
    header, rows, best = table(runs, "val_acc")
    assert "lr" in header and "optimizer" not in header  # only the settings that differ
    accs = [r.best("val_acc") for r in runs]
    assert best == accs.index(max(accs))  # higher accuracy is better
    _, _, best_loss = table(runs, "val_loss")
    losses = [r.best("val_loss") for r in runs]
    assert best_loss == losses.index(min(losses))  # lower loss is better
    assert [len(r.series("val_loss")) for r in sorted(runs, key=lambda r: r.label)] == [5, 2, 3]


def test_cli_compare(trained, capsys):
    capsys.readouterr()
    assert trained("compare", "small", "big", "--metric", "val_acc") == 0
    out = capsys.readouterr().out
    assert "Validation accuracy" in out and "★" in out and "big #2" in out
    assert trained("compare", "big", "--last", 1) == 0
    out = capsys.readouterr().out
    assert "big #2" in out and "big #1" not in out
    assert trained("compare", "small", "--metric", "val_mae") == 1  # classifiers have no MAE
    assert "These all do" in capsys.readouterr().err


def test_compare_screen(trained):
    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.chart import LineChart

    async def scenario():
        app = AIForgeApp(trained.ws.root)
        async with app.run_test(size=(140, 45)) as pilot:
            app.screen.query_one("#models").move_cursor(row=0)
            await pilot.pause()
            selected_model = app.screen.selected
            await pilot.press("m")
            await pilot.pause(0.3)
            screen = app.screen
            chart = screen.query_one("#cmp-chart", LineChart)
            mine = [r.label for r in screen.runs if r.model == selected_model]
            assert chart.labels == mine  # the model's own runs are picked first
            assert screen.query_one("#metric").value == "val_loss"
            runs = screen.query_one("#runs")
            runs.select_all()
            await pilot.pause(0.2)
            assert len(chart.series) == 3 and screen.query_one("#cmp-table").row_count == 3
            screen.query_one("#metric").value = "val_acc"
            await pilot.pause(0.2)
            assert chart.title.startswith("Validation accuracy (higher")
            runs.deselect_all()
            await pilot.pause(0.2)
            assert chart.series == [] and chart.title == "Tick some runs"

    asyncio.run(scenario())


def test_chart_shares_the_epoch_axis():
    from aiforge.tui.chart import LineChart

    chart = LineChart("x")
    chart.set_series([("long", [1.0, 0.5, 0.2, 0.1]), ("short", [1.0, 0.4])])
    assert chart.labels == ["long", "short"] and chart.palette[:2] == ["cyan", "magenta"]
