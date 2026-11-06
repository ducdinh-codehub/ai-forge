"""Seed, balanced classes and the learning-rate finder: on the command line, the Train screen,
the goal wizard and the exported training script."""
import asyncio
import csv
import subprocess
import sys

import torch
from torch.utils.data import TensorDataset

from aiforge import pipeline
from aiforge.cli import main
from aiforge.model_io import export
from aiforge.spec import Workspace, from_template
from aiforge.trainer import Trainer, TrainConfig, apply_seed, class_weights

SPIRALS = {"source": "synthetic", "kind": "spirals"}


def _spirals(ws):
    ws.save_spec(from_template("MLP classifier (tabular)", "m"))
    spec = ws.load_spec("m")
    data = pipeline.load_data(ws, spec, SPIRALS)
    pipeline.prepare(spec, data.info)
    return spec, data


def _val_losses(ws, seed):
    pipeline.train_model(ws, "m", pipeline.load_data(ws, ws.load_spec("m"), SPIRALS),
                         TrainConfig(epochs=3, seed=seed, device="cpu"), on_log=lambda m: None)
    run = ws.load_history("m")[-1]
    return run["config"]["seed"], [e["val_loss"] for e in run["epochs"]]


def test_the_same_seed_trains_the_same_model(tmp_path):
    ws = Workspace(tmp_path)
    ws.save_spec(from_template("MLP classifier (tabular)", "m"))
    assert _val_losses(ws, 7) == _val_losses(ws, 7)
    seed, losses = _val_losses(ws, -1)  # a new seed each run, recorded so it can be repeated
    assert seed >= 0 and _val_losses(ws, seed) == (seed, losses)


def test_class_weights_make_every_class_count_equally():
    ds = TensorDataset(torch.zeros(10, 2), torch.tensor([0] * 8 + [1] * 2))
    w = class_weights(ds, 3)  # class 2 has no examples
    assert torch.allclose(w, torch.tensor([10 / 24, 10 / 6, 0.0]))
    assert torch.isclose((w[0] * 8), (w[1] * 2))


def test_balancing_helps_a_rare_class(tmp_path):
    # Two overlapping clouds, 95% of class 0: unbalanced, the model rarely predicts class 1.
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(950, 2, generator=g), torch.randn(50, 2, generator=g) + 1.5])
    y = torch.cat([torch.zeros(950, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    spec = from_template("MLP classifier (tabular)", "m")
    spec.input_shape, spec.class_names = [2], ["common", "rare"]
    next(l for l in reversed(spec.layers) if l.type == "Linear").config["out_features"] = 2
    ds = TensorDataset(x, y)
    found = {}
    for balance in (False, True):
        cfg = TrainConfig(epochs=15, lr=0.01, seed=1, balance=balance, device="cpu")
        apply_seed(cfg)
        model, logs = spec.build(), []
        Trainer(spec, model, ds, TensorDataset(x[:0], y[:0]), cfg, on_log=logs.append).run()
        with torch.no_grad():
            found[balance] = (model(x[y == 1]).argmax(1) == 1).float().mean().item()
        assert any("Balancing classes" in m for m in logs) == balance
    assert found[True] > found[False] + 0.3, found


def test_the_lr_finder_suggests_a_rate_and_leaves_the_model_alone(tmp_path):
    spec, data = _spirals(Workspace(tmp_path))
    model = spec.build()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    steps = []
    res = Trainer(spec, model, data.train, data.val, TrainConfig(device="cpu"), on_log=lambda m: None).find_lr(
        on_step=lambda i, n, lr, loss: steps.append(lr))
    assert res["suggestion"] and 1e-4 <= res["suggestion"] <= 0.1
    assert res["suggestion"] < res["lowest"]
    assert steps == sorted(steps) and len(res["losses"]) == len(steps)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_cli_find_lr_trains_nothing(tmp_path, capsys):
    ws = tmp_path / "ws"
    assert main(["-w", str(ws), "new", "m", "-t", "mlp classifier"]) == 0
    assert main(["-w", str(ws), "train", "m", "-d", "spirals", "--find-lr", "--device", "cpu"]) == 0
    assert "Suggested learning rate" in capsys.readouterr().out
    assert not Workspace(ws).has_weights("m") and not Workspace(ws).load_history("m")


def _table(path, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "weight", "label"])
        for i, label in enumerate(labels):
            w.writerow([i % 7, i % 5, label])
    return str(path)


def test_the_wizard_balances_unbalanced_classes(tmp_path):
    from aiforge.wizard import plan

    skewed = plan("predict the label", _table(tmp_path / "a.csv", ["ok"] * 90 + ["fraud"] * 10))
    assert skewed.task == "classification" and skewed.train["balance"] is True
    assert any("Balance classes on" in s for s in skewed.settings)
    even = plan("predict the label", _table(tmp_path / "b.csv", ["ok"] * 50 + ["fraud"] * 50))
    assert "balance" not in even.train


def test_the_script_repeats_the_seed_and_the_balancing(tmp_path):
    ws_dir = tmp_path / "ws"
    data = _table(tmp_path / "t.csv", ["ok"] * 90 + ["fraud"] * 10)
    assert main(["-w", str(ws_dir), "new", "m", "-t", "mlp classifier"]) == 0
    assert main(["-w", str(ws_dir), "train", "m", "-d", data, "--target", "label", "-e", "1", "--device", "cpu",
                 "--seed", "3", "--balance"]) == 0
    script = export(Workspace(ws_dir), "m", "train_py", tmp_path)
    text = script.read_text()
    assert "'train_seed': 3" in text and "'balance': True" in text
    proc = subprocess.run([sys.executable, str(script), "--epochs", "1", "--device", "cpu",
                           "--out", str(tmp_path / "w.pt")], capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr


def test_train_screen_finds_a_learning_rate_and_records_the_seed(tmp_path):
    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.train import TrainScreen

    ws = Workspace(tmp_path)
    ws.save_spec(from_template("MLP classifier (tabular)", "m"))

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 45)) as pilot:
            await app.push_screen(TrainScreen("m", finetune=False))
            screen = app.screen
            await pilot.pause()
            screen.query_one("#device").value = "cpu"
            screen.query_one("#load").press()
            for _ in range(200):
                await pilot.pause(0.05)
                if screen.info:
                    break
            assert screen.query_one("#col-balance").display  # spirals: classes to balance
            screen.query_one("#autofit").press()
            await pilot.pause()
            screen.query_one("#find-lr").press()
            for _ in range(400):
                await pilot.pause(0.05)
                if screen.query_one("#lr").value != "0.001" and not screen.trainer:
                    break
            assert screen.query_one("#lr").value != "0.001", "no learning rate suggested"
            assert not ws.has_weights("m")  # finding a rate trains nothing

            screen.query_one("#epochs").value = "2"
            screen.query_one("#seed").value = "11"
            screen.query_one("#start").press()
            for _ in range(400):
                await pilot.pause(0.05)
                if ws.load_history("m"):
                    break

    asyncio.run(scenario())
    run = ws.load_history("m")[-1]
    assert run["config"]["seed"] == 11 and run["config"]["lr"] != 0.001
