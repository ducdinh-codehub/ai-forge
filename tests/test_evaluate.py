"""Evaluation: metrics against hand-computed values, significance tests, and a full report."""
import json

import numpy as np
import pytest
import torch

from aiforge import evaluate as ev
from aiforge.pipeline import data_cfg_from_arg, load_data, train_model
from aiforge.trainer import TrainConfig
from conftest import make_spec


def _logits(pred, c):
    """Confident logits that predict `pred`."""
    out = np.zeros((len(pred), c))
    out[np.arange(len(pred)), pred] = 5.0
    return out


def test_classification_metrics_by_hand():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 0])
    m = ev.classification_metrics(_logits(pred, 2), y, ["neg", "pos"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion"] == [[3, 1], [1, 1]]
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2) / 2)
    assert m["per_class"][1]["precision"] == pytest.approx(0.5)
    assert m["imbalance"] == 2.0
    lo, hi = m["accuracy_ci"]
    assert lo <= m["accuracy"] <= hi


def test_ranking_metrics():
    pos = np.array([False, False, True, True])
    assert ev.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), pos) == 1.0
    assert ev.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), pos) == 0.0
    assert ev.roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), pos) == 0.5
    assert ev.roc_auc(np.ones(3), np.ones(3, bool)) is None  # only one class: undefined
    assert ev.average_precision(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, False, True, False])) \
        == pytest.approx((1 + 2 / 3) / 2)


def test_regression_metrics_by_hand():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = ev.regression_metrics(y + np.array([0.5, -0.5, 0.5, -0.5]), y)
    assert m["mae"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 1.0 / 5.0)
    assert m["mape"] == pytest.approx(np.mean([0.5, 0.25, 0.5 / 3, 0.125]))
    assert ev.regression_metrics(np.zeros(3), np.zeros(3))["mape"] is None


def test_mcnemar():
    same = np.array([True, False, True])
    assert ev.mcnemar(same, same)["p_value"] == 1.0
    a = np.array([True] * 30 + [False] * 2)
    b = np.array([False] * 30 + [True] * 2)
    assert ev.mcnemar(a, b)["p_value"] < 0.001  # A is clearly better


@pytest.fixture
def trained(ws):
    spec = make_spec("MLP classifier (tabular)", "spiral")
    ws.save_spec(spec)
    data = load_data(ws, spec, data_cfg_from_arg("spirals", spec, samples=900))
    train_model(ws, "spiral", data, TrainConfig(epochs=15, lr=0.01, device="cpu", seed=0), on_log=lambda m: None)
    return ws


def test_full_report(trained, tmp_path):
    ws = trained
    spec, model, _ = ws.load_model("spiral")
    ds, ref, desc, units = ev.load_eval_data(spec, ws.data_dir)
    assert desc.startswith("validation split")

    other = make_spec("MLP classifier (tabular)", "random")
    other.input_shape, other.class_names = spec.input_shape, spec.class_names
    other.layers[-1].config["out_features"] = 3
    rep = ev.evaluate_model(spec, model.eval(), ds, ref, desc, units, torch.device("cpu"),
                            history=ws.load_history("spiral"), other=("random", other, other.build().eval()))

    m = rep["metrics"]
    assert m["n"] == len(ds) and m["accuracy"] > rep["baseline"]["accuracy"]
    assert [r["noise"] for r in rep["robustness"]] == list(ev.NOISE_LEVELS)
    assert rep["comparison"]["other"] == "random" and 0 <= rep["comparison"]["p_value"] <= 1
    assert all(e["true"] != e["predicted"] for e in rep["errors"])
    assert rep["verdict"]

    md = ev.save_report(rep, tmp_path)
    assert "spiral" in md.read_text(encoding="utf-8")
    assert json.loads(md.with_suffix(".json").read_text())["model"] == "spiral"
    ev.remember(rep, ws.model_dir("spiral"))
    assert json.loads((ws.model_dir("spiral") / "evaluations.json").read_text())[-1]["accuracy"] == m["accuracy"]


def test_model_without_aiforge_training_needs_a_file(ws):
    spec = make_spec("MLP classifier (tabular)")
    with pytest.raises(ValueError, match="wasn't trained in AIForge"):
        ev.load_eval_data(spec, ws.data_dir)
