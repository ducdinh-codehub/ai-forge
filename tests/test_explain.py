"""Explanations point at what the model really uses: built with models whose answer we know."""
import asyncio

import pytest
import torch
from PIL import Image

from aiforge.cli import main
from aiforge.explain import explain, render
from aiforge.spec import LayerSpec, ModelSpec
from conftest import make_spec

TABLE = {"type": "tabular", "features": ["signal", "noise"], "mean": [0.0, 0.0], "std": [1.0, 1.0],
         "categorical": {}}


def _table_model(task, weight):
    spec = ModelSpec("t", [2], task=task, layers=[LayerSpec.new("Linear", out_features=len(weight))],
                     class_names=["yes", "no"] if task == "classification" else [], data={"preprocess": dict(TABLE)})
    model = spec.build()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(weight))
        model[0].bias.zero_()
    return spec, model


def test_table_feature_that_matters_comes_first():
    spec, model = _table_model("classification", [[3.0, 0.0], [-3.0, 0.0]])  # only "signal" counts
    exp = explain(spec, model, "1, 5")
    assert exp.answer.startswith("yes")
    (top, value, effect), (other, _, rest) = exp.items
    assert (top, value, other) == ("signal", "1", "noise")
    assert effect > 40 and rest == pytest.approx(0, abs=1e-6)  # points of confidence
    assert "signal" in render(exp).plain


def test_regression_effect_is_in_target_units():
    spec, model = _table_model("regression", [[2.0, 0.0]])
    spec.data["preprocess"].update(target_mean=100.0, target_std=10.0)
    exp = explain(spec, model, "1.5, 0")
    assert exp.answer == "130" and exp.unit == "target units"
    assert exp.items[0][:2] == ("signal", "1.5") and exp.items[0][2] == pytest.approx(30.0)  # 2 × 1.5 × 10


def test_words_that_matter_stand_out():
    spec = make_spec("Bag of words (Embedding + average, fastest)")
    spec.class_names = ["neg", "pos"]
    spec.data["preprocess"] = {"type": "text", "vocab": {"good": 2, "bad": 3, "movie": 4}, "seq_len": 128}
    model = spec.build()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[2] = 5.0  # "good" points to pos
        model[0].weight[3] = -5.0  # "bad" points to neg
        model[3].weight.copy_(torch.stack([-torch.ones(64), torch.ones(64)]))
        model[3].bias.zero_()
    exp = explain(spec, model, "a good movie")
    assert exp.answer.startswith("pos")
    effects = {token: effect for token, _, effect in exp.items}
    assert effects["good"] > 1 and abs(effects["movie"]) < 1e-3
    assert dict((t, n) for t, n, _ in exp.items)["a"] == "unknown word"


def test_image_saliency_is_where_the_model_looks(tmp_path):
    spec = ModelSpec("img", [1, 8, 8], class_names=["dark", "light"],
                     layers=[LayerSpec.new("Flatten"), LayerSpec.new("Linear", out_features=2)],
                     data={"preprocess": {"type": "image", "size": [8, 8], "channels": 1, "mean": [0.5], "std": [0.5]}})
    model = spec.build()
    left = torch.zeros(1, 8, 8)
    left[..., :4] = 1.0  # only the left half of the picture counts
    with torch.no_grad():
        model[1].weight.copy_(torch.stack([-left.flatten(), left.flatten()]))
        model[1].bias.zero_()
    Image.new("L", (8, 8), 200).save(tmp_path / "x.png")
    exp = explain(spec, model, str(tmp_path / "x.png"))
    heat = torch.tensor(exp.heat)
    assert heat[:, :4].mean() > 0.9 and heat[:, 4:].max() < 0.05
    assert render(exp).plain.count("▀") == 8 * 4 * 2  # image + heat map, two pixels per character


def test_language_models_are_not_explained():
    spec = make_spec("Tiny GPT (characters, fast)")
    with pytest.raises(ValueError, match="language model"):
        explain(spec, spec.build(), "hello")


def test_cli_and_predict_screen_explain(tmp_path, capsys):
    ws_dir = tmp_path / "ws"
    assert main(["-w", str(ws_dir), "new", "spiral", "-t", "mlp classifier"]) == 0
    assert main(["-w", str(ws_dir), "train", "spiral", "-d", "spirals", "-e", "3", "--device", "cpu"]) == 0
    capsys.readouterr()
    assert main(["-w", str(ws_dir), "explain", "spiral", "0.1, 0.45"]) == 0
    out = capsys.readouterr().out
    assert "Why" in out and "x" in out and "pts" in out

    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.predict import PredictScreen

    async def scenario():
        app = AIForgeApp(ws_dir)
        async with app.run_test(size=(140, 45)) as pilot:
            await app.push_screen(PredictScreen("spiral"))
            screen = app.screen
            screen.query_one("#raw").value = "0.1, 0.45"
            screen.query_one("#explain").press()
            box = screen.query_one("#explanation")
            for _ in range(100):
                await pilot.pause(0.05)
                if "Why" in str(box.render()):
                    break
            assert box.display and "Why" in str(box.render())
            assert "→" in str(screen.query_one("#result").render())  # the answer is shown too

    asyncio.run(scenario())
