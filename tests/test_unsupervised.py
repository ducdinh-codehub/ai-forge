"""Learning without labels: k-means and PCA (classic, fitted) and autoencoders (trained), from the
data they read to the groups, oddities and reports they give back."""
import asyncio
import csv

import numpy as np
import pytest
import torch
from PIL import Image

from aiforge import EXAMPLES_DIR, pipeline
from aiforge import data as datamod
from aiforge import evaluate as ev
from aiforge import unsupervised as un
from aiforge.cli import main
from aiforge.layers import KMeans, PCA
from aiforge.predict import describe, predict_batch, predict_one
from aiforge.spec import LayerSpec, Workspace, from_template
from aiforge.trainer import TrainConfig, Trainer, data_mismatch, fit_spec_to_data


def _blobs(n=800, seed=0):
    ds, _ = datamod._synthetic("blobs", n)
    return ds.tensors


def _trained(ws, template, data, name="m", **cfg):
    ws.save_spec(from_template(template, name))
    spec = ws.load_spec(name)
    loaded = pipeline.load_data(ws, spec, pipeline.data_cfg_from_arg(str(data), spec))
    pipeline.train_model(ws, name, loaded, TrainConfig(**{"device": "cpu", **cfg}), on_log=lambda m: None)
    return ws.load_model(name)


# -- the algorithms -------------------------------------------------------------------------------


def test_kmeans_finds_the_four_blobs():
    x, y = _blobs()
    layer = KMeans(2, 4)
    rounds = list(un.fit_kmeans(layer, x, 50))
    assert rounds[-1][2] == 0, "should stop once no example changes group"
    groups = layer(x).argmin(1).numpy()
    assert un.purity(groups, y.numpy()) > 0.95
    assert un.adjusted_rand(groups, y.numpy()) > 0.9


@pytest.mark.parametrize("seed", range(4))
def test_kmeans_does_not_get_stuck_with_two_centres_in_one_group(seed):
    x, y = _blobs()
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    x, y = x[order], y[order]
    layer = KMeans(2, 4)
    list(un.fit_kmeans(layer, x, 50, seed=seed))
    assert un.purity(layer(x).argmin(1).numpy(), y.numpy()) > 0.95


def test_kmeans_needs_enough_examples():
    with pytest.raises(ValueError, match="at least 4 examples"):
        list(un.fit_kmeans(KMeans(2, 4), torch.zeros(3, 2), 10))


def test_silhouette_tells_clear_groups_from_random_ones():
    x, y = _blobs()
    assert un.silhouette(x, y) > 0.5
    shuffled = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(0))]
    assert abs(un.silhouette(x, shuffled)) < 0.1
    assert un.silhouette(x, torch.zeros(len(x), dtype=torch.long)) is None  # one group: undefined


def test_pca_finds_the_main_directions():
    g = torch.Generator().manual_seed(0)
    # 5 columns that really vary along 2 directions, plus a little noise
    x = torch.randn(500, 2, generator=g) @ torch.randn(2, 5, generator=g) + torch.randn(500, 5, generator=g) * 0.01
    layer = PCA(5, 2)
    ratio = un.fit_pca(layer, x)
    assert ratio.sum() > 0.99 and ratio[0] >= ratio[1]
    assert torch.allclose(layer.components @ layer.components.T, torch.eye(2), atol=1e-4)  # orthonormal
    rebuilt = layer.reconstruct(layer(x))
    assert ((rebuilt - x) ** 2).mean() < 1e-3
    again = PCA(5, 2)
    un.fit_pca(again, x)
    assert torch.allclose(again.components, layer.components)  # the same map every time, not a mirror


def test_text_map_draws_every_kind():
    pts = np.array([[0, 0], [1, 1], [0, 1]])
    text = un.text_map(pts, np.array([0, 1, 2]), width=10, height=4, highlight=np.array([False, False, True]))
    assert "●" in text and "▲" in text and "✖" in text and "■" not in text


# -- the data -------------------------------------------------------------------------------------


def _table(tmp_path):
    path = tmp_path / "shop.csv"
    rows = [["spend", "visits", "segment"]] + [[str(10 + i % 7), str(i % 3), "a" if i % 2 else "b"] for i in range(40)]
    path.write_text("\n".join(",".join(r) for r in rows))
    return path


def test_tables_need_no_label_column(tmp_path):
    spec = from_template("k-means clustering (4 groups)", "m")
    path = _table(tmp_path)
    tr, _, info = datamod.load({"source": "csv", "path": path}, tmp_path, spec)
    assert info.task == "clustering" and info.input_shape == [3] and info.class_names == []
    assert "no labels used" in info.description
    # A chosen label column is left out of the inputs and kept aside to check the groups.
    tr, _, info = datamod.load({"source": "csv", "path": path, "target": "segment"}, tmp_path, spec)
    assert info.input_shape == [2] and info.class_names == ["a", "b"] and "kept aside" in info.description


def test_text_is_refused(tmp_path):
    spec = from_template("k-means clustering (4 groups)", "m")
    for cfg in ({"source": "text_corpus", "path": datamod.EXAMPLE_CORPUS},
                {"source": "csv_text", "path": EXAMPLES_DIR / "movie_reviews.csv"}):
        with pytest.raises(ValueError, match="not text"):
            datamod.load(cfg, tmp_path, spec)


def test_a_plain_image_folder_is_enough(tmp_path):
    for i in range(6):
        Image.new("L", (28, 28), i * 40).save(tmp_path / f"{i}.png")
    spec = from_template("Image autoencoder (28×28 grayscale, e.g. MNIST)", "m")
    tr, va, info = datamod.load({"source": "image_folder", "path": tmp_path, "val_split": 0}, tmp_path, spec)
    assert len(tr) == 6 and info.class_names == []
    with pytest.raises(ValueError, match="sub-folders"):  # a classifier still needs its classes
        datamod.load({"source": "image_folder", "path": tmp_path}, tmp_path,
                     from_template("Small CNN (28×28 grayscale, e.g. MNIST)", "c"))


# -- the model spec -------------------------------------------------------------------------------


def test_classic_models_are_a_single_layer():
    spec = from_template("k-means clustering (4 groups)", "m")
    assert spec.validate() == []
    spec.layers.insert(0, LayerSpec.new("Linear", out_features=2))
    assert "single KMeans layer" in spec.validate()[0]


def test_autoencoders_must_rebuild_the_input_shape():
    spec = from_template("Autoencoder (tables, 2-number code)", "m")
    spec.layers[-1].config["out_features"] = 3
    assert "rebuild its input" in spec.validate()[0]
    assert spec.bottleneck() == 5  # the Tanh after the 2-number Linear: that's the code


@pytest.mark.parametrize("template, shape", [("Autoencoder (tables, 2-number code)", [7]),
                                             ("Image autoencoder (28×28 grayscale, e.g. MNIST)", [3, 32, 32])])
def test_auto_fit_makes_an_autoencoder_rebuild_new_data(tmp_path, template, shape):
    spec = from_template(template, "m")
    info = datamod.DataInfo(shape, "float", "autoencoder", [])
    assert data_mismatch(spec, info)
    fit_spec_to_data(spec, info)
    assert data_mismatch(spec, info) == [] and spec.validate() == []
    assert list(spec.output_shape()) == shape


# -- training, using and evaluating ---------------------------------------------------------------


def test_kmeans_model_trains_and_judges_new_examples(ws):
    spec, model, _ = _trained(ws, "k-means clustering (4 groups)", "blobs")
    run = ws.load_history("m")[-1]
    assert run["epochs"][-1]["val_silhouette"] > 0.5
    stats = spec.data["unsupervised"]
    assert len(stats["quantiles"]) == 101 and sum(stats["sizes"]) == 1600
    near, far = describe(spec, model, "2, 2"), describe(spec, model, "9, 9")
    assert near["group"] == far["group"]  # both closest to the (2, 2) blob
    assert near["percentile"] < 0.5 and far["percentile"] == 1.0 and "unusual" in far["verdict"]
    assert [label for label, _ in predict_one(spec, model, "2, 2", top_k=2)] == \
        [label for label, _ in near["pairs"][:2]]


def test_pca_model_trains_in_one_step(ws):
    spec, model, _ = _trained(ws, "PCA (2 components, for a 2-D map)", "regression")
    ep = ws.load_history("m")[-1]["epochs"]
    assert len(ep) == 1 and 0 < ep[0]["val_kept"] <= 1 and len(ep[0]["explained"]) == 2
    assert len(describe(spec, model, "0, 0, 0, 0")["code"]) == 2


def test_autoencoder_learns_and_spots_an_odd_row(ws):
    spec, model, _ = _trained(ws, "Autoencoder (tables, 2-number code)", EXAMPLES_DIR / "house_prices.csv",
                              epochs=20, lr=0.01)
    kept = [e["val_kept"] for e in ws.load_history("m")[-1]["epochs"]]
    assert kept[-1] > kept[0] and kept[-1] > 0.5
    assert describe(spec, model, "4, 110.8, 9000")["percentile"] == 1.0


def test_classic_training_can_be_stopped(tmp_path):
    spec = from_template("k-means clustering (4 groups)", "m")
    tr, va, info = datamod.load({"source": "synthetic", "kind": "blobs"}, tmp_path, spec)
    fit_spec_to_data(spec, info)
    trainer = Trainer(spec, spec.build(), tr, va, TrainConfig(device="cpu"), on_log=lambda m: None)
    trainer.stop()
    assert trainer.run()["epochs"] == []


def test_batch_marks_the_unusual_rows(ws, tmp_path):
    spec, model, _ = _trained(ws, "k-means clustering (4 groups)", "blobs")
    src = tmp_path / "new.csv"
    src.write_text("x,y\n2,2\n9,9\n")
    out, n = predict_batch(spec, model, str(src))
    rows = list(csv.DictReader(out.open()))
    assert n == 2 and list(rows[0]) == ["x", "y", "group", "distance", "unusual_percentile"]
    assert float(rows[1]["unusual_percentile"]) == 1.0 > float(rows[0]["unusual_percentile"])


def test_evaluation_report(ws):
    spec, model, _ = _trained(ws, "k-means clustering (4 groups)", "blobs")
    ds, ref, desc, orig = ev.load_eval_data(spec, ws.data_dir, "val")
    rep = ev.evaluate_model(spec, model, ds, ref, desc, orig, torch.device("cpu"), ws.load_history("m"))
    assert rep["metrics"]["purity"] > 0.95 and rep["verdict"].startswith("✅")
    assert len(rep["unusual"]) == 10 and rep["unusual"][0]["score"] >= rep["unusual"][-1]["score"]
    md = ev.report_markdown(rep)
    for part in ("Silhouette", "Groups vs. the labels", "Map of the data", "most unusual"):
        assert part in md
    ev.remember(rep, ws.model_dir("m"))
    assert ev.summary(ev.json.loads((ws.model_dir("m") / "evaluations.json").read_text())[-1]).startswith("silhouette")


def test_evaluating_on_a_table_without_labels(ws, tmp_path):
    spec, model, _ = _trained(ws, "Autoencoder (tables, 2-number code)", EXAMPLES_DIR / "house_prices.csv",
                              epochs=3)
    src = tmp_path / "check.csv"
    src.write_text("rooms,area,price\n4,110.8,422.6\n6,112.7,533.4\n3,80,9000\n")
    ds, ref, desc, orig = ev.load_eval_data(spec, ws.data_dir, "file", str(src))
    rep = ev.evaluate_model(spec, model, ds, ref, desc, orig, torch.device("cpu"))
    assert "price=9000" in rep["unusual"][0]["example"]
    assert "by_label" not in rep and "kept" in rep["metrics"]


def test_explain_and_training_script_say_what_they_cover(ws):
    from aiforge.explain import explain
    from aiforge.scriptgen import training_script

    spec, model, _ = _trained(ws, "k-means clustering (4 groups)", "blobs")
    with pytest.raises(ValueError, match="without labels"):
        explain(spec, model, "1, 1")
    with pytest.raises(ValueError, match="supervised models"):
        training_script(spec, TrainConfig())


# -- command line, web API and app ----------------------------------------------------------------


def test_command_line(tmp_path, capsys):
    run = lambda *a: main(["-w", str(tmp_path), *map(str, a)])  # noqa: E731
    assert run("new", "groups", "-t", "k-means") == 0
    assert run("train", "groups", "-d", "blobs", "--device", "cpu") == 0
    assert run("predict", "groups", "9, 9") == 0
    out = capsys.readouterr().out
    assert "Converged" in out and "unusual" in out
    assert run("evaluate", "groups") == 0
    assert "Silhouette" in capsys.readouterr().out


def test_web_api(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from aiforge.serve import create_app

    _trained(Workspace(tmp_path), "k-means clustering (4 groups)", "blobs")
    client = TestClient(create_app(Workspace(tmp_path), "m"))
    res = client.post("/predict", json={"input": "2, 2"}).json()
    assert res["predictions"][0]["label"].startswith("group ")


def test_new_model_dialog_and_train_screen(tmp_path):
    from aiforge.tui.app import AIForgeApp
    from aiforge.tui.modals import NewModelDialog
    from aiforge.tui.train import TrainScreen

    ws = Workspace(tmp_path)
    ws.set_setting("star_answered", True)
    created = {}

    async def scenario():
        app = AIForgeApp(tmp_path)
        async with app.run_test(size=(140, 50)) as pilot:
            await app.push_screen(NewModelDialog(ws), lambda spec: created.setdefault("spec", spec))
            await pilot.pause()
            assert app.screen.query_one("#style-reinforcement").disabled
            await pilot.click("#style-unsupervised")
            await pilot.pause()
            await pilot.click("#ok")
            await pilot.pause()
            ws.save_spec(created["spec"])
            await app.push_screen(TrainScreen(created["spec"].name, finetune=False,
                                              preset={"source": "synthetic", "kind": "blobs"}))
            screen = app.screen
            assert not screen.query_one("#row-lr").display  # no learning rate for k-means
            for _ in range(200):
                await pilot.pause(0.05)
                if screen.info:
                    break
            screen.action_start()
            for _ in range(400):
                await pilot.pause(0.05)
                if ws.load_history(created["spec"].name):
                    break

    asyncio.run(scenario())
    assert created["spec"].task == "clustering"
    assert ws.load_history(created["spec"].name)[-1]["epochs"][-1]["val_silhouette"] > 0.5


GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(GPU is None, reason="no GPU")
@pytest.mark.parametrize("template, data", [("Autoencoder (tables, 2-number code)", "blobs"),
                                            ("k-means clustering (4 groups)", "blobs")])
def test_trains_on_the_gpu(ws, template, data):
    # Apple GPUs (MPS) have no float64: every metric must still work there.
    spec, model, _ = _trained(ws, template, data, device=GPU, epochs=2)
    last = ws.load_history("m")[-1]["epochs"][-1]
    assert last.get("val_kept", last.get("val_silhouette")) is not None
    assert describe(spec, model, "1, 1")["percentile"] is not None
