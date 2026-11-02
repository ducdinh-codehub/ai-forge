"""Loading data: toy sets, CSV tables and text, plain text corpora, images, and raw prediction input."""
import pickle

import pytest
import torch
from PIL import Image
from torch.utils.data import Subset, TensorDataset

from aiforge import EXAMPLES_DIR
from aiforge import data as datamod
from aiforge.data import encode_text, limit, prepare_input, tokenize
from conftest import make_spec


@pytest.mark.parametrize("kind", ["moons", "circles", "spirals", "blobs", "regression"])
def test_synthetic_split_is_reproducible(tmp_path, kind):
    cfg = {"source": "synthetic", "kind": kind, "samples": 600, "val_split": 0.25}
    tr, va, info = datamod.load(cfg, tmp_path)
    assert (info.n_train, info.n_val) == (len(tr), len(va)) and len(va) == len(tr) // 3
    assert tuple(tr[0][0].shape) == tuple(info.input_shape)
    _, va2, _ = datamod.load(cfg, tmp_path)
    assert torch.equal(va[0][0], va2[0][0])  # same seed, same split


def test_csv_table_with_categories(tmp_path):
    tr, va, info = datamod.load({"source": "csv", "path": EXAMPLES_DIR / "fruit_size.csv"}, tmp_path)
    assert info.task == "classification" and info.class_names == ["big", "small"]
    pre = info.preprocess
    assert pre["features"] == ["size", "weight", "color"] and "color" in pre["categorical"]
    x = torch.stack([tr[i][0] for i in range(len(tr))] + [va[i][0] for i in range(len(va))])
    assert torch.allclose(x.mean(0), torch.zeros(3), atol=1e-5)  # standardised


def test_csv_regression_and_missing_values(tmp_path):
    rows = [f"{i},{'' if i % 7 == 0 else i * 2},{i * 10 + 0.5}" for i in range(40)]
    (tmp_path / "t.csv").write_text("a,b,y\n" + "\n".join(rows))
    _, _, info = datamod.load({"source": "csv", "path": tmp_path / "t.csv"}, tmp_path)
    assert info.task == "regression" and info.n_outputs == 1
    assert info.preprocess["target_std"] > 1  # predictions are scaled back to real units
    with pytest.raises(ValueError, match="not in CSV columns"):
        datamod.load({"source": "csv", "path": tmp_path / "t.csv", "target": "zzz"}, tmp_path)


def test_csv_text(tmp_path):
    spec = make_spec("Text classifier (Embedding + LSTM)")
    tr, _, info = datamod.load({"source": "csv_text", "path": EXAMPLES_DIR / "movie_reviews.csv"},
                               tmp_path, spec)
    assert info.class_names == ["neg", "pos"] and info.input_dtype == "long"
    assert tr[0][0].shape == (spec.input_shape[0],)


def test_text_corpus_keeps_validation_text_apart(tmp_path):
    spec = make_spec("Tiny GPT (characters, fast)")
    tr, va, info = datamod.load({"source": "text_corpus", "path": datamod.EXAMPLE_CORPUS,
                                 "tokenizer": "char", "val_split": 0.1}, tmp_path, spec)
    assert info.task == "language_model" and len(tr) and len(va)
    x, y = tr[0]
    assert torch.equal(x[1:], y[:-1])  # the target is the input shifted by one token
    (tmp_path / "short.txt").write_text("too short")
    with pytest.raises(ValueError, match="needs more than"):
        datamod.load({"source": "text_corpus", "path": tmp_path / "short.txt"}, tmp_path, spec)


def test_image_folder(tmp_path):
    for cls, colour in (("cat", "red"), ("dog", "blue")):
        (tmp_path / "img" / cls).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (50, 40), colour).save(tmp_path / "img" / cls / f"{i}.png")
    spec = make_spec("CNN (32×32 colour, e.g. CIFAR-10)")
    tr, va, info = datamod.load({"source": "image_folder", "path": tmp_path / "img"}, tmp_path, spec)
    assert info.class_names == ["cat", "dog"] and len(tr) + len(va) == 6
    assert tuple(tr[0][0].shape) == tuple(info.input_shape) == (3, 32, 32)  # resized


def test_limit():
    ds = list(range(10))
    assert len(limit(ds, 3)) == 3 and limit(ds, 0) is ds and limit(ds, 99) is ds


def test_tokenize_and_left_pad():
    assert tokenize("Great movie, isn't it?") == ["great", "movie", ",", "isn't", "it", "?"]
    assert encode_text("great unknown", {"great": 5}, 4) == [0, 0, 5, 1]


def test_prepare_tabular_input():
    spec = make_spec("MLP classifier (tabular)")
    spec.input_shape = [3]
    spec.data["preprocess"] = {"type": "tabular", "features": ["size", "weight", "color"],
                               "mean": [1.0, 0.0, 0.0], "std": [2.0, 1.0, 1.0],
                               "categorical": {"color": ["green", "red"]}}
    assert prepare_input(spec, "3, 4.5, red").tolist() == [[1.0, 4.5, 1.0]]
    with pytest.raises(ValueError, match="Expected 3 values"):
        prepare_input(spec, "1, 2")
    with pytest.raises(ValueError, match="not a known value"):
        prepare_input(spec, "1, 2, blue")


def _image_items(tmp_path, n):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    return [(tmp_path / "a.png", 0)] * n


def test_auto_workers_only_for_enough_images_on_linux(tmp_path, monkeypatch):
    pre = {"size": [8, 8], "channels": 3, "mean": [0.5] * 3, "std": [0.5] * 3}
    big = Subset(datamod._ImageFiles(_image_items(tmp_path, 2000), datamod._image_transform(pre)), range(1500))
    small = datamod._ImageFiles(_image_items(tmp_path, 50), datamod._image_transform(pre))
    tensors = TensorDataset(torch.zeros(5000, 2))
    monkeypatch.setattr(datamod.os, "cpu_count", lambda: 8)

    monkeypatch.setattr(datamod.sys, "platform", "linux")
    assert datamod.loader_workers(big) == 4
    assert datamod.loader_workers(small) == 0 and datamod.loader_workers(tensors) == 0
    monkeypatch.setattr(datamod.sys, "platform", "darwin")  # workers start and stop slowly there
    assert datamod.loader_workers(big) == 0
    assert datamod.loader_workers(tensors, requested=2) == 2  # an explicit choice always wins


def test_image_dataset_can_go_to_worker_processes(tmp_path):
    pre = {"size": [8, 8], "channels": 3, "mean": [0.5] * 3, "std": [0.5] * 3}
    ds = Subset(datamod._ImageFiles(_image_items(tmp_path, 4), datamod._image_transform(pre)), range(2))
    copy = pickle.loads(pickle.dumps(ds))  # what spawn does for every worker
    assert torch.equal(copy[0][0], ds[0][0])
