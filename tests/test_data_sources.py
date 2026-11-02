"""More ways to bring data: Excel and Parquet tables, tables of image paths, Hugging Face datasets."""
import csv
import sys

import pytest
import torch
from PIL import Image

from aiforge import EXAMPLES_DIR
from aiforge import data as datamod
from aiforge.cli import main
from aiforge.pipeline import data_cfg_from_arg
from aiforge.spec import Workspace
from conftest import make_spec

FRUIT = EXAMPLES_DIR / "fruit_size.csv"


def _fruit_rows():
    with FRUIT.open() as f:
        return list(csv.reader(f))


def _as_numbers(rows):
    return [[float(v) if v.replace(".", "", 1).isdigit() else v for v in r] for r in rows]


def _excel(path, rows):
    import openpyxl

    book = openpyxl.Workbook()
    for r in rows:
        book.active.append(r)
    book.save(path)


def _parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    header, data = rows[0], rows[1:]
    pq.write_table(pa.table({h: [r[i] for r in data] for i, h in enumerate(header)}), path)


@pytest.mark.parametrize("suffix, write", [(".xlsx", _excel), (".parquet", _parquet)])
def test_excel_and_parquet_load_like_the_csv(tmp_path, suffix, write):
    pytest.importorskip("openpyxl" if suffix == ".xlsx" else "pyarrow")
    path = tmp_path / f"fruit{suffix}"
    write(path, [_fruit_rows()[0]] + _as_numbers(_fruit_rows()[1:]))  # real numbers, as Excel stores them
    a_tr, _, a = datamod.load({"source": "csv", "path": FRUIT}, tmp_path)
    b_tr, _, b = datamod.load({"source": "csv", "path": path}, tmp_path)
    assert b.class_names == a.class_names and b.preprocess["categorical"] == a.preprocess["categorical"]
    assert torch.allclose(b_tr[0][0], a_tr[0][0])
    assert datamod.csv_columns(str(path)) == ["size", "weight", "color", "label"]


def test_missing_optional_package_says_what_to_install(tmp_path, monkeypatch):
    real = __import__("importlib").import_module

    def no_openpyxl(name, *a):
        if name == "openpyxl":
            raise ImportError(name)
        return real(name, *a)

    monkeypatch.setattr("importlib.import_module", no_openpyxl)
    (tmp_path / "t.xlsx").write_bytes(b"")
    with pytest.raises(ValueError, match="pip install openpyxl"):
        datamod.read_table(tmp_path / "t.xlsx")


def _image_table(tmp_path, missing=False):
    for cls, colour in (("cat", "red"), ("dog", "blue")):
        (tmp_path / "pics" / cls).mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (40, 30), colour).save(tmp_path / "pics" / cls / f"{i}.png")
    rows = [["file", "animal"]] + [[f"pics/{c}/{i}.png", c] for c in ("cat", "dog") for i in range(4)]
    if missing:
        rows.append(["pics/cat/nope.png", "cat"])
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def test_table_of_image_paths(tmp_path):
    spec = make_spec("CNN (32×32 colour, e.g. CIFAR-10)")
    tr, va, info = datamod.load({"source": "image_csv", "path": _image_table(tmp_path)}, tmp_path, spec)
    assert info.class_names == ["cat", "dog"] and len(tr) + len(va) == 8
    assert tuple(tr[0][0].shape) == (3, 32, 32)  # resized to the model's input
    assert data_cfg_from_arg(str(tmp_path / "labels.csv"), spec)["source"] == "image_csv"


def test_table_of_image_paths_reports_missing_files(tmp_path):
    spec = make_spec("CNN (32×32 colour, e.g. CIFAR-10)")
    with pytest.raises(FileNotFoundError, match="1 image"):
        datamod.load({"source": "image_csv", "path": _image_table(tmp_path, missing=True)}, tmp_path, spec)


def test_huggingface_table(tmp_path, hub):
    spec = make_spec("MLP classifier (tabular)")
    tr, va, info = datamod.load({"source": "hf", "name": "iris"}, tmp_path, spec)
    assert info.class_names == ["setosa", "virginica"]  # names from the dataset, not 0 / 1
    assert info.input_shape == [2] and len(tr) + len(va) == 40
    assert hub == [("iris", None, "train")]


def test_huggingface_text(tmp_path, hub):
    spec = make_spec("Text classifier (Embedding + LSTM)")
    cfg = data_cfg_from_arg("hf:imdb:plain_text", spec, split="test")
    assert cfg == {"source": "hf", "name": "imdb", "subset": "plain_text", "split": "test"}
    tr, _, info = datamod.load(cfg, tmp_path, spec)
    assert info.class_names == ["neg", "pos"] and info.input_dtype == "long"
    assert hub == [("imdb", "plain_text", "test")]


def test_huggingface_images(tmp_path, hub):
    spec = make_spec("CNN (32×32 colour, e.g. CIFAR-10)")
    tr, _, info = datamod.load({"source": "hf", "name": "pets"}, tmp_path, spec)
    assert info.class_names == ["cat", "dog"] and tuple(tr[0][0].shape) == (3, 32, 32)
    assert datamod.decodes_per_sample(tr)  # decoded on the fly, like image files


def test_huggingface_errors(tmp_path, hub, monkeypatch):
    with pytest.raises(ValueError, match="Couldn't load 'nope'"):
        datamod.load({"source": "hf", "name": "nope"}, tmp_path, make_spec("MLP classifier (tabular)"))
    with pytest.raises(ValueError, match="Enter a Hugging Face dataset name"):
        datamod.load({"source": "hf", "name": " "}, tmp_path)
    monkeypatch.delitem(sys.modules, "datasets")
    monkeypatch.setattr("importlib.import_module", lambda name, *a: (_ for _ in ()).throw(ImportError(name)))
    with pytest.raises(ValueError, match="pip install datasets"):
        datamod.load({"source": "hf", "name": "imdb"}, tmp_path)


def test_cli_trains_from_excel_and_from_a_table_of_images(tmp_path):
    pytest.importorskip("openpyxl")
    _excel(tmp_path / "fruit.xlsx", [_fruit_rows()[0]] + _as_numbers(_fruit_rows()[1:]))
    ws = tmp_path / "ws"
    assert main(["-w", str(ws), "new", "fruit", "-t", "mlp classifier"]) == 0
    assert main(["-w", str(ws), "train", "fruit", "-d", str(tmp_path / "fruit.xlsx"), "-e", "1", "--device", "cpu"]) == 0
    assert Workspace(ws).load_spec("fruit").class_names == ["big", "small"]

    labels = _image_table(tmp_path)
    assert main(["-w", str(ws), "new", "pets", "-t", "small cnn"]) == 0
    assert main(["-w", str(ws), "train", "pets", "-d", str(labels), "-e", "1", "--device", "cpu"]) == 0
    assert Workspace(ws).load_spec("pets").data["dataset_cfg"]["source"] == "image_csv"
