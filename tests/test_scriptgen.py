"""The exported training script runs on its own and trains the very same network."""
import subprocess
import sys

import pytest
import torch

from aiforge import EXAMPLES_DIR
from aiforge.cli import main
from aiforge.model_io import export
from aiforge.spec import Workspace


@pytest.mark.parametrize("template, data, extra, expect", [
    ("mlp classifier", "spirals", [], "val acc"),
    ("mlp classifier", EXAMPLES_DIR / "fruit_size.csv", [], "val acc"),
    ("mlp regressor", EXAMPLES_DIR / "house_prices.csv", [], "target units"),
    ("bag of words", EXAMPLES_DIR / "movie_reviews.csv", [], "val acc"),
    ("lstm language model", EXAMPLES_DIR / "nursery_rhymes.txt", ["--max-samples", 64], "val perplexity"),
    ("small cnn", EXAMPLES_DIR / "color_images", [], "val acc"),
])
def test_script_trains_the_same_network(tmp_path, template, data, extra, expect):
    ws_dir = tmp_path / "ws"
    assert main(["-w", str(ws_dir), "new", "m", "-t", template]) == 0
    assert main(["-w", str(ws_dir), "train", "m", "-d", str(data), "-e", "1", "--device", "cpu",
                 "--scheduler", "cosine", *map(str, extra)]) == 0
    ws = Workspace(ws_dir)
    script = export(ws, "m", "train_py", tmp_path)
    assert script.name == "m_train.py"
    text = script.read_text()
    assert "'scheduler': 'cosine'" in text and "def build_model" in text

    out = tmp_path / "w.pt"
    proc = subprocess.run([sys.executable, str(script), "--epochs", "2", "--device", "cpu", "--out", str(out)],
                          capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "epoch   2" in proc.stdout and expect in proc.stdout

    _, model, _ = ws.load_model("m")
    model.load_state_dict(torch.load(out, weights_only=True))  # strict: identical layers and shapes


def test_untrained_model_has_no_script(tmp_path):
    ws_dir = tmp_path / "ws"
    main(["-w", str(ws_dir), "new", "m", "-t", "mlp classifier"])
    with pytest.raises(ValueError, match="Train the model in AIForge first"):
        export(Workspace(ws_dir), "m", "train_py", tmp_path)


def _run_script(tmp_path, template, data):
    ws_dir = tmp_path / "ws"
    assert main(["-w", str(ws_dir), "new", "m", "-t", template]) == 0
    assert main(["-w", str(ws_dir), "train", "m", "-d", str(data), "-e", "1", "--device", "cpu"]) == 0
    ws = Workspace(ws_dir)
    script = export(ws, "m", "train_py", tmp_path)
    out = tmp_path / "w.pt"
    proc = subprocess.run([sys.executable, str(script), "--epochs", "1", "--device", "cpu", "--out", str(out)],
                          capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    _, model, _ = ws.load_model("m")
    model.load_state_dict(torch.load(out, weights_only=True))


def test_script_reads_parquet_and_tables_of_images(tmp_path):
    from test_data_sources import _as_numbers, _fruit_rows, _image_table, _parquet

    pytest.importorskip("pyarrow")
    _parquet(tmp_path / "fruit.parquet", [_fruit_rows()[0]] + _as_numbers(_fruit_rows()[1:]))
    _run_script(tmp_path / "a", "mlp classifier", tmp_path / "fruit.parquet")
    (tmp_path / "b").mkdir()
    _run_script(tmp_path / "b", "small cnn", _image_table(tmp_path / "b"))


def test_script_for_huggingface_text_uses_datasets():
    from aiforge.scriptgen import training_script
    from aiforge.trainer import TrainConfig
    from conftest import make_spec

    spec = make_spec("Text classifier (Embedding + LSTM)")
    spec.data["dataset_cfg"] = {"source": "hf", "name": "imdb", "split": "train"}
    text = training_script(spec, TrainConfig())
    assert "from datasets import load_dataset" in text and "the Hugging Face dataset 'imdb'" in text
    compile(text, "train.py", "exec")


def test_windows_paths_dont_break_the_script():
    from aiforge.scriptgen import training_script
    from aiforge.trainer import TrainConfig
    from conftest import make_spec

    spec = make_spec("MLP classifier (tabular)")
    spec.data["dataset_cfg"] = {"source": "csv", "path": r"C:\Users\new\table.csv"}  # \U and \n
    text = training_script(spec, TrainConfig())
    namespace = {}
    exec(compile(text.replace('if __name__ == "__main__":', "if False:"), "train.py", "exec"), namespace)
    assert r"C:\Users\new\table.csv" in namespace["__doc__"] and namespace["DATA"]["path"] == r"C:\Users\new\table.csv"
