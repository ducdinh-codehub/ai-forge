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
