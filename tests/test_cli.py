"""End-to-end checks of the command line: every command against a throwaway workspace."""
import csv
import json

import pytest

from aiforge import EXAMPLES_DIR
from aiforge.cli import main


@pytest.fixture
def aiforge(tmp_path):
    def run(*args):
        return main(["-w", str(tmp_path), *map(str, args)])
    run.ws = tmp_path
    return run


@pytest.fixture
def spiral(aiforge):
    assert aiforge("new", "spiral", "-t", "mlp classifier") == 0
    assert aiforge("train", "spiral", "-d", "spirals", "-e", 3, "--lr", 0.005, "--device", "cpu") == 0
    return aiforge


def history(ws, name):
    return json.loads((ws / "models" / name / "history.json").read_text())


def test_new_train_predict_export(spiral, capsys):
    spec = json.loads((spiral.ws / "models/spiral/spec.json").read_text())
    assert spec["class_names"] == ["arm-0", "arm-1", "arm-2"]  # auto-fitted to the data
    run = history(spiral.ws, "spiral")[-1]
    assert run["kind"] == "train" and len(run["epochs"]) == 3

    capsys.readouterr()
    assert spiral("predict", "spiral", "0.1, 0.45") == 0
    assert "arm-" in capsys.readouterr().out

    assert spiral("export", "spiral", "-F", "onnx,bundle") == 0
    assert (spiral.ws / "exports/spiral/spiral.onnx").exists()
    assert (spiral.ws / "exports/spiral/spiral.aiforge.zip").exists()
    assert spiral("export", "spiral", "-F", "nonsense") == 1


def test_retrain_reuses_data_and_resumes(spiral, capsys):
    assert spiral("train", "spiral", "-e", 1, "--resume", "--device", "cpu") == 0
    assert "loaded trained weights" in capsys.readouterr().out
    assert len(history(spiral.ws, "spiral")) == 2


def test_fine_tune_saves_a_copy(spiral):
    assert spiral("train", "spiral", "-d", "moons", "-e", 1, "--save-as", "moons", "--device", "cpu") == 0
    run = history(spiral.ws, "moons")[-1]
    assert run["kind"] == "fine-tune" and run["parent"] == "spiral"
    assert len(history(spiral.ws, "spiral")) == 1  # the original is untouched
    assert spiral("train", "spiral", "-d", "moons", "-e", 1, "--save-as", "moons") == 1  # name taken


def test_regression_csv_batch_predict(aiforge, tmp_path):
    data = EXAMPLES_DIR / "house_prices.csv"
    assert aiforge("new", "house", "-t", "regressor") == 0
    assert aiforge("train", "house", "-d", data, "-e", 2, "--device", "cpu") == 0
    out = tmp_path / "pred.csv"
    assert aiforge("predict", "house", "-f", data, "-o", out) == 0
    rows = list(csv.reader(out.open()))
    assert rows[0][-1] == "prediction" and len(rows) == 501


def test_text_classifier_keeps_class_names(aiforge, capsys):
    # The template already has 2 outputs, so nothing "mismatches" - the names must still be saved.
    assert aiforge("new", "reviews", "-t", "bag of words") == 0
    assert aiforge("train", "reviews", "-d", EXAMPLES_DIR / "movie_reviews.csv", "-e", 1, "--device", "cpu") == 0
    capsys.readouterr()
    assert aiforge("predict", "reviews", "a great film") == 0
    out = capsys.readouterr().out
    assert "pos" in out and "class 0" not in out


def test_language_model_generates(aiforge, capsys):
    assert aiforge("new", "gpt", "-t", "tiny gpt") == 0
    assert aiforge("train", "gpt", "-d", EXAMPLES_DIR / "nursery_rhymes.txt", "-e", 1,
                   "--max-samples", 200, "--device", "cpu") == 0
    capsys.readouterr()
    assert aiforge("generate", "gpt", "Mary", "-n", 20, "--seed", 1) == 0
    assert capsys.readouterr().out.startswith("Mary")
    assert aiforge("predict", "gpt", "Mar") == 0


def test_evaluate_writes_report(spiral, capsys):
    assert spiral("evaluate", "spiral", "--no-robustness") == 0
    assert "Accuracy" in capsys.readouterr().out
    assert list((spiral.ws / "exports/spiral").glob("evaluation-*.md"))


@pytest.mark.parametrize("args, message", [
    (("train", "ghost", "-d", "moons"), "No model called 'ghost'"),
    (("new", "x", "-t", "cnn"), "matches"),
    (("new", "x", "-t", "nothing like this"), "No template matches"),
])
def test_errors_are_one_line(aiforge, capsys, args, message):
    assert aiforge(*args) == 1
    assert message in capsys.readouterr().err


def test_untrained_and_bad_settings(aiforge, capsys):
    aiforge("new", "m", "-t", 1)
    assert aiforge("predict", "m", "1, 2") == 1
    assert aiforge("train", "m") == 1  # never trained, so --data is needed
    assert aiforge("train", "m", "-d", "moons", "-e", 0) == 1
    assert aiforge("train", "m", "-d", "/no/such/file.csv") == 1
    assert "isn't trained" in capsys.readouterr().err
