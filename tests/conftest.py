import sys
import types

import pytest
import torch
from PIL import Image

from aiforge.spec import TEMPLATES, ModelSpec, Workspace, from_template

# Every template that describes a real model ("Blank" has no layers on purpose).
MODEL_TEMPLATES = [t for t in TEMPLATES if t != "Blank"]


def make_spec(template: str, name: str = "m") -> ModelSpec:
    """A template's spec, without ImageNet downloads (same structure, random weights)."""
    spec = from_template(template, name)
    for layer in spec.layers:
        if "pretrained" in layer.config:
            layer.config["pretrained"] = False
    return spec


def example_input(spec: ModelSpec, batch: int = 2) -> torch.Tensor:
    if spec.input_dtype == "long":
        return torch.randint(0, spec.vocab_size() or 2, (batch, *spec.input_shape))
    return torch.randn(batch, *spec.input_shape)


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


# -- Hugging Face (a stand-in for the `datasets` package, with the same interface) -------------------


class Value:
    def __init__(self, dtype):
        self.dtype = dtype


class ClassLabel:
    def __init__(self, names):
        self.names = names


class Image_:  # the class name is what AIForge checks
    pass


Image_.__name__ = "Image"


class FakeDataset:
    def __init__(self, columns, features):
        self.columns, self.features = columns, features
        self.column_names = list(columns)

    def __len__(self):
        return len(next(iter(self.columns.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.columns[key]
        return {c: v[key] for c, v in self.columns.items()}


@pytest.fixture
def hub(monkeypatch):
    """A fake Hugging Face Hub with three datasets; records what was asked for."""
    asked = []
    n = 40
    sets = {
        "iris": FakeDataset(
            {"sepal": [float(i % 7) for i in range(n)], "petal": [float(i % 3) for i in range(n)],
             "species": [i % 2 for i in range(n)]},
            {"sepal": Value("float32"), "petal": Value("float32"), "species": ClassLabel(["setosa", "virginica"])}),
        "imdb": FakeDataset(
            {"text": ["a great film" if i % 2 else "dull and awful" for i in range(n)], "label": [i % 2 for i in range(n)]},
            {"text": Value("string"), "label": ClassLabel(["neg", "pos"])}),
        "pets": FakeDataset(
            {"img": [Image.new("RGB", (20, 20), "red" if i % 2 else "blue") for i in range(n)],
             "label": [i % 2 for i in range(n)]},
            {"img": Image_(), "label": ClassLabel(["cat", "dog"])}),
    }

    def load_dataset(name, subset=None, split="train"):
        asked.append((name, subset, split))
        if name not in sets:
            raise FileNotFoundError(f"Dataset '{name}' doesn't exist on the Hub")
        return sets[name]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return asked
