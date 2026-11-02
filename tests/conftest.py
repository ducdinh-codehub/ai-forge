import pytest
import torch

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
