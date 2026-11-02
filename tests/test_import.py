"""Importing torch.save()'d files: supported models load, anything that could run code is refused."""
import os

import pytest
import torch
from torch import nn

from aiforge.model_io import import_path
from aiforge.spec import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


def test_imports_pickled_sequential(ws, tmp_path):
    src = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Sequential(nn.Dropout(0.1), nn.Linear(8, 3)))
    torch.save(src, tmp_path / "mlp.pt")
    name, msg = import_path(ws, str(tmp_path / "mlp.pt"))
    assert name == "mlp" and "weights 4/4" in msg
    _, model, _ = ws.load_model(name)
    x = torch.randn(2, 4)
    assert torch.allclose(model.eval()(x), src.eval()(x))


class _RunsCode:
    def __reduce__(self):
        return os.system, ("echo pwned > " + os.devnull,)


def test_refuses_file_that_runs_code(ws, tmp_path):
    torch.save({"x": _RunsCode()}, tmp_path / "evil.pt")
    with pytest.raises(ValueError, match="can't load safely"):
        import_path(ws, str(tmp_path / "evil.pt"))


def test_refuses_unsupported_layer(ws, tmp_path):
    torch.save(nn.Sequential(nn.Linear(2, 2), nn.PReLU()), tmp_path / "prelu.pt")
    with pytest.raises(ValueError, match="can't load safely"):
        import_path(ws, str(tmp_path / "prelu.pt"))


def test_state_dict_and_torchscript_messages(ws, tmp_path):
    torch.save(nn.Linear(2, 2).state_dict(), tmp_path / "sd.pt")
    with pytest.raises(ValueError, match="only weights"):
        import_path(ws, str(tmp_path / "sd.pt"))
    torch.jit.script(nn.Sequential(nn.Linear(2, 2))).save(str(tmp_path / "ts.pt"))
    with pytest.raises(ValueError, match="TorchScript"):
        import_path(ws, str(tmp_path / "ts.pt"))
