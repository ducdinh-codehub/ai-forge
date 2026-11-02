"""Export to every format and import back: structure, weights and history survive the trip."""
import json

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from aiforge.model_io import EXPORT_FORMATS, export, import_path, load_weights_into
from conftest import MODEL_TEMPLATES, example_input, make_spec


def _store(ws, template, name="m"):
    spec = make_spec(template, name)
    ws.save_spec(spec)
    model = spec.build()
    ws.save_weights(name, model)
    return spec, model.eval()


@pytest.mark.parametrize("template", MODEL_TEMPLATES)
def test_torchscript_and_onnx_export(ws, template):
    spec, model = _store(ws, template)
    x = example_input(spec)

    ts = torch.jit.load(str(export(ws, "m", "torchscript")))
    with torch.no_grad():
        assert torch.allclose(ts(x), model(x), atol=1e-5)

    proto = onnx.load(str(export(ws, "m", "onnx")))
    onnx.checker.check_model(proto)
    dims = proto.graph.output[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "batch"  # any batch size works
    # The exporter may leave a sequence length symbolic; every fixed size must be right.
    assert all(d.dim_param or d.dim_value == n for d, n in zip(dims[1:], spec.output_shape()))
    assert len(dims) == 1 + len(spec.output_shape())


@pytest.mark.parametrize("template", ["MLP classifier (tabular)", "Small CNN (28×28 grayscale, e.g. MNIST)",
                                      "Text classifier (Embedding + LSTM)", "Tiny GPT (characters, fast)"])
def test_onnx_gives_the_same_answers(ws, template):
    spec, model = _store(ws, template)
    x = example_input(spec, batch=3)  # exported with batch 1
    session = ReferenceEvaluator(str(export(ws, "m", "onnx")))
    out = session.run(None, {"input": x.numpy()})[0]
    with torch.no_grad():
        assert np.allclose(out, model(x).numpy(), atol=1e-4)


def test_every_format_writes_a_file(ws):
    _store(ws, "Small CNN (28×28 grayscale, e.g. MNIST)")
    for fmt in EXPORT_FORMATS:
        path = export(ws, "m", fmt, ws.root / "out")
        assert path.exists() and path.stat().st_size > 0, fmt
    with pytest.raises(ValueError):
        export(ws, "m", "nope")


def test_bundle_round_trip(ws):
    spec, model = _store(ws, "CNN (32×32 colour, e.g. CIFAR-10)")
    ws.append_history("m", {"kind": "train", "epochs": []})
    name, msg = import_path(ws, str(export(ws, "m", "bundle")))
    assert name == "m-2" and "WARNING" not in msg

    copy_spec, copy_model, _ = ws.load_model(name)
    assert [l.type for l in copy_spec.layers] == [l.type for l in spec.layers]
    assert ws.load_history(name) == ws.load_history("m")
    x = example_input(spec)
    with torch.no_grad():
        assert torch.equal(copy_model.eval()(x), model(x))


def test_safetensors_round_trip(ws):
    _, model = _store(ws, "MLP classifier (tabular)")
    name, msg = import_path(ws, str(export(ws, "m", "weights_safetensors")))
    assert "weights 6/6" in msg
    _, copy_model, _ = ws.load_model(name)
    for a, b in zip(copy_model.state_dict().values(), model.state_dict().values()):
        assert torch.equal(a, b)


def test_spec_json_imports_structure_only(ws):
    spec, _ = _store(ws, "Text Transformer classifier")
    name, _ = import_path(ws, str(export(ws, "m", "spec_json")))
    assert ws.load_spec(name).to_dict() | {"name": "m"} == spec.to_dict()
    assert not ws.has_weights(name)


def test_load_weights_into_matching_model(ws):
    _, model = _store(ws, "MLP classifier (tabular)")
    path = export(ws, "m", "weights_pt")
    ws.save_spec(make_spec("MLP classifier (tabular)", "other"))
    assert load_weights_into(ws, "other", str(path)).startswith("Loaded 6/6")
    with pytest.raises(ValueError, match="No tensor matched"):
        ws.save_spec(make_spec("Small CNN (28×28 grayscale, e.g. MNIST)", "cnn"))
        load_weights_into(ws, "cnn", str(path))


def test_summary_lists_every_layer(ws):
    spec, _ = _store(ws, "MLP classifier (tabular)")
    text = export(ws, "m", "summary_txt").read_text()
    assert f"{spec.total_params():,}" in text
    assert all(l.type in text for l in spec.layers)


def test_import_rejects_unknown_files(ws, tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(ValueError, match="Supported"):
        import_path(ws, str(tmp_path / "x.txt"))
    with pytest.raises(FileNotFoundError):
        import_path(ws, str(tmp_path / "missing.zip"))
    (tmp_path / "folder").mkdir()
    with pytest.raises(ValueError, match="spec.json"):
        import_path(ws, str(tmp_path / "folder"))
    (tmp_path / "s.json").write_text(json.dumps({"name": "s", "input_shape": [2]}))
    assert import_path(ws, str(tmp_path / "s.json"))[0] == "s"
