"""Model specs: shape analysis, building, persistence, generated code and the workspace."""
import pytest
import torch

from aiforge.layers import count_params
from aiforge.spec import LayerSpec, ModelSpec, safe_name, to_pytorch_source
from conftest import MODEL_TEMPLATES, example_input, make_spec


@pytest.mark.parametrize("template", MODEL_TEMPLATES)
def test_template_builds_and_matches_its_analysis(template):
    spec = make_spec(template)
    assert spec.validate() == []
    model = spec.build().eval()
    with torch.no_grad():
        out = model(example_input(spec))
    assert tuple(out.shape) == (2, *spec.output_shape())
    assert spec.total_params() == count_params(model)[0]


@pytest.mark.parametrize("template", MODEL_TEMPLATES)
def test_generated_source_rebuilds_the_same_model(template):
    spec = make_spec(template)
    model = spec.build().eval()
    namespace = {}
    exec(compile(to_pytorch_source(spec), "<generated>", "exec"), namespace)
    rebuilt = namespace["build_model"]()
    rebuilt.load_state_dict(model.state_dict())  # strict: same keys and shapes
    rebuilt.eval()
    x = example_input(spec)
    with torch.no_grad():
        assert torch.allclose(rebuilt(x), model(x), atol=1e-6)


def test_dict_round_trip():
    spec = make_spec("MLP classifier (tabular)")
    spec.class_names, spec.frozen, spec.notes = ["a", "b"], [spec.layers[0].id], "hi"
    assert ModelSpec.from_dict(spec.to_dict()) == spec
    # Unknown keys from newer versions are ignored instead of crashing.
    assert ModelSpec.from_dict({**spec.to_dict(), "from_the_future": 1}) == spec


def test_shape_error_is_reported_and_stops_analysis():
    spec = ModelSpec("bad", [10], layers=[LayerSpec.new("Linear", out_features=4),
                                          LayerSpec.new("Conv2d"), LayerSpec.new("ReLU")])
    infos = spec.analyze()
    assert infos[0].error is None and infos[0].out_shape == (4,)
    assert infos[1].error  # a Conv2d needs an image, not a flat vector
    assert infos[2].error == "fix the layer above first"
    assert spec.validate()[0].startswith("Layer 2 (Conv2d)")


def test_validate_catches_head_and_class_mismatch():
    spec = make_spec("MLP classifier (tabular)")
    spec.class_names = ["a", "b", "c"]
    assert any("out_features to 3" in p for p in spec.validate())
    spec = make_spec("Small CNN (28×28 grayscale, e.g. MNIST)")
    spec.layers = [l for l in spec.layers if l.type not in ("Flatten", "GlobalAvgPool", "Linear")]
    assert any("flat vector" in p for p in spec.validate())


def test_freeze_stops_gradients():
    spec = make_spec("MLP classifier (tabular)")
    spec.frozen = [spec.layers[0].id]
    model = spec.build()
    assert not any(p.requires_grad for p in model[0].parameters())
    assert all(p.requires_grad for p in model[-1].parameters())


def test_workspace_save_rename_duplicate(ws):
    spec = make_spec("MLP classifier (tabular)", "first")
    ws.save_spec(spec)
    ws.save_weights("first", spec.build())
    ws.append_history("first", {"kind": "train"})
    assert ws.unique_name("first") == "first-2"

    ws.duplicate("first", "copy")
    ws.rename("first", "renamed")
    assert {m["name"] for m in ws.list_models()} == {"copy", "renamed"}
    assert ws.load_spec("renamed").name == "renamed"
    assert ws.has_weights("copy") and ws.load_history("copy") == [{"kind": "train"}]
    ws.delete("copy")
    assert not ws.exists("copy")


def test_weights_follow_layer_ids_across_edits(ws):
    spec = make_spec("MLP classifier (tabular)", "net")
    ws.save_spec(spec)
    trained = spec.build()
    ws.save_weights("net", trained)

    edited = spec.clone()
    edited.layers.insert(0, LayerSpec.new("Dropout"))  # shifts every position by one
    _, model, note = ws.load_model("net", edited)
    assert note.startswith("loaded trained weights")
    assert torch.equal(model[-1].weight, trained[-1].weight)


@pytest.mark.parametrize("raw, clean", [("my model", "my-model"), ("../etc/passwd", "etc-passwd"),
                                        ("  ok_v1.2 ", "ok_v1.2"), ("***", "")])
def test_safe_name(raw, clean):
    assert safe_name(raw) == clean
