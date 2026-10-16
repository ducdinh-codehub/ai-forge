"""Export a model to common formats and import models from disk."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
from torch import nn

from .spec import DTYPES, LayerSpec, ModelSpec, Workspace, safe_name, to_pytorch_source, transfer_state

EXPORT_FORMATS = {
    "bundle": ("AIForge bundle (.zip)", "Structure + weights + history; re-import anywhere"),
    "weights_pt": ("Weights - PyTorch state_dict (.pt)", "torch.load() then model.load_state_dict()"),
    "weights_safetensors": ("Weights - safetensors (.safetensors)", "Safe, fast tensor format (HF ecosystem)"),
    "spec_json": ("Structure - JSON spec (.json)", "Layer list, shapes, classes, preprocessing"),
    "pytorch_py": ("Structure - PyTorch code (.py)", "Plain nn.Sequential source you can edit"),
    "summary_txt": ("Structure - summary table (.txt)", "Layers, output shapes, parameter counts"),
    "torchscript": ("Full model - TorchScript (.pt)", "Runs with torch.jit.load(), no Python class needed"),
    "onnx": ("Full model - ONNX (.onnx)", "Deploy with ONNX Runtime, TensorRT, OpenVINO, ..."),
}


def summary_text(spec: ModelSpec) -> str:
    infos = spec.analyze()
    w = [4, 19, 15, 15, 13]
    head = f"{'#':<{w[0]}}{'Layer':<{w[1]}}{'Input':<{w[2]}}{'Output':<{w[3]}}{'Params':>{w[4]}}"
    lines = [f"Model: {spec.name}   task: {spec.task}", "=" * sum(w), head, "-" * sum(w),
             f"{'':<{w[0]}}{'Input':<{w[1]}}{'':<{w[2]}}{str(tuple(spec.input_shape)):<{w[3]}}{'':>{w[4]}}"]
    for i in infos:
        lines.append(f"{i.index + 1:<{w[0]}}{i.spec.type:<{w[1]}}{str(i.in_shape):<{w[2]}}"
                     f"{str(i.out_shape) if i.out_shape else 'ERROR':<{w[3]}}{i.params:>{w[4]},}")
    total = sum(i.params for i in infos)
    frozen = sum(i.params for i in infos if i.spec.id in spec.frozen)
    lines += ["=" * sum(w), f"Total params: {total:,}", f"Trainable:    {total - frozen:,}",
              f"Size (fp32):  {total * 4 / 1024 ** 2:.2f} MB"]
    if spec.class_names:
        lines.append(f"Classes: {', '.join(spec.class_names)}")
    return "\n".join(lines)


def export(ws: Workspace, name: str, fmt: str, out_dir: Path | None = None) -> Path:
    spec, model, _ = ws.load_model(name)
    model.eval()
    out_dir = Path(out_dir or ws.export_dir / name).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / safe_name(name)
    example = torch.zeros((1, *spec.input_shape), dtype=DTYPES[spec.input_dtype])

    if fmt == "bundle":
        path = base.with_suffix(".aiforge.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("spec.json", "weights.pt", "history.json"):
                src = ws.model_dir(name) / f
                if src.exists():
                    z.write(src, f)
        return path
    if fmt == "weights_pt":
        path = base.with_name(base.name + "_weights.pt")
        torch.save(model.state_dict(), path)
        return path
    if fmt == "weights_safetensors":
        from safetensors.torch import save_file

        path = base.with_suffix(".safetensors")
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(path), metadata={"aiforge_spec": json.dumps(spec.to_dict())[:60000]})
        return path
    if fmt == "spec_json":
        path = base.with_name(base.name + "_spec.json")
        path.write_text(json.dumps(spec.to_dict(), indent=2))
        return path
    if fmt == "pytorch_py":
        path = base.with_name(base.name.replace("-", "_").replace(".", "_") + "_model.py")
        path.write_text(to_pytorch_source(spec))
        return path
    if fmt == "summary_txt":
        path = base.with_name(base.name + "_summary.txt")
        path.write_text(summary_text(spec) + "\n")
        return path
    if fmt == "torchscript":
        path = base.with_name(base.name + "_torchscript.pt")
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        traced.save(str(path))
        return path
    if fmt == "onnx":
        path = base.with_suffix(".onnx")
        torch.onnx.export(model, (example,), str(path), input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                          opset_version=17, dynamo=False)
        return path
    raise ValueError(f"Unknown export format {fmt}")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_path(ws: Workspace, path: str, name: str | None = None,
                input_shape: list[int] | None = None) -> tuple[str, str]:
    """Import a model into the workspace. Returns (model name, message)."""
    p = Path(path.strip().strip("'\"")).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    if p.is_dir():
        sf = p / "spec.json"
        if not sf.exists():
            raise ValueError("Folder has no spec.json (expected an AIForge model folder)")
        spec = ModelSpec.from_dict(json.loads(sf.read_text()))
        return _store(ws, spec, name, (p / "weights.pt") if (p / "weights.pt").exists() else None,
                      (p / "history.json") if (p / "history.json").exists() else None)

    if p.suffix == ".zip":
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(p) as z:
            z.extractall(tmp)
            return import_path(ws, tmp, name or p.name.split(".")[0], input_shape)

    if p.suffix == ".json":
        spec = ModelSpec.from_dict(json.loads(p.read_text()))
        return _store(ws, spec, name, None, None)

    if p.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(p), "pt") as f:
            meta = f.metadata() or {}
        if "aiforge_spec" not in meta:
            raise ValueError("This safetensors file has no AIForge structure. Open a model and use "
                             "'Load weights' to attach it to a matching structure.")
        spec = ModelSpec.from_dict(json.loads(meta["aiforge_spec"]))
        return _store(ws, spec, name, p, None)

    if p.suffix in (".pt", ".pth", ".bin"):
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            raise ValueError("This file holds only weights (a state_dict). Open or create a model with "
                             "the same structure and use 'Load weights'.")
        if isinstance(obj, torch.jit.ScriptModule):
            raise ValueError("TorchScript models can't be edited. Export the original nn.Sequential instead.")
        if not isinstance(obj, nn.Module):
            raise ValueError(f"Unsupported object in file: {type(obj).__name__}")
        spec, model = sequential_to_spec(obj, name or p.stem, input_shape)
        tmp = ws.root / ".import_weights.pt"
        torch.save(model.state_dict(), tmp)
        try:
            return _store(ws, spec, name or p.stem, tmp, None)
        finally:
            tmp.unlink(missing_ok=True)

    raise ValueError("Supported: AIForge .zip / folder / .json, .safetensors from AIForge, "
                     "or a torch.save()'d nn.Sequential (.pt/.pth)")


def _store(ws: Workspace, spec: ModelSpec, name: str | None, weights: Path | None,
           history: Path | None) -> tuple[str, str]:
    spec.name = ws.unique_name(name or spec.name)
    problems = [i.error for i in spec.analyze() if i.error]
    ws.save_spec(spec)
    msg = f"Imported '{spec.name}' ({len(spec.layers)} layers, {spec.total_params():,} params)"
    if weights:
        model = spec.build()
        if weights.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(weights))
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        n, total = transfer_state(state, model)
        ws.save_weights(spec.name, model)
        msg += f", weights {n}/{total} tensors"
    if history:
        (ws.model_dir(spec.name) / "history.json").write_text(history.read_text())
    if problems:
        msg += f" - WARNING: {problems[0]}"
    return spec.name, msg


def load_weights_into(ws: Workspace, name: str, path: str) -> str:
    p = Path(path.strip().strip("'\"")).expanduser()
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(p))
    else:
        state = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    spec, model, _ = ws.load_model(name)
    n, total = transfer_state(state, model)
    if n == 0:
        raise ValueError("No tensor matched this model's layer names/shapes.")
    ws.save_weights(name, model)
    return f"Loaded {n}/{total} tensors into '{name}'"


_SIMPLE = {nn.ReLU: "ReLU", nn.GELU: "GELU", nn.SiLU: "SiLU", nn.Tanh: "Tanh",
           nn.Sigmoid: "Sigmoid", nn.Flatten: "Flatten"}


def _flatten_modules(m: nn.Module) -> list[nn.Module]:
    if isinstance(m, nn.Sequential):
        return [x for c in m for x in _flatten_modules(c)]
    return [m]


def sequential_to_spec(module: nn.Module, name: str, input_shape: list[int] | None):
    if not isinstance(module, nn.Sequential):
        raise ValueError(f"Only nn.Sequential models can be converted (got {type(module).__name__}).")
    mods = _flatten_modules(module)
    layers: list[LayerSpec] = []
    sources: list[nn.Module | None] = []
    dtype = "float"
    i = 0
    while i < len(mods):
        m = mods[i]
        t = type(m)
        if t is nn.AdaptiveAvgPool2d and i + 1 < len(mods) and isinstance(mods[i + 1], nn.Flatten):
            layers.append(LayerSpec.new("GlobalAvgPool"))
            sources.append(None)
            i += 2
            continue
        if t in _SIMPLE:
            ls = LayerSpec.new(_SIMPLE[t])
        elif t is nn.Linear:
            ls = LayerSpec.new("Linear", out_features=m.out_features, bias=m.bias is not None)
            input_shape = input_shape or ([m.in_features] if not layers else None)
        elif t in (nn.Conv2d, nn.Conv1d):
            ls = LayerSpec.new(t.__name__, out_channels=m.out_channels, kernel_size=m.kernel_size[0],
                               stride=m.stride[0], padding=int(m.padding[0]), bias=m.bias is not None)
        elif t in (nn.MaxPool2d, nn.AvgPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            ls = LayerSpec.new(t.__name__, kernel_size=k, stride=s)
        elif t in (nn.BatchNorm1d, nn.BatchNorm2d):
            ls = LayerSpec.new("BatchNorm")
        elif t is nn.LayerNorm:
            ls = LayerSpec.new("LayerNorm")
        elif t is nn.Dropout:
            ls = LayerSpec.new("Dropout", p=m.p)
        elif t is nn.LeakyReLU:
            ls = LayerSpec.new("LeakyReLU", negative_slope=m.negative_slope)
        elif t is nn.Embedding:
            ls = LayerSpec.new("Embedding", num_embeddings=m.num_embeddings, embedding_dim=m.embedding_dim)
            dtype = "long" if not layers else dtype
        else:
            raise ValueError(f"Layer type {t.__name__} is not supported by the builder yet.")
        layers.append(ls)
        sources.append(m)
        i += 1
    if not input_shape:
        raise ValueError("Couldn't infer the input shape - enter it in the import dialog (e.g. 3,32,32).")
    spec = ModelSpec(name=name, input_shape=list(input_shape), input_dtype=dtype, layers=layers)
    last = layers[-1] if layers else None
    if last and last.type == "Linear" and last.config["out_features"] == 1:
        spec.task = "regression"
    model = spec.build()
    for src, dst in zip(sources, model):
        if src is not None:
            dst.load_state_dict(src.state_dict())
    return spec, model
