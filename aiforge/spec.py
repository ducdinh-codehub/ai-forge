"""Model specification: a JSON-serialisable description of a sequential network."""
from __future__ import annotations

import copy
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from .layers import REGISTRY, Shape, count_params, output_shape

DTYPES = {"float": torch.float32, "long": torch.long}


@dataclass
class LayerSpec:
    type: str
    config: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @staticmethod
    def new(type_name: str, **overrides) -> "LayerSpec":
        cfg = REGISTRY[type_name].defaults()
        cfg.update(overrides)
        return LayerSpec(type_name, cfg)


@dataclass
class LayerInfo:
    index: int
    spec: LayerSpec
    in_shape: Shape | None
    out_shape: Shape | None
    params: int
    formula: str
    error: str | None = None


@dataclass
class ModelSpec:
    name: str
    input_shape: list[int]
    input_dtype: str = "float"  # "float" or "long" (token ids)
    task: str = "classification"  # "regression" or "language_model" (predict the next token)
    layers: list[LayerSpec] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    # Preprocessing learned from the training data (normalisation stats, image size...).
    data: dict = field(default_factory=dict)
    frozen: list[str] = field(default_factory=list)  # layer ids frozen during fine-tuning
    notes: str = ""

    # -- analysis ----------------------------------------------------------

    def analyze(self) -> list[LayerInfo]:
        """Propagate shapes through every layer; stop marking after the first error."""
        infos: list[LayerInfo] = []
        shape: Shape | None = tuple(self.input_shape)
        dtype = self.input_dtype
        for i, layer in enumerate(self.layers):
            if shape is None:
                infos.append(LayerInfo(i, layer, None, None, 0, "", "fix the layer above first"))
                continue
            try:
                out, params, formula = _analyze_layer(layer.type, _freeze(layer.config), shape, dtype)
                infos.append(LayerInfo(i, layer, shape, out, params, formula))
                shape, dtype = out, "float"
            except Exception as e:  # noqa: BLE001 - shown to the user
                infos.append(LayerInfo(i, layer, shape, None, 0, "", _friendly(e)))
                shape = None
        return infos

    def output_shape(self) -> Shape | None:
        infos = self.analyze()
        if not infos:
            return tuple(self.input_shape)
        return infos[-1].out_shape

    def total_params(self) -> int:
        return sum(i.params for i in self.analyze())

    def validate(self) -> list[str]:
        problems = [f"Layer {i.index + 1} ({i.spec.type}): {i.error}"
                    for i in self.analyze() if i.error]
        if not self.layers:
            problems.append("The model has no layers.")
        out = None if problems else self.output_shape()
        if out is not None and self.task == "language_model":
            problems += self._language_model_problems(out)
            return problems
        if out is not None and len(out) != 1:
            problems.append(f"Final output must be a flat vector, got shape {out}. "
                            "Add Flatten / GlobalAvgPool before the head.")
        if out is not None and self.task == "classification" and self.class_names \
                and out[0] != len(self.class_names):
            problems.append(f"Output has {out[0]} values but there are "
                            f"{len(self.class_names)} classes. Set the last Linear's "
                            f"out_features to {len(self.class_names)}.")
        return problems

    def vocab_size(self) -> int | None:
        """Number of tokens the first Embedding knows (None without one)."""
        emb = next((l for l in self.layers if l.type == "Embedding"), None)
        return int(emb.config["num_embeddings"]) if emb else None

    def _language_model_problems(self, out: Shape) -> list[str]:
        if self.input_dtype != "long" or len(self.input_shape) != 1:
            return ["A language model reads token ids: set the input type to 'long' with one dimension "
                    "(the number of tokens it reads at once)."]
        if len(out) != 2 or out[0] != self.input_shape[0]:
            return [f"A language model must output a score for every vocabulary token at every "
                    f"position, shape ({self.input_shape[0]}, vocab), but this one outputs {out}. "
                    "Remove SequencePool / Flatten and end with a Linear layer (for LSTM/GRU, turn on "
                    "return_sequence)."]
        vocab = self.vocab_size()
        if vocab and out[1] != vocab:
            return [f"The last layer outputs {out[1]} scores per position but the vocabulary has {vocab} "
                    f"tokens (the Embedding size). Set the last Linear's out_features to {vocab}."]
        return []

    # -- building ----------------------------------------------------------

    def build(self) -> nn.Sequential:
        shape: Shape = tuple(self.input_shape)
        dtype = DTYPES[self.input_dtype]
        mods = []
        for layer in self.layers:
            mod = REGISTRY[layer.type].build(shape, layer.config)
            shape = output_shape(mod, shape, dtype)
            dtype = torch.float32
            mods.append(mod)
        model = nn.Sequential(*mods)
        self.apply_freeze(model)
        return model

    def apply_freeze(self, model: nn.Sequential) -> None:
        for layer, mod in zip(self.layers, model):
            for p in mod.parameters():
                p.requires_grad = layer.id not in self.frozen

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ModelSpec":
        d = copy.deepcopy(d)
        d["layers"] = [LayerSpec(**l) for l in d.get("layers", [])]
        known = ModelSpec.__dataclass_fields__
        return ModelSpec(**{k: v for k, v in d.items() if k in known})

    def clone(self, name: str | None = None) -> "ModelSpec":
        s = ModelSpec.from_dict(self.to_dict())
        if name:
            s.name = name
        return s


def _freeze(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


@lru_cache(maxsize=512)
def _analyze_layer(type_name: str, cfg_json: str, shape: Shape, dtype: str):
    lt = REGISTRY[type_name]
    cfg = json.loads(cfg_json)
    if lt.ranks is not None and len(shape) not in lt.ranks:
        need = " or ".join(f"{r}D" for r in lt.ranks)
        raise ValueError(f"needs {need} input, got {len(shape)}D {shape}")
    if dtype == "long" and type_name != "Embedding":
        raise ValueError("token-id input must go through an Embedding layer first")
    if type_name == "Embedding" and dtype != "long":
        raise ValueError("Embedding needs token-id input (set input type to 'long')")
    if type_name == "PretrainedBackbone":
        cfg = {**cfg, "pretrained": False}  # same structure, no download while editing
    mod = lt.build(shape, cfg)
    out = output_shape(mod, shape, DTYPES[dtype])
    if any(d <= 0 for d in out):
        raise ValueError(f"output shape {out} is empty - input too small")
    return out, count_params(mod)[0], lt.formula(shape, cfg)


def _friendly(e: Exception) -> str:
    msg = str(e).split("\n")[0]
    if "Kernel size can't be greater" in msg or "Output size is too small" in msg:
        return "input is smaller than the kernel - remove a pooling layer or reduce kernel_size"
    return msg[:160]


# ---------------------------------------------------------------------------
# Workspace (model library on disk)
# ---------------------------------------------------------------------------


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.data_dir = self.root / "data"
        self.export_dir = self.root / "exports"
        for d in (self.models_dir, self.data_dir, self.export_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- UI preferences (e.g. panel sizes), kept in workspace/settings.json --

    def get_setting(self, key: str, default=None):
        try:
            return json.loads((self.root / "settings.json").read_text()).get(key, default)
        except (OSError, ValueError):
            return default

    def set_setting(self, key: str, value) -> None:
        f = self.root / "settings.json"
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            data = {}
        data[key] = value
        f.write_text(json.dumps(data, indent=2))

    def model_dir(self, name: str) -> Path:
        return self.models_dir / name

    def list_models(self) -> list[dict]:
        rows = []
        for d in sorted(self.models_dir.iterdir()):
            f = d / "spec.json"
            if not f.exists():
                continue
            try:
                spec = ModelSpec.from_dict(json.loads(f.read_text()))
            except Exception:  # noqa: BLE001
                continue
            hist = self.load_history(spec.name)
            rows.append({
                "name": spec.name,
                "task": spec.task,
                "params": spec.total_params(),
                "layers": len(spec.layers),
                "trained": (d / "weights.pt").exists(),
                "epochs": sum(len(r.get("epochs", [])) for r in hist),
                "modified": time.strftime("%b %d %H:%M", time.localtime(f.stat().st_mtime)),
            })
        return rows

    def exists(self, name: str) -> bool:
        return (self.model_dir(name) / "spec.json").exists()

    def save_spec(self, spec: ModelSpec) -> None:
        d = self.model_dir(spec.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))

    def load_spec(self, name: str) -> ModelSpec:
        return ModelSpec.from_dict(json.loads((self.model_dir(name) / "spec.json").read_text()))

    def has_weights(self, name: str) -> bool:
        return (self.model_dir(name) / "weights.pt").exists()

    def save_weights(self, name: str, model: nn.Module) -> None:
        torch.save(model.state_dict(), self.model_dir(name) / "weights.pt")

    def load_model(self, name: str, spec: ModelSpec | None = None) -> tuple[ModelSpec, nn.Sequential, str]:
        """Build the model and load whatever stored weights still fit its structure.

        Weights follow layer ids, so inserting/removing layers (e.g. a Flatten added by the
        fix dialog) keeps the weights of every unchanged layer."""
        stored = self.load_spec(name)
        spec = spec or stored
        model = spec.build()
        note = "fresh random weights"
        wf = self.model_dir(name) / "weights.pt"
        if wf.exists():
            state = torch.load(wf, map_location="cpu", weights_only=True)
            kept, total = remap_state(stored, state, spec, model)
            if kept == 0:  # e.g. ids changed by an external tool - fall back to matching by name
                n, t = transfer_state(state, model)
                note = f"loaded {n}/{t} weight tensors"
            else:
                note = f"loaded trained weights for {kept}/{total} layers"
        return spec, model, note

    def load_history(self, name: str) -> list[dict]:
        f = self.model_dir(name) / "history.json"
        return json.loads(f.read_text()) if f.exists() else []

    def append_history(self, name: str, run: dict) -> None:
        hist = self.load_history(name)
        hist.append(run)
        (self.model_dir(name) / "history.json").write_text(json.dumps(hist, indent=2))

    def delete(self, name: str) -> None:
        shutil.rmtree(self.model_dir(name), ignore_errors=True)

    def duplicate(self, src: str, dst: str) -> None:
        shutil.copytree(self.model_dir(src), self.model_dir(dst))
        spec = self.load_spec(dst)
        spec.name = dst
        self.save_spec(spec)

    def rename(self, src: str, dst: str) -> None:
        self.model_dir(src).rename(self.model_dir(dst))
        spec = self.load_spec(dst)
        spec.name = dst
        self.save_spec(spec)

    def unique_name(self, base: str) -> str:
        base = safe_name(base) or "model"
        name, i = base, 2
        while self.exists(name):
            name, i = f"{base}-{i}", i + 1
        return name


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.")


def transfer_state(state: dict, model: nn.Module) -> tuple[int, int]:
    """Copy tensors whose key and shape match; returns (copied, total in model)."""
    own = model.state_dict()
    fit = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(fit, strict=False)
    return len(fit), len(own)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _L(t, **kw):
    return LayerSpec.new(t, **kw)


TEMPLATES: dict[str, dict] = {
    "Blank": dict(input_shape=[10], layers=lambda: []),
    "MLP classifier (tabular)": dict(
        input_shape=[10],
        layers=lambda: [_L("Linear", out_features=64), _L("ReLU"), _L("Dropout", p=0.1),
                        _L("Linear", out_features=32), _L("ReLU"), _L("Linear", out_features=2)]),
    "MLP regressor (tabular)": dict(
        input_shape=[10], task="regression",
        layers=lambda: [_L("Linear", out_features=64), _L("ReLU"),
                        _L("Linear", out_features=32), _L("ReLU"), _L("Linear", out_features=1)]),
    "Small CNN (28×28 grayscale, e.g. MNIST)": dict(
        input_shape=[1, 28, 28],
        layers=lambda: [_L("Conv2d", out_channels=16), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Conv2d", out_channels=32), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Flatten"), _L("Linear", out_features=128), _L("ReLU"),
                        _L("Dropout", p=0.25), _L("Linear", out_features=10)]),
    "CNN (32×32 colour, e.g. CIFAR-10)": dict(
        input_shape=[3, 32, 32],
        layers=lambda: [_L("Conv2d", out_channels=32), _L("BatchNorm"), _L("ReLU"),
                        _L("Conv2d", out_channels=32), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Conv2d", out_channels=64), _L("BatchNorm"), _L("ReLU"),
                        _L("Conv2d", out_channels=64), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("GlobalAvgPool"), _L("Dropout", p=0.3), _L("Linear", out_features=10)]),
    "Transfer learning (ResNet18 backbone)": dict(
        input_shape=[3, 224, 224],
        layers=lambda: [_L("PretrainedBackbone"), _L("Dropout", p=0.2), _L("Linear", out_features=2)]),
    "Text Transformer classifier": dict(
        input_shape=[128], input_dtype="long",
        layers=lambda: [_L("Embedding", num_embeddings=20000, embedding_dim=128),
                        _L("PositionalEncoding"),
                        _L("TransformerEncoder", num_layers=2, num_heads=4, ff_dim=256),
                        _L("SequencePool"), _L("Dropout", p=0.1), _L("Linear", out_features=2)]),
    "Vision Transformer (ViT, 32×32 colour)": dict(
        input_shape=[3, 32, 32],
        layers=lambda: [_L("PatchEmbedding", patch_size=4, embed_dim=128), _L("PositionalEncoding"),
                        _L("TransformerEncoder", num_layers=4, num_heads=4, ff_dim=256),
                        _L("SequencePool"), _L("Linear", out_features=10)]),
    "Text classifier (Embedding + LSTM)": dict(
        input_shape=[100], input_dtype="long",
        layers=lambda: [_L("Embedding", num_embeddings=20000, embedding_dim=64),
                        _L("LSTM", hidden_size=128), _L("Dropout", p=0.2),
                        _L("Linear", out_features=2)]),
    # --- more computer vision ---
    "Deep CNN (64×64 colour photos)": dict(
        input_shape=[3, 64, 64],
        layers=lambda: [_L("Conv2d", out_channels=32), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Conv2d", out_channels=64), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Conv2d", out_channels=128), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Conv2d", out_channels=128), _L("BatchNorm"), _L("ReLU"),
                        _L("GlobalAvgPool"), _L("Dropout", p=0.3), _L("Linear", out_features=2)]),
    "Transfer learning (MobileNetV3, fast)": dict(
        input_shape=[3, 224, 224],
        layers=lambda: [_L("PretrainedBackbone", arch="mobilenet_v3_small"), _L("Dropout", p=0.2),
                        _L("Linear", out_features=2)]),
    "Transfer learning (EfficientNet-B0, accurate)": dict(
        input_shape=[3, 224, 224],
        layers=lambda: [_L("PretrainedBackbone", arch="efficientnet_b0"), _L("Dropout", p=0.3),
                        _L("Linear", out_features=2)]),
    "Vision Transformer (ViT, 28×28 grayscale)": dict(
        input_shape=[1, 28, 28],
        layers=lambda: [_L("PatchEmbedding", patch_size=4, embed_dim=64), _L("PositionalEncoding"),
                        _L("TransformerEncoder", num_layers=3, num_heads=4, ff_dim=128),
                        _L("SequencePool"), _L("Linear", out_features=10)]),
    "Hybrid CNN + Transformer (32×32 colour)": dict(
        input_shape=[3, 32, 32],
        # the convs find local features; PatchEmbedding with patch_size 1 turns each 8×8 cell into a token
        layers=lambda: [_L("Conv2d", out_channels=32), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("Conv2d", out_channels=64), _L("BatchNorm"), _L("ReLU"), _L("MaxPool2d"),
                        _L("PatchEmbedding", patch_size=1, embed_dim=128), _L("PositionalEncoding"),
                        _L("TransformerEncoder", num_layers=2, num_heads=4, ff_dim=256),
                        _L("SequencePool"), _L("Linear", out_features=10)]),
    # --- more language / NLP ---
    "Bag of words (Embedding + average, fastest)": dict(
        input_shape=[128], input_dtype="long",
        layers=lambda: [_L("Embedding", num_embeddings=20000, embedding_dim=64), _L("SequencePool"),
                        _L("Dropout", p=0.2), _L("Linear", out_features=2)]),
    "Text classifier (bidirectional GRU)": dict(
        input_shape=[100], input_dtype="long",
        layers=lambda: [_L("Embedding", num_embeddings=20000, embedding_dim=64),
                        _L("GRU", hidden_size=128, bidirectional=True), _L("Dropout", p=0.2),
                        _L("Linear", out_features=2)]),
    "BERT-style Transformer (deeper encoder)": dict(
        input_shape=[128], input_dtype="long",
        layers=lambda: [_L("Embedding", num_embeddings=20000, embedding_dim=256),
                        _L("PositionalEncoding"), _L("LayerNorm"),
                        _L("TransformerEncoder", num_layers=4, num_heads=8, ff_dim=512),
                        _L("SequencePool"), _L("Dropout", p=0.1), _L("Linear", out_features=2)]),
    "GPT-style Transformer (causal attention)": dict(
        input_shape=[128], input_dtype="long",
        # causal: each word only sees earlier words, so the last position has read the whole text
        layers=lambda: [_L("Embedding", num_embeddings=20000, embedding_dim=128),
                        _L("PositionalEncoding"),
                        _L("TransformerEncoder", num_layers=4, num_heads=4, ff_dim=512, causal=True),
                        _L("SequencePool", mode="last"), _L("Linear", out_features=2)]),
    # --- language models (generate text) ---
    # Vocabulary sizes are placeholders: training sets them from the text (auto-fit).
    "Tiny GPT (characters, fast)": dict(
        input_shape=[64], input_dtype="long", task="language_model", tokenizer="char",
        # causal: while learning, position i may only look at tokens 0..i - otherwise it would just
        # read the answer (the next token) instead of predicting it
        layers=lambda: [_L("Embedding", num_embeddings=96, embedding_dim=128),
                        _L("PositionalEncoding"), _L("Dropout", p=0.1),
                        _L("TransformerEncoder", num_layers=4, num_heads=4, ff_dim=512, causal=True),
                        _L("Linear", out_features=96)]),
    "GPT-small (words, needs more text)": dict(
        input_shape=[64], input_dtype="long", task="language_model", tokenizer="word",
        layers=lambda: [_L("Embedding", num_embeddings=8000, embedding_dim=192),
                        _L("PositionalEncoding"), _L("Dropout", p=0.1),
                        _L("TransformerEncoder", num_layers=4, num_heads=6, ff_dim=768, causal=True),
                        _L("Linear", out_features=8000)]),
    "LSTM language model (characters)": dict(
        input_shape=[64], input_dtype="long", task="language_model", tokenizer="char",
        layers=lambda: [_L("Embedding", num_embeddings=96, embedding_dim=64),
                        _L("LSTM", hidden_size=256, num_layers=2, return_sequence=True, dropout=0.1),
                        _L("Linear", out_features=96)]),
}


# Model types shown first in the "new model" dialog; each lists its templates, best first.
MANUAL = "✎ Build manually (blank)"
MODEL_TYPES: dict[str, list[str]] = {
    "📊 Tabular data (CSV)": ["MLP classifier (tabular)", "MLP regressor (tabular)"],
    "🖼 Computer vision (images)": ["Small CNN (28×28 grayscale, e.g. MNIST)",
                                   "CNN (32×32 colour, e.g. CIFAR-10)",
                                   "Deep CNN (64×64 colour photos)",
                                   "Transfer learning (ResNet18 backbone)",
                                   "Transfer learning (MobileNetV3, fast)",
                                   "Transfer learning (EfficientNet-B0, accurate)",
                                   "Vision Transformer (ViT, 32×32 colour)",
                                   "Vision Transformer (ViT, 28×28 grayscale)",
                                   "Hybrid CNN + Transformer (32×32 colour)"],
    "🧠 Language model (generate text, tiny GPT)": ["Tiny GPT (characters, fast)",
                                                    "GPT-small (words, needs more text)",
                                                    "LSTM language model (characters)"],
    "💬 Text classification (sentiment, spam, topics)": ["Text Transformer classifier",
                                                        "BERT-style Transformer (deeper encoder)",
                                                        "GPT-style Transformer (causal attention)",
                                                        "Text classifier (Embedding + LSTM)",
                                                        "Text classifier (bidirectional GRU)",
                                                        "Bag of words (Embedding + average, fastest)"],
    MANUAL: ["Blank"],
}


def from_template(template: str, name: str) -> ModelSpec:
    t = TEMPLATES[template]
    return ModelSpec(name=name, input_shape=list(t["input_shape"]),
                     input_dtype=t.get("input_dtype", "float"),
                     task=t.get("task", "classification"), layers=t["layers"](),
                     data={"tokenizer": t["tokenizer"]} if "tokenizer" in t else {})


# ---------------------------------------------------------------------------
# Code generation (export model structure as plain PyTorch)
# ---------------------------------------------------------------------------


def to_pytorch_source(spec: ModelSpec) -> str:
    model = spec.build()
    body = repr(model).replace("\n", "\n    ")
    custom_types = {"LSTM": "RNNBlock", "GRU": "RNNBlock", "PretrainedBackbone": "PretrainedBackbone",
                    "PositionalEncoding": "PositionalEncoding", "TransformerEncoder": "TransformerBlock",
                    "SequencePool": "SequencePool", "PatchEmbedding": "PatchEmbedding"}
    custom = sorted({custom_types[l.type] for l in spec.layers if l.type in custom_types})
    lines = [
        f'"""{spec.name} - generated by AIForge.',
        "",
        f"Input shape (without batch): {tuple(spec.input_shape)}  dtype: {spec.input_dtype}",
        f"Task: {spec.task}" + (f"  classes: {spec.class_names}" if spec.class_names else ""),
        '"""',
        "import torch",
        "from torch import nn",
        "",
    ]
    if custom:
        lines += [f"from aiforge.layers import {', '.join(custom)}  # custom blocks", ""]
    lines += [
        "",
        "def build_model() -> nn.Sequential:",
        "    layers = [",
    ]
    for l in spec.layers:
        lines.append(f"        {_layer_ctor(spec, l)},")
    lines += [
        "    ]",
        "    return nn.Sequential(*layers)",
        "",
        "",
        "# Printed structure:",
        "#   " + body.replace("\n", "\n#"),
        "",
        "if __name__ == '__main__':",
        "    model = build_model()",
        "    model.load_state_dict(torch.load('weights.pt', map_location='cpu'))",
        "    model.eval()",
        f"    x = torch.zeros(1, *{tuple(spec.input_shape)}" +
        (", dtype=torch.long)" if spec.input_dtype == "long" else ")"),
        "    print(model(x))",
        "",
    ]
    return "\n".join(lines)


def _layer_ctor(spec: ModelSpec, layer: LayerSpec) -> str:
    idx = spec.layers.index(layer)
    info = spec.analyze()[idx]
    s, c = info.in_shape, layer.config
    t = layer.type
    if t == "Linear":
        return f"nn.Linear({s[-1]}, {c['out_features']}, bias={c['bias']})"
    if t in ("Conv2d", "Conv1d"):
        return (f"nn.{t}({s[0]}, {c['out_channels']}, kernel_size={c['kernel_size']}, "
                f"stride={c['stride']}, padding={c['padding']}, bias={c['bias']})")
    if t in ("MaxPool2d", "AvgPool2d"):
        return f"nn.{t}(kernel_size={c['kernel_size']}, stride={c['stride']})"
    if t == "GlobalAvgPool":
        return "nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())"
    if t == "BatchNorm":
        return f"nn.BatchNorm{'2d' if len(s) == 3 else '1d'}({s[0]})"
    if t == "LayerNorm":
        return f"nn.LayerNorm({s[-1]})"
    if t == "Dropout":
        return f"nn.Dropout({c['p']})"
    if t == "LeakyReLU":
        return f"nn.LeakyReLU({c['negative_slope']})"
    if t == "Embedding":
        return f"nn.Embedding({c['num_embeddings']}, {c['embedding_dim']})"
    if t in ("LSTM", "GRU"):
        return (f"RNNBlock('{t}', {s[-1]}, {c['hidden_size']}, {c['num_layers']}, "
                f"{c['bidirectional']}, {c['return_sequence']}, {c['dropout']})")
    if t == "PatchEmbedding":
        return f"PatchEmbedding({s[0]}, {c['patch_size']}, {c['embed_dim']})"
    if t == "PositionalEncoding":
        return f"PositionalEncoding({s[0]}, {s[1]}, '{c['kind']}')"
    if t == "TransformerEncoder":
        return (f"TransformerBlock({s[-1]}, {c['num_layers']}, {c['num_heads']}, {c['ff_dim']}, "
                f"{c['dropout']}, {c['causal']})")
    if t == "SequencePool":
        return f"SequencePool('{c['mode']}')"
    if t == "PretrainedBackbone":
        return f"PretrainedBackbone('{c['arch']}', pretrained=False)"
    return f"nn.{t}()"


def remap_state(old_spec: ModelSpec, old_state: dict, new_spec: ModelSpec,
                new_model: nn.Sequential) -> tuple[int, int]:
    """Carry weights across a structure edit by following layer ids, not positions.

    Returns (layers whose weights were kept, layers that have weights)."""
    old_idx = {l.id: i for i, l in enumerate(old_spec.layers)}
    own = new_model.state_dict()
    fit, kept, with_params = {}, set(), set()
    for k, v in own.items():
        idx, rest = k.split(".", 1)
        lid = new_spec.layers[int(idx)].id
        with_params.add(lid)
        ok = f"{old_idx[lid]}.{rest}" if lid in old_idx else None
        if ok in old_state and old_state[ok].shape == v.shape:
            fit[k] = old_state[ok]
            kept.add(lid)
    new_model.load_state_dict(fit, strict=False)
    # A layer only counts as kept when all of its tensors carried over.
    partial = {k.split(".", 1)[0] for k in own if k not in fit}
    kept -= {new_spec.layers[int(i)].id for i in partial}
    return len(kept), len(with_params)
