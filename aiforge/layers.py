"""Layer registry.

Every layer the builder offers is described by a ``LayerType``: its editable
parameters, how to build the torch module from the incoming tensor shape, and a
human readable formula for its parameter count. Shapes exclude the batch dim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import torch
from torch import nn

Shape = tuple[int, ...]


@dataclass
class Param:
    name: str
    kind: str  # "int" | "float" | "bool" | "choice"
    default: Any
    help: str = ""
    choices: list[str] = field(default_factory=list)
    minimum: float | None = None


@dataclass
class LayerType:
    name: str
    category: str
    description: str
    params: list[Param]
    build: Callable[[Shape, dict], nn.Module]
    formula: Callable[[Shape, dict], str] = lambda s, c: ""
    # Which input ranks the layer accepts, None = any.
    ranks: tuple[int, ...] | None = None

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}


# ---------------------------------------------------------------------------
# Custom modules
# ---------------------------------------------------------------------------


class RNNBlock(nn.Module):
    """LSTM/GRU over (seq, features); returns last step or the full sequence."""

    def __init__(self, kind: str, input_size: int, hidden_size: int, num_layers: int,
                 bidirectional: bool, return_sequence: bool, dropout: float):
        super().__init__()
        cls = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = cls(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                       bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.return_sequence = return_sequence

    def forward(self, x):
        out, _ = self.rnn(x)
        return out if self.return_sequence else out[:, -1, :]


class PretrainedBackbone(nn.Module):
    """A torchvision network with its classifier head removed -> feature vector."""

    FEATURES = {"resnet18": 512, "resnet34": 512, "resnet50": 2048,
                "mobilenet_v3_small": 576, "efficientnet_b0": 1280}

    def __init__(self, arch: str, pretrained: bool):
        super().__init__()
        import torchvision.models as tvm

        weights = "DEFAULT" if pretrained else None
        net = getattr(tvm, arch)(weights=weights)
        if arch.startswith("resnet"):
            net.fc = nn.Identity()
        else:
            net.classifier = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Add position information to a (seq_len, dim) sequence."""

    def __init__(self, seq_len: int, dim: int, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            pos = torch.arange(seq_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
            pe = torch.zeros(1, seq_len, dim)
            pe[0, :, 0::2] = torch.sin(pos * div)
            pe[0, :, 1::2] = torch.cos(pos * div)[:, : dim // 2]
            self.register_buffer("pos", pe)

    def forward(self, x):
        return x + self.pos[:, : x.shape[1]]


class TransformerBlock(nn.Module):
    """A stack of Transformer encoder layers (self-attention + feed-forward)."""

    def __init__(self, dim: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float,
                 causal: bool):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, num_heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.causal = causal

    def forward(self, x):
        mask = None
        if self.causal:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf"), device=x.device), diagonal=1)
        return self.norm(self.encoder(x, mask=mask))


class SequencePool(nn.Module):
    """Collapse a (seq_len, dim) sequence into one (dim,) vector."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "mean":
            return x.mean(1)
        if self.mode == "max":
            return x.max(1).values
        return x[:, 0] if self.mode == "first" else x[:, -1]


class PatchEmbedding(nn.Module):
    """Cut an image into square patches and project each to a vector (Vision Transformer input)."""

    def __init__(self, channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_out(size: int, k: int, s: int, p: int) -> int:
    return (size + 2 * p - k) // s + 1


def _fmt(n: int) -> str:
    return f"{n:,}"


REGISTRY: dict[str, LayerType] = {}


def register(lt: LayerType) -> None:
    REGISTRY[lt.name] = lt


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------

register(LayerType(
    "Linear", "Core", "Fully connected layer: y = xW + b. Acts on the last dimension.",
    [Param("out_features", "int", 64, "Number of output neurons", minimum=1),
     Param("bias", "bool", True, "Learn an additive bias")],
    build=lambda s, c: nn.Linear(s[-1], c["out_features"], bias=c["bias"]),
    formula=lambda s, c: f"{s[-1]}×{c['out_features']}" + (f" + {c['out_features']}" if c["bias"] else ""),
))

register(LayerType(
    "Flatten", "Core", "Flatten everything except the batch into one vector.",
    [], build=lambda s, c: nn.Flatten(),
))

register(LayerType(
    "Dropout", "Regularization", "Randomly zero activations during training.",
    [Param("p", "float", 0.2, "Drop probability (0-1)", minimum=0)],
    build=lambda s, c: nn.Dropout(c["p"]),
))

# ---------------------------------------------------------------------------
# Convolution & pooling
# ---------------------------------------------------------------------------

register(LayerType(
    "Conv2d", "Convolution", "2D convolution over (channels, height, width) images.",
    [Param("out_channels", "int", 32, "Number of filters", minimum=1),
     Param("kernel_size", "int", 3, "Filter size", minimum=1),
     Param("stride", "int", 1, "Step between filter positions", minimum=1),
     Param("padding", "int", 1, "Zero padding on each side", minimum=0),
     Param("bias", "bool", True, "Learn a bias per filter")],
    build=lambda s, c: nn.Conv2d(s[0], c["out_channels"], c["kernel_size"], c["stride"],
                                 c["padding"], bias=c["bias"]),
    formula=lambda s, c: f"{s[0]}×{c['out_channels']}×{c['kernel_size']}²"
                         + (f" + {c['out_channels']}" if c["bias"] else ""),
    ranks=(3,),
))

register(LayerType(
    "Conv1d", "Convolution", "1D convolution over (channels, length) sequences.",
    [Param("out_channels", "int", 32, "Number of filters", minimum=1),
     Param("kernel_size", "int", 3, "Filter size", minimum=1),
     Param("stride", "int", 1, "Step between filter positions", minimum=1),
     Param("padding", "int", 1, "Zero padding on each side", minimum=0),
     Param("bias", "bool", True, "Learn a bias per filter")],
    build=lambda s, c: nn.Conv1d(s[0], c["out_channels"], c["kernel_size"], c["stride"],
                                 c["padding"], bias=c["bias"]),
    formula=lambda s, c: f"{s[0]}×{c['out_channels']}×{c['kernel_size']}"
                         + (f" + {c['out_channels']}" if c["bias"] else ""),
    ranks=(2,),
))

register(LayerType(
    "MaxPool2d", "Convolution", "Keep the max value in each window (downsamples).",
    [Param("kernel_size", "int", 2, "Window size", minimum=1),
     Param("stride", "int", 2, "Step between windows", minimum=1)],
    build=lambda s, c: nn.MaxPool2d(c["kernel_size"], c["stride"]),
    ranks=(3,),
))

register(LayerType(
    "AvgPool2d", "Convolution", "Average each window (downsamples).",
    [Param("kernel_size", "int", 2, "Window size", minimum=1),
     Param("stride", "int", 2, "Step between windows", minimum=1)],
    build=lambda s, c: nn.AvgPool2d(c["kernel_size"], c["stride"]),
    ranks=(3,),
))

register(LayerType(
    "GlobalAvgPool", "Convolution", "Average each channel to one value -> (channels,).",
    [], build=lambda s, c: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()),
    ranks=(3,),
))

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _batchnorm(s: Shape, c: dict) -> nn.Module:
    return nn.BatchNorm1d(s[0]) if len(s) in (1, 2) else nn.BatchNorm2d(s[0])


register(LayerType(
    "BatchNorm", "Normalization", "Normalize each channel/feature over the batch.",
    [], build=_batchnorm, formula=lambda s, c: f"2×{s[0]}", ranks=(1, 2, 3),
))

register(LayerType(
    "LayerNorm", "Normalization", "Normalize over the last dimension per sample.",
    [], build=lambda s, c: nn.LayerNorm(s[-1]), formula=lambda s, c: f"2×{s[-1]}",
))

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

for _name, _cls, _desc in [
    ("ReLU", nn.ReLU, "max(0, x) - the default choice."),
    ("GELU", nn.GELU, "Smooth ReLU used in transformers."),
    ("SiLU", nn.SiLU, "x·sigmoid(x), a.k.a. Swish."),
    ("Tanh", nn.Tanh, "Squash to (-1, 1)."),
    ("Sigmoid", nn.Sigmoid, "Squash to (0, 1)."),
]:
    register(LayerType(_name, "Activation", _desc, [],
                       build=(lambda cls: lambda s, c: cls())(_cls)))

register(LayerType(
    "LeakyReLU", "Activation", "ReLU with a small slope for negative inputs.",
    [Param("negative_slope", "float", 0.01, "Slope for x < 0", minimum=0)],
    build=lambda s, c: nn.LeakyReLU(c["negative_slope"]),
))

# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

register(LayerType(
    "Embedding", "Sequence", "Map integer token ids to learned vectors. Needs token-id input.",
    [Param("num_embeddings", "int", 10000, "Vocabulary size", minimum=1),
     Param("embedding_dim", "int", 64, "Vector size per token", minimum=1)],
    build=lambda s, c: nn.Embedding(c["num_embeddings"], c["embedding_dim"]),
    formula=lambda s, c: f"{c['num_embeddings']}×{c['embedding_dim']}",
))


def _rnn_formula(kind: str):
    gates = 4 if kind == "LSTM" else 3

    def f(s: Shape, c: dict) -> str:
        h = c["hidden_size"]
        d = "×2 dirs" if c["bidirectional"] else ""
        return f"{gates}×({s[-1]}×{h} + {h}×{h} + 2×{h}) first layer{d}"
    return f


for _kind in ("LSTM", "GRU"):
    register(LayerType(
        _kind, "Sequence", f"{_kind} recurrent layer over (seq_len, features).",
        [Param("hidden_size", "int", 128, "Hidden units", minimum=1),
         Param("num_layers", "int", 1, "Stacked layers", minimum=1),
         Param("bidirectional", "bool", False, "Read the sequence both ways"),
         Param("return_sequence", "bool", False, "Output every step (else only the last)"),
         Param("dropout", "float", 0.0, "Dropout between stacked layers", minimum=0)],
        build=(lambda k: lambda s, c: RNNBlock(k, s[-1], c["hidden_size"], c["num_layers"],
                                               c["bidirectional"], c["return_sequence"],
                                               c["dropout"]))(_kind),
        formula=_rnn_formula(_kind),
        ranks=(2,),
    ))

# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def _patch_build(s: Shape, c: dict) -> nn.Module:
    p = c["patch_size"]
    if s[1] % p or s[2] % p:
        raise ValueError(f"image {s[1]}×{s[2]} isn't divisible by patch_size {p}")
    return PatchEmbedding(s[0], p, c["embed_dim"])


register(LayerType(
    "PatchEmbedding", "Transformer",
    "Split an image into patches and turn each into a vector: (C,H,W) → (patches, dim). "
    "The first layer of a Vision Transformer.",
    [Param("patch_size", "int", 4, "Patch side in pixels (must divide H and W)", minimum=1),
     Param("embed_dim", "int", 64, "Vector size per patch", minimum=1)],
    build=_patch_build,
    formula=lambda s, c: f"{s[0]}×{c['embed_dim']}×{c['patch_size']}² + {c['embed_dim']}",
    ranks=(3,),
))

register(LayerType(
    "PositionalEncoding", "Transformer",
    "Tell the Transformer where each token/patch is. Attention alone ignores order.",
    [Param("kind", "choice", "learned", "learned = trainable, sinusoidal = fixed",
           choices=["learned", "sinusoidal"])],
    build=lambda s, c: PositionalEncoding(s[0], s[1], c["kind"]),
    formula=lambda s, c: f"{s[0]}×{s[1]}" if c["kind"] == "learned" else "fixed (no params)",
    ranks=(2,),
))


def _transformer_build(s: Shape, c: dict) -> nn.Module:
    d, h = s[-1], c["num_heads"]
    if d % h:
        ok = [n for n in range(1, min(d, 32) + 1) if d % n == 0]
        raise ValueError(f"vector size {d} must be divisible by num_heads {h} (try {', '.join(map(str, ok[-4:]))})")
    return TransformerBlock(d, c["num_layers"], h, c["ff_dim"], c["dropout"], c["causal"])


def _transformer_formula(s: Shape, c: dict) -> str:
    d, f = s[-1], c["ff_dim"]
    return f"{c['num_layers']}×(attn 4×{d}²+4×{d} + ff 2×{d}×{f}+{f}+{d} + norms 4×{d}) + 2×{d}"


register(LayerType(
    "TransformerEncoder", "Transformer",
    "Self-attention layers: every position looks at every other one. Input (seq_len, dim).",
    [Param("num_layers", "int", 2, "Stacked encoder layers (depth)", minimum=1),
     Param("num_heads", "int", 4, "Attention heads (must divide the vector size)", minimum=1),
     Param("ff_dim", "int", 256, "Hidden size of each feed-forward block", minimum=1),
     Param("dropout", "float", 0.1, "Dropout inside attention / feed-forward", minimum=0),
     Param("causal", "bool", False, "Only look at earlier positions (GPT-style)")],
    build=_transformer_build, formula=_transformer_formula, ranks=(2,),
))

register(LayerType(
    "SequencePool", "Transformer",
    "Turn a sequence into one vector for the classifier head: (seq_len, dim) → (dim,).",
    [Param("mode", "choice", "mean", "mean / max over positions, or take the first / last",
           choices=["mean", "max", "first", "last"])],
    build=lambda s, c: SequencePool(c["mode"]), ranks=(2,),
))

# ---------------------------------------------------------------------------
# Transfer learning
# ---------------------------------------------------------------------------

register(LayerType(
    "PretrainedBackbone", "Transfer",
    "A torchvision network (ImageNet weights) without its head. Input: (3, H, W) images.",
    [Param("arch", "choice", "resnet18", "Backbone architecture",
           choices=list(PretrainedBackbone.FEATURES)),
     Param("pretrained", "bool", True, "Download ImageNet weights")],
    build=lambda s, c: PretrainedBackbone(c["arch"], c["pretrained"]),
    ranks=(3,),
))


CATEGORIES = ["Core", "Convolution", "Normalization", "Activation", "Regularization",
              "Sequence", "Transformer", "Transfer"]


def by_category() -> dict[str, list[LayerType]]:
    out: dict[str, list[LayerType]] = {c: [] for c in CATEGORIES}
    for lt in REGISTRY.values():
        out[lt.category].append(lt)
    return out


def coerce(param: Param, raw: Any) -> Any:
    """Convert user input to the param's type, raising ValueError with a friendly message."""
    if param.kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")
    if param.kind == "choice":
        if raw not in param.choices:
            raise ValueError(f"{param.name} must be one of {', '.join(param.choices)}")
        return raw
    try:
        val = int(raw) if param.kind == "int" else float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{param.name} must be a{'n integer' if param.kind == 'int' else ' number'}")
    if param.minimum is not None and val < param.minimum:
        raise ValueError(f"{param.name} must be ≥ {param.minimum:g}")
    return val


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def output_shape(module: nn.Module, in_shape: Shape, dtype: torch.dtype) -> Shape:
    module.eval()
    with torch.no_grad():
        x = torch.zeros((1, *in_shape), dtype=dtype)
        y = module(x)
    return tuple(y.shape[1:])
