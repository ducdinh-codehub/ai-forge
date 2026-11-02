"""Why did the model give this answer? Simple, honest explanations for one prediction.

- Tables: each feature is set to its average (0 after standardisation) and we measure how much
  the answer moves. Big moves = the feature mattered for this example.
- Text: each word is left out in turn, with the same measurement.
- Images: a saliency map - how strongly each pixel would change the answer (the gradient),
  averaged over a few slightly noisy copies of the image (SmoothGrad) to make it less speckled.

These show what the model *reacts to*, not a proof of why it is right.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from rich.text import Text
from torch import nn

from .data import prepare_input, tokenize
from .spec import ModelSpec

GRID_WIDTH = 32  # image explanations: cells across (each cell is one character, two pixels tall)


@dataclass
class Explanation:
    kind: str  # "features" | "tokens" | "image"
    answer: str  # what is being explained, e.g. "arm-0 (98.7%)"
    unit: str  # what the effects are measured in
    items: list[tuple[str, str, float]] = field(default_factory=list)  # (name, value, effect)
    heat: list[list[float]] = field(default_factory=list)  # image: saliency per cell, 0..1
    pixels: list[list[list[float]]] = field(default_factory=list)  # image: RGB per cell, 0..1


def explain(spec: ModelSpec, model: nn.Module, raw: str) -> Explanation:
    if spec.task == "language_model":
        raise ValueError("Explanations are for classifiers and regressors - a language model writes text.")
    model.eval()
    pre = spec.data.get("preprocess", {})
    kind = pre.get("type") or ("image" if len(spec.input_shape) == 3 else "tabular")
    x = prepare_input(spec, raw)
    with torch.no_grad():
        out = model(x)[0]
    if spec.task == "classification":
        cls = int(out.argmax())
        names = spec.class_names or [f"class {i}" for i in range(len(out))]
        answer, unit = f"{names[cls]} ({out.softmax(0)[cls]:.1%})", "points of confidence"
    else:
        cls = 0
        answer = f"{_unscale(out[0].item(), pre):.4g}"
        unit = "target units"

    def score(batch: torch.Tensor) -> torch.Tensor:
        """The explained number for each input: the answer's probability (%), or the prediction."""
        with torch.no_grad():
            o = model(batch)
        if spec.task == "classification":
            return o.softmax(1)[:, cls] * 100
        return o[:, 0] * pre.get("target_std", 1.0)

    if kind == "image":
        return _image(model, x, cls, spec.task, pre, answer, unit)
    if kind == "text":
        return _text(spec, raw, pre, score, answer, unit)
    if len(spec.input_shape) != 1:
        raise ValueError("Explanations need a table, text or an image as input.")
    return _features(raw, x, pre, score, answer, unit)


def _unscale(v: float, pre: dict) -> float:
    return v * pre.get("target_std", 1.0) + pre.get("target_mean", 0.0)


def _features(raw, x, pre, score, answer, unit) -> Explanation:
    n = x.shape[1]
    names = pre.get("features") or [f"x{i + 1}" for i in range(n)]
    values = [v.strip() for v in raw.replace(";", ",").split(",") if v.strip()]
    if len(values) != n:
        values = [f"{v:.3g}" for v in x[0].tolist()]
    variants = x.repeat(n, 1)
    variants[torch.arange(n), torch.arange(n)] = 0.0  # one feature at its average per row
    effects = (score(x) - score(variants)).tolist()
    items = sorted(zip(names, values, effects), key=lambda t: -abs(t[2]))
    return Explanation("features", answer, unit, items)


def _text(spec, raw, pre, score, answer, unit) -> Explanation:
    vocab, seq_len = pre["vocab"], int(pre["seq_len"])
    tokens = tokenize(raw)[:seq_len]
    if not tokens:
        raise ValueError("Type some text to explain.")

    def encode(ids):
        return [0] * (seq_len - len(ids)) + ids

    ids = [vocab.get(t, 1) for t in tokens]
    base = score(torch.tensor([encode(ids)]))
    without = torch.tensor([encode(ids[:i] + ids[i + 1:]) for i in range(len(ids))])
    effects = (base - score(without)).tolist()
    known = ["" if vocab.get(t) else "unknown word" for t in tokens]
    return Explanation("tokens", answer, unit, list(zip(tokens, known, effects)))


def _image(model, x, cls, task, pre, answer, unit, samples: int = 12) -> Explanation:
    grads = torch.zeros_like(x)
    noise = 0.1 * (x.max() - x.min()).item()
    g = torch.Generator().manual_seed(0)
    for _ in range(samples):  # SmoothGrad: average the gradient over slightly noisy copies
        xi = (x + torch.randn(x.shape, generator=g) * noise).requires_grad_(True)
        model(xi)[0, cls].backward()
        grads += xi.grad
    sal = grads.abs().amax(1, keepdim=True) / samples  # (1, 1, H, W): strongest channel per pixel
    h, w = x.shape[2:]
    cols = min(GRID_WIDTH, w)
    rows = max(2, round(h * cols / w / 2) * 2)  # an even number: two pixels per character
    heat = nn.functional.adaptive_avg_pool2d(sal, (rows, cols))[0, 0]
    heat = (heat / heat.flatten().quantile(0.99).clamp_min(1e-12)).clamp(0, 1)
    mean = torch.tensor(pre.get("mean", [0.5] * x.shape[1])).view(1, -1, 1, 1)
    std = torch.tensor(pre.get("std", [0.5] * x.shape[1])).view(1, -1, 1, 1)
    img = nn.functional.adaptive_avg_pool2d((x * std + mean).clamp(0, 1), (rows, cols))[0]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return Explanation("image", answer, "gradient strength", heat=heat.tolist(),
                       pixels=img.permute(1, 2, 0).tolist())


# ---------------------------------------------------------------------------
# Rendering (Rich text: the command line and the Predict screen)
# ---------------------------------------------------------------------------

BAR = 24


def render(exp: Explanation) -> Text:
    out = Text()
    out.append("Why ", style="bold").append(exp.answer, style="bold cyan").append("?\n", style="bold")
    if exp.kind == "features":
        _render_features(exp, out)
    elif exp.kind == "tokens":
        _render_tokens(exp, out)
    else:
        _render_image(exp, out)
    return out


def _fmt(v: float, unit: str) -> str:
    return f"{v:+.1f} pts" if unit == "points of confidence" else f"{v:+.3g}"


def _render_features(exp: Explanation, out: Text) -> None:
    out.append(f"How much the answer moves when a feature is set to its average ({exp.unit}).\n"
               "Green pushed towards this answer, red pushed away from it.\n\n", style="dim")
    biggest = max((abs(e) for *_, e in exp.items), default=0) or 1
    name_w = min(22, max(len(n) for n, *_ in exp.items))
    val_w = min(14, max(len(v) for _, v, _ in exp.items))
    for name, value, effect in exp.items:
        n = round(abs(effect) / biggest * BAR)
        left = ("█" * n).rjust(BAR) if effect < 0 else " " * BAR
        right = ("█" * n).ljust(BAR) if effect > 0 else " " * BAR
        out.append(f"  {name[:name_w]:<{name_w}} {value[:val_w]:>{val_w}}  ")
        out.append(left, style="red").append("│", style="dim").append(right, style="green")
        out.append(f" {_fmt(effect, exp.unit)}\n", style="green" if effect > 0 else "red" if effect < 0 else "dim")


def _render_tokens(exp: Explanation, out: Text) -> None:
    out.append(f"How much the answer moves when a word is left out ({exp.unit}).\n"
               "Green words pushed towards this answer, red ones away from it.\n\n", style="dim")
    biggest = max((abs(e) for *_, e in exp.items), default=0) or 1
    out.append("  ")
    for token, note, effect in exp.items:
        a = min(1.0, abs(effect) / biggest)
        bg = _mix((30, 30, 30), (34, 139, 34) if effect > 0 else (190, 40, 40), a)
        out.append(f" {token} ", style=f"#ffffff on {bg}" + (" italic" if note else ""))
        out.append(" ")
    out.append("\n\n")
    for token, note, effect in sorted(exp.items, key=lambda t: -abs(t[2]))[:5]:
        out.append(f"  {token[:18]:<18} ")
        out.append(_fmt(effect, exp.unit), style="green" if effect > 0 else "red")
        out.append(f"  {note}\n" if note else "\n", style="dim")


def _render_image(exp: Explanation, out: Text) -> None:
    out.append("Left: the image. Right: where the model looks - brighter = a change there would move "
               "the answer more.\n\n", style="dim")
    px, heat = exp.pixels, exp.heat
    for r in range(0, len(px) - 1, 2):
        out.append("  ")
        for c in range(len(px[0])):
            out.append("▀", style=f"{_rgb(px[r][c])} on {_rgb(px[r + 1][c])}")
        out.append("   ")
        for c in range(len(px[0])):
            top, bottom = _heat(px[r][c], heat[r][c]), _heat(px[r + 1][c], heat[r + 1][c])
            out.append("▀", style=f"{top} on {bottom}")
        out.append("\n")


def _mix(a: tuple, b: tuple, t: float) -> str:
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _rgb(p: list[float]) -> str:
    return "#%02x%02x%02x" % tuple(round(max(0.0, min(1.0, v)) * 255) for v in p)


def _heat(p: list[float], h: float) -> str:
    """A dimmed grey copy of the image with the saliency drawn over it in yellow-red."""
    grey = (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) * 0.35
    hot = (1.0, 0.85 * (1 - h) + 0.15, 0.1) if h > 0.05 else (grey, grey, grey)
    return _rgb([grey + (c - grey) * h for c in hot])
