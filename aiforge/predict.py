"""Run a trained model on raw inputs (single value or a whole file/folder)."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from .data import IMAGE_EXT, _read_csv, prepare_input
from .spec import ModelSpec


def show_token(token: str) -> str:
    """Make whitespace tokens visible in a list of guesses."""
    return {"\n": "↵ (new line)", " ": "␣ (space)", "\t": "⇥ (tab)"}.get(token, token)


@torch.no_grad()
def predict_one(spec: ModelSpec, model: nn.Module, raw: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Class probabilities, regression values, or (language models) the likeliest next tokens."""
    model.eval()
    pre = spec.data.get("preprocess", {})
    if spec.task == "language_model":
        if pre.get("type") != "corpus":
            raise ValueError("This language model has no vocabulary saved - train it in AIForge first.")
        from .textgen import next_token_probs

        vals, idx = next_token_probs(model, pre, raw).topk(min(top_k, len(pre["vocab"]) - 1))
        return [(show_token(pre["vocab"][i]), float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    out = model(prepare_input(spec, raw))[0]
    if spec.task == "classification":
        probs = out.softmax(0)
        names = spec.class_names or [f"class {i}" for i in range(len(probs))]
        k = min(top_k, len(probs))
        vals, idx = probs.topk(k)
        return [(names[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    out = out * pre.get("target_std", 1.0) + pre.get("target_mean", 0.0)
    return [(f"output {i}" if len(out) > 1 else "prediction", float(v)) for i, v in enumerate(out.tolist())]


def input_hint(spec: ModelSpec) -> str:
    pre = spec.data.get("preprocess", {})
    kind = pre.get("type")
    if kind == "image" or (kind is None and len(spec.input_shape) == 3):
        return "Path to an image file (png/jpg)"
    if kind == "text":
        return "A sentence / paragraph of text"
    if kind == "corpus" or spec.task == "language_model":
        return "The start of a text (a prompt) - the model writes what comes next. Leave empty to let it begin."
    feats = pre.get("features")
    if feats:
        cats = pre.get("categorical", {})
        return "Comma-separated values: " + ", ".join(
            f"{f} ({'/'.join(cats[f][:4])}{'…' if len(cats[f]) > 4 else ''})" if f in cats else f
            for f in feats)
    return f"{int(torch.tensor(spec.input_shape).prod())} comma-separated numbers"


def predict_batch(spec: ModelSpec, model: nn.Module, src: str, dst: str | None = None) -> tuple[Path, int]:
    """Predict every row of a CSV (tabular/text) or every image in a folder; write a CSV."""
    if spec.task == "language_model":
        raise ValueError("Batch prediction is for classifiers and regressors - a language model writes text.")
    p = Path(src.strip().strip("'\"")).expanduser()
    pre = spec.data.get("preprocess", {})
    rows_out: list[list[str]] = []
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        inputs = [(str(f.relative_to(p)), str(f)) for f in files]
        header = ["file"]
    else:
        cols, rows = _read_csv(p)
        header = cols
        if pre.get("type") == "text":
            tcol = next((c for c in cols if c.lower() in ("text", "sentence", "review", "content")), cols[0])
            ti = cols.index(tcol)
            inputs = [(r, r[ti]) for r in rows]
        else:
            feats = pre.get("features") or []
            if feats and all(f in cols for f in feats):
                ix = [cols.index(f) for f in feats]
            else:
                ix = list(range(len(cols)))
            inputs = [(r, ",".join(r[i] for i in ix)) for r in rows]
    if not inputs:
        raise ValueError("Nothing to predict (no rows / images found)")
    is_cls = spec.task == "classification"
    for ident, raw in inputs:
        res = predict_one(spec, model, raw, top_k=1)
        base = ident if isinstance(ident, list) else [ident]
        rows_out.append(base + ([res[0][0], f"{res[0][1]:.4f}"] if is_cls else [f"{v:.6g}" for _, v in res]))
    out = Path(dst).expanduser() if dst else (p if p.is_dir() else p.parent) / f"{p.stem}_predictions.csv"
    if out.is_dir():
        out = out / "predictions.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + (["prediction", "confidence"] if is_cls else ["prediction"]))
        w.writerows(rows_out)
    return out, len(rows_out)
