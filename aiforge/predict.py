"""Run a trained model on raw inputs (single value or a whole file/folder)."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn

from .data import IMAGE_EXT, prepare_input, read_table
from .spec import ModelSpec, is_reinforcement, is_unsupervised


def show_token(token: str) -> str:
    """Make whitespace tokens visible in a list of guesses."""
    return {"\n": "↵ (new line)", " ": "␣ (space)", "\t": "⇥ (tab)"}.get(token, token)


@torch.no_grad()
def predict_one(spec: ModelSpec, model: nn.Module, raw: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Class probabilities, regression values, (language models) the likeliest next tokens, or
    (unsupervised models) the nearest groups / the rebuild error and code, see ``describe``."""
    model.eval()
    pre = spec.data.get("preprocess", {})
    if spec.task == "language_model":
        if pre.get("type") != "corpus":
            raise ValueError("This language model has no vocabulary saved - train it in AIForge first.")
        from .textgen import next_token_probs

        vals, idx = next_token_probs(model, pre, raw).topk(min(top_k, len(pre["vocab"]) - 1))
        return [(show_token(pre["vocab"][i]), float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    if is_reinforcement(spec.task):  # an observation in, every action's score out (best first)
        q = model(prepare_input(spec, raw))[0]
        names = spec.class_names or [f"action {i}" for i in range(len(q))]
        order = q.argsort(descending=True).tolist()[:top_k]
        return [(names[i], float(q[i])) for i in order]
    if is_unsupervised(spec.task):
        pairs = describe(spec, model, raw)["pairs"]
        return pairs[:top_k] if spec.task == "clustering" else pairs
    out = model(prepare_input(spec, raw))[0]
    if spec.task == "classification":
        probs = out.softmax(0)
        names = spec.class_names or [f"class {i}" for i in range(len(probs))]
        k = min(top_k, len(probs))
        vals, idx = probs.topk(k)
        return [(names[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    out = out * pre.get("target_std", 1.0) + pre.get("target_mean", 0.0)
    return [(f"output {i}" if len(out) > 1 else "prediction", float(v)) for i, v in enumerate(out.tolist())]


@torch.no_grad()
def describe(spec: ModelSpec, model: nn.Module, raw: str) -> dict:
    """What an unsupervised model makes of one input: its unusualness score (and how it compares
    with the training data), its group (k-means) and its compressed code (PCA, autoencoder).
    ``pairs`` lists the numbers as (label, value): the nearest groups first, or the error first."""
    from . import unsupervised as un

    model.eval()
    x = prepare_input(spec, raw)
    score = float(un.scores(spec, model, x)[0])
    res = {"score": score, "score_name": un.SCORE_NAMES[spec.task], "percentile": un.percentile(spec, score),
           "verdict": un.verdict(spec, score)}
    if spec.task == "clustering":
        dist = model(x)[0]
        order = dist.argsort().tolist()
        res["group"] = order[0]
        res["pairs"] = [(un.group_name(i), float(dist[i])) for i in order]
    else:
        code = un.codes(spec, model, x)[0].tolist()
        res["code"] = code
        name = "component" if spec.task == "pca" else "code"
        res["pairs"] = [("rebuild error", score)] + [(f"{name} {i + 1}", v) for i, v in enumerate(code[:8])]
    return res


def input_hint(spec: ModelSpec) -> str:
    pre = spec.data.get("preprocess", {})
    kind = pre.get("type")
    if kind == "image" or (kind is None and len(spec.input_shape) == 3):
        return "Path to an image file (png/jpg)"
    if kind == "text":
        return "A sentence / paragraph of text"
    if kind == "env":
        return (f"{spec.input_shape[0]} comma-separated numbers: what the agent observes. Easier: "
                "watch it play on the Use screen (or 'aiforge play').")
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
    if is_reinforcement(spec.task):
        raise ValueError("An agent acts in its environment instead: watch it play, or evaluate it.")
    p = Path(src.strip().strip("'\"")).expanduser()
    pre = spec.data.get("preprocess", {})
    rows_out: list[list[str]] = []
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        inputs = [(str(f.relative_to(p)), str(f)) for f in files]
        header = ["file"]
    else:
        cols, rows = read_table(p)
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
    if is_unsupervised(spec.task):
        return _describe_batch(spec, model, inputs, header, p, dst)
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


def _output_path(p: Path, dst: str | None) -> Path:
    out = Path(dst).expanduser() if dst else (p if p.is_dir() else p.parent) / f"{p.stem}_predictions.csv"
    return out / "predictions.csv" if out.is_dir() else out


def _describe_batch(spec: ModelSpec, model: nn.Module, inputs, header, p: Path, dst) -> tuple[Path, int]:
    """Unsupervised models: group and/or unusualness of every row or image, in the input's order.
    The percentile says how unusual it is (0.99 = more unusual than 99% of the training data)."""
    from . import unsupervised as un

    cluster = spec.task == "clustering"
    rows_out = []
    for ident, raw in inputs:
        d = describe(spec, model, raw)
        base = ident if isinstance(ident, list) else [ident]
        pct = "" if d["percentile"] is None else f"{d['percentile']:.3f}"
        rows_out.append(base + ([un.group_name(d["group"]), f"{d['score']:.6g}", pct] if cluster
                                else [f"{d['score']:.6g}", pct]))
    out = _output_path(p, dst)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + (["group", "distance", "unusual_percentile"] if cluster
                             else ["rebuild_error", "unusual_percentile"]))
        w.writerows(rows_out)
    return out, len(rows_out)
