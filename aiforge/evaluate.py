"""Evaluate a trained model: metrics, baseline, calibration, error analysis, slices, robustness and
model-vs-model significance. Independent of the UI; ``report_markdown`` renders the result."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from . import data as datamod
from .spec import ModelSpec

BOOTSTRAP = 200  # resamples for confidence intervals
NOISE_LEVELS = (0.1, 0.3)  # robustness test: Gaussian noise, in units of a normalised input
TOP_ERRORS = 10


# ---------------------------------------------------------------------------
# Evaluation data
# ---------------------------------------------------------------------------


class _Relabel(Dataset):
    """Map the loader's class indices onto the model's class order (they can differ when an
    evaluation file is missing a class or lists classes in another order)."""

    def __init__(self, ds, mapping: list[int]):
        self.ds, self.mapping = ds, mapping

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, self.mapping[int(y)]


def _unwrap(ds, i: int):
    """(base dataset, index in it) through Subset / _Relabel wrappers."""
    while True:
        if isinstance(ds, Subset):
            ds, i = ds.dataset, ds.indices[i]
        elif isinstance(ds, _Relabel):
            ds = ds.ds
        else:
            return ds, i


def _labels(ds) -> np.ndarray | None:
    """All targets without running the (possibly slow, e.g. image-decoding) transforms."""
    mapping = None
    if isinstance(ds, _Relabel):
        ds, mapping = ds.ds, np.array(ds.mapping)
    idx = np.arange(len(ds))
    while isinstance(ds, Subset):
        idx = np.asarray(ds.indices)[idx]
        ds = ds.dataset
    if isinstance(ds, TensorDataset):
        y = ds.tensors[1].numpy()[idx]
    elif hasattr(ds, "items"):  # data._ImageFiles
        y = np.array([ds.items[i][1] for i in idx])
    elif hasattr(ds, "targets"):  # torchvision datasets
        y = np.asarray(ds.targets)[idx]
    else:
        return None
    return mapping[y] if mapping is not None and y.dtype.kind in "iu" else y


def _class_mapping(spec: ModelSpec, names: list[str]) -> list[int] | None:
    if not spec.class_names or names == spec.class_names:
        return None
    unknown = [n for n in names if n not in spec.class_names]
    if unknown:
        raise ValueError(f"The data has classes the model doesn't know: {', '.join(unknown[:8])}. "
                         f"Model classes: {', '.join(spec.class_names[:12])}")
    return [spec.class_names.index(n) for n in names]


def _tabular_file(spec: ModelSpec, path: Path, target: str | None):
    """A new CSV, normalised with the *model's* statistics and category codes (recomputing them
    from the new file would silently shift every input)."""
    pre = spec.data["preprocess"]
    header, rows = datamod._read_csv(path)
    feats, cats = pre["features"], pre.get("categorical", {})
    missing = [f for f in feats if f not in header]
    if missing:
        raise ValueError(f"The CSV is missing the model's input columns: {', '.join(missing)}")
    target = target or next((h for h in reversed(header) if h not in feats), None)
    if target not in header:
        raise ValueError("Choose the target column (the true answers)")
    fi, ti = [header.index(f) for f in feats], header.index(target)
    mean, std = np.array(pre["mean"]), np.array(pre["std"])
    xs, ys, skipped = [], [], 0
    for r in rows:
        if len(r) != len(header):
            skipped += 1
            continue
        try:
            vals = []
            for j, (f, c) in enumerate(zip(feats, fi)):
                v = r[c].strip()
                if f in cats:
                    vals.append(float(cats[f].index(v)))  # ValueError for unseen categories
                else:
                    vals.append(float(v) if v != "" else float(mean[j]))
            label = r[ti].strip()
            y = spec.class_names.index(label) if spec.task == "classification" else float(label)
        except ValueError:
            skipped += 1
            continue
        xs.append(vals)
        ys.append(y)
    if not xs:
        raise ValueError("No usable rows (unknown categories or labels in every row?)")
    x = torch.tensor((np.array(xs) - mean) / std, dtype=torch.float)
    y = torch.tensor(ys) if spec.task == "classification" else torch.tensor(ys, dtype=torch.float).unsqueeze(1)
    note = f" ({skipped} rows skipped: unknown category/label or bad value)" if skipped else ""
    return TensorDataset(x, y), f"{path.name}: {len(xs):,} rows{note}"


def load_eval_data(spec: ModelSpec, data_dir: Path, source: str = "val", path: str = "",
                   target: str | None = None, text_column: str | None = None):
    """Returns (eval dataset, reference labels for the baseline or None, description,
    whether regression targets are already in original units)."""
    pre = spec.data.get("preprocess", {})
    if source == "val":
        cfg = spec.data.get("dataset_cfg")
        if not cfg:
            raise ValueError("This model wasn't trained in AIForge, so there is no validation data. "
                             "Choose 'A different file / folder'.")
        tr, va, info = datamod.load(dict(cfg), data_dir, spec)
        if len(va) == 0:
            raise ValueError("The model was trained without a validation split (0). "
                             "Evaluate on a different file instead.")
        mapping = _class_mapping(spec, info.class_names) if spec.task == "classification" else None
        if mapping:
            tr, va = _Relabel(tr, mapping), _Relabel(va, mapping)
        kind = "test set" if cfg["source"] in datamod.TORCHVISION else "validation split"
        return va, _labels(tr), f"{kind} of {info.description.split(':')[0]} ({len(va):,} samples)", False
    p = Path(path).expanduser()
    if pre.get("type") == "tabular" and pre.get("features"):
        ds, desc = _tabular_file(spec, p, target)
        return ds, None, desc, True
    src = {"text": "csv_text", "image": "image_folder"}.get(pre.get("type"), "csv")
    ds, _, info = datamod.load({"source": src, "path": str(p), "target": target,
                                "text_column": text_column, "val_split": 0}, data_dir, spec)
    if spec.task == "classification":
        mapping = _class_mapping(spec, info.class_names)
        if mapping:
            ds = _Relabel(ds, mapping)
    return ds, None, info.description, False


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_model(model: nn.Module, ds, device, noise: float = 0.0, keep_x: bool = False,
              progress: Callable[[int, int], None] | None = None, batch_size: int = 256):
    model.eval().to(device)
    gen = torch.Generator().manual_seed(0)
    outs, ys, xs = [], [], []
    done = 0
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        if noise:
            x = x + torch.randn(x.shape, generator=gen) * noise
        outs.append(model(x.to(device)).float().cpu())
        ys.append(torch.as_tensor(y))
        if keep_x:
            xs.append(x)
        done += len(x)
        if progress:
            progress(done, len(ds))
    model.cpu()
    out = torch.cat(outs).numpy()
    return out, torch.cat(ys).numpy(), (torch.cat(xs).numpy() if keep_x else None)


# ---------------------------------------------------------------------------
# Metrics (numpy, no extra dependencies)
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    _, inv, counts = np.unique(a[order], return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = np.empty(len(a))
    ranks[order] = ((end - counts + 1 + end) / 2)[inv]  # average rank for ties
    return ranks


def roc_auc(score: np.ndarray, pos: np.ndarray) -> float | None:
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((_rankdata(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(score: np.ndarray, pos: np.ndarray) -> float | None:
    n1 = int(pos.sum())
    if n1 == 0:
        return None
    hits = pos[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / n1)


def _confusion(y: np.ndarray, pred: np.ndarray, c: int) -> np.ndarray:
    return np.bincount(y * c + pred, minlength=c * c).reshape(c, c)


def _per_class(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, support


def _macro_f1(y, pred, c) -> float:
    _, _, f1, support = _per_class(_confusion(y, pred, c))
    return float(f1[support > 0].mean())


def _ci(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def classification_metrics(logits: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    probs = _softmax(logits)
    n, c = probs.shape
    names = names if len(names) == c else [f"class {i}" for i in range(c)]
    pred = probs.argmax(1)
    cm = _confusion(y, pred, c)
    precision, recall, f1, support = _per_class(cm)
    present = support > 0
    correct = pred == y
    conf = probs.max(1)

    per_class = [{"class": names[i], "precision": float(precision[i]), "recall": float(recall[i]),
                  "f1": float(f1[i]), "support": int(support[i]),
                  "roc_auc": roc_auc(probs[:, i], y == i), "pr_auc": average_precision(probs[:, i], y == i)}
                 for i in range(c)]
    if c == 2:
        auc, ap = per_class[1]["roc_auc"], per_class[1]["pr_auc"]
    else:
        aucs = [p["roc_auc"] for p in per_class if p["roc_auc"] is not None]
        aps = [p["pr_auc"] for p in per_class if p["pr_auc"] is not None]
        auc = float(np.mean(aucs)) if aucs else None
        ap = float(np.mean(aps)) if aps else None

    # Calibration: do 80%-confident answers turn out right 80% of the time?
    bins, ece = [], 0.0
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += gap * m.sum() / n
            bins.append({"range": f"{lo:.0%}-{hi:.0%}", "confidence": float(conf[m].mean()),
                         "accuracy": float(correct[m].mean()), "count": int(m.sum())})

    rng = np.random.default_rng(0)
    boot_acc, boot_f1 = [], []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, n, n)
        boot_acc.append(correct[i].mean())
        boot_f1.append(_macro_f1(y[i], pred[i], c))

    counts = support[present]
    return {
        "n": int(n), "classes": names,
        "accuracy": float(correct.mean()), "accuracy_ci": _ci(boot_acc),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()), "macro_f1_ci": _ci(boot_f1),
        "weighted_f1": float((f1 * support).sum() / n),
        "macro_precision": float(precision[present].mean()), "macro_recall": float(recall[present].mean()),
        "roc_auc": auc, "pr_auc": ap,
        "log_loss": float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1)).mean()),
        "ece": float(ece), "reliability": bins,
        "imbalance": float(counts.max() / counts.min()) if len(counts) else 1.0,
        "per_class": per_class, "confusion": cm.tolist(),
    }


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    pred, y = pred.reshape(-1).astype(float), y.reshape(-1).astype(float)
    err = pred - y
    abs_err = np.abs(err)
    ss_tot = ((y - y.mean()) ** 2).sum()
    near_zero = np.abs(y) <= 1e-6 * max(1.0, float(np.abs(y).max()))
    mape = float((abs_err[~near_zero] / np.abs(y[~near_zero])).mean()) if (~near_zero).any() else None
    rng = np.random.default_rng(0)
    boot_mae, boot_r2 = [], []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        boot_mae.append(abs_err[i].mean())
        tot = ((y[i] - y[i].mean()) ** 2).sum()
        boot_r2.append(1 - (err[i] ** 2).sum() / tot if tot > 0 else 0.0)
    return {
        "n": int(len(y)),
        "mae": float(abs_err.mean()), "mae_ci": _ci(boot_mae),
        "mse": float((err ** 2).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else None, "r2_ci": _ci(boot_r2),
        "mape": mape, "mape_excluded": float(near_zero.mean()),
        "median_ae": float(np.median(abs_err)), "max_error": float(abs_err.max()),
        "target_mean": float(y.mean()), "target_std": float(y.std()),
    }


def mcnemar(right_a: np.ndarray, right_b: np.ndarray) -> dict:
    """Is the accuracy difference between two models on the same examples real?"""
    b = int((right_a & ~right_b).sum())  # only A right
    c = int((~right_a & right_b).sum())  # only B right
    n = b + c
    if n == 0:
        p = 1.0
    elif n < 25:  # exact binomial test
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    else:  # chi-squared with continuity correction, 1 degree of freedom
        p = math.erfc(math.sqrt((abs(b - c) - 1) ** 2 / n / 2))
    return {"test": "McNemar", "only_a_right": b, "only_b_right": c, "p_value": float(p)}


def paired_bootstrap_mae(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    diff = np.abs(err_a) - np.abs(err_b)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOTSTRAP * 5)]
    boots = np.array(boots)
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"test": "paired bootstrap (MAE)", "mae_difference": float(diff.mean()),
            "ci": _ci(list(boots)), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Describing examples (error analysis)
# ---------------------------------------------------------------------------


def _describe(ds, i: int, spec: ModelSpec, x_row: np.ndarray | None) -> str:
    base, j = _unwrap(ds, i)
    if hasattr(base, "items"):
        return Path(base.items[j][0]).name
    pre = spec.data.get("preprocess", {})
    if pre.get("type") == "text":
        inv = {v: k for k, v in pre.get("vocab", {}).items()}
        ids = base.tensors[0][j].tolist() if isinstance(base, TensorDataset) else []
        text = " ".join(inv.get(t, "?") for t in ids if t > 1)
        return (text[:70] + "…") if len(text) > 70 else text
    if pre.get("type") == "tabular" and x_row is not None and pre.get("features"):
        raw = x_row * np.array(pre["std"]) + np.array(pre["mean"])
        cats = pre.get("categorical", {})
        parts = []
        for f, v in zip(pre["features"], raw):
            if f in cats:
                k = int(round(v))
                parts.append(f"{f}={cats[f][k] if 0 <= k < len(cats[f]) else '?'}")
            else:
                parts.append(f"{f}={v:.4g}")
        text = ", ".join(parts)
        return (text[:90] + "…") if len(text) > 90 else text
    return f"example #{j}"


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def _baseline(spec, ref, y_eval, c=None):
    labels = ref if ref is not None and len(ref) else y_eval
    source = "training data" if ref is not None and len(ref) else "evaluation data"
    if spec.task == "classification":
        counts = np.bincount(labels.astype(int), minlength=c)
        maj = int(counts.argmax())
        prior = np.clip(counts / counts.sum(), 1e-12, 1)
        y = y_eval.astype(int)
        return {"kind": "always predict the most common class", "class": int(maj), "source": source,
                "accuracy": float((y == maj).mean()),
                "macro_f1": _macro_f1(y, np.full_like(y, maj), c),
                "log_loss": float(-np.log(prior[y]).mean())}
    mean = float(labels.reshape(-1).mean())
    y = y_eval.reshape(-1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"kind": "always predict the average", "value": mean, "source": source,
            "mae": float(np.abs(y - mean).mean()),
            "r2": float(1 - ((y - mean) ** 2).sum() / ss_tot) if ss_tot > 0 else None}


def _learning_diagnosis(history: list[dict], task: str) -> dict | None:
    runs = [r for r in history if r.get("epochs")]
    if not runs:
        return None
    ep = runs[-1]["epochs"]
    if "val_loss" not in ep[-1]:
        return {"verdict": "unknown", "text": "The last run had no validation data, so over/underfitting "
                "can't be judged from the training history."}
    val = [e["val_loss"] for e in ep]
    best = int(np.argmin(val))
    res = {"epochs": len(ep), "best_epoch": best + 1, "train_loss": ep[-1]["train_loss"],
           "val_loss": ep[-1]["val_loss"], "min_val_loss": val[best]}
    gap = None
    if task == "classification" and "train_acc" in ep[best]:
        gap = ep[best]["train_acc"] - ep[best].get("val_acc", 0)
        res["acc_gap"] = gap
    # rose clearly after the best epoch (a 10% rise from a near-zero loss is just noise)
    rising = best < len(ep) - 2 and val[-1] > val[best] * 1.1 and val[-1] - val[best] > 0.02
    if rising or (gap is not None and gap > 0.1):
        res["verdict"] = "overfitting"
        res["text"] = (f"Validation loss was lowest at epoch {best + 1} of {len(ep)} and then rose"
                       if rising else f"Training accuracy is {gap:.0%} higher than validation accuracy") + \
            ": the model memorizes training examples. AIForge kept the best epoch; try Dropout, " \
            "Early stop, weight decay or more data."
    elif best == len(ep) - 1 and len(ep) > 1 and val[-2] - val[-1] > 0.01 * val[-1]:
        res["verdict"] = "still improving"
        res["text"] = "Validation loss was still falling at the last epoch: more epochs would likely help."
    else:
        res["verdict"] = "healthy"
        res["text"] = "Training and validation curves levelled off together: no sign of overfitting."
    return res


def evaluate_model(spec: ModelSpec, model: nn.Module, ds, ref_labels, description: str,
                   original_units: bool, device, history: list[dict] | None = None,
                   robustness: bool = True, other: tuple[str, ModelSpec, nn.Module] | None = None,
                   progress: Callable[[str, int, int], None] | None = None) -> dict:
    progress = progress or (lambda *a: None)
    pre = spec.data.get("preprocess", {})
    tabular = pre.get("type") == "tabular"
    out, y, x = run_model(model, ds, device, keep_x=tabular,
                          progress=lambda d, t: progress("Running the model", d, t))
    rep: dict = {"model": spec.name, "task": spec.task, "data": description,
                 "date": time.strftime("%Y-%m-%d %H:%M"), "warnings": []}

    if spec.task == "classification":
        y = y.astype(int)
        m = classification_metrics(out, y, spec.class_names)
        pred, conf = _softmax(out).argmax(1), _softmax(out).max(1)
        right = pred == y
        rep["metrics"] = m
        rep["baseline"] = _baseline(spec, ref_labels, y, out.shape[1])
        wrong = np.where(~right)[0]
        worst = wrong[np.argsort(-conf[wrong])][:TOP_ERRORS]
        rep["errors"] = [{"index": int(i), "true": m["classes"][y[i]], "predicted": m["classes"][pred[i]],
                          "confidence": float(conf[i]),
                          "example": _describe(ds, int(i), spec, x[i] if x is not None else None)}
                         for i in worst]
        if m["imbalance"] > 3:
            rep["warnings"].append(f"Classes are imbalanced (the biggest is {m['imbalance']:.0f}× the "
                                   "smallest): plain accuracy is misleading. Look at balanced accuracy, "
                                   "macro F1, per-class recall and PR-AUC.")
        if m["ece"] > 0.1:
            rep["warnings"].append(f"Probabilities are poorly calibrated (ECE {m['ece']:.2f}): the "
                                   "model's confidence can't be taken at face value.")
        if m["accuracy"] - rep["baseline"]["accuracy"] < 0.05:
            rep["warnings"].append("The model barely beats always guessing the most common class.")
    else:
        if not original_units:  # targets were standardised for training; scale both back
            s, mu = pre.get("target_std", 1.0), pre.get("target_mean", 0.0)
            out, y = out * s + mu, y * s + mu
            if ref_labels is not None:
                ref_labels = ref_labels * s + mu
        else:
            s, mu = pre.get("target_std", 1.0), pre.get("target_mean", 0.0)
            out = out * s + mu
        m = regression_metrics(out, y)
        rep["metrics"] = m
        rep["baseline"] = _baseline(spec, ref_labels, y)
        err = np.abs(out.reshape(-1) - y.reshape(-1))
        worst = np.argsort(-err)[:TOP_ERRORS]
        rep["errors"] = [{"index": int(i), "true": float(y.reshape(-1)[i]), "predicted": float(out.reshape(-1)[i]),
                          "error": float(err[i]),
                          "example": _describe(ds, int(i), spec, x[i] if x is not None else None)}
                         for i in worst]
        if m["mape_excluded"] > 0:
            rep["warnings"].append(f"{m['mape_excluded']:.1%} of true values are (near) zero and were left "
                                   "out of MAPE: percentage errors explode near zero.")
        if m["r2"] is not None and m["r2"] < 0.1:
            rep["warnings"].append("R² is close to 0: the model barely beats always predicting the average.")
        right = None

    # Slices: performance per category of each categorical input column (tables only)
    rep["slices"] = []
    if tabular and x is not None:
        cats = pre.get("categorical", {})
        overall = m["accuracy"] if spec.task == "classification" else m["mae"]
        for f in [f for f in pre["features"] if f in cats][:5]:
            j = pre["features"].index(f)
            codes = np.rint(x[:, j] * pre["std"][j] + pre["mean"][j]).astype(int)
            rows = []
            for code in np.argsort(-np.bincount(codes.clip(0), minlength=len(cats[f])))[:10]:
                mask = codes == code
                if mask.sum() < 5:
                    continue
                if spec.task == "classification":
                    score = float(right[mask].mean())
                    flag = score < overall - 0.1
                else:
                    score = float(err[mask].mean())
                    flag = score > overall * 1.5
                rows.append({"value": cats[f][code] if code < len(cats[f]) else "?", "n": int(mask.sum()),
                             "score": score, "flag": bool(flag)})
            if rows:
                rep["slices"].append({"feature": f, "rows": rows})
        if any(r["flag"] for s in rep["slices"] for r in s["rows"]):
            rep["warnings"].append("Some groups do much worse than average - see 'Slices'. This can hide "
                                   "real-world failures or fairness problems.")

    # Robustness: how much does a little input noise hurt?
    rep["robustness"] = []
    if robustness and spec.input_dtype == "float":
        for level in NOISE_LEVELS:
            o, yy, _ = run_model(model, ds, device, noise=level,
                                 progress=lambda d, t: progress(f"Robustness test (noise {level})", d, t))
            if spec.task == "classification":
                score = float((o.argmax(1) == yy.astype(int)).mean())
                drop = m["accuracy"] - score
            else:
                o = o * pre.get("target_std", 1.0) + pre.get("target_mean", 0.0)
                score = float(np.abs(o.reshape(-1) - y.reshape(-1)).mean())
                drop = score - m["mae"]
            rep["robustness"].append({"noise": level, "score": score, "drop": float(drop)})
        if spec.task == "classification" and rep["robustness"] and rep["robustness"][0]["drop"] > 0.1:
            rep["warnings"].append("A little noise costs more than 10 accuracy points: the model is fragile. "
                                   "More (or augmented) training data usually helps.")

    rep["learning"] = _learning_diagnosis(history or [], spec.task)
    if rep["learning"] and rep["learning"]["verdict"] == "overfitting":
        rep["warnings"].append("The training history shows overfitting - see 'Learning curves'.")

    # Model A vs model B on the very same examples
    rep["comparison"] = None
    if other:
        name_b, spec_b, model_b = other
        ob, _, _ = run_model(model_b, ds, device, progress=lambda d, t: progress(f"Running {name_b}", d, t))
        if spec.task == "classification":
            right_b = ob.argmax(1) == y
            rep["comparison"] = {"other": name_b, "score_a": float(right.mean()), "score_b": float(right_b.mean()),
                                 "metric": "accuracy", **mcnemar(right, right_b)}
        else:
            pb = spec_b.data.get("preprocess", {})
            ob = ob.reshape(-1) * pb.get("target_std", 1.0) + pb.get("target_mean", 0.0)
            rep["comparison"] = {"other": name_b, "score_a": m["mae"],
                                 "score_b": float(np.abs(ob - y.reshape(-1)).mean()), "metric": "MAE",
                                 **paired_bootstrap_mae(out.reshape(-1) - y.reshape(-1), ob - y.reshape(-1))}
    rep["verdict"] = _verdict(rep)
    return rep


def compatible(spec_a: ModelSpec, spec_b: ModelSpec) -> bool:
    """Can model B be run on model A's evaluation data and compared fairly?"""
    pa, pb = spec_a.data.get("preprocess", {}), spec_b.data.get("preprocess", {})
    return (spec_a.name != spec_b.name and spec_a.task == spec_b.task
            and spec_a.input_shape == spec_b.input_shape and spec_a.input_dtype == spec_b.input_dtype
            and spec_a.class_names == spec_b.class_names and pa.get("type") == pb.get("type")
            and (pa.get("type") != "text" or pa.get("vocab") == pb.get("vocab")))


def _verdict(rep: dict) -> str:
    m, b = rep["metrics"], rep["baseline"]
    if rep["task"] == "classification":
        lift = m["accuracy"] - b["accuracy"]
        if lift < 0.05:
            return "❌ Not useful yet: barely better than always guessing the most common class."
        if m["macro_f1"] >= 0.9 and m["balanced_accuracy"] >= 0.9:
            return "✅ Strong: high accuracy across every class."
        if m["macro_f1"] >= 0.7:
            return "👍 Good: clearly better than the baseline, with room to improve on some classes."
        return "⚠️ Weak: better than the baseline, but many classes are often wrong."
    r2 = m["r2"] if m["r2"] is not None else 0
    if r2 < 0.1:
        return "❌ Not useful yet: barely better than always predicting the average."
    if r2 >= 0.8:
        return "✅ Strong: explains most of the variation in the target."
    if r2 >= 0.5:
        return "👍 Good: explains a fair share of the variation, with room to improve."
    return "⚠️ Weak: better than predicting the average, but errors are large."


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(v) -> str:
    return "–" if v is None else f"{v:.1%}"


def _num(v) -> str:
    return "–" if v is None else f"{v:,.4g}"


def report_markdown(rep: dict) -> str:
    m, b = rep["metrics"], rep["baseline"]
    cls = rep["task"] == "classification"
    md = [f"# Evaluation: {rep['model']}", "",
          f"**Data:** {rep['data']} · {rep['date']}", "",
          f"## {rep['verdict']}", ""]
    if rep["warnings"]:
        md += [f"- ⚠️ {w}" for w in rep["warnings"]] + [""]

    md += ["## Metrics", "", "| Metric | Model | Baseline* | What it tells you |", "|---|---|---|---|"]
    if cls:
        lo, hi = m["accuracy_ci"]
        flo, fhi = m["macro_f1_ci"]
        md += [
            f"| **Accuracy** | **{_pct(m['accuracy'])}** ({_pct(lo)}-{_pct(hi)}) | {_pct(b['accuracy'])} | "
            "Share of correct answers. Only reliable when classes are balanced. |",
            f"| Balanced accuracy | {_pct(m['balanced_accuracy'])} | {_pct(1 / len(m['classes']))} | "
            "Average recall over classes: every class counts equally. |",
            f"| Macro F1 | {m['macro_f1']:.3f} ({flo:.3f}-{fhi:.3f}) | {b['macro_f1']:.3f} | "
            "Balance of precision and recall, averaged over classes. |",
            f"| Weighted F1 | {m['weighted_f1']:.3f} | | F1 weighted by class size. |",
            f"| ROC-AUC | {_num(m['roc_auc'])} | 0.5 | How well it ranks right above wrong, at any threshold. |",
            f"| PR-AUC | {_num(m['pr_auc'])} | | Like ROC-AUC, but honest on rare classes. |",
            f"| Log loss | {m['log_loss']:.4f} | {b['log_loss']:.4f} | Penalises confident wrong answers. Lower is better. |",
            f"| Calibration error (ECE) | {m['ece']:.3f} | | Gap between confidence and real accuracy. Below 0.05 is good. |",
        ]
    else:
        lo, hi = m["mae_ci"]
        rlo, rhi = m["r2_ci"]
        md += [
            f"| **MAE** | **{_num(m['mae'])}** ({_num(lo)}-{_num(hi)}) | {_num(b['mae'])} | "
            "Average error, in the target's own units. Robust to outliers. |",
            f"| RMSE | {_num(m['rmse'])} | | Like MAE but punishes big errors more. |",
            f"| MSE | {_num(m['mse'])} | | Squared error (what training minimises). |",
            f"| **R²** | **{_num(m['r2'])}** ({_num(rlo)}-{_num(rhi)}) | {_num(b['r2'])} | "
            "Share of the variation explained: 1 = perfect, 0 = no better than the average. |",
            f"| MAPE | {_pct(m['mape'])} | | Average % error. Unreliable when true values are near zero. |",
            f"| Median error | {_num(m['median_ae'])} | | Half of the errors are smaller than this. |",
            f"| Largest error | {_num(m['max_error'])} | | The single worst prediction. |",
        ]
    md += ["", f"\\*Baseline: {b['kind']} (learned from the {b['source']}). "
           "Brackets: 95% bootstrap confidence interval.", ""]

    if cls:
        md += ["## Per class", "", "| Class | Precision | Recall | F1 | ROC-AUC | Examples |",
               "|---|---|---|---|---|---|"]
        rows = sorted(m["per_class"], key=lambda r: r["f1"])
        for r in rows[:30]:
            md.append(f"| {r['class']} | {_pct(r['precision'])} | {_pct(r['recall'])} | {r['f1']:.3f} | "
                      f"{_num(r['roc_auc'])} | {r['support']:,} |")
        if len(rows) > 30:
            md.append(f"| … {len(rows) - 30} more classes | | | | | |")
        md += ["", "*Precision: when it says X, how often is it right? Recall: of all real X, how many "
               "did it find? Worst classes first.*", ""]
        cm, names = m["confusion"], m["classes"]
        md += ["## Confusion matrix", ""]
        if len(names) <= 10:
            md += ["Rows = true class, columns = predicted class.", "",
                   "| true ＼ predicted | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
            for i, row in enumerate(cm):
                md.append(f"| **{names[i]}** | " + " | ".join(f"**{v}**" if i == j else (str(v) if v else "·")
                                                             for j, v in enumerate(row)) + " |")
        else:
            pairs = sorted(((cm[i][j], names[i], names[j]) for i in range(len(names))
                            for j in range(len(names)) if i != j and cm[i][j]), reverse=True)[:12]
            md += ["Most common mix-ups:", "", "| True | Predicted as | Times |", "|---|---|---|"]
            md += [f"| {t} | {p} | {n:,} |" for n, t, p in pairs]
        md += ["", "## Calibration", "", "| Confidence | Avg. confidence | Actually right | Examples |",
               "|---|---|---|---|"]
        md += [f"| {r['range']} | {_pct(r['confidence'])} | {_pct(r['accuracy'])} | {r['count']:,} |"
               for r in m["reliability"]]
        md += ["", "*Well calibrated: 'Actually right' ≈ 'Avg. confidence' in every row.*", ""]

    if rep["errors"]:
        md += ["## Error analysis: the worst mistakes", ""]
        if cls:
            md += ["Confident mistakes are the most informative: look for a pattern (mislabelled data, a "
                   "confusing class, an unusual input).", "",
                   "| # | True | Predicted | Confidence | Example |", "|---|---|---|---|---|"]
            md += [f"| {e['index']} | {e['true']} | {e['predicted']} | {_pct(e['confidence'])} | "
                   f"{e['example'].replace('|', '/')} |" for e in rep["errors"]]
        else:
            md += ["| # | True | Predicted | Error | Example |", "|---|---|---|---|---|"]
            md += [f"| {e['index']} | {_num(e['true'])} | {_num(e['predicted'])} | {_num(e['error'])} | "
                   f"{e['example'].replace('|', '/')} |" for e in rep["errors"]]
        md.append("")

    if rep["slices"]:
        metric = "Accuracy" if cls else "MAE"
        md += ["## Slices: performance per group", "",
               f"{metric} for each value of the table's category columns. ⚠️ = much worse than overall.", ""]
        for s in rep["slices"]:
            md += [f"**{s['feature']}**", "", f"| Value | Examples | {metric} |", "|---|---|---|"]
            md += [f"| {'⚠️ ' if r['flag'] else ''}{r['value']} | {r['n']:,} | "
                   f"{_pct(r['score']) if cls else _num(r['score'])} |" for r in s["rows"]]
            md.append("")

    if rep["robustness"]:
        metric = "Accuracy" if cls else "MAE"
        base = _pct(m["accuracy"]) if cls else _num(m["mae"])
        md += ["## Robustness: inputs with added noise", "",
               f"| Noise | {metric} | Change |", "|---|---|---|", f"| none | {base} | |"]
        for r in rep["robustness"]:
            change = f"{-r['drop'] * 100:+.1f} points" if cls else f"{r['drop']:+,.4g}"
            md.append(f"| {r['noise']} | {_pct(r['score']) if cls else _num(r['score'])} | {change} |")
        md += ["", "*Noise is measured in normalised units (0.3 ≈ a third of a typical value's spread). "
               "A robust model loses little.*", ""]

    if rep["learning"]:
        lc = rep["learning"]
        md += ["## Learning curves", "", f"**{lc['verdict'].capitalize()}.** {lc['text']}", ""]
        if "best_epoch" in lc:
            md += [f"Best epoch {lc['best_epoch']} of {lc['epochs']} · final train loss {lc['train_loss']:.4f} · "
                   f"final validation loss {lc['val_loss']:.4f}", ""]

    if rep["comparison"]:
        c = rep["comparison"]
        fmt = _pct if cls else _num
        better = (c["score_a"] > c["score_b"]) if cls else (c["score_a"] < c["score_b"])
        winner = rep["model"] if better else c["other"]
        sig = c["p_value"] < 0.05
        md += [f"## Compared with {c['other']}", "", f"| Model | {c['metric']} |", "|---|---|",
               f"| {rep['model']} | {fmt(c['score_a'])} |", f"| {c['other']} | {fmt(c['score_b'])} |", "",
               f"{c['test']}: p = {c['p_value']:.4f}. " +
               (f"**{winner} is significantly better** (p < 0.05)." if sig else
                "**The difference is not statistically significant** - it could be chance."), ""]
        if "only_a_right" in c:
            md += [f"Only {rep['model']} right: {c['only_a_right']:,} · only {c['other']} right: "
                   f"{c['only_b_right']:,}", ""]
    return "\n".join(md)


def save_report(rep: dict, export_dir: Path) -> Path:
    """Write the report (Markdown + JSON) to exports/<model>/ and remember a summary."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = export_dir / rep["model"]
    out.mkdir(parents=True, exist_ok=True)
    md = out / f"evaluation-{stamp}.md"
    md.write_text(report_markdown(rep), encoding="utf-8")
    (out / f"evaluation-{stamp}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return md


def remember(rep: dict, model_dir: Path) -> None:
    """Keep a one-line summary per evaluation next to the model (shown on the home screen)."""
    f = model_dir / "evaluations.json"
    try:
        hist = json.loads(f.read_text())
    except (OSError, ValueError):
        hist = []
    m = rep["metrics"]
    keys = ("accuracy", "macro_f1", "roc_auc") if rep["task"] == "classification" else ("mae", "rmse", "r2")
    hist.append({"date": rep["date"], "data": rep["data"], **{k: m[k] for k in keys}})
    f.write_text(json.dumps(hist[-50:], indent=2))
