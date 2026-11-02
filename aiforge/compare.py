"""Compare training runs: of one model over time, or of different models on the same task."""
from __future__ import annotations

from dataclasses import dataclass

from .spec import Workspace

# Metrics a run can have per epoch, in the order they are offered, with how to show them.
METRICS = {
    "val_loss": ("Validation loss", "{:.4f}", "min"),
    "train_loss": ("Training loss", "{:.4f}", "min"),
    "val_acc": ("Validation accuracy", "{:.1%}", "max"),
    "train_acc": ("Training accuracy", "{:.1%}", "max"),
    "val_mae": ("Validation MAE", "{:.4g}", "min"),
    "val_ppl": ("Validation perplexity", "{:.3g}", "min"),
    "train_ppl": ("Training perplexity", "{:.3g}", "min"),
}
SETTINGS = ("lr", "batch_size", "optimizer", "scheduler", "weight_decay", "precision")


@dataclass
class Run:
    model: str
    index: int  # position in the model's history
    started: str
    kind: str
    dataset: str
    epochs: list[dict]
    config: dict

    @property
    def label(self) -> str:
        return f"{self.model} #{self.index + 1}"

    def series(self, metric: str) -> list[float]:
        return [e[metric] for e in self.epochs if metric in e]

    def best(self, metric: str) -> float | None:
        values = self.series(metric)
        if not values:
            return None
        return min(values) if METRICS.get(metric, ("", "", "min"))[2] == "min" else max(values)


def collect_runs(ws: Workspace, models: list[str] | None = None) -> list[Run]:
    """Every training run (train / fine-tune) of the given models, oldest first per model."""
    runs = []
    for name in models if models is not None else [m["name"] for m in ws.list_models()]:
        for i, r in enumerate(ws.load_history(name)):
            if r.get("kind") in ("train", "fine-tune") and r.get("epochs"):
                runs.append(Run(name, i, r.get("started", ""), r["kind"], r.get("dataset", ""),
                                r["epochs"], r.get("config", {})))
    return runs


def common_metrics(runs: list[Run]) -> list[str]:
    """Metrics every run has, best first (validation before training)."""
    return [m for m in METRICS if runs and all(r.series(m) for r in runs)]


def fmt(metric: str, value: float | None) -> str:
    return "-" if value is None else METRICS.get(metric, ("", "{:.4g}", ""))[1].format(value)


def table(runs: list[Run], metric: str) -> tuple[list[str], list[list[str]], int | None]:
    """Header, rows and the index of the best run for a side-by-side summary. Settings that are
    the same for every run are left out, so the differences stand out."""
    varying = [k for k in SETTINGS if len({str(r.config.get(k)) for r in runs}) > 1]
    header = ["Run", "Date", "Data", "Epochs", "Best", "Last"] + varying  # the metric is in the title
    rows = []
    for r in runs:
        values = r.series(metric)
        rows.append([r.label, r.started[5:16], r.dataset.split(":")[0][:18], str(len(r.epochs)),
                     fmt(metric, r.best(metric)), fmt(metric, values[-1] if values else None)]
                    + [str(r.config.get(k, "-")) for k in varying])
    bests = [r.best(metric) for r in runs]
    scored = [i for i, b in enumerate(bests) if b is not None]
    if not scored:
        return header, rows, None
    pick = min if METRICS[metric][2] == "min" else max
    # Compare scores only: on a tie the earliest run wins (min/max return the first best item).
    return header, rows, pick(scored, key=lambda i: bests[i])
