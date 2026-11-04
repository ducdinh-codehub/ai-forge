"""Learning without labels: fitting the classic algorithms (k-means, PCA), and what every
unsupervised model is used for afterwards: how unusual an example is, which group it belongs to,
and a 2-D map of the data. Independent of the UI; plain torch/numpy, no extra dependencies."""
from __future__ import annotations

import math
import threading
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .spec import ModelSpec

MAX_VALUES = 50_000_000  # classic fitting holds the data in memory: about 200 MB of float32
CHUNK = 4096  # rows per distance computation
SCORE_NAMES = {"clustering": "distance to its group's centre", "pca": "rebuild error",
               "autoencoder": "rebuild error"}
MAP_SYMBOLS = "●▲■◆✚★○△□◇"


def group_name(i: int) -> str:
    return f"group {i + 1}"


# ---------------------------------------------------------------------------
# Data in memory
# ---------------------------------------------------------------------------


def gather(ds, limit: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs and labels of (at most ``limit`` random) examples, as two tensors."""
    if len(ds) > limit:
        gen = torch.Generator().manual_seed(seed)
        ds = Subset(ds, torch.randperm(len(ds), generator=gen)[:limit].tolist())
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=1024, shuffle=False):
        xs.append(x.float())
        ys.append(torch.as_tensor(y))
    if not xs:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(xs), torch.cat(ys)


def example_limit(input_shape) -> int:
    """How many examples the classic algorithms keep in memory at once."""
    return max(2000, MAX_VALUES // max(1, math.prod(input_shape)))


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------


def _sq_dist(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Squared distances, (n, k), in chunks so big datasets don't need an n×k×d tensor."""
    cc = (c * c).sum(1)
    out = [((xb * xb).sum(1, keepdim=True) - 2 * xb @ c.T + cc).clamp_min(0) for xb in x.split(CHUNK)]
    return torch.cat(out) if out else torch.zeros(0, len(c))


def _kmeans_pp(pool: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Greedy k-means++: each new centre is drawn far from the ones chosen so far (a few
    candidates are drawn and the one that helps most is kept)."""
    centres = [pool[torch.randint(len(pool), (1,), generator=gen)].squeeze(0)]
    d = _sq_dist(pool, centres[0][None]).squeeze(1)
    trials = 2 + int(math.log(k))
    for _ in range(1, k):
        if d.sum() <= 0:  # fewer distinct points than groups
            idx = torch.randint(len(pool), (trials,), generator=gen)
        else:
            idx = torch.multinomial(d / d.sum(), trials, replacement=True, generator=gen)
        options = torch.minimum(d[None], _sq_dist(pool, pool[idx]).T)  # (trials, n)
        best = int(options.sum(1).argmin())
        centres.append(pool[idx[best]])
        d = options[best]
    return torch.stack(centres)


def _lloyd_step(x: torch.Tensor, c: torch.Tensor):
    """Assign every example to its nearest centre, then move each centre to the middle of its
    examples. Returns (new centres, squared distance to the old nearest centre, assignment)."""
    near, assign = _sq_dist(x, c).min(1)
    sums = torch.zeros_like(c).index_add_(0, assign, x)
    counts = torch.bincount(assign, minlength=len(c))
    new = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], c)
    for j in torch.nonzero(counts == 0).flatten().tolist():  # an empty group restarts far away
        far = int(near.argmax())
        new[j], near[far] = x[far], 0
    return new, near, assign


def _start(x: torch.Tensor, k: int, gen: torch.Generator, tries: int = 4) -> torch.Tensor:
    """k-means can get stuck with two centres in one real group. Try a few k-means++ starts on a
    sample, run each briefly, and keep the one that ends with the tightest groups."""
    pool = x[torch.randperm(len(x), generator=gen)[:5000]]
    best, best_score = None, math.inf
    for _ in range(tries):
        c = _kmeans_pp(pool, k, gen)
        for _ in range(10):
            c, near, _ = _lloyd_step(pool, c)
        if float(near.mean()) < best_score:
            best, best_score = c, float(near.mean())
    return best


def fit_kmeans(layer: nn.Module, x: torch.Tensor, max_iter: int, seed: int = 0,
               stop: threading.Event | None = None) -> Iterator[tuple[int, float, int]]:
    """Lloyd's algorithm on the flat inputs ``x``: assign every example to its nearest centre, move
    each centre to the middle of its examples, repeat. Writes the centres into ``layer`` and yields
    (iteration, mean squared distance to the centre, examples that changed group) after each round;
    stops early once no example changes group."""
    x = x.flatten(1)
    k = layer.centroids.shape[0]
    if len(x) < k:
        raise ValueError(f"k-means needs at least {k} examples to find {k} groups, got {len(x)}")
    gen = torch.Generator().manual_seed(seed)
    c = _start(x, k, gen)
    prev = None
    for it in range(1, max_iter + 1):
        if stop is not None and stop.is_set():
            return
        c, near, assign = _lloyd_step(x, c)
        layer.centroids.copy_(c)
        moved = len(x) if prev is None else int((assign != prev).sum())
        prev = assign
        yield it, float(near.mean()), moved
        if moved == 0:
            return


def assign(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The nearest group of every example (a k-means model)."""
    return _batched(model, x).argmin(1)


def silhouette(x: torch.Tensor, labels: torch.Tensor, max_n: int = 1000, seed: int = 0) -> float | None:
    """How well separated the groups are, from -1 (wrong groups) to 1 (tight, far-apart groups).
    For each example: (distance to the nearest other group - distance to its own group) / the larger
    of the two, averaged. Uses a random sample: the exact value needs every pair of examples."""
    x, labels = x.flatten(1), labels.long()
    if len(x) > max_n:
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:max_n]
        x, labels = x[idx], labels[idx]
    k = int(labels.max()) + 1 if len(labels) else 0
    if k < 2 or len(x) < 3:
        return None
    d = _sq_dist(x, x).sqrt()
    onehot = torch.nn.functional.one_hot(labels, k).float()
    sums, counts = d @ onehot, onehot.sum(0)
    own = counts[labels]
    a = sums.gather(1, labels[:, None]).squeeze(1) / (own - 1).clamp_min(1)
    other = sums / counts.clamp_min(1)
    other[torch.arange(len(x)), labels] = math.inf
    other[:, counts == 0] = math.inf
    b = other.min(1).values
    s = torch.where(own > 1, (b - a) / torch.maximum(a, b).clamp_min(1e-12), torch.zeros_like(a))
    return float(s.mean())


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def fit_pca(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Find the directions of largest variation (eigenvectors of the covariance matrix) and write
    them into ``layer``. Returns the share of the total variation each component keeps."""
    x = x.flatten(1).double()
    n_comp = layer.components.shape[0]
    if len(x) < 2:
        raise ValueError("PCA needs at least 2 examples")
    mean = x.mean(0)
    xc = x - mean
    total = (xc * xc).sum() / (len(x) - 1)
    if x.shape[1] <= 4096:
        vals, vecs = torch.linalg.eigh(xc.T @ xc / (len(x) - 1))  # ascending
        vals, vecs = vals.flip(0)[:n_comp], vecs.flip(1)[:, :n_comp].T
    else:  # very wide inputs: a randomised SVD instead of a d×d matrix
        _, s, v = torch.pca_lowrank(xc, q=min(n_comp + 10, *x.shape), center=False)
        vals, vecs = (s[:n_comp] ** 2) / (len(x) - 1), v[:, :n_comp].T
    # Make each direction's largest entry positive, so refitting gives the same map, not a mirror.
    flip = torch.sign(vecs.gather(1, vecs.abs().argmax(1, keepdim=True))).clamp_min(0) * 2 - 1
    layer.mean.copy_(mean.float())
    layer.components.copy_((vecs * flip).float())
    return (vals.clamp_min(0) / total.clamp_min(1e-12)).float()


# ---------------------------------------------------------------------------
# Using a fitted model
# ---------------------------------------------------------------------------


@torch.no_grad()
def _batched(fn, x: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    if isinstance(fn, nn.Module):
        fn.eval()
    return torch.cat([fn(xb).float() for xb in x.split(batch)]) if len(x) else torch.zeros(0)


def rebuild(spec: ModelSpec, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """What a PCA model or an autoencoder makes of ``x`` after squeezing it (flat, one row each)."""
    if spec.task == "pca":
        layer = model[0]
        return _batched(lambda xb: layer.reconstruct(layer(xb)), x)
    return _batched(model, x).flatten(1)


def scores(spec: ModelSpec, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """How unusual each example is: the distance to its group's centre (k-means) or how badly the
    model rebuilds it (PCA, autoencoder). Higher = more unusual."""
    if spec.task == "clustering":
        return _batched(model, x).min(1).values
    return ((rebuild(spec, model, x) - x.flatten(1)) ** 2).mean(1)


def variance_kept(spec: ModelSpec, model: nn.Module, x: torch.Tensor) -> float | None:
    """Share of the data's variation the rebuilt examples keep (R²): 1 = perfect, 0 = no better
    than rebuilding every example as the average one."""
    flat = x.flatten(1)
    total = ((flat - flat.mean(0)) ** 2).sum()
    if total <= 0:
        return None
    return float(1 - ((rebuild(spec, model, x) - flat) ** 2).sum() / total)


def codes(spec: ModelSpec, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The model's compressed view of each example: the PCA coordinates, the autoencoder's
    bottleneck values, or (k-means, which compresses nothing) the inputs themselves."""
    if spec.task == "pca":
        return _batched(model, x)
    idx = spec.bottleneck()
    if spec.task == "autoencoder" and idx is not None:
        return _batched(model[: idx + 1], x).flatten(1)
    return x.flatten(1)


def project_2d(points: torch.Tensor) -> np.ndarray:
    """Points flattened onto the two directions along which they vary most (a PCA to 2-D)."""
    p = points.flatten(1).double()
    if p.shape[1] <= 2:
        out = p if p.shape[1] == 2 else torch.cat([p, torch.zeros(len(p), 2 - p.shape[1], dtype=p.dtype)], 1)
        return out.numpy()
    from .layers import PCA

    layer = PCA(p.shape[1], 2)
    fit_pca(layer, p)
    return layer(p.float()).numpy()


# ---------------------------------------------------------------------------
# Reference statistics: what is "normal" for this model
# ---------------------------------------------------------------------------


def reference_stats(spec: ModelSpec, model: nn.Module, x: torch.Tensor) -> dict:
    """Stored with the model after training (spec.data['unsupervised']) so a new example's score can
    be compared with the training data's."""
    s = scores(spec, model, x)
    stats = {"score": SCORE_NAMES[spec.task], "n": len(s),
             "quantiles": [float(v) for v in torch.quantile(s.double(), torch.linspace(0, 1, 101, dtype=torch.double))]
             if len(s) else []}
    if spec.task == "clustering":
        stats["sizes"] = torch.bincount(assign(model, x), minlength=model[0].centroids.shape[0]).tolist()
    return stats


def percentile(spec: ModelSpec, score: float) -> float | None:
    """Share of the training examples that scored lower: 0.97 = more unusual than 97% of them."""
    q = spec.data.get("unsupervised", {}).get("quantiles")
    if not q:
        return None
    return float(np.interp(score, q, np.linspace(0, 1, len(q)), left=0.0, right=1.0))


def verdict(spec: ModelSpec, score: float) -> str:
    p = percentile(spec, score)
    if p is None:
        return "train the model in AIForge to compare with its training data"
    if p >= 0.99:
        return f"⚠️ unusual: more unusual than {p:.0%} of the training examples"
    if p >= 0.95:
        return f"🟡 a bit unusual: more unusual than {p:.0%} of the training examples"
    return f"✅ looks normal: more unusual than {p:.0%} of the training examples"


# ---------------------------------------------------------------------------
# Comparing groups with labels that were kept aside
# ---------------------------------------------------------------------------


def purity(groups: np.ndarray, labels: np.ndarray) -> float:
    """Share of examples whose label is the most common one in their group (1 = every group holds
    a single label)."""
    total = 0
    for g in np.unique(groups):
        total += np.bincount(labels[groups == g]).max()
    return float(total / max(1, len(groups)))


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float | None:
    """Agreement between two groupings, corrected for chance: 1 = the same, ~0 = random."""
    n = len(a)
    if n < 2:
        return None
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    comb = lambda v: v * (v - 1) / 2  # noqa: E731
    index, rows, cols = comb(table).sum(), comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    expected = rows * cols / comb(n)
    top = (rows + cols) / 2
    return float((index - expected) / (top - expected)) if top != expected else None


# ---------------------------------------------------------------------------
# A map of the data in plain text
# ---------------------------------------------------------------------------


def text_map(points: np.ndarray, kinds: np.ndarray | None = None, width: int = 60, height: int = 18,
             highlight: np.ndarray | None = None) -> str:
    """Scatter plot drawn with characters. ``kinds`` (small ints) picks each point's symbol; a cell
    shows its most common symbol. ``highlight`` marks points with ✖ (drawn on top)."""
    if len(points) == 0:
        return "(no points)"
    lo, hi = points.min(0), points.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1)
    col = np.clip(((points[:, 0] - lo[0]) / span[0] * (width - 1)).round().astype(int), 0, width - 1)
    row = np.clip(((hi[1] - points[:, 1]) / span[1] * (height - 1)).round().astype(int), 0, height - 1)
    kinds = np.zeros(len(points), int) if kinds is None else kinds
    counts: dict[tuple[int, int], np.ndarray] = {}
    for r, c, k in zip(row, col, kinds):
        counts.setdefault((r, c), np.zeros(int(kinds.max()) + 1, int))[k] += 1
    grid = [[" "] * width for _ in range(height)]
    single = kinds.max() == 0
    for (r, c), cnt in counts.items():
        if single:  # density: more points, darker dot
            grid[r][c] = "·" if cnt[0] == 1 else "•" if cnt[0] < 5 else "●"
        else:
            grid[r][c] = MAP_SYMBOLS[int(cnt.argmax()) % len(MAP_SYMBOLS)]
    for i in np.flatnonzero(highlight) if highlight is not None else []:
        grid[row[i]][col[i]] = "✖"
    return "\n".join("│" + "".join(line) for line in grid) + "\n└" + "─" * width
