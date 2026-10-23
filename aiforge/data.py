"""Datasets. Every loader returns train/val datasets plus a ``DataInfo`` describing
what the model must look like and how raw inputs are preprocessed for prediction."""
from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, TensorDataset, random_split

SOURCES = {
    "synthetic": "Synthetic toy data (no files needed)",
    "csv": "CSV table - numeric/categorical columns + a target column",
    "csv_text": "CSV with a text column + label column (text classification)",
    "text_corpus": "Plain text file (.txt) - language model / text generation",
    "image_folder": "Image folder - one sub-folder per class",
    "mnist": "MNIST handwritten digits (auto-download, 28×28×1)",
    "fashion_mnist": "Fashion-MNIST clothing (auto-download, 28×28×1)",
    "cifar10": "CIFAR-10 objects (auto-download, 32×32×3)",
    "cifar100": "CIFAR-100 objects, 100 classes (auto-download, 32×32×3)",
}
SYNTHETIC = ["moons", "spirals", "circles", "blobs", "regression"]
FILE_SOURCES = ("csv", "csv_text", "image_folder", "text_corpus")  # need a path
TORCHVISION = ("mnist", "fashion_mnist", "cifar10", "cifar100")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TEXT_EXT = {".txt", ".md", ".text"}
EXAMPLE_CORPUS = Path(__file__).parent / "examples" / "nursery_rhymes.txt"


@dataclass
class DataInfo:
    input_shape: list[int]
    input_dtype: str
    task: str
    class_names: list[str]
    preprocess: dict = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    description: str = ""

    @property
    def n_outputs(self) -> int:
        if self.task == "language_model":
            return len(self.preprocess["vocab"])
        return len(self.class_names) if self.task == "classification" else 1

    @property
    def output_shape(self) -> tuple[int, ...]:
        """What the model must output: one score per vocabulary token at every position for a
        language model, else one flat vector."""
        if self.task == "language_model":
            return (self.input_shape[0], self.n_outputs)
        return (self.n_outputs,)


def load(cfg: dict, data_dir: Path, spec=None) -> tuple[Dataset, Dataset, DataInfo]:
    src = cfg["source"]
    val_split = float(cfg.get("val_split", 0.2))
    if src == "synthetic":
        ds, info = _synthetic(cfg.get("kind", "moons"), int(cfg.get("samples", 2000)))
    elif src == "csv":
        ds, info = _csv_table(Path(cfg["path"]).expanduser(), cfg.get("target") or None,
                              cfg.get("task") or None)
    elif src == "csv_text":
        seq_len = int(spec.input_shape[0]) if spec and spec.input_dtype == "long" else 100
        prev = spec.data.get("preprocess", {}) if spec else {}
        ds, info = _csv_text(Path(cfg["path"]).expanduser(), cfg.get("target") or None,
                             cfg.get("text_column") or None, seq_len, _vocab_size(spec),
                             prev.get("vocab") if prev.get("type") == "text" else None)
    elif src == "image_folder":
        ds, info = _image_folder(Path(cfg["path"]).expanduser(), spec)
    elif src == "text_corpus":
        return _text_corpus(Path(cfg["path"]).expanduser(), cfg.get("tokenizer") or "char", spec, val_split)
    elif src in TORCHVISION:
        return _torchvision(src, data_dir, spec)
    else:
        raise ValueError(f"Unknown data source {src}")
    n_val = max(1, int(len(ds) * val_split)) if val_split > 0 else 0
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train, val = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    info.n_train, info.n_val = len(train), len(val)
    return train, val, info


def _vocab_size(spec, default: int = 20000) -> int:
    if spec:
        for l in spec.layers:
            if l.type == "Embedding":
                return int(l.config["num_embeddings"])
    return default


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def _synthetic(kind: str, n: int):
    g = torch.Generator().manual_seed(0)
    if kind == "regression":
        x = torch.rand(n, 4, generator=g) * 4 - 2
        y = (torch.sin(x[:, 0]) + x[:, 1] ** 2 * 0.5 - x[:, 2] * x[:, 3]
             + torch.randn(n, generator=g) * 0.1)
        info = DataInfo([4], "float", "regression", [], {"type": "tabular",
                        "features": ["x1", "x2", "x3", "x4"], "mean": [0.0] * 4, "std": [1.0] * 4,
                        "categorical": {}}, description="y = sin(x1) + x2²/2 - x3·x4 + noise")
        return TensorDataset(x, y.unsqueeze(1)), info
    if kind == "moons":
        h = n // 2
        t = torch.rand(h, generator=g) * math.pi
        a = torch.stack([torch.cos(t), torch.sin(t)], 1)
        b = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)
        x = torch.cat([a, b]) + torch.randn(2 * h, 2, generator=g) * 0.1
        y = torch.cat([torch.zeros(h), torch.ones(h)]).long()
        classes = ["upper", "lower"]
    elif kind == "circles":
        h = n // 2
        t = torch.rand(2 * h, generator=g) * 2 * math.pi
        r = torch.cat([torch.ones(h), torch.full((h,), 0.4)])
        x = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1) + torch.randn(2 * h, 2, generator=g) * 0.06
        y = torch.cat([torch.zeros(h), torch.ones(h)]).long()
        classes = ["outer", "inner"]
    elif kind == "spirals":
        k, per = 3, n // 3
        xs, ys = [], []
        for c in range(k):
            r = torch.linspace(0.05, 1, per)
            t = c * 2 * math.pi / k + r * 3.5 + torch.randn(per, generator=g) * 0.15
            xs.append(torch.stack([r * torch.cos(t), r * torch.sin(t)], 1))
            ys.append(torch.full((per,), c))
        x, y = torch.cat(xs), torch.cat(ys).long()
        classes = ["arm-0", "arm-1", "arm-2"]
    elif kind == "blobs":
        k, per = 4, n // 4
        centers = torch.tensor([[-2, -2], [2, 2], [-2, 2], [2, -2]], dtype=torch.float)
        x = torch.cat([centers[c] + torch.randn(per, 2, generator=g) * 0.8 for c in range(k)])
        y = torch.arange(k).repeat_interleave(per)
        classes = [f"blob-{c}" for c in range(k)]
    else:
        raise ValueError(f"unknown synthetic dataset {kind}")
    info = DataInfo([2], "float", "classification", classes,
                    {"type": "tabular", "features": ["x", "y"], "mean": [0.0, 0.0],
                     "std": [1.0, 1.0], "categorical": {}}, description=f"2D {kind}")
    return TensorDataset(x.float(), y), info


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("CSV needs a header row and at least one data row")
    return [h.strip() for h in rows[0]], rows[1:]


def csv_columns(path: str) -> list[str]:
    try:
        return _read_csv(Path(path).expanduser())[0]
    except Exception:  # noqa: BLE001
        return []


def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _csv_table(path: Path, target: str | None, task: str | None):
    header, rows = _read_csv(path)
    target = target or header[-1]
    if target not in header:
        raise ValueError(f"Target column '{target}' not in CSV columns: {', '.join(header)}")
    ti = header.index(target)
    rows = [r for r in rows if len(r) == len(header)]
    feats = [h for i, h in enumerate(header) if i != ti]
    cols = {h: [r[header.index(h)].strip() for r in rows] for h in header}

    categorical: dict[str, list[str]] = {}
    matrix = []
    for h in feats:
        vals = cols[h]
        if all(_is_num(v) for v in vals if v != ""):
            nums = [float(v) if v != "" else math.nan for v in vals]
            known = [v for v in nums if not math.isnan(v)]
            fill = sum(known) / len(known) if known else 0.0
            matrix.append([fill if math.isnan(v) else v for v in nums])
        else:
            cats = sorted(set(vals))
            categorical[h] = cats
            matrix.append([float(cats.index(v)) for v in vals])
    x = torch.tensor(matrix, dtype=torch.float).T
    mean, std = x.mean(0), x.std(0).clamp_min(1e-8)
    x = (x - mean) / std

    yv = cols[target]
    numeric_target = all(_is_num(v) for v in yv)
    if task is None:
        task = "regression" if numeric_target and len(set(yv)) > 20 else "classification"
    if task == "regression":
        if not numeric_target:
            raise ValueError("Regression needs a numeric target column")
        y = torch.tensor([float(v) for v in yv]).unsqueeze(1)
        y_mean, y_std = y.mean().item(), max(y.std().item(), 1e-8)
        y = (y - y_mean) / y_std  # train on a standardised target; predictions are scaled back
        classes = []
    else:
        classes = sorted(set(yv), key=lambda v: (float(v) if numeric_target else 0, v))
        y = torch.tensor([classes.index(v) for v in yv])
    info = DataInfo([len(feats)], "float", task, classes, {
        "type": "tabular", "features": feats, "mean": mean.tolist(), "std": std.tolist(),
        "categorical": categorical, **({"target_mean": y_mean, "target_std": y_std}
                                       if task == "regression" else {})},
        description=f"{path.name}: {len(rows)} rows, "
                                                  f"{len(feats)} features, target '{target}'")
    return TensorDataset(x, y), info


# ---------------------------------------------------------------------------
# Text CSV
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def encode_text(text: str, vocab: dict[str, int], seq_len: int) -> list[int]:
    ids = [vocab.get(t, 1) for t in tokenize(text)][:seq_len]
    # Left-pad so the real tokens end the sequence (RNNs read the last step). 0 = <pad>, 1 = <unk>
    return [0] * (seq_len - len(ids)) + ids


def _csv_text(path: Path, target: str | None, text_col: str | None, seq_len: int,
              vocab_size: int, vocab: dict[str, int] | None = None):
    header, rows = _read_csv(path)
    target = target or header[-1]
    text_col = text_col or next(h for h in header if h != target)
    for c in (target, text_col):
        if c not in header:
            raise ValueError(f"Column '{c}' not in CSV columns: {', '.join(header)}")
    ti, xi = header.index(target), header.index(text_col)
    rows = [r for r in rows if len(r) == len(header)]
    if vocab is None:  # keep an existing vocab when fine-tuning so embeddings stay meaningful
        counts = Counter(t for r in rows for t in tokenize(r[xi]))
        words = [w for w, _ in counts.most_common(max(0, vocab_size - 2))]
        vocab = {w: i + 2 for i, w in enumerate(words)}
    x = torch.tensor([encode_text(r[xi], vocab, seq_len) for r in rows])
    classes = sorted(set(r[ti].strip() for r in rows))
    y = torch.tensor([classes.index(r[ti].strip()) for r in rows])
    info = DataInfo([seq_len], "long", "classification", classes, {
        "type": "text", "vocab": vocab, "seq_len": seq_len, "vocab_size": vocab_size},
        description=f"{path.name}: {len(rows)} texts, vocab {len(vocab) + 2}")
    return TensorDataset(x, y), info


# ---------------------------------------------------------------------------
# Plain text (language models)
# ---------------------------------------------------------------------------


class TextWindows(Dataset):
    """Slices of a token stream: x = tokens[i : i+L], y = tokens[i+1 : i+L+1] (the next token at
    every position)."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, stride: int):
        self.tokens, self.seq_len = tokens, seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(0, (len(self.tokens) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        chunk = self.tokens[s: s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if not text.strip():
        raise ValueError(f"{path.name} is empty")
    return text


def _text_corpus(path: Path, tokenizer: str, spec, val_split: float):
    from . import textgen

    if tokenizer not in textgen.TOKENIZERS:
        raise ValueError(f"Unknown tokenizer '{tokenizer}' (use char or word)")
    seq_len = int(spec.input_shape[0]) if spec and spec.input_dtype == "long" and len(spec.input_shape) == 1 \
        else 64
    tokens = textgen.split(read_text(path), tokenizer)
    prev = spec.data.get("preprocess", {}) if spec else {}
    if prev.get("type") == "corpus" and prev.get("tokenizer") == tokenizer:
        vocab = prev["vocab"]  # keep the vocabulary when continuing/fine-tuning: ids must not move
    else:
        # Every character gets an id; for words the Embedding size caps the vocabulary.
        size = _vocab_size(spec, 8000) if tokenizer == "word" else textgen.MAX_CHARS
        vocab = textgen.build_vocab(tokens, tokenizer, size)
    index = {t: i for i, t in enumerate(vocab)}
    ids = torch.tensor([index.get(t, 0) for t in tokens], dtype=torch.long)
    # Split the text itself (not overlapping windows) so validation text is never seen in training.
    n_val = int(len(ids) * val_split) if val_split > 0 else 0
    if n_val and n_val < seq_len + 1:
        n_val = min(len(ids) // 2, seq_len + 1)
    train_ids, val_ids = (ids[:-n_val], ids[-n_val:]) if n_val else (ids, ids[:0])
    if len(train_ids) < seq_len + 1:
        raise ValueError(f"{path.name} has only {len(ids):,} {tokenizer} tokens - it needs more than "
                         f"{seq_len} (the model's input length) for training. Use a longer text or a "
                         "shorter input length.")
    # Overlapping windows: a short text still gives ~2,000 examples per epoch, a long one starts a
    # window every half input length.
    stride = max(1, min(seq_len // 2, len(train_ids) // 2000))
    train = TextWindows(train_ids, seq_len, stride)
    val = TextWindows(val_ids, seq_len, seq_len)
    unk = (ids == 0).float().mean().item()
    info = DataInfo([seq_len], "long", "language_model", [], {
        "type": "corpus", "tokenizer": tokenizer, "vocab": vocab, "seq_len": seq_len},
        n_train=len(train), n_val=len(val),
        description=f"{path.name}: {len(ids):,} {tokenizer} tokens, vocabulary {len(vocab):,}"
                    + (f" ({unk:.1%} unknown)" if unk >= 0.005 else ""))
    return train, val, info


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _image_transform(pre: dict, train: bool = False):
    from torchvision import transforms as T

    ops = [T.Resize(tuple(pre["size"]))]
    if pre["channels"] == 1:
        ops.append(T.Grayscale(1))
    else:
        ops.append(T.Lambda(lambda im: im.convert("RGB")))
    if train and pre.get("augment"):
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(pre["mean"], pre["std"])]
    return T.Compose(ops)


class _ImageFiles(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path, label = self.items[i]
        with Image.open(path) as im:
            return self.transform(im), label


def _image_folder(root: Path, spec):
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if len(classes) < 2:
        raise ValueError("Image folder needs at least 2 sub-folders (one per class)")
    items = [(p, ci) for ci, c in enumerate(classes) for p in sorted((root / c).rglob("*"))
             if p.suffix.lower() in IMAGE_EXT]
    if not items:
        raise ValueError("No images found (.png/.jpg/...)")
    shape = list(spec.input_shape) if spec and len(spec.input_shape) == 3 else [3, 64, 64]
    c = shape[0] if shape[0] in (1, 3) else 3
    imagenet = c == 3
    pre = {"type": "image", "size": shape[1:], "channels": c,
           "mean": [0.485, 0.456, 0.406] if imagenet else [0.5],
           "std": [0.229, 0.224, 0.225] if imagenet else [0.5]}
    ds = _ImageFiles(items, _image_transform(pre))
    info = DataInfo([c, *shape[1:]], "float", "classification", classes, pre,
                    description=f"{root.name}: {len(items)} images, {len(classes)} classes")
    return ds, info


def _torchvision(name: str, data_dir: Path, spec):
    from torchvision import datasets, transforms as T

    meta = {
        "mnist": (datasets.MNIST, [1, 28, 28], [0.1307], [0.3081]),
        "fashion_mnist": (datasets.FashionMNIST, [1, 28, 28], [0.2860], [0.3530]),
        "cifar10": (datasets.CIFAR10, [3, 32, 32], [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        "cifar100": (datasets.CIFAR100, [3, 32, 32], [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    }[name]
    cls, shape, mean, std = meta
    pre = {"type": "image", "size": shape[1:], "channels": shape[0], "mean": mean, "std": std}
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = data_dir / name
    train = cls(str(root), train=True, download=True, transform=tf)
    val = cls(str(root), train=False, download=True, transform=tf)
    info = DataInfo(shape, "float", "classification", list(train.classes), pre,
                    n_train=len(train), n_val=len(val),
                    description=f"{name}: {len(train)} train / {len(val)} test images")
    return train, val, info


def limit(ds: Dataset, n: int | None) -> Dataset:
    """Use only the first n samples (quick experiments)."""
    if not n or n >= len(ds):
        return ds
    return Subset(ds, range(n))


# ---------------------------------------------------------------------------
# Raw input -> tensor (for prediction)
# ---------------------------------------------------------------------------


def prepare_input(spec, raw: str) -> torch.Tensor:
    pre = spec.data.get("preprocess", {})
    kind = pre.get("type")
    if kind == "image" or (kind is None and len(spec.input_shape) == 3):
        if kind is None:
            c = spec.input_shape[0]
            pre = {"size": spec.input_shape[1:], "channels": c, "mean": [0.5] * c, "std": [0.5] * c}
        from PIL import Image

        p = Path(raw.strip().strip("'\"")).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        with Image.open(p) as im:
            return _image_transform(pre)(im).unsqueeze(0)
    if kind == "text":
        return torch.tensor([encode_text(raw, pre["vocab"], pre["seq_len"])])
    if kind == "corpus":
        from .textgen import _context, encode

        return torch.tensor([_context(pre, encode(raw, pre["vocab"], pre["tokenizer"]))])
    # tabular / generic numbers
    parts = [v.strip() for v in re.split(r"[,\s;]+", raw.strip()) if v.strip()]
    feats = pre.get("features")
    n = int(torch.tensor(spec.input_shape).prod())
    if len(parts) != n:
        hint = f" ({', '.join(feats)})" if feats else ""
        raise ValueError(f"Expected {n} values{hint}, got {len(parts)}")
    vals = []
    cats = pre.get("categorical", {})
    for i, v in enumerate(parts):
        name = feats[i] if feats else None
        if name in cats:
            if v not in cats[name]:
                raise ValueError(f"'{v}' is not a known value for {name}: {', '.join(cats[name])}")
            vals.append(float(cats[name].index(v)))
        else:
            vals.append(float(v))
    x = torch.tensor(vals)
    if "mean" in pre:
        x = (x - torch.tensor(pre["mean"])) / torch.tensor(pre["std"])
    return x.reshape(1, *spec.input_shape)
