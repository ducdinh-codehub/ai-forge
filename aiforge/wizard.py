""""Just tell me what you want": turn a goal in plain words and a data file into a ready plan.

The person writes what the model should do ("sort photos of cats and dogs", "predict house
prices") and points at their data. ``plan`` looks at both and picks everything the New-model
dialog and the Train screen would otherwise ask for: the learning style, the template, the label
column and the training settings, each with a reason in plain words.

It reads words, not minds: simple keyword rules, then the data decides (a numeric label with
many values is a number to predict, a folder of sub-folders is classes of images...). Every
choice can still be changed afterwards in the builder and on the Train screen.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import data as datamod
from .spec import TEMPLATES, ModelSpec, from_template, safe_name


class PlanError(ValueError):
    """The goal and data don't give enough to go on; the message says what to add.
    ``suggestions``: keys of famous datasets (see hub.CATALOG) that fit the goal, best first.
    ``needs_data``: nothing is wrong yet, the data just hasn't been chosen."""

    def __init__(self, msg: str, suggestions: list[str] | None = None, needs_data: bool = False):
        super().__init__(msg)
        self.suggestions = suggestions or []
        self.needs_data = needs_data


# -- reading the goal --------------------------------------------------------------------------

# Word stems (see _stem) that hint at what the model should do.
HINTS = {
    "regression": {"price", "cost", "value", "amount", "much", "many", "estimate", "forecast",
                   "number", "sale", "salary", "income", "temperature", "age", "weight", "height",
                   "score", "rating", "revenue", "demand", "quantity", "duration", "time", "predict"},
    "classification": {"sort", "classify", "classifie", "category", "categorise", "categorize",
                       "detect", "recognise", "recognize", "tell", "which", "whether", "spam",
                       "sentiment", "positive", "negative", "label", "kind", "type", "identify", "yes"},
    "clustering": {"group", "cluster", "segment", "similar", "together", "bucket"},
    "autoencoder": {"unusual", "odd", "oddity", "oddities", "anomaly", "anomalie", "outlier", "strange",
                    "weird", "abnormal", "rare", "fraud"},
    "pca": {"map", "visualise", "visualize", "visualisation", "visualization", "reduce", "compress",
            "dimension", "plot", "overview"},
    "language_model": {"write", "generate", "story", "storie", "poem", "lyric", "song", "continue",
                       "imitate", "chatbot", "invent"},
    "reinforcement": {"game", "play", "agent", "reward", "balance", "pole", "cartpole", "maze", "grid",
                      "gridworld", "navigate", "trap"},
}
IMAGE_WORDS = {"photo", "image", "picture", "pic", "img", "jpg", "png", "scan", "x-ray", "xray", "camera"}
TEXT_WORDS = {"text", "review", "email", "message", "sms", "tweet", "comment", "sentence", "headline",
              "article", "post", "word", "document", "news"}
FAST_WORDS = {"fast", "quick", "quickly", "small", "tiny", "light", "laptop", "simple"}
ACCURATE_WORDS = {"accurate", "best", "precise", "good", "high", "quality", "strong"}
# Built-in data for goals that come without a file.
BUILTIN = [({"digit", "handwritten", "handwriting", "mnist"}, "mnist"),
           ({"clothe", "clothing", "fashion", "shoe", "shirt", "dress"}, "fashion_mnist"),
           ({"cifar", "airplane", "aeroplane", "vehicle", "truck", "ship", "frog"}, "cifar10")]
STOP = {"a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "from", "my", "our", "your", "me",
        "i", "we", "it", "is", "are", "be", "that", "this", "these", "with", "by", "into", "want", "need",
        "would", "like", "should", "can", "could", "model", "ai", "make", "build", "train", "learn",
        "some", "out", "what", "if", "its", "s", "please", "help", "using", "use", "based", "them", "will",
        # verbs: the name says what it works on ("house-prices", not "predict-house-prices")
        "sort", "predict", "classify", "find", "detect", "tell", "write", "generate", "recognise",
        "recognize", "spot", "estimate", "forecast", "group", "apart", "which", "whether", "how"}
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}
# Column names that usually hold the answer to learn.
LABEL_NAMES = {"label", "labels", "target", "class", "category", "y", "output", "outcome", "result",
               "answer", "sentiment", "species", "type", "diagnosis", "tag", "price", "saleprice"}


def _stem(word: str) -> str:
    """Crude singular: prices → price, categories → categorie (good enough to compare words)."""
    word = word.lower()
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "ie"
    if len(word) > 3 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list[str]:
    """Lower-case word stems; camelCase and snake_case names are split (SalePrice → sale, price)."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [_stem(w) for w in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text)]


@dataclass
class Goal:
    text: str
    words: set[str]
    order: list[str]  # the stems in the order written
    tasks: list[str]  # hinted tasks, strongest first
    image: bool = False
    text_data: bool = False
    fast: bool = False
    accurate: bool = False
    groups: int | None = None  # "into 5 groups"
    builtin: str | None = None  # a famous dataset or environment it names

    @property
    def task(self) -> str | None:
        return self.tasks[0] if self.tasks else None


def read_goal(text: str) -> Goal:
    words = _words(text)
    ws = set(words)
    scores = {task: len(ws & hints) for task, hints in HINTS.items()}
    # "predict" alone says supervised, not which kind: then the label column decides.
    if scores["regression"] == 1 and "predict" in ws:
        scores["regression"] = 0
    if "how" in ws and ws & {"much", "many", "long"}:
        scores["regression"] += 2
    if ws & {"like", "style"} and ws & {"write", "generate", "sound"}:
        scores["language_model"] += 1
    tasks = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0]
    groups = None
    for i, w in enumerate(words):
        n = int(w) if w.isdigit() else NUMBER_WORDS.get(w)
        if n and 2 <= n <= 50 and i + 1 < len(words) and words[i + 1] in ("group", "cluster", "segment",
                                                                              "bucket", "type", "kind"):
            groups = n
    builtin = next((name for keys, name in BUILTIN if ws & keys), None)
    if ws & {"cartpole", "pole", "balance"}:
        builtin = "cartpole"
    elif ws & {"maze", "gridworld", "grid"} and "reinforcement" in tasks:
        builtin = "gridworld"
    return Goal(text, ws, words, tasks, image=bool(ws & IMAGE_WORDS), text_data=bool(ws & TEXT_WORDS),
                fast=bool(ws & FAST_WORDS), accurate=bool(ws & ACCURATE_WORDS), groups=groups,
                builtin=builtin)


# -- the plan ----------------------------------------------------------------------------------

WHY_TEMPLATE = {
    "MLP classifier (tabular)": "a small fully-connected network, the standard choice for tables",
    "MLP regressor (tabular)": "a small fully-connected network that outputs one number",
    "Small CNN (28×28 grayscale, e.g. MNIST)": "a small convolutional network for small grey images",
    "CNN (32×32 colour, e.g. CIFAR-10)": "a convolutional network, the standard choice for small images",
    "Deep CNN (64×64 colour photos)": "a deeper convolutional network for photos, trained from scratch",
    "Transfer learning (MobileNetV3, fast)":
        "starts from a network that already learned from 1.2 million photos, so a few hundred "
        "of yours are enough (downloads about 10 MB the first time)",
    "Transfer learning (EfficientNet-B0, accurate)":
        "starts from a strong network pre-trained on 1.2 million photos: the most accurate choice "
        "(downloads about 20 MB the first time; slower without a GPU)",
    "Bag of words (Embedding + average, fastest)":
        "looks at which words appear: fast, and good for short or few texts",
    "Text Transformer classifier": "reads the words in context with self-attention",
    "Tiny GPT (characters, fast)": "a small GPT that learns to write letter by letter",
    "GPT-small (words, needs more text)": "a GPT that writes word by word: more fluent, needs more text",
    "k-means clustering (4 groups)": "k-means: a classic algorithm that finds groups in seconds",
    "PCA (2 components, for a 2-D map)": "PCA: squeezes every example into 2 numbers you can draw as a map",
    "Autoencoder (tables, 2-number code)":
        "squeezes each row through 2 numbers and rebuilds it; rows it rebuilds badly are the unusual ones",
    "Conv autoencoder (28×28 grayscale)":
        "squeezes each image and rebuilds it; images it rebuilds badly are the unusual ones",
    "CartPole agent (DQN)": "a small network that scores each move (a DQN agent)",
    "GridWorld agent (DQN)": "a small network that scores each move (a DQN agent)",
}
TASK_WORDS = {"classification": "sort examples into classes", "regression": "predict a number",
              "language_model": "write text like yours", "clustering": "find groups by itself",
              "pca": "draw a 2-D map of the data", "autoencoder": "spot unusual examples",
              "reinforcement": "learn to act by trial and error"}


@dataclass
class Plan:
    goal: str
    name: str
    template: str
    task: str
    data_cfg: dict
    train: dict = field(default_factory=dict)  # Train-screen fields: epochs, batch_size, lr, early_stop
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)  # a table's columns, to change the label
    label: str | None = None  # the chosen label column of a table
    groups: int | None = None  # k-means: how many groups
    settings: list[str] = field(default_factory=list)  # each training setting: what it does, why this value

    def spec(self, name: str | None = None) -> ModelSpec:
        """The model to create. It remembers the data, so the Train screen opens with it."""
        spec = from_template(self.template, name or self.name)
        if self.groups:
            for layer in spec.layers:
                if layer.type == "KMeans":
                    layer.config["k"] = self.groups
        spec.data["dataset_cfg"] = copy.deepcopy(self.data_cfg)
        spec.notes = f"Goal: {self.goal}"
        return spec

    @property
    def title(self) -> str:
        """The template's name, with the number of groups it will really look for."""
        return re.sub(r"\(\d+ groups\)", f"({self.groups} groups)", self.template) if self.groups else self.template

    def summary(self) -> str:
        """The plan in plain words (Rich markup)."""
        lines = [f"[b]Plan:[/b] {TASK_WORDS.get(self.task, self.task)} with [b]{self.title}[/b]"]
        lines += [f"  • {r}" for r in self.reasons]
        lines += [f"  [yellow]⚠ {w}[/]" for w in self.warnings]
        if self.settings:
            lines += ["", "[b]Training settings[/b]: what each one means, and why this value"]
            lines += [f"  • {s}" for s in self.settings]
            if self.train:
                lines.append("  [dim]Change any of them on the Train screen. Guide (g) → lesson 6 explains every "
                             "setting; F1 on the Train screen explains them with your values.[/]")
        return "\n".join(lines)


def plan(goal: str, path: str | None = None, label: str | None = None) -> Plan:
    """Pick a template, data settings and training settings for ``goal`` on the data at ``path``.

    ``label``: a table column to learn, chosen by the person (else it is guessed)."""
    g = read_goal(goal or "")
    if not g.words and not path:
        raise PlanError("Describe what the model should do, e.g. \"sort photos of cats and dogs\".")
    if path and path.strip().lower() in datamod.TORCHVISION:  # "mnist", picked in the dataset hub
        g.builtin = path.strip().lower()
        out = _builtin_plan(g)
    elif path and path.strip():
        p = Path(path.strip()).expanduser()
        if not p.exists():
            raise PlanError(f"Can't find {p}. Check the path, or press Browse.")
        if p.is_dir():
            out = _image_folder_plan(g, p)
        elif p.suffix.lower() in datamod.TEXT_EXT:
            out = _corpus_plan(g, p)
        elif p.suffix.lower() in datamod.TABLE_EXT:
            out = _table_plan(g, p, label)
        else:
            raise PlanError(f"AIForge reads tables (.csv, .xlsx, .parquet), text (.txt) and folders of "
                            f"images, not {p.suffix or 'this kind of'} files.")
    else:
        out = _builtin_plan(g)
    out.goal = goal.strip()
    out.name = _name(g, path)
    return out


def _name(g: Goal, path: str | None) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", g.text.lower()) if w not in STOP and len(w) > 1]
    if not words and path:
        words = [Path(path.strip()).expanduser().stem]
    return safe_name("-".join(words[:3]))[:40] or "my-model"


VAL_SPLIT = 0.2  # the Train screen's default: the share of the data kept aside to test the model


def _settings(n: int, kind: str) -> tuple[dict, list[str]]:
    """Training settings for ``n`` examples of a kind of model, each explained in plain words:
    what the setting does and why this value suits this data."""
    if kind == "classic":
        return {}, ["No training settings to choose: a classic algorithm is fitted to the data in seconds, "
                    "not trained over many passes."]
    if kind == "agent":
        return {"epochs": 300}, [
            "[b]Episodes 300[/b]: an episode is one game from start to end. The agent plays 300, "
            "trying random moves at first and its own best moves more and more.",
            "[b]Learning rate 0.001[/b]: how big each correction of the agent's network is; the safe "
            "start for the Adam optimizer.",
            "AIForge tests the agent every 10 episodes and keeps the best tested version."]
    if kind == "table":
        epochs, batch = (60, 32) if n < 1000 else (30, 64) if n < 20000 else (15, 128)
        cfg = dict(epochs=epochs, batch_size=batch, lr=0.001, early_stop=8)
        size = "a small table" if n < 1000 else "a medium table" if n < 20000 else "a big table"
        epochs_why = (f"{size}, so it needs many passes to learn" if n < 1000 else
                      f"{size}: each pass already teaches it a lot")
    elif kind == "transfer":
        cfg = dict(epochs=10, batch_size=32, lr=0.0003, early_stop=3)
        epochs_why = "the pre-trained network already knows what photos look like, so a few passes are enough"
    elif kind == "image":
        cfg = dict(epochs=30 if n < 5000 else 15, batch_size=32 if n < 5000 else 64, lr=0.001, early_stop=5)
        epochs_why = ("few images, so it needs more passes over them" if n < 5000 else
                      "many images: each pass already teaches it a lot")
    elif kind == "text":
        cfg = dict(epochs=20 if n < 5000 else 10, batch_size=32, lr=0.001, early_stop=4)
        epochs_why = "texts are learned quickly; more passes mostly memorise them"
    elif kind == "language_model":
        cfg = dict(epochs=10, batch_size=64, lr=0.001, early_stop=3)
        epochs_why = "after a few passes a small text model starts copying the text instead of learning its style"
    elif kind == "autoencoder":
        cfg = dict(epochs=40 if n < 5000 else 20, batch_size=64, lr=0.001, early_stop=5)
        epochs_why = "rebuilding examples well takes a good number of passes"
    else:
        return {}, []
    batch = cfg["batch_size"]
    what = "characters" if kind == "language_model" else "examples"
    lines = [f"[b]Epochs {cfg['epochs']}[/b]: how many times the model goes through all your training "
             f"{what}. Why {cfg['epochs']}: {epochs_why}."]
    if kind == "language_model" or not n:
        lines.append(f"[b]Batch size {batch}[/b]: how many pieces it looks at together before each small "
                     "correction. Bigger batches are faster; smaller ones correct more often.")
    else:
        n_train = n - max(1, int(n * VAL_SPLIT))
        steps = max(1, -(-n_train // batch))
        why = ("small data, so smaller batches give it more corrections per pass" if batch == 32 else
               "a big dataset, so bigger batches keep each pass fast" if batch >= 128 else
               "a common middle value: fast, with enough corrections per pass")
        lines.append(f"[b]Batch size {batch}[/b]: how many examples it looks at together before each small "
                     f"correction, here {steps:,} corrections per pass. Why {batch}: {why}.")
    lines.append(f"[b]Learning rate {cfg['lr']:g}[/b]: how big each correction is, the most important "
                 "setting. Too big and learning jumps around; too small and it crawls. " +
                 ("Why 0.0003: smaller than the usual 0.001, so the pre-trained network's knowledge isn't "
                  "overwritten." if kind == "transfer" else "Why 0.001: the safe start for the Adam optimizer."))
    lines.append(f"[b]Validation split {VAL_SPLIT:g}[/b]: {VAL_SPLIT:.0%} of your data is kept aside and never "
                 "trained on. It's the honest test of how the model does on data it hasn't seen.")
    lines.append(f"[b]Early stop {cfg['early_stop']}[/b]: if the score on that kept-aside data hasn't improved "
                 f"for {cfg['early_stop']} epochs in a row, training stops. Either way AIForge keeps the best "
                 "epoch, not the last.")
    return cfg, lines


def _plan(template: str, task: str, data_cfg: dict, n: int, kind: str, reasons: list[str],
          warnings: list[str] | None = None, **extra) -> Plan:
    train, settings = _settings(n, kind)
    reasons = reasons + [f"Model: {WHY_TEMPLATE.get(template, 'a good starting point')}."]
    return Plan("", "", template, task, data_cfg, train, reasons, warnings or [], settings=settings, **extra)


def _unsupervised(g: Goal) -> str | None:
    return next((t for t in g.tasks if t in ("clustering", "autoencoder", "pca")), None)


# -- tables ------------------------------------------------------------------------------------


@dataclass
class Column:
    name: str
    numeric: bool
    distinct: int
    avg_words: float
    image_paths: bool


def _columns(header: list[str], rows: list[list[str]]) -> list[Column]:
    rows = [r for r in rows if len(r) == len(header)]
    cols = []
    for i, h in enumerate(header):
        vals = [r[i].strip() for r in rows if r[i].strip()]
        numeric = bool(vals) and all(datamod._is_num(v) for v in vals)
        avg_words = sum(len(v.split()) for v in vals) / len(vals) if vals else 0
        images = bool(vals) and sum(Path(v).suffix.lower() in datamod.IMAGE_EXT for v in vals) > len(vals) / 2
        cols.append(Column(h, numeric, len(set(vals)), avg_words, images))
    return cols


FEATURE_WORDS = ("from", "using", "use", "based", "given", "with")


def _match(g: Goal, names: list[str]) -> str | None:
    """The column the goal names as the answer: "predict price from area" → price. Words after
    "from" / "using" / "based on" name the inputs, so they only count when nothing comes before."""
    cut = next((i for i, w in enumerate(g.order) if w in FEATURE_WORDS), len(g.order))
    for words in (g.order[:cut], g.order):
        hits = [(min(words.index(w) for w in set(_words(n)) & set(words)), n) for n in names
                if set(_words(n)) & set(words)]
        if hits:
            return min(hits)[1]
    return None


def _id_like(c: Column, n_rows: int) -> bool:
    words = _words(c.name)
    return bool(words) and (words[-1] in ("id", "index", "uuid") or c.name.lower() in ("no", "row", "#")) \
        and c.distinct >= n_rows * 0.9


def _table_plan(g: Goal, p: Path, label: str | None) -> Plan:
    try:
        header, rows = datamod.read_table(p)
    except Exception as e:  # noqa: BLE001 - shown to the person
        raise PlanError(f"Couldn't read {p.name}: {e}") from e
    cols = _columns(header, rows)
    names = [c.name for c in cols]
    if label is not None and label not in names:
        label = None
    n = len(rows)
    about = f"{p.name} is a table: {n:,} rows, {len(cols)} columns."
    unsup = _unsupervised(g)
    image_col = next((c for c in cols if c.image_paths), None)
    text_col = max((c for c in cols if not c.numeric and c.avg_words >= 3 and not c.image_paths),
                   key=lambda c: c.avg_words, default=None)
    if unsup:
        return _table_unsupervised(g, p, cols, label, about, unsup, n)
    if image_col:
        return _image_table_plan(g, p, cols, image_col, label, about, n)
    if text_col and (g.text_data or len(cols) <= 3):  # a text and its label (a long "address" is no text to sort)
        return _text_table_plan(g, p, cols, text_col, label, about, n)

    target, why = _pick_label(g, cols, label, exclude=())
    tc = next(c for c in cols if c.name == target)
    reasons = [about, f"Label column: [b]{target}[/b] ({why}). The model learns it from the other columns."]
    warnings = []
    wants_number = g.task == "regression" or (g.task is None and tc.numeric and tc.distinct > 20)
    if wants_number and not tc.numeric:
        warnings.append(f"'{target}' holds words, not numbers, so the model sorts rows into its "
                        f"{tc.distinct} values instead of predicting a number.")
        wants_number = False
    if not wants_number and tc.distinct > 50:
        warnings.append(f"'{target}' has {tc.distinct:,} different values: that's many classes to learn. "
                        "Is it the right label column?")
    task = "regression" if wants_number else "classification"
    reasons.append("It's a number to predict, so this is regression: the model outputs a value."
                   if task == "regression" else
                   f"It has {tc.distinct} values, so this is classification: the model picks one of them.")
    warnings += _id_warnings(cols, n, target)
    if n < 50:
        warnings.append(f"Only {n} rows: the model may just memorise them. A few hundred rows work much better.")
    template = "MLP regressor (tabular)" if task == "regression" else "MLP classifier (tabular)"
    cfg = {"source": "csv", "path": str(p.resolve()), "target": target, "task": task}
    return _plan(template, task, cfg, n, "table", reasons, warnings, columns=names, label=target)


def _pick_label(g: Goal, cols: list[Column], label: str | None, exclude) -> tuple[str, str]:
    names = [c.name for c in cols if c.name not in exclude]
    if label in names:
        return label, "you chose it"
    if hit := _match(g, names):
        return hit, "your goal mentions it"
    if hit := next((n for n in names if n.strip().lower() in LABEL_NAMES), None):
        return hit, "its name says it's the answer"
    return names[-1], "the last column; change it below if that's wrong"


def _id_warnings(cols: list[Column], n: int, target: str | None) -> list[str]:
    ids = [c.name for c in cols if c.name != target and _id_like(c, n)]
    return [f"'{c}' looks like an ID number. Every other column is used as an input, and an ID only "
            "lets the model memorise rows: remove that column from the file for a better model."
            for c in ids]


def _text_table_plan(g: Goal, p, cols, text_col, label, about, n) -> Plan:
    target, why = _pick_label(g, cols, label, exclude=(text_col.name,))
    tc = next(c for c in cols if c.name == target)
    reasons = [about, f"Text column: [b]{text_col.name}[/b] (about {text_col.avg_words:.0f} words each).",
               f"Label column: [b]{target}[/b] ({why}), {tc.distinct} values: the model sorts each text "
               "into one of them."]
    warnings = []
    if g.task == "regression":
        warnings.append("AIForge sorts texts into classes; it doesn't predict numbers from text yet.")
    if tc.distinct > 50:
        warnings.append(f"'{target}' has {tc.distinct:,} different values. Is it the right label column?")
    template = ("Bag of words (Embedding + average, fastest)" if g.fast or n < 2000 and not g.accurate
                else "Text Transformer classifier")
    cfg = {"source": "csv_text", "path": str(p.resolve()), "text_column": text_col.name, "target": target}
    return _plan(template, "classification", cfg, n, "text", reasons, warnings,
                 columns=[c.name for c in cols if c.name != text_col.name], label=target)


def _image_table_plan(g: Goal, p, cols, image_col, label, about, n) -> Plan:
    target, why = _pick_label(g, cols, label, exclude=(image_col.name,))
    reasons = [about, f"Image column: [b]{image_col.name}[/b] (paths to image files).",
               f"Label column: [b]{target}[/b] ({why}): the model sorts each image into its values."]
    template, kind = _image_template(g, None, n)
    cfg = {"source": "image_csv", "path": str(p.resolve()), "image_column": image_col.name, "target": target}
    return _plan(template, "classification", cfg, n, kind, reasons,
                 columns=[c.name for c in cols if c.name != image_col.name], label=target)


def _table_unsupervised(g: Goal, p, cols, label, about, task, n) -> Plan:
    numeric_or_cat = [c for c in cols if c.avg_words < 3 and not c.image_paths]
    if len(numeric_or_cat) < len(cols):
        raise PlanError("Finding groups, oddities or a map works on tables of numbers and categories, "
                        "not on long text or image paths.")
    names = [c.name for c in cols]
    kept = label if label in names else next((c for c in names if c.strip().lower() in LABEL_NAMES), None)
    reasons = [about, "Every column is an input; no right answers are needed."]
    if kept:
        reasons.append(f"[b]{kept}[/b] is kept aside, only to check the result afterwards, never learned from.")
    template = {"clustering": "k-means clustering (4 groups)", "pca": "PCA (2 components, for a 2-D map)",
                "autoencoder": "Autoencoder (tables, 2-number code)"}[task]
    groups = None
    if task == "clustering":
        groups = g.groups or 4
        reasons.append(f"It looks for {groups} groups" + (" (from your goal)." if g.groups else
                       ": try other numbers with the KMeans layer's k, Evaluate shows how well they separate."))
    cfg = {"source": "csv", "path": str(p.resolve()), **({"target": kept} if kept else {})}
    return _plan(template, task, cfg, n, "autoencoder" if task == "autoencoder" else "classic", reasons,
                 _id_warnings(cols, n, kept), columns=names, label=kept, groups=groups)


# -- images ------------------------------------------------------------------------------------


def _image_size(path: Path) -> tuple[int, int, str] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.width, im.height, im.mode
    except Exception:  # noqa: BLE001 - a broken file just gives no hint
        return None


def _image_template(g: Goal, sample: Path | None, n: int) -> tuple[str, str]:
    size = _image_size(sample) if sample else None
    small = size is not None and max(size[:2]) <= 40
    grey = size is not None and size[2] in ("L", "1", "LA", "I;16")
    if small and grey:
        return "Small CNN (28×28 grayscale, e.g. MNIST)", "image"
    if small:
        return "CNN (32×32 colour, e.g. CIFAR-10)", "image"
    if g.accurate and not g.fast:
        return "Transfer learning (EfficientNet-B0, accurate)", "transfer"
    if n <= 20000:
        return "Transfer learning (MobileNetV3, fast)", "transfer"
    return "Deep CNN (64×64 colour photos)", "image"


def _image_folder_plan(g: Goal, root: Path) -> Plan:
    classes = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    counts = {d.name: sum(1 for f in d.rglob("*") if f.suffix.lower() in datamod.IMAGE_EXT) for d in classes}
    counts = {k: v for k, v in counts.items() if v}
    unsup = _unsupervised(g)
    if len(counts) >= 2 and not unsup:
        n = sum(counts.values())
        sample = next(f for f in (root / next(iter(counts))).rglob("*") if f.suffix.lower() in datamod.IMAGE_EXT)
        shown = ", ".join(f"{k} ({v})" for k, v in list(counts.items())[:6]) + ("…" if len(counts) > 6 else "")
        reasons = [f"{root.name} has {n:,} images in {len(counts)} sub-folders: {shown}.",
                   "Each sub-folder is a class, so this is classification: the model learns to tell them apart."]
        warnings = []
        if min(counts.values()) < 20:
            warnings.append(f"'{min(counts, key=counts.get)}' has only {min(counts.values())} images. "
                            "Aim for at least 50 per class.")
        if max(counts.values()) > 5 * min(counts.values()):
            warnings.append("Some classes have far more images than others; the model will favour them.")
        template, kind = _image_template(g, sample, n)
        if kind == "transfer":
            reasons.append("Photos are resized to 224×224, the size the pre-trained network knows.")
        cfg = {"source": "image_folder", "path": str(root.resolve())}
        return _plan(template, "classification", cfg, n, kind, reasons, warnings)

    images = [f for f in root.rglob("*") if f.suffix.lower() in datamod.IMAGE_EXT]
    if not images:
        raise PlanError(f"No images (.png, .jpg…) found in {root.name}.")
    n = len(images)
    task = unsup or "clustering"
    reasons = [f"{root.name} has {n:,} images" + ("." if len(counts) < 2 else
               f"; its {len(counts)} sub-folder names are kept aside, only to check the result.")]
    warnings = []
    if not unsup:
        reasons.append("There are no sub-folders, so no labels: the model finds groups by itself.")
        warnings.append("To teach it named classes instead, put each class in its own sub-folder "
                        "(e.g. cats/ and dogs/) and make a new plan.")
    if task == "autoencoder":
        template = "Conv autoencoder (28×28 grayscale)"
        reasons.append("Images are shrunk to 28×28 grey, enough to spot ones that look different.")
    elif task == "pca":
        template = "PCA (2 components, for a 2-D map)"
    else:
        template = "k-means clustering (4 groups)"
    groups = (g.groups or 4) if task == "clustering" else None
    if groups:
        reasons.append(f"It looks for {groups} groups" + (" (from your goal)." if g.groups else "."))
    cfg = {"source": "image_folder", "path": str(root.resolve())}
    return _plan(template, task, cfg, n, "autoencoder" if task == "autoencoder" else "classic", reasons,
                 warnings, groups=groups)


# -- text to imitate ---------------------------------------------------------------------------


def _corpus_plan(g: Goal, p: Path) -> Plan:
    if _unsupervised(g):
        raise PlanError("Finding groups, oddities or a map works on tables and images, not plain text.")
    try:
        text = datamod.read_text(p)
    except Exception as e:  # noqa: BLE001
        raise PlanError(f"Couldn't read {p.name}: {e}") from e
    size = len(text)
    reasons = [f"{p.name} is plain text ({size:,} characters): the model learns to write like it, "
               "one piece at a time, by predicting what comes next."]
    warnings = []
    if g.task not in (None, "language_model"):
        warnings.append("A .txt file can only teach the model to write. To sort texts into classes, use a "
                        "table with a text column and a label column.")
    words = bool(g.words & {"word", "sentence"}) and size >= 1_000_000
    template = "GPT-small (words, needs more text)" if words else "Tiny GPT (characters, fast)"
    if size < 100_000:
        warnings.append("That's a short text: the model learns its look (lines, rhymes, common words) but "
                        "not real sentences. 1 MB or more works much better.")
    tokenizer = TEMPLATES[template]["tokenizer"]
    cfg = {"source": "text_corpus", "path": str(p.resolve()), "tokenizer": tokenizer}
    return _plan(template, "language_model", cfg, size, "language_model", reasons, warnings)


# -- no file -----------------------------------------------------------------------------------


def _builtin_plan(g: Goal) -> Plan:
    if g.builtin in ("cartpole", "gridworld") or g.task == "reinforcement":
        env = g.builtin if g.builtin in ("cartpole", "gridworld") else "gridworld"
        template = "CartPole agent (DQN)" if env == "cartpole" else "GridWorld agent (DQN)"
        world = ("CartPole: keep a pole upright by moving the cart left and right" if env == "cartpole" else
                 "GridWorld: walk to the goal and avoid the traps")
        reasons = [f"No data needed: the agent learns in a built-in world, {world}.",
                   "Good moves earn rewards, so it gets better by trial and error (reinforcement learning)."]
        cfg = {"source": "env", "env": env, **TEMPLATES[template].get("env", {})}
        return _plan(template, "reinforcement", cfg, 0, "agent", reasons)
    if g.builtin in datamod.TORCHVISION:
        what = {"mnist": "MNIST: 70,000 handwritten digits", "fashion_mnist":
                "Fashion-MNIST: 70,000 pictures of clothes in 10 kinds",
                "cifar10": "CIFAR-10: 60,000 small photos of 10 kinds of objects",
                "cifar100": "CIFAR-100: 60,000 small photos of 100 kinds of objects"}[g.builtin]
        template = "CNN (32×32 colour, e.g. CIFAR-10)" if g.builtin.startswith("cifar") else \
            "Small CNN (28×28 grayscale, e.g. MNIST)"
        reasons = [f"Uses the famous {what} (downloads by itself the first time)."]
        return _plan(template, "classification", {"source": g.builtin}, 60000, "image", reasons)
    if g.task == "language_model":
        reasons = ["No text given, so it starts on the bundled nursery rhymes. For your own style, point at "
                   "a .txt file; for real sentences, get Tiny Shakespeare from 🗄 Datasets."]
        cfg = {"source": "text_corpus", "path": str(datamod.EXAMPLE_CORPUS), "tokenizer": "char"}
        return _plan("Tiny GPT (characters, fast)", "language_model", cfg, 0, "language_model", reasons)
    hint = {"image": "a folder of images with one sub-folder per class (e.g. cats/ and dogs/)",
            "text": "a table with a text column and a label column"}
    own = hint.get("image" if g.image else "text" if g.text_data else "",
                   "a table (.csv, .xlsx), a folder of images or a .txt file")
    fits = suggest_datasets(g.text)
    if fits:
        from . import hub

        names = ", ".join(hub.BY_KEY[k].name for k in fits)
        raise PlanError(f"Now choose your data: 📂 Browse for your own ({own}), or 🗄 get a famous dataset "
                        f"that fits your goal: {names}.", fits, needs_data=True)
    raise PlanError(f"Now choose your data: 📂 Browse for your own ({own}), or 🗄 get a famous dataset.",
                    needs_data=True)


# Words too common in the dataset descriptions to say which one fits.
_GENERIC = {_stem(w) for w in STOP} | {"predict", "classe", "class", "classification", "data", "dataset",
                                       "label", "labelled", "real", "small", "model", "regression", "one"} \
    | IMAGE_WORDS | TEXT_WORDS  # the kind of data counts once, below


def suggest_datasets(goal: str, limit: int = 3) -> list[str]:
    """Famous datasets (hub keys) that fit a goal, best first: shared words, then the right kind."""
    from . import hub

    g = read_goal(goal)
    words = g.words - _GENERIC
    scored = []
    for i, d in enumerate(hub.CATALOG):
        score = len(words & set(_words(f"{d.name} {d.summary} {d.task}")))
        if not score:
            continue
        kind = (d.group == hub.VISION and g.image or d.group == hub.TEXT and g.text_data
                or d.group == hub.LANGUAGE_MODEL and g.task == "language_model"
                or d.task == "regression" and g.task == "regression")
        scored.append((-(score + bool(kind)), i, d.key))
    return [key for *_, key in sorted(scored)[:limit]]
