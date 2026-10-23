"""Dataset hub: a short list of well-known, trustworthy public datasets, grouped by model type.

Every dataset is downloaded from its official home (or a long-standing mirror run by the
original authors' community) and converted once into a format the trainer already reads:
a CSV table, a CSV with a text column, an image folder (one sub-folder per class), or one of
the built-in torchvision sources. ``train_cfg`` returns the data settings for the Train screen.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# progress(stage, done, total). total is None when the size is unknown.
Progress = Callable[[str, int, "int | None"], None]

MARKER = ".aiforge-dataset.json"
USER_AGENT = "aiforge-studio dataset hub (+https://github.com/ducdinh-codehub/ai-forge)"


class Cancelled(Exception):
    pass


@dataclass
class HubDataset:
    key: str
    group: str
    name: str
    summary: str
    source: str  # who publishes it
    url: str  # homepage, shown to the user
    license: str
    size: str  # download size
    samples: str
    task: str  # what the model learns
    templates: list[str]
    format: str  # "csv" | "csv_text" | "image_folder" | "text_corpus" | a torchvision source
    target: str | None = None
    text_column: str | None = None
    task_kind: str | None = None  # force "classification"/"regression" for CSV tables
    notes: str = ""
    fetch: Callable | None = field(default=None, repr=False)  # fetch(dest, progress, cancel) -> str


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, progress: Progress, cancel: Callable[[], bool],
              stage: str = "Downloading") -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        done = 0
        while chunk := r.read(1 << 16):
            if cancel():
                f.close()
                tmp.unlink(missing_ok=True)
                raise Cancelled()
            f.write(chunk)
            done += len(chunk)
            progress(stage, done, total)
    tmp.replace(dest)
    return dest


def _fetch_text(url: str, dest: Path, progress, cancel) -> str:
    return _download(url, dest / "raw.tmp", progress, cancel).read_text(encoding="utf-8", errors="replace")


def _write_csv(path: Path, header: list[str], rows) -> int:
    n = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
            n += 1
    return n


def _lines(text: str, sep: str = ","):
    for line in text.splitlines():
        if line.strip():
            yield [c.strip() for c in line.split(sep)]


def _check_count(n: int, expected: int, what: str) -> None:
    # Guard against a moved/changed upstream file silently producing a broken dataset.
    if n < expected * 0.95:
        raise ValueError(f"Expected about {expected:,} {what}, got {n:,} - the source may have changed")


UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


# ---------------------------------------------------------------------------
# Tabular
# ---------------------------------------------------------------------------


def _iris(dest, progress, cancel):
    text = _fetch_text(UCI + "iris/iris.data", dest, progress, cancel)
    rows = [r[:4] + [r[4].removeprefix("Iris-")] for r in _lines(text) if len(r) == 5]
    n = _write_csv(dest / "iris.csv", ["sepal_length", "sepal_width", "petal_length", "petal_width",
                                       "species"], rows)
    _check_count(n, 150, "rows")
    return "iris.csv"


WINE_COLS = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols",
             "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue",
             "od280_od315", "proline"]


def _wine(dest, progress, cancel):
    text = _fetch_text(UCI + "wine/wine.data", dest, progress, cancel)
    rows = [r[1:] + [f"cultivar_{r[0]}"] for r in _lines(text) if len(r) == 14]
    n = _write_csv(dest / "wine.csv", WINE_COLS + ["cultivar"], rows)
    _check_count(n, 178, "rows")
    return "wine.csv"


def _breast_cancer(dest, progress, cancel):
    text = _fetch_text(UCI + "breast-cancer-wisconsin/wdbc.data", dest, progress, cancel)
    base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",
            "concave_points", "symmetry", "fractal_dimension"]
    cols = [f"{b}_{s}" for s in ("mean", "se", "worst") for b in base]
    label = {"M": "malignant", "B": "benign"}
    rows = [r[2:] + [label[r[1]]] for r in _lines(text) if len(r) == 32]  # drop the patient id
    n = _write_csv(dest / "breast_cancer.csv", cols + ["diagnosis"], rows)
    _check_count(n, 569, "rows")
    return "breast_cancer.csv"


def _adult(dest, progress, cancel):
    text = _fetch_text(UCI + "adult/adult.data", dest, progress, cancel)
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"]
    rows = [r for r in _lines(text) if len(r) == 15]
    n = _write_csv(dest / "adult.csv", cols, rows)
    _check_count(n, 32561, "rows")
    return "adult.csv"


def _wine_quality(dest, progress, cancel):
    text = _fetch_text(UCI + "wine-quality/winequality-red.csv", dest, progress, cancel)
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    header = [h.replace(" ", "_") for h in rows[0]]
    n = _write_csv(dest / "wine_quality_red.csv", header, rows[1:])
    _check_count(n, 1599, "rows")
    return "wine_quality_red.csv"


def _california(dest, progress, cancel):
    text = _fetch_text("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/"
                       "housing.csv", dest, progress, cancel)
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    t = header.index("median_house_value")  # move the target to the last column
    order = [i for i in range(len(header)) if i != t] + [t]
    n = _write_csv(dest / "california_housing.csv", [header[i] for i in order],
                   ([r[i] for i in order] for r in body if len(r) == len(header)))
    _check_count(n, 20640, "rows")
    return "california_housing.csv"


# ---------------------------------------------------------------------------
# Computer vision
# ---------------------------------------------------------------------------


def _torchvision(name: str):
    def fetch(dest, progress, cancel):
        from torchvision import datasets

        cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST,
               "cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
        progress("Downloading (torchvision, checksum-verified)", 0, None)
        for train in (True, False):
            cls(str(dest), train=train, download=True)
        return ""
    return fetch


def _save_member(tf: tarfile.TarFile, member: tarfile.TarInfo, out: Path) -> None:
    src = tf.extractfile(member)
    if src is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with src, out.open("wb") as f:
        shutil.copyfileobj(src, f)


IMAGENETTE = {"n01440764": "tench", "n02102040": "English_springer", "n02979186": "cassette_player",
              "n03000684": "chain_saw", "n03028079": "church", "n03394916": "French_horn",
              "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball",
              "n03888257": "parachute"}


def _imagenette(dest, progress, cancel):
    archive = _download("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
                        dest / "imagenette2-160.tgz", progress, cancel)
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            parts = m.name.split("/")  # imagenette2-160/{train,val}/<wnid>/<file>.JPEG
            if not m.isfile() or len(parts) != 4 or parts[2] not in IMAGENETTE:
                continue
            if cancel():
                raise Cancelled()
            # Merge train+val; AIForge makes its own validation split.
            _save_member(tf, m, dest / "images" / IMAGENETTE[parts[2]] / f"{parts[1]}_{parts[3]}")
            n += 1
            if n % 200 == 0:
                progress("Unpacking images", n, 13394)
    archive.unlink()
    _check_count(n, 13394, "images")
    return "images"


def _pets(dest, progress, cancel):
    archive = _download("https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
                        dest / "images.tar.gz", progress, cancel)
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]  # images/<Breed_Name>_<n>.jpg
            if not m.isfile() or not name.lower().endswith(".jpg") or "_" not in name:
                continue
            if cancel():
                raise Cancelled()
            _save_member(tf, m, dest / "images" / name.rsplit("_", 1)[0] / name)
            n += 1
            if n % 200 == 0:
                progress("Unpacking images", n, 7390)
    archive.unlink()
    _check_count(n, 7390, "images")
    return "images"


# ---------------------------------------------------------------------------
# Language / NLP
# ---------------------------------------------------------------------------


def _sms_spam(dest, progress, cancel):
    archive = _download("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
                        dest / "sms.zip", progress, cancel)
    with zipfile.ZipFile(archive) as z:
        text = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
    archive.unlink()
    rows = [line.split("\t", 1)[::-1] for line in text.splitlines() if "\t" in line]
    n = _write_csv(dest / "sms_spam.csv", ["text", "label"], rows)
    _check_count(n, 5574, "messages")
    return "sms_spam.csv"


def _sst2(dest, progress, cancel):
    archive = _download("https://dl.fbaipublicfiles.com/glue/data/SST-2.zip", dest / "sst2.zip",
                        progress, cancel)
    label = {"0": "negative", "1": "positive"}
    rows = []
    with zipfile.ZipFile(archive) as z:
        for part in ("SST-2/train.tsv", "SST-2/dev.tsv"):  # the GLUE test split has no labels
            lines = z.read(part).decode("utf-8").splitlines()[1:]
            rows += [[s.strip(), label[y.strip()]] for s, y in (l.rsplit("\t", 1) for l in lines if "\t" in l)]
    archive.unlink()
    n = _write_csv(dest / "sst2.csv", ["text", "label"], rows)
    _check_count(n, 68221, "sentences")
    return "sst2.csv"


def _imdb(dest, progress, cancel):
    archive = _download("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                        dest / "aclImdb_v1.tar.gz", progress, cancel)
    n = 0
    with tarfile.open(archive, "r:gz") as tf, (dest / "imdb.csv").open("w", newline="",
                                                                        encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for m in tf:
            parts = m.name.split("/")  # aclImdb/{train,test}/{pos,neg}/<id>.txt
            if not m.isfile() or len(parts) != 4 or parts[2] not in ("pos", "neg"):
                continue
            if cancel():
                raise Cancelled()
            text = tf.extractfile(m).read().decode("utf-8", errors="replace").replace("<br />", " ")
            w.writerow([text, "positive" if parts[2] == "pos" else "negative"])
            n += 1
            if n % 500 == 0:
                progress("Converting reviews", n, 50000)
    archive.unlink()
    _check_count(n, 50000, "reviews")
    return "imdb.csv"


def _ag_news(dest, progress, cancel):
    archive = _download("https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz", dest / "ag_news.tgz",
                        progress, cancel)
    classes = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}
    rows = []
    with tarfile.open(archive, "r:gz") as tf:
        for part in ("ag_news_csv/train.csv", "ag_news_csv/test.csv"):
            text = tf.extractfile(part).read().decode("utf-8")
            for r in csv.reader(io.StringIO(text)):
                if len(r) == 3:
                    # "\" stands for a line break in the original files
                    rows.append([f"{r[1]}. {r[2]}".replace("\\", " "), classes[r[0]]])
    archive.unlink()
    n = _write_csv(dest / "ag_news.csv", ["text", "label"], rows)
    _check_count(n, 127600, "articles")
    return "ag_news.csv"


# ---------------------------------------------------------------------------
# Plain text (language models)
# ---------------------------------------------------------------------------


def _tiny_shakespeare(dest, progress, cancel):
    out = _download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                    dest / "tiny_shakespeare.txt", progress, cancel)
    _check_count(len(out.read_text(encoding="utf-8")), 1_115_394, "characters")
    return "tiny_shakespeare.txt"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

TABULAR = "📊 Tabular data (CSV)"
VISION = "🖼 Computer vision (images)"
TEXT = "💬 Text classification (NLP)"
LANGUAGE_MODEL = "🧠 Language model (generate text)"
GROUPS = [TABULAR, VISION, TEXT, LANGUAGE_MODEL]

_MLP_C, _MLP_R = "MLP classifier (tabular)", "MLP regressor (tabular)"
_TEXT_T = ["Text Transformer classifier", "Text classifier (bidirectional GRU)",
           "Bag of words (Embedding + average, fastest)"]
_TRANSFER = ["Transfer learning (MobileNetV3, fast)", "Transfer learning (ResNet18 backbone)",
             "Transfer learning (EfficientNet-B0, accurate)"]

CATALOG: list[HubDataset] = [
    # --- tabular ---
    HubDataset("iris", TABULAR, "Iris flowers",
               "The classic first dataset: measure a flower's petals and sepals, predict its species.",
               "UCI Machine Learning Repository (R. A. Fisher, 1936)", "https://archive.ics.uci.edu/dataset/53/iris",
               "CC BY 4.0", "5 KB", "150 rows · 4 features · 3 classes", "classification", [_MLP_C],
               "csv", target="species", task_kind="classification", fetch=_iris,
               notes="Tiny - trains in seconds. Expect ~95% accuracy."),
    HubDataset("wine", TABULAR, "Wine cultivars",
               "Chemical analysis of Italian wines; predict which of 3 grape cultivars it came from.",
               "UCI Machine Learning Repository", "https://archive.ics.uci.edu/dataset/109/wine",
               "CC BY 4.0", "11 KB", "178 rows · 13 features · 3 classes", "classification", [_MLP_C],
               "csv", target="cultivar", task_kind="classification", fetch=_wine,
               notes="Small and clean. A good model reaches ~97% accuracy."),
    HubDataset("breast_cancer", TABULAR, "Breast cancer diagnosis",
               "30 measurements of cell nuclei from a biopsy image; predict benign or malignant.",
               "UCI Machine Learning Repository (Univ. of Wisconsin)",
               "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
               "CC BY 4.0", "125 KB", "569 rows · 30 features · 2 classes", "classification", [_MLP_C],
               "csv", target="diagnosis", task_kind="classification", fetch=_breast_cancer,
               notes="A well-studied medical benchmark. Expect ~95-98% accuracy. For learning only."),
    HubDataset("adult", TABULAR, "Adult census income",
               "US census records (age, education, job, hours…); predict whether income is above $50K.",
               "UCI Machine Learning Repository (US Census Bureau, 1994)",
               "https://archive.ics.uci.edu/dataset/2/adult",
               "CC BY 4.0", "3.8 MB", "32,561 rows · 14 features · 2 classes", "classification", [_MLP_C],
               "csv", target="income", task_kind="classification", fetch=_adult,
               notes="Mixes numbers and categories, like real business data. Expect ~85% accuracy."),
    HubDataset("wine_quality", TABULAR, "Red wine quality",
               "Lab tests of Portuguese red wines; predict the tasters' quality score (0-10).",
               "UCI Machine Learning Repository (Cortez et al., 2009)",
               "https://archive.ics.uci.edu/dataset/186/wine+quality",
               "CC BY 4.0", "84 KB", "1,599 rows · 11 features · score", "regression", [_MLP_R],
               "csv", target="quality", task_kind="regression", fetch=_wine_quality,
               notes="A small regression problem. A mean error (MAE) around 0.5 points is good."),
    HubDataset("california_housing", TABULAR, "California housing prices",
               "1990 census districts (income, rooms, location…); predict the median house value.",
               "StatLib / Pace & Barry (1997), via Hands-On Machine Learning (A. Géron)",
               "https://github.com/ageron/handson-ml2/tree/master/datasets/housing",
               "Public domain", "1.4 MB", "20,640 rows · 9 features · price", "regression", [_MLP_R],
               "csv", target="median_house_value", task_kind="regression", fetch=_california,
               notes="The standard regression benchmark. Has a few missing values (filled automatically)."),
    # --- computer vision ---
    HubDataset("mnist", VISION, "MNIST handwritten digits",
               "Black-and-white images of handwritten digits 0-9. The 'hello world' of deep learning.",
               "Yann LeCun, Corinna Cortes, Chris Burges", "http://yann.lecun.com/exdb/mnist/",
               "CC BY-SA 3.0", "12 MB", "70,000 images · 28×28 grayscale · 10 classes", "classification",
               ["Small CNN (28×28 grayscale, e.g. MNIST)", "Vision Transformer (ViT, 28×28 grayscale)"],
               "mnist", fetch=_torchvision("mnist"), notes="A small CNN reaches ~99% accuracy in a few epochs."),
    HubDataset("fashion_mnist", VISION, "Fashion-MNIST clothing",
               "Photos of clothes and shoes in 10 categories - a harder drop-in replacement for MNIST.",
               "Zalando Research", "https://github.com/zalandoresearch/fashion-mnist",
               "MIT", "30 MB", "70,000 images · 28×28 grayscale · 10 classes", "classification",
               ["Small CNN (28×28 grayscale, e.g. MNIST)", "Vision Transformer (ViT, 28×28 grayscale)"],
               "fashion_mnist", fetch=_torchvision("fashion_mnist"),
               notes="A small CNN reaches ~90-92% accuracy."),
    HubDataset("cifar10", VISION, "CIFAR-10 objects",
               "Small colour photos of planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks.",
               "University of Toronto (A. Krizhevsky, 2009)", "https://www.cs.toronto.edu/~kriz/cifar.html",
               "Free for research", "170 MB", "60,000 images · 32×32 colour · 10 classes", "classification",
               ["CNN (32×32 colour, e.g. CIFAR-10)", "Hybrid CNN + Transformer (32×32 colour)",
                "Vision Transformer (ViT, 32×32 colour)"],
               "cifar10", fetch=_torchvision("cifar10"),
               notes="The standard colour-image benchmark. A good CNN reaches ~80-85%; a GPU helps."),
    HubDataset("cifar100", VISION, "CIFAR-100 objects",
               "Like CIFAR-10 but with 100 classes (apple, bicycle, whale, …), 600 images each.",
               "University of Toronto (A. Krizhevsky, 2009)", "https://www.cs.toronto.edu/~kriz/cifar.html",
               "Free for research", "170 MB", "60,000 images · 32×32 colour · 100 classes", "classification",
               ["CNN (32×32 colour, e.g. CIFAR-10)", "Hybrid CNN + Transformer (32×32 colour)"],
               "cifar100", fetch=_torchvision("cifar100"),
               notes="Much harder than CIFAR-10: 50-60% accuracy is a solid result for a small model."),
    HubDataset("imagenette", VISION, "Imagenette photos",
               "Real-world photos of 10 easy-to-tell ImageNet classes (tench, church, parachute, …).",
               "fast.ai (subset of ImageNet)", "https://github.com/fastai/imagenette",
               "Non-commercial research (ImageNet terms)", "99 MB", "13,394 photos · 160 px · 10 classes", "classification",
               _TRANSFER + ["Deep CNN (64×64 colour photos)"],
               "image_folder", fetch=_imagenette,
               notes="Perfect for transfer learning: a pretrained backbone reaches ~95%+ quickly."),
    HubDataset("oxford_pets", VISION, "Oxford-IIIT Pets",
               "Photos of 25 dog and 12 cat breeds - a real fine-grained classification task.",
               "Visual Geometry Group, University of Oxford", "https://www.robots.ox.ac.uk/~vgg/data/pets/",
               "CC BY-SA 4.0", "790 MB", "7,390 photos · various sizes · 37 classes", "classification",
               _TRANSFER, "image_folder", fetch=_pets,
               notes="Hard to learn from scratch - use a transfer-learning template (~85-90% accuracy)."),
    # --- language ---
    HubDataset("sms_spam", TEXT, "SMS Spam Collection",
               "Real text messages labelled as spam or ham (normal).",
               "UCI Machine Learning Repository (Almeida & Hidalgo, 2011)",
               "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
               "CC BY 4.0", "200 KB", "5,574 messages · 2 classes", "classification", _TEXT_T,
               "csv_text", target="label", text_column="text", fetch=_sms_spam,
               notes="Small and fast. Mostly ham, so look at accuracy above 87% (the 'always ham' score)."),
    HubDataset("sst2", TEXT, "SST-2 movie sentiment (GLUE)",
               "Short movie-review phrases and sentences labelled positive or negative.",
               "Stanford NLP (Socher et al., 2013), part of the GLUE benchmark", "https://nlp.stanford.edu/sentiment/",
               "Free for research", "7 MB", "68,221 sentences · 2 classes", "classification", _TEXT_T,
               "csv_text", target="label", text_column="text", fetch=_sst2,
               notes="A classic NLP benchmark. Small models reach ~80-85% accuracy."),
    HubDataset("imdb", TEXT, "IMDB movie reviews",
               "Full-length movie reviews from IMDB, labelled positive or negative.",
               "Stanford AI Lab (Maas et al., 2011)", "https://ai.stanford.edu/~amaas/data/sentiment/",
               "Free for research", "84 MB", "50,000 reviews · 2 classes", "classification",
               _TEXT_T + ["GPT-style Transformer (causal attention)"],
               "csv_text", target="label", text_column="text", fetch=_imdb,
               notes="Reviews are long: raise the input length (e.g. 256) for better results. ~85-88% is good."),
    HubDataset("ag_news", TEXT, "AG News topics",
               "News headlines + descriptions sorted into World, Sports, Business and Sci/Tech.",
               "A. Gulli's AG corpus, prepared by Zhang, Zhao & LeCun (2015); mirrored by fast.ai",
               "http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
               "Free for non-commercial use", "12 MB", "127,600 articles · 4 classes", "classification",
               _TEXT_T + ["BERT-style Transformer (deeper encoder)"],
               "csv_text", target="label", text_column="text", fetch=_ag_news,
               notes="A large topic-classification benchmark. Small models reach ~88-91%."),
    # --- language models ---
    HubDataset("tiny_shakespeare", LANGUAGE_MODEL, "Tiny Shakespeare",
               "40,000 lines from Shakespeare's plays in one text file. The classic first dataset for "
               "a character-level language model: watch it learn to write dialogue in old English.",
               "Andrej Karpathy's char-rnn project (text from Project Gutenberg)",
               "https://github.com/karpathy/char-rnn", "Public domain (the plays); MIT (the repository)",
               "1.1 MB", "1,115,394 characters · 65 different characters", "language model (next character)",
               ["Tiny GPT (characters, fast)", "LSTM language model (characters)"], "text_corpus",
               fetch=_tiny_shakespeare,
               notes="On a laptop CPU an epoch takes a minute or two. After ~5 epochs the samples look like "
                     "a play (names, line breaks); a validation perplexity around 4-5 is good."),
]
BY_KEY = {d.key: d for d in CATALOG}


# ---------------------------------------------------------------------------
# Workspace storage
# ---------------------------------------------------------------------------


def _dir(d: HubDataset, data_dir: Path) -> Path:
    # torchvision sources live where the Train screen already looks for them (data/<name>)
    from .data import TORCHVISION

    return data_dir / d.key if d.format in TORCHVISION else data_dir / "hub" / d.key


def status(d: HubDataset, data_dir: Path) -> dict | None:
    """Download record if the dataset is ready to use, else None."""
    try:
        return json.loads((_dir(d, data_dir) / MARKER).read_text())
    except (OSError, ValueError):
        return None


def disk_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def download(d: HubDataset, data_dir: Path, progress: Progress = lambda *a: None,
             cancel: Callable[[], bool] = lambda: False) -> dict:
    dest = _dir(d, data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        rel = d.fetch(dest, progress, cancel)
    finally:
        for tmp in dest.glob("*.part"):
            tmp.unlink(missing_ok=True)
        (dest / "raw.tmp").unlink(missing_ok=True)
    rec = {"key": d.key, "path": rel, "downloaded": time.strftime("%Y-%m-%d %H:%M"),
           "bytes": disk_size(dest)}
    (dest / MARKER).write_text(json.dumps(rec, indent=2))
    return rec


def remove(d: HubDataset, data_dir: Path) -> None:
    shutil.rmtree(_dir(d, data_dir), ignore_errors=True)


def train_cfg(d: HubDataset, data_dir: Path) -> dict:
    """Data settings for the Train screen (same keys as its ``_data_cfg``)."""
    rec = status(d, data_dir) or {}
    path = str(_dir(d, data_dir) / rec["path"]) if rec.get("path") else ""
    return {"source": d.format, "path": path, "target": d.target, "text_column": d.text_column,
            "task": d.task_kind, "hub": d.key}
