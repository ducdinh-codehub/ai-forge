"""Train-and-save without a UI: load data, fit the model to it, train, store the result.

The command line uses this; it does the same steps as the Train screen in the TUI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from torch import nn

from . import data as datamod
from .data import DataInfo
from .spec import ModelSpec, Workspace, safe_name
from .trainer import Trainer, TrainConfig, data_mismatch, fit_spec_to_data, reset_head

TEXT_SUFFIXES = datamod.TEXT_EXT


def data_cfg_from_arg(value: str, spec: ModelSpec, **opts) -> dict:
    """Turn ``--data`` into a data config: a toy set (``spirals``), a famous dataset (``mnist``),
    a .csv file, a .txt file or an image folder. ``opts`` adds target / text_column / tokenizer /
    val_split / source (to override the guess)."""
    opts = {k: v for k, v in opts.items() if v not in (None, "")}
    source = opts.pop("source", None)
    if source is None:
        key = value.strip().lower()
        if key in datamod.SYNTHETIC:
            return {"source": "synthetic", "kind": key, **opts}
        if key in datamod.TORCHVISION:
            return {"source": key, **opts}
        p = Path(value).expanduser()
        if not p.exists():
            raise ValueError(f"'{value}' is not a file, a folder, a toy dataset "
                             f"({', '.join(datamod.SYNTHETIC)}) or a built-in dataset "
                             f"({', '.join(datamod.TORCHVISION)})")
        if p.is_dir():
            source = "image_folder"
        elif p.suffix.lower() in TEXT_SUFFIXES:
            source = "text_corpus"
        elif spec.input_dtype == "long" and spec.task != "language_model":
            source = "csv_text"  # a text classifier reads a text column, not numbers
        else:
            source = "csv"
    if source == "synthetic":
        return {"source": source, "kind": value.strip().lower(), **opts}
    if source in datamod.FILE_SOURCES:
        return {"source": source, "path": str(Path(value).expanduser().resolve()), **opts}
    return {"source": source, **opts}


def check_config(cfg: TrainConfig) -> None:
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.lr <= 0:
        raise ValueError("epochs and batch size must be at least 1, and the learning rate above 0")


@dataclass
class LoadedData:
    cfg: dict
    train: object
    val: object
    info: DataInfo


def load_data(ws: Workspace, spec: ModelSpec, cfg: dict, max_samples: int = 0) -> LoadedData:
    tr, va, info = datamod.load(cfg, ws.data_dir, spec)
    return LoadedData(cfg, datamod.limit(tr, max_samples), va, info)


def prepare(spec: ModelSpec, info: DataInfo, auto_fit: bool = True) -> list[str]:
    """Make the model fit the data, like the 'Auto-fit model to data' button. Returns the changes;
    raises ValueError when the model can't be trained on this data."""
    changes = []
    if probs := data_mismatch(spec, info):
        if not auto_fit:
            raise ValueError(probs[0])
        changes = fit_spec_to_data(spec, info)
    if spec.class_names != info.class_names:
        # A head that is already the right size isn't a mismatch, but the names still change.
        spec.class_names = list(info.class_names)
    if probs := spec.validate():
        raise ValueError(f"{probs[0]} - open the model in AIForge's builder to fix it.")
    return changes


def train_model(ws: Workspace, name: str, data: LoadedData, cfg: TrainConfig, *,
                save_as: str | None = None, resume: bool = False, auto_fit: bool = True,
                on_log: Callable[[str], None] = print,
                on_batch: Callable[[int, int, int, float], None] | None = None,
                on_epoch: Callable[[dict], None] | None = None,
                on_trainer: Callable[[Trainer], None] | None = None) -> tuple[dict, ModelSpec]:
    """Train ``name`` on ``data`` and save it (as ``save_as`` if given: a fine-tuned copy).

    ``resume`` continues from the saved weights instead of random ones; saving under a new name
    always starts from them. Returns the run record (also appended to the history) and the spec.
    Nothing is saved when no epoch finished."""
    check_config(cfg)
    save_as = safe_name(save_as) if save_as else name
    if not save_as:
        raise ValueError("Pick a name with letters or digits for --save-as")
    finetune = save_as != name
    if finetune and ws.exists(save_as):
        raise ValueError(f"'{save_as}' already exists - pick another name")
    spec = ws.load_spec(name)
    old_classes = list(spec.class_names)
    for change in prepare(spec, data.info, auto_fit):
        on_log(f"auto-fit: {change}")
    if not finetune:
        spec.frozen = []

    model: nn.Module
    if resume or finetune:
        if not ws.has_weights(name):
            raise ValueError(f"'{name}' has no trained weights yet - train it without --resume first")
        _, model, note = ws.load_model(name, spec)
        if old_classes and data.info.class_names != old_classes and reset_head(spec, model):
            note += ", output layer re-initialised for the new classes"
    else:
        model, note = spec.build(), "fresh random weights"
    spec.apply_freeze(model)
    on_log(f"{'Fine-tuning' if finetune else 'Training'} '{name}'"
           f"{f' → {save_as!r}' if finetune else ''} ({note})")

    trainer = Trainer(spec, model, data.train, data.val, cfg, on_log=on_log, on_batch=on_batch,
                      on_epoch=on_epoch, target_std=data.info.preprocess.get("target_std", 1.0),
                      preprocess=data.info.preprocess)
    if on_trainer:
        on_trainer(trainer)
    run = trainer.run()
    if not run["epochs"]:
        return run, spec

    spec.data["preprocess"] = data.info.preprocess
    stored = {k: v for k, v in data.cfg.items() if v not in (None, "")}
    if stored.get("path"):
        stored["path"] = str(Path(stored["path"]).expanduser())
    spec.data["dataset_cfg"] = stored
    run["kind"] = "fine-tune" if finetune else "train"
    run["dataset"] = data.info.description
    run["source"] = "cli"
    if finetune:
        run["parent"] = name
    spec.name = save_as
    ws.save_spec(spec)
    ws.save_weights(save_as, model)
    ws.append_history(save_as, run)
    return run, spec
