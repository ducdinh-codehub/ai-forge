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
from .trainer import (CHECKPOINT, Trainer, TrainConfig, data_mismatch, fit_spec_to_data, load_checkpoint,
                      reset_head)

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


def checkpoint_path(ws: Workspace, name: str) -> Path:
    return ws.model_dir(name) / CHECKPOINT


def interrupted_run(ws: Workspace, name: str) -> dict | None:
    """The checkpoint of ``name``'s last unfinished run (stopped, crashed or closed), if any."""
    return load_checkpoint(checkpoint_path(ws, name))


def resume_problem(ws: Workspace, name: str, ck: dict) -> str | None:
    """Why the interrupted run can't continue (the model was edited since), or None."""
    then = ModelSpec.from_dict(ck["meta"]["spec"]).layers
    now = ws.load_spec(name).layers
    if [(l.id, l.type) for l in then] != [(l.id, l.type) for l in now]:
        return (f"'{name}' was edited after its last run was interrupted, so that run can't "
                "continue. Start a new run instead.")
    return None


def stored_data_cfg(cfg: dict) -> dict:
    """The data config as remembered in the spec: no empty values, absolute paths."""
    stored = {k: v for k, v in cfg.items() if v not in (None, "")}
    if stored.get("path"):
        stored["path"] = str(Path(stored["path"]).expanduser())
    return stored


def train_model(ws: Workspace, name: str, data: LoadedData, cfg: TrainConfig, *,
                save_as: str | None = None, resume: bool = False, auto_fit: bool = True,
                on_log: Callable[[str], None] = print,
                on_batch: Callable[[int, int, int, float], None] | None = None,
                on_epoch: Callable[[dict], None] | None = None,
                on_trainer: Callable[[Trainer], None] | None = None) -> tuple[dict, ModelSpec]:
    """Train ``name`` on ``data`` and save it (as ``save_as`` if given: a fine-tuned copy).

    ``resume`` continues the last interrupted run where it stopped (same data and settings, see
    ``interrupted_run``), or else starts from the saved weights instead of random ones; saving
    under a new name always starts from them. Returns the run record (also appended to the
    history) and the spec. Nothing is saved when no epoch finished."""
    check_config(cfg)
    ck = interrupted_run(ws, name) if resume else None
    if ck:
        return _resume(ws, name, data, cfg, ck, on_log, on_batch, on_epoch, on_trainer)
    checkpoint_path(ws, name).unlink(missing_ok=True)  # a new run replaces an unfinished one
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
    meta = {"spec": spec.to_dict(), "save_as": save_as if finetune else None,
            "data_cfg": stored_data_cfg(data.cfg)}
    return _train_and_save(ws, name, save_as, spec, model, data, cfg, meta, None,
                           on_log, on_batch, on_epoch, on_trainer)


def _resume(ws, name, data, cfg, ck, on_log, on_batch, on_epoch, on_trainer):
    if problem := resume_problem(ws, name, ck):
        raise ValueError(problem)
    meta = ck["meta"]
    spec = ModelSpec.from_dict(meta["spec"])
    spec.name = name  # the model may have been renamed since
    if probs := data_mismatch(spec, data.info):
        raise ValueError(f"The data doesn't match the interrupted run: {probs[0]}")
    # The original settings, so the learning-rate schedule and epoch count carry on unchanged.
    cfg = TrainConfig(**{**ck["config"], "device": cfg.device, "workers": cfg.workers})
    model = spec.build()
    spec.apply_freeze(model)
    save_as = meta["save_as"] or name
    on_log(f"Continuing the interrupted run of '{name}'"
           f"{f' → {save_as!r}' if save_as != name else ''} ({cfg.epochs} epochs, lr {cfg.lr:g})")
    return _train_and_save(ws, name, save_as, spec, model, data, cfg, meta, ck,
                           on_log, on_batch, on_epoch, on_trainer)


def _train_and_save(ws, name, save_as, spec, model, data, cfg, meta, ck,
                    on_log, on_batch, on_epoch, on_trainer):
    trainer = Trainer(spec, model, data.train, data.val, cfg, on_log=on_log, on_batch=on_batch,
                      on_epoch=on_epoch, target_std=data.info.preprocess.get("target_std", 1.0),
                      preprocess=data.info.preprocess, checkpoint=checkpoint_path(ws, name),
                      checkpoint_meta=meta)
    if on_trainer:
        on_trainer(trainer)
    run = trainer.run(resume=ck)
    if not run["epochs"]:
        return run, spec

    finetune = save_as != name
    spec.data["preprocess"] = data.info.preprocess
    spec.data["dataset_cfg"] = meta["data_cfg"]
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
