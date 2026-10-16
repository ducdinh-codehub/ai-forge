"""Training / fine-tuning loop, independent of the UI (talks through callbacks)."""
from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DataInfo
from .spec import ModelSpec


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "Adam"  # Adam | AdamW | SGD
    weight_decay: float = 0.0
    scheduler: str = "none"  # none | cosine | step
    early_stop: int = 0  # patience in epochs, 0 = off
    device: str = "auto"
    max_samples: int = 0  # 0 = use everything


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available_devices() -> list[str]:
    devs = ["auto", "cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def fit_spec_to_data(spec: ModelSpec, info: DataInfo) -> list[str]:
    """Adjust the input shape, task and output head to match a dataset. Returns a change log."""
    changes = []
    if list(spec.input_shape) != list(info.input_shape) or spec.input_dtype != info.input_dtype:
        changes.append(f"input {tuple(spec.input_shape)}/{spec.input_dtype} → "
                       f"{tuple(info.input_shape)}/{info.input_dtype}")
        spec.input_shape, spec.input_dtype = list(info.input_shape), info.input_dtype
    if spec.task != info.task:
        changes.append(f"task {spec.task} → {info.task}")
        spec.task = info.task
    if spec.class_names != info.class_names:
        if info.class_names:
            changes.append(f"classes → {len(info.class_names)} ({', '.join(info.class_names[:5])}"
                           f"{'…' if len(info.class_names) > 5 else ''})")
        spec.class_names = list(info.class_names)
    head = next((l for l in reversed(spec.layers) if l.type == "Linear"), None)
    if head and head.config["out_features"] != info.n_outputs:
        changes.append(f"output layer {head.config['out_features']} → {info.n_outputs} units")
        head.config["out_features"] = info.n_outputs
    return changes


def data_mismatch(spec: ModelSpec, info: DataInfo) -> list[str]:
    probs = []
    if list(spec.input_shape) != list(info.input_shape):
        probs.append(f"Model input is {tuple(spec.input_shape)} but data is {tuple(info.input_shape)}")
    if spec.input_dtype != info.input_dtype:
        probs.append(f"Model input type is '{spec.input_dtype}' but data is '{info.input_dtype}'")
    out = spec.output_shape()
    if out is not None and out != (info.n_outputs,):
        probs.append(f"Model outputs {out} but the data needs ({info.n_outputs},)")
    if spec.task != info.task:
        probs.append(f"Model task is {spec.task} but data is {info.task}")
    return probs


class Trainer:
    """Runs synchronously; call ``run`` from a worker thread and ``stop`` from anywhere."""

    def __init__(self, spec: ModelSpec, model: nn.Module, train_ds, val_ds, cfg: TrainConfig,
                 on_log: Callable[[str], None] = print,
                 on_batch: Callable[[int, int, int, float], None] | None = None,
                 on_epoch: Callable[[dict], None] | None = None, target_std: float = 1.0):
        self.spec, self.model, self.cfg = spec, model, cfg
        self.train_ds, self.val_ds = train_ds, val_ds
        self.on_log, self.on_batch, self.on_epoch = on_log, on_batch, on_epoch
        self.target_std = target_std  # report regression error in the original units
        self._stop = threading.Event()
        self.best_state: dict | None = None
        self.epochs: list[dict] = []

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _loss_fn(self):
        return nn.CrossEntropyLoss() if self.spec.task == "classification" else nn.MSELoss()

    def _optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Every layer is frozen - unfreeze at least one layer to train.")
        c = self.cfg
        if c.optimizer == "SGD":
            return torch.optim.SGD(params, lr=c.lr, momentum=0.9, weight_decay=c.weight_decay)
        cls = torch.optim.AdamW if c.optimizer == "AdamW" else torch.optim.Adam
        return cls(params, lr=c.lr, weight_decay=c.weight_decay)

    def _set_train_mode(self):
        self.model.train()
        # Frozen layers keep their running statistics (BatchNorm) and skip dropout.
        for layer, mod in zip(self.spec.layers, self.model):
            if layer.id in self.spec.frozen:
                mod.eval()

    def run(self) -> dict:
        c = self.cfg
        dev = pick_device(c.device)
        self.model.to(dev)
        loader_kw = dict(batch_size=c.batch_size, num_workers=0)
        train_dl = DataLoader(self.train_ds, shuffle=True, drop_last=len(self.train_ds) > c.batch_size,
                              **loader_kw)
        val_dl = DataLoader(self.val_ds, shuffle=False, **loader_kw) if len(self.val_ds) else None
        loss_fn, opt = self._loss_fn(), self._optimizer()
        sched = None
        if c.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, c.epochs))
        elif c.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, c.epochs // 3), gamma=0.1)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.on_log(f"Device: {dev} · trainable params {trainable:,} / {total:,} · "
                    f"{len(self.train_ds):,} train / {len(self.val_ds):,} val samples")
        best, bad, started = math.inf, 0, time.time()
        steps = len(train_dl)
        for epoch in range(1, c.epochs + 1):
            self._set_train_mode()
            run_loss, run_correct, seen = 0.0, 0, 0
            t0 = time.time()
            for step, (x, y) in enumerate(train_dl, 1):
                if self.stopped:
                    break
                x, y = x.to(dev), y.to(dev)
                out = self.model(x)
                loss = loss_fn(out, y.float() if self.spec.task == "regression" else y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                n = x.shape[0]
                run_loss += loss.item() * n
                seen += n
                if self.spec.task == "classification":
                    run_correct += (out.argmax(1) == y).sum().item()
                if self.on_batch and (step % max(1, steps // 50) == 0 or step == steps):
                    self.on_batch(epoch, step, steps, run_loss / seen)
            if self.stopped and seen == 0:
                break
            if sched:
                sched.step()
            rec = {"epoch": epoch, "train_loss": run_loss / max(1, seen),
                   "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2)}
            if self.spec.task == "classification":
                rec["train_acc"] = run_correct / max(1, seen)
            if val_dl:
                rec.update(evaluate(self.model, val_dl, self.spec.task, dev, loss_fn, self.target_std))
            self.epochs.append(rec)
            if self.on_epoch:
                self.on_epoch(rec)
            score = rec.get("val_loss", rec["train_loss"])
            if score < best:
                best, bad = score, 0
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if c.early_stop and bad >= c.early_stop:
                    self.on_log(f"Early stopping: no improvement for {bad} epochs.")
                    break
            if self.stopped:
                break
        if self.stopped:
            self.on_log("Training stopped by user.")
        self.model.cpu()
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "duration_s": round(time.time() - started, 1),
            "config": asdict(c),
            "epochs": self.epochs,
            "best_score": best if best < math.inf else None,
            "stopped": self.stopped,
        }


@torch.no_grad()
def evaluate(model: nn.Module, loader, task: str, dev, loss_fn=None, target_std: float = 1.0) -> dict:
    model.eval()
    loss_fn = loss_fn or (nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss())
    tot, correct, n, abs_err = 0.0, 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        tot += loss_fn(out, y.float() if task == "regression" else y).item() * x.shape[0]
        n += x.shape[0]
        if task == "classification":
            correct += (out.argmax(1) == y).sum().item()
        else:
            abs_err += (out - y).abs().sum().item()
    res = {"val_loss": tot / max(1, n)}
    if task == "classification":
        res["val_acc"] = correct / max(1, n)
    else:
        res["val_mae"] = abs_err / max(1, n * (y.shape[1] if y.dim() > 1 else 1)) * target_std
    return res


def reset_head(spec: ModelSpec, model: nn.Sequential) -> str | None:
    """Re-initialise the last Linear layer (start a fresh classifier on top of learned features)."""
    for layer, mod in zip(reversed(spec.layers), reversed(list(model))):
        if layer.type == "Linear":
            mod.reset_parameters()
            return layer.id
    return None


def copy_spec(spec: ModelSpec) -> ModelSpec:
    return copy.deepcopy(spec)
