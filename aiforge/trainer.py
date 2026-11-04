"""Training / fine-tuning loop, independent of the UI (talks through callbacks)."""
from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DataInfo, loader_workers
from .spec import CLASSIC_TASKS, ModelSpec, is_unsupervised


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
    precision: str = "auto"  # auto | fp32 | bf16 | fp16 (see amp_dtype)
    workers: int = -1  # background data-loading processes, -1 = auto (see data.loader_workers)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


PRECISIONS = {"auto": "auto", "fp32": "full fp32", "bf16": "mixed bf16", "fp16": "mixed fp16"}


def amp_dtype(dev: torch.device, precision: str = "auto") -> torch.dtype | None:
    """The 16-bit type to train in on ``dev`` (mixed precision), or None for full float32.

    NVIDIA GPUs do most maths several times faster in 16 bits and need half the memory for
    activations, so ``auto`` uses it there. Apple GPUs (MPS) and CPUs showed no speed-up, so
    ``auto`` keeps float32 on them; ``bf16`` / ``fp16`` force it anywhere."""
    if precision == "auto":
        if dev.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


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
    if is_unsupervised(info.task):  # no output head to size: the model never predicts a label
        if spec.task == "autoencoder":
            changes += _fit_autoencoder_output(spec)
        return changes
    if info.task == "language_model":
        emb = next((l for l in spec.layers if l.type == "Embedding"), None)
        if emb and emb.config["num_embeddings"] != info.n_outputs:
            changes.append(f"vocabulary {emb.config['num_embeddings']} → {info.n_outputs} tokens")
            emb.config["num_embeddings"] = info.n_outputs
    head = next((l for l in reversed(spec.layers) if l.type == "Linear"), None)
    if head and head.config["out_features"] != info.n_outputs:
        changes.append(f"output layer {head.config['out_features']} → {info.n_outputs} units")
        head.config["out_features"] = info.n_outputs
    return changes


def _fit_autoencoder_output(spec: ModelSpec) -> list[str]:
    """Make an autoencoder's last layers output the input's shape again."""
    shape, changes = list(spec.input_shape), []
    unflatten = next((l for l in reversed(spec.layers) if l.type == "Unflatten"), None)
    if unflatten and len(shape) == 3:
        new = dict(zip(("channels", "height", "width"), shape))
        if any(unflatten.config[k] != v for k, v in new.items()):
            changes.append(f"rebuilt image → {'×'.join(map(str, shape))}")
            unflatten.config.update(new)
    size = math.prod(shape) if unflatten or len(shape) == 1 else None
    linear = next((l for l in reversed(spec.layers) if l.type == "Linear"), None)
    if size and linear and linear.config["out_features"] != size:
        changes.append(f"output layer {linear.config['out_features']} → {size} units")
        linear.config["out_features"] = size
    conv = next((l for l in reversed(spec.layers) if l.type in ("ConvTranspose2d", "Conv2d")), None)
    if len(shape) == 3 and not unflatten and conv and conv.config["out_channels"] != shape[0]:
        changes.append(f"output channels {conv.config['out_channels']} → {shape[0]}")
        conv.config["out_channels"] = shape[0]
    return changes


def data_mismatch(spec: ModelSpec, info: DataInfo) -> list[str]:
    probs = []
    if list(spec.input_shape) != list(info.input_shape):
        probs.append(f"Model input is {tuple(spec.input_shape)} but data is {tuple(info.input_shape)}")
    if spec.input_dtype != info.input_dtype:
        probs.append(f"Model input type is '{spec.input_dtype}' but data is '{info.input_dtype}'")
    out = spec.output_shape()
    if out is not None and not is_unsupervised(info.task) and out != info.output_shape:
        probs.append(f"Model outputs {out} but the data needs {info.output_shape}")
    if spec.task == "autoencoder" and out is not None and list(out) != list(info.input_shape):
        probs.append(f"An autoencoder must output the input's shape {tuple(info.input_shape)}, "
                     f"but this one outputs {out}")
    if spec.task != info.task:
        probs.append(f"Model task is {spec.task} but data is {info.task}")
    vocab = spec.vocab_size()
    if info.task == "language_model" and vocab is not None and vocab != info.n_outputs:
        probs.append(f"The Embedding knows {vocab} tokens but the text's vocabulary has {info.n_outputs}")
    return probs


CHECKPOINT = "checkpoint.pt"  # in the model folder while a run is unfinished
CHECKPOINT_VERSION = 1


def save_checkpoint(path: Path, state: dict) -> None:
    """Write via a temporary file so a crash mid-write never leaves a broken checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: someone is reading the old checkpoint right now
            if attempt == 19:
                raise
            time.sleep(0.05)


def load_checkpoint(path: Path) -> dict | None:
    """The interrupted run saved at ``path``, or None when there is none (or it can't be read)."""
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a damaged checkpoint just means "nothing to resume"
        return None
    return state if isinstance(state, dict) and state.get("version") == CHECKPOINT_VERSION else None


def lm_loss(out, y, loss_fn):
    """Cross-entropy over every position: (batch, seq, vocab) vs (batch, seq)."""
    return loss_fn(out.reshape(-1, out.shape[-1]), y.reshape(-1))


class Trainer:
    """Runs synchronously; call ``run`` from a worker thread and ``stop`` from anywhere."""

    def __init__(self, spec: ModelSpec, model: nn.Module, train_ds, val_ds, cfg: TrainConfig,
                 on_log: Callable[[str], None] = print,
                 on_batch: Callable[[int, int, int, float], None] | None = None,
                 on_epoch: Callable[[dict], None] | None = None, target_std: float = 1.0,
                 preprocess: dict | None = None, checkpoint: Path | None = None,
                 checkpoint_meta: dict | None = None):
        self.spec, self.model, self.cfg = spec, model, cfg
        # Saved after every epoch so an interrupted run can continue; removed when the run finishes.
        self.checkpoint, self.checkpoint_meta = checkpoint, checkpoint_meta or {}
        self.train_ds, self.val_ds = train_ds, val_ds
        self.on_log, self.on_batch, self.on_epoch = on_log, on_batch, on_epoch
        self.target_std = target_std  # report regression error in the original units
        self.preprocess = preprocess or {}  # language models: vocabulary, for the sample text
        self._stop = threading.Event()
        self.best_state: dict | None = None
        self.epochs: list[dict] = []

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _loss_fn(self):
        return nn.MSELoss() if self.spec.task in ("regression", "autoencoder") else nn.CrossEntropyLoss()

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

    def run(self, resume: dict | None = None) -> dict:
        """Train; ``resume`` is a checkpoint (see load_checkpoint) to continue instead of starting over.

        Unsupervised models also store what is normal for them (see unsupervised.reference_stats)
        in ``spec.data["unsupervised"]``, so the caller saves it with the weights."""
        if self.spec.task in CLASSIC_TASKS:
            return self._run_classic()
        run = self._run_network(resume)
        if self.spec.task == "autoencoder" and run["epochs"]:
            self._store_reference()
        return run

    def _store_reference(self, x: torch.Tensor | None = None) -> None:
        from . import unsupervised as un

        if x is None:
            x, _ = un.gather(self.train_ds, min(20_000, un.example_limit(self.spec.input_shape)))
        self.spec.data["unsupervised"] = un.reference_stats(self.spec, self.model, x)

    def _run_classic(self) -> dict:
        """Fit k-means or PCA on the training data, held in memory, on the CPU (fast enough, and
        works everywhere). The epochs setting is k-means' maximum number of rounds."""
        from . import unsupervised as un

        c, layer, t_start = self.cfg, self.model[0], time.time()
        self.model.cpu()
        limit = un.example_limit(self.spec.input_shape)
        x, _ = un.gather(self.train_ds, limit)
        xv, _ = un.gather(self.val_ds, limit)
        note = f" (a random {len(x):,} of {len(self.train_ds):,})" if len(x) < len(self.train_ds) else ""
        self.on_log(f"Device: cpu (classic algorithms) · fitting on {len(x):,} training examples{note} "
                    f"× {math.prod(self.spec.input_shape):,} values · {len(xv):,} validation examples")
        if self.spec.task == "clustering":
            k = layer.centroids.shape[0]
            for it, inertia, moved in un.fit_kmeans(layer, x, c.epochs, stop=self._stop):
                t0 = time.time()
                rec = {"epoch": it, "train_loss": inertia, "moved": moved}
                check = xv if len(xv) else x
                groups = un.assign(self.model, check)
                if len(xv):
                    rec["val_loss"] = float((un.scores(self.spec, self.model, xv) ** 2).mean())
                rec["val_silhouette" if len(xv) else "train_silhouette"] = un.silhouette(check, groups)
                rec["seconds"] = round(time.time() - t0, 2)
                self.epochs.append(rec)
                if self.on_batch:
                    self.on_batch(it, 1, 1, inertia)
                if self.on_epoch:
                    self.on_epoch(rec)
                if moved == 0:
                    self.on_log(f"Converged after {it} rounds: no example changed group any more.")
            if self.epochs and self.epochs[-1]["moved"] and not self.stopped:
                self.on_log(f"Stopped after {c.epochs} rounds while examples were still changing group "
                            f"({self.epochs[-1]['moved']:,} in the last round): raise Max rounds.")
            if self.epochs:
                sizes = torch.bincount(un.assign(self.model, x), minlength=k).tolist()
                self.on_log("Group sizes: " + " · ".join(f"{un.group_name(i)}: {n:,}" for i, n in enumerate(sizes)))
        else:
            ratio = un.fit_pca(layer, x)
            rec = {"epoch": 1, "train_loss": float(un.scores(self.spec, self.model, x).mean()),
                   "train_kept": un.variance_kept(self.spec, self.model, x),
                   "explained": [round(float(v), 6) for v in ratio]}
            if len(xv):
                rec["val_loss"] = float(un.scores(self.spec, self.model, xv).mean())
                rec["val_kept"] = un.variance_kept(self.spec, self.model, xv)
            rec["seconds"] = round(time.time() - t_start, 2)
            self.epochs.append(rec)
            if self.on_epoch:
                self.on_epoch(rec)
            self.on_log("Variation kept by each component: " +
                        " · ".join(f"{i + 1}: {v:.1%}" for i, v in enumerate(ratio.tolist())))
        if self.checkpoint:
            self.checkpoint.unlink(missing_ok=True)  # one quick fit: nothing to resume
        if self.epochs:
            self._store_reference(x)
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        last = self.epochs[-1] if self.epochs else {}
        return {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
            "duration_s": round(time.time() - t_start, 1),
            "config": asdict(c),
            "epochs": self.epochs,
            "best_score": last.get("val_loss", last.get("train_loss")),
            "stopped": self.stopped,
        }

    def _run_network(self, resume: dict | None) -> dict:
        c = self.cfg
        dev = pick_device(c.device)
        self.model.to(dev)
        workers = loader_workers(self.train_ds, c.workers)
        # Workers stay alive between epochs: starting one re-imports PyTorch (seconds on macOS/Windows).
        loader_kw = dict(batch_size=c.batch_size, num_workers=workers, persistent_workers=workers > 0,
                         pin_memory=dev.type == "cuda")
        train_dl = DataLoader(self.train_ds, shuffle=True, drop_last=len(self.train_ds) > c.batch_size,
                              **loader_kw)
        val_dl = DataLoader(self.val_ds, shuffle=False, **loader_kw) if len(self.val_ds) else None
        loss_fn, opt = self._loss_fn(), self._optimizer()
        amp = amp_dtype(dev, c.precision)
        # float16 can round tiny gradients to 0; the scaler multiplies the loss up first (bfloat16 needn't).
        scaler = torch.amp.GradScaler(dev.type, enabled=amp == torch.float16)
        sched = None
        if c.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, c.epochs))
        elif c.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, c.epochs // 3), gamma=0.1)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        speed = [f"mixed precision ({str(amp).removeprefix('torch.')})"] if amp else []
        speed += [f"{workers} data workers"] if workers else []
        self.on_log(f"Device: {dev}{''.join(f' · {x}' for x in speed)} · trainable params {trainable:,} / "
                    f"{total:,} · {len(self.train_ds):,} train / {len(self.val_ds):,} val samples")
        best, bad, started, earlier = math.inf, 0, time.time(), 0.0
        first = 1
        if resume:
            self.model.load_state_dict(resume["model"])
            opt.load_state_dict(resume["optimizer"])
            if sched and resume.get("scheduler"):
                sched.load_state_dict(resume["scheduler"])
            self.epochs = list(resume["epochs"])
            self.best_state = resume["best_state"]
            best, bad, first = resume["best"], resume["bad"], resume["epoch"] + 1
            started, earlier = resume["started"], resume["elapsed"]
            if scaler.is_enabled() and resume.get("scaler"):
                scaler.load_state_dict(resume["scaler"])
            if resume.get("rng") is not None:  # same shuffles and dropout as an uninterrupted run
                torch.set_rng_state(resume["rng"])
            self.on_log(f"Resuming the interrupted run after epoch {resume['epoch']}/{c.epochs}.")
        resumed_at = resume["epoch"] if resume else None
        t_start, complete = time.time(), False
        steps = len(train_dl)
        lm = self.spec.task == "language_model"
        ae = self.spec.task == "autoencoder"  # learns to rebuild its input: the input is the answer
        for epoch in range(first, c.epochs + 1):
            self._set_train_mode()
            run_loss, run_correct, seen, done = 0.0, 0, 0, 0
            t0 = time.time()
            for step, (x, y) in enumerate(train_dl, 1):
                if self.stopped:
                    break
                done = step
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                    out = self.model(x)
                    if lm:
                        loss = lm_loss(out, y, loss_fn)
                    elif ae:
                        loss = loss_fn(out, x)
                    else:
                        loss = loss_fn(out, y.float() if self.spec.task == "regression" else y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if lm:  # deep Transformers trained from scratch can take the odd huge step
                    scaler.unscale_(opt)  # clip the real gradients, not the scaled-up ones
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)  # skipped when float16 overflowed; the scale then shrinks
                scaler.update()
                n = x.shape[0]
                run_loss += loss.item() * n
                seen += n
                if self.spec.task == "classification":
                    run_correct += (out.argmax(1) == y).sum().item()
                elif lm:
                    run_correct += (out.argmax(-1) == y).float().mean(1).sum().item()
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
            elif lm:
                rec["train_token_acc"] = run_correct / max(1, seen)
                rec["train_ppl"] = perplexity(rec["train_loss"])
            if val_dl:
                rec.update(evaluate(self.model, val_dl, self.spec.task, dev, loss_fn, self.target_std, amp))
            if lm and self.preprocess.get("vocab"):
                from .textgen import generate

                # A short sample after every epoch shows the model learning to write.
                rec["sample"] = generate(self.model, self.preprocess, length=160, temperature=0.8)
            self.epochs.append(rec)
            if self.on_epoch:
                self.on_epoch(rec)
            score = rec.get("val_loss", rec["train_loss"])
            if score < best:
                best, bad = score, 0
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            early = bool(c.early_stop and bad >= c.early_stop)
            complete = early or (epoch == c.epochs and done == steps)
            # A stopped, half-done epoch isn't checkpointed: resuming repeats it in full.
            if self.checkpoint and not complete and done == steps:
                save_checkpoint(self.checkpoint, {
                    "version": CHECKPOINT_VERSION, "epoch": epoch, "config": asdict(c),
                    "model": self.model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched else None,
                    "best_state": self.best_state, "best": best, "bad": bad, "epochs": self.epochs,
                    "started": started, "elapsed": earlier + time.time() - t_start,
                    "rng": torch.get_rng_state(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "meta": self.checkpoint_meta})
            if early:
                self.on_log(f"Early stopping: no improvement for {bad} epochs.")
                break
            if self.stopped:
                break
        can_resume = bool(self.checkpoint and not complete and self.checkpoint.exists())
        if self.stopped:
            self.on_log("Training stopped by user." +
                        (" Resume it later to continue where it left off." if can_resume else ""))
        if self.checkpoint and not can_resume:
            self.checkpoint.unlink(missing_ok=True)  # finished: nothing left to resume
        self.model.cpu()
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        run = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "duration_s": round(earlier + time.time() - t_start, 1),
            "config": asdict(c),
            "epochs": self.epochs,
            "best_score": best if best < math.inf else None,
            "stopped": self.stopped,
        }
        if resumed_at is not None:
            run["resumed_at"] = resumed_at
        return run


def perplexity(loss: float) -> float:
    """exp(cross-entropy): roughly how many tokens the model is still choosing between."""
    return math.exp(min(loss, 50.0))


@torch.no_grad()
def evaluate(model: nn.Module, loader, task: str, dev, loss_fn=None, target_std: float = 1.0,
             amp: torch.dtype | None = None) -> dict:
    model.eval()
    loss_fn = loss_fn or (nn.MSELoss() if task in ("regression", "autoencoder") else nn.CrossEntropyLoss())
    tot, correct, n, abs_err = 0.0, 0, 0, 0.0
    sse, sums, squares = 0.0, 0.0, 0.0  # autoencoder: how much of the variation it rebuilds
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            out = model(x)
        out = out.float()  # metrics in full precision
        if task == "autoencoder":
            flat = x.flatten(1).float()
            tot += loss_fn(out, x).item() * x.shape[0]
            sse += ((out.flatten(1) - flat) ** 2).sum().item()
            # Running totals in float64 on the CPU: Apple GPUs (MPS) have no float64.
            sums = sums + flat.sum(0).cpu().double()
            squares = squares + (flat ** 2).sum(0).cpu().double()
            n += x.shape[0]
            continue
        if task == "language_model":
            tot += lm_loss(out, y, loss_fn).item() * x.shape[0]
            correct += (out.argmax(-1) == y).float().mean(1).sum().item()
        else:
            tot += loss_fn(out, y.float() if task == "regression" else y).item() * x.shape[0]
        n += x.shape[0]
        if task == "classification":
            correct += (out.argmax(1) == y).sum().item()
        elif task == "regression":
            abs_err += (out - y).abs().sum().item()
    res = {"val_loss": tot / max(1, n)}
    if task == "autoencoder":
        total = float((squares - sums ** 2 / max(1, n)).sum()) if n else 0.0
        res["val_kept"] = 1 - sse / total if total > 0 else None
        return res
    if task == "classification":
        res["val_acc"] = correct / max(1, n)
    elif task == "language_model":
        res["val_token_acc"] = correct / max(1, n)
        res["val_ppl"] = perplexity(res["val_loss"])
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
