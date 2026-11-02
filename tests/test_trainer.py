"""Training: models learn, the best epoch is kept, and the controls (early stop, freeze, stop) work."""
import math

import pytest
import torch

from aiforge import data as datamod
from aiforge.trainer import (TrainConfig, Trainer, amp_dtype, data_mismatch, fit_spec_to_data,
                             load_checkpoint, reset_head)
from conftest import make_spec


def _setup(tmp_path, template, cfg):
    spec = make_spec(template)
    tr, va, info = datamod.load(cfg, tmp_path, spec)
    fit_spec_to_data(spec, info)
    return spec, tr, va, info


def _train(spec, tr, va, info=None, quiet=True, **cfg):
    cfg = TrainConfig(device="cpu", **cfg)
    torch.manual_seed(0)
    trainer = Trainer(spec, spec.build(), tr, va, cfg, on_log=lambda m: None if quiet else print(m),
                      preprocess=info.preprocess if info else None)
    return trainer, trainer.run()


def test_fit_spec_to_data_removes_every_mismatch(tmp_path):
    spec = make_spec("MLP classifier (tabular)")
    _, _, info = datamod.load({"source": "synthetic", "kind": "spirals"}, tmp_path)
    assert data_mismatch(spec, info)
    changes = fit_spec_to_data(spec, info)
    assert any("output layer" in c for c in changes)
    assert data_mismatch(spec, info) == [] and spec.validate() == []


def test_classifier_learns_spirals(tmp_path):
    spec, tr, va, _ = _setup(tmp_path, "MLP classifier (tabular)",
                             {"source": "synthetic", "kind": "spirals", "samples": 900})
    trainer, run = _train(spec, tr, va, epochs=15, lr=0.01)
    ep = run["epochs"]
    assert len(ep) == 15 and not run["stopped"]
    assert ep[-1]["train_loss"] < ep[0]["train_loss"] * 0.7
    assert max(e["val_acc"] for e in ep) > 0.6  # 3 classes: guessing gets ~0.33
    # The model ends with the weights of its best epoch.
    assert run["best_score"] == min(e["val_loss"] for e in ep)


def test_regressor_learns(tmp_path):
    spec, tr, va, _ = _setup(tmp_path, "MLP regressor (tabular)",
                             {"source": "synthetic", "kind": "regression", "samples": 800})
    _, run = _train(spec, tr, va, epochs=10, lr=0.01)
    ep = run["epochs"]
    assert ep[-1]["val_loss"] < ep[0]["val_loss"]


def test_language_model_trains_and_writes(tmp_path):
    spec, tr, va, info = _setup(tmp_path, "Tiny GPT (characters, fast)",
                                {"source": "text_corpus", "path": datamod.EXAMPLE_CORPUS, "val_split": 0.1})
    _, run = _train(spec, datamod.limit(tr, 256), va, info, epochs=2, lr=0.003, batch_size=32)
    ep = run["epochs"]
    assert ep[-1]["train_loss"] < ep[0]["train_loss"]
    assert "train_ppl" in ep[0] and isinstance(ep[-1]["sample"], str)


@pytest.mark.parametrize("optimizer, scheduler", [("SGD", "step"), ("AdamW", "cosine")])
def test_optimizers_and_schedulers(tmp_path, optimizer, scheduler):
    spec, tr, va, _ = _setup(tmp_path, "MLP classifier (tabular)",
                             {"source": "synthetic", "kind": "moons", "samples": 300})
    _, run = _train(spec, tr, va, epochs=3, optimizer=optimizer, scheduler=scheduler)
    lrs = [e["lr"] for e in run["epochs"]]
    assert lrs[-1] < lrs[0]  # the scheduler lowered the learning rate


def test_early_stopping(tmp_path):
    spec, tr, va, _ = _setup(tmp_path, "MLP classifier (tabular)",
                             {"source": "synthetic", "kind": "moons", "samples": 300})
    # A learning rate of ~0 means nothing improves after the first epoch.
    _, run = _train(spec, tr, va, epochs=20, lr=1e-12, early_stop=2)
    assert len(run["epochs"]) == 3


def test_stop_before_first_epoch_saves_nothing(tmp_path):
    spec, tr, va, _ = _setup(tmp_path, "MLP classifier (tabular)",
                             {"source": "synthetic", "kind": "moons", "samples": 300})
    trainer = Trainer(spec, spec.build(), tr, va, TrainConfig(device="cpu"), on_log=lambda m: None)
    trainer.stop()
    run = trainer.run()
    assert run["stopped"] and run["epochs"] == [] and run["best_score"] is None


def test_frozen_layers_do_not_change(tmp_path):
    spec, tr, va, _ = _setup(tmp_path, "MLP classifier (tabular)",
                             {"source": "synthetic", "kind": "moons", "samples": 300})
    spec.frozen = [spec.layers[0].id]
    model = spec.build()
    before = model[0].weight.clone()
    Trainer(spec, model, tr, va, TrainConfig(epochs=2, device="cpu"), on_log=lambda m: None).run()
    assert torch.equal(model[0].weight, before)
    assert not torch.equal(model[-1].weight, spec.build()[-1].weight)

    spec.frozen = [l.id for l in spec.layers]
    with pytest.raises(ValueError, match="Every layer is frozen"):
        Trainer(spec, spec.build(), tr, va, TrainConfig(device="cpu"), on_log=lambda m: None).run()


def test_reset_head():
    spec = make_spec("MLP classifier (tabular)")
    model = spec.build()
    before = model[-1].weight.clone()
    assert reset_head(spec, model)
    assert not torch.equal(model[-1].weight, before)


# -- checkpoints: an interrupted run continues where it stopped ----------------------------------


def _moons(tmp_path):
    return _setup(tmp_path, "MLP classifier (tabular)", {"source": "synthetic", "kind": "moons", "samples": 300})


def _run_with_checkpoint(spec, tr, va, ck_path, stop_after=None, resume=None, **cfg):
    cfg = TrainConfig(**{"device": "cpu", "epochs": 6, "lr": 0.01, "scheduler": "cosine", **cfg})
    model = spec.build()
    trainer = Trainer(spec, model, tr, va, cfg, on_log=lambda m: None, checkpoint=ck_path,
                      checkpoint_meta={"note": "hi"},
                      on_epoch=lambda r: trainer.stop() if r["epoch"] == stop_after else None)
    return trainer.run(resume=resume), model


def test_resumed_run_matches_an_uninterrupted_one(tmp_path):
    spec, tr, va, _ = _moons(tmp_path)
    torch.manual_seed(0)
    full, full_model = _run_with_checkpoint(spec, tr, va, tmp_path / "a.pt")
    assert not (tmp_path / "a.pt").exists()  # finished: nothing to resume

    torch.manual_seed(0)
    part, _ = _run_with_checkpoint(spec, tr, va, tmp_path / "b.pt", stop_after=3)
    assert part["stopped"] and len(part["epochs"]) == 3
    ck = load_checkpoint(tmp_path / "b.pt")
    assert ck["epoch"] == 3 and ck["meta"] == {"note": "hi"}

    torch.manual_seed(123)  # the checkpoint restores the random state itself
    rest, model = _run_with_checkpoint(spec, tr, va, tmp_path / "b.pt", resume=ck)
    assert rest["resumed_at"] == 3 and rest["started"] == part["started"]
    assert [e["epoch"] for e in rest["epochs"]] == [1, 2, 3, 4, 5, 6]
    assert [e["lr"] for e in rest["epochs"]] == pytest.approx([e["lr"] for e in full["epochs"]])
    assert rest["best_score"] == pytest.approx(full["best_score"])
    for a, b in zip(model.state_dict().values(), full_model.state_dict().values()):
        assert torch.allclose(a, b)
    assert not (tmp_path / "b.pt").exists()


def test_half_done_epoch_is_not_checkpointed(tmp_path):
    spec, tr, va, _ = _moons(tmp_path)
    ck_path = tmp_path / "ck.pt"
    model = spec.build()
    cfg = TrainConfig(device="cpu", epochs=5, batch_size=16)
    trainer = Trainer(spec, model, tr, va, cfg, on_log=lambda m: None, checkpoint=ck_path,
                      on_epoch=lambda r: None,
                      on_batch=lambda e, s, n, l: trainer.stop() if (e, s) >= (2, n // 2) else None)
    trainer.run()
    assert load_checkpoint(ck_path)["epoch"] == 1  # epoch 2 was cut short: it is redone in full


def test_early_stop_leaves_no_checkpoint(tmp_path):
    spec, tr, va, _ = _moons(tmp_path)
    ck_path = tmp_path / "ck.pt"
    run, _ = _run_with_checkpoint(spec, tr, va, ck_path, lr=1e-12, early_stop=2)
    assert len(run["epochs"]) == 3 and not ck_path.exists()


def test_damaged_checkpoint_is_ignored(tmp_path):
    (tmp_path / "ck.pt").write_bytes(b"not a checkpoint")
    assert load_checkpoint(tmp_path / "ck.pt") is None
    assert load_checkpoint(tmp_path / "missing.pt") is None


def test_stop_after_the_last_epoch_leaves_nothing_to_resume(tmp_path):
    spec, tr, va, _ = _moons(tmp_path)
    run, _ = _run_with_checkpoint(spec, tr, va, tmp_path / "ck.pt", stop_after=6)
    assert run["stopped"] and len(run["epochs"]) == 6 and not (tmp_path / "ck.pt").exists()


# -- speed: mixed precision ------------------------------------------------------------------------

MPS = torch.backends.mps.is_available()


def test_amp_dtype(monkeypatch):
    cpu, mps, cuda = torch.device("cpu"), torch.device("mps"), torch.device("cuda")
    assert amp_dtype(cpu) is None and amp_dtype(mps) is None  # no measured gain there
    assert amp_dtype(cuda, "fp32") is None
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert amp_dtype(cuda) == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    assert amp_dtype(cuda) == torch.float16  # older cards: float16 with loss scaling
    assert amp_dtype(cpu, "bf16") == torch.bfloat16 and amp_dtype(mps, "fp16") == torch.float16


def test_bfloat16_training_learns(tmp_path):
    spec, tr, va, _ = _setup(tmp_path, "MLP classifier (tabular)",
                             {"source": "synthetic", "kind": "spirals", "samples": 900})
    logs = []
    torch.manual_seed(0)
    trainer = Trainer(spec, spec.build(), tr, va, TrainConfig(epochs=15, lr=0.01, device="cpu", precision="bf16"),
                      on_log=logs.append)
    run = trainer.run()
    assert any("mixed precision (bfloat16)" in m for m in logs)
    assert max(e["val_acc"] for e in run["epochs"]) > 0.6
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())  # weights stay full precision


@pytest.mark.skipif(not MPS, reason="needs an Apple GPU")
def test_float16_with_loss_scaling_on_mps_resumes(tmp_path):
    spec, tr, va, info = _setup(tmp_path, "Tiny GPT (characters, fast)",
                                {"source": "text_corpus", "path": datamod.EXAMPLE_CORPUS, "val_split": 0.1})
    tr = datamod.limit(tr, 256)
    cfg = TrainConfig(epochs=3, lr=0.003, batch_size=32, device="mps", precision="fp16")
    ck_path = tmp_path / "ck.pt"
    trainer = Trainer(spec, spec.build(), tr, va, cfg, on_log=lambda m: None, checkpoint=ck_path,
                      preprocess=info.preprocess, on_epoch=lambda r: trainer.stop() if r["epoch"] == 1 else None)
    trainer.run()
    ck = load_checkpoint(ck_path)
    assert ck["scaler"]["scale"] > 0  # the loss scale is saved for resuming
    rest = Trainer(spec, spec.build(), tr, va, cfg, on_log=lambda m: None, checkpoint=ck_path,
                   preprocess=info.preprocess).run(resume=ck)
    losses = [e["train_loss"] for e in rest["epochs"]]
    assert len(losses) == 3 and all(math.isfinite(l) for l in losses) and losses[-1] < losses[0]


def test_checkpoint_save_waits_for_a_reader(tmp_path, monkeypatch):
    from pathlib import Path

    from aiforge.trainer import save_checkpoint

    real, calls = Path.replace, []

    def busy_twice(self, target):  # what Windows does while another process reads the file
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError("file in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", busy_twice)
    save_checkpoint(tmp_path / "ck.pt", {"version": 1, "epoch": 3})
    assert len(calls) == 3 and load_checkpoint(tmp_path / "ck.pt")["epoch"] == 3
