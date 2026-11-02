"""Training: models learn, the best epoch is kept, and the controls (early stop, freeze, stop) work."""
import pytest
import torch

from aiforge import data as datamod
from aiforge.trainer import TrainConfig, Trainer, data_mismatch, fit_spec_to_data, reset_head
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
