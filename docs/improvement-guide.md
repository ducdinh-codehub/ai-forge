# AIForge improvement guide

A prioritised plan for making AIForge sturdier and more useful. Tick items off as they land.

## 1. Fix first (small, high impact)

- [x] **Safe `.pt` import.** `model_io.import_path` unpickled `.pt`/`.pth`/`.bin` files with
  `torch.load(weights_only=False)`, so a malicious file could run code on import. Load with
  `weights_only=True` and allow-list only the `nn` layer classes the builder can convert; anything
  else is refused with a clear message.
- [x] **Tests for the core modules.** `spec.py`, `trainer.py`, `data.py`, `evaluate.py`, `layers.py`
  and `model_io.py` have no direct tests. Add:
  - every template builds, runs a forward pass, and exports to ONNX + TorchScript with the right shapes
  - spec → generated `.py` → re-import round trip
  - 1–2 epoch training on the spirals data, loss goes down
  - bundle export → import round trip (structure, weights, history)

## 2. Training

- [x] **Resume / pause.** Checkpoint model + optimizer + scheduler + epoch every epoch; offer
  "Resume" when a run was interrupted.
- [x] **Speed.** "Auto" mixed precision (`torch.autocast`) on CUDA/MPS; `num_workers > 0` for image
  folders (currently hard-coded to 0 in `trainer.py`).
- [x] **Learning-rate finder.** One button that sweeps the LR for a few hundred steps and suggests a value.
- [ ] **Mini sweep.** Try a few learning rates / batch sizes, compare side by side, keep the best.
- [x] **Balanced classes.** Weight rare classes up in the loss; the goal wizard turns it on for
  unbalanced data (Evaluate already warned about it).
- [x] **Reproducibility.** A seed field, recorded with each run in the history.

## 3. Product

- [x] **Experiment comparison.** Overlay the curves of several runs or models on one chart.
- [x] **More data sources.** Excel / Parquet; any Hugging Face dataset by name; image datasets given
  as a CSV of paths + labels.
- [x] **Explain a prediction.** Feature importance (tables), saliency heatmap (images, as terminal
  block art), highlighted tokens (text).
- [x] **"Training script" export.** A standalone `train.py` reproducing the run: data loading,
  config and loop. The path from no-code to real code.
- [x] **Guided first run.** Offer the spirals quick start on first launch with on-screen highlights.

## 4. Docs & distribution

- [x] Short GIF / asciinema demo at the top of the README.
- [x] `CHANGELOG.md`.
- [x] Document the CPU-only PyTorch install to cut the ~2 GB download for users without a GPU.
