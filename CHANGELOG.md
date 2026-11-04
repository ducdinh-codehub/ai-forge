# Changelog

All notable changes to AIForge. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/) (still `0.x`: anything may change).

## [Unreleased]

### Added
- **Reinforcement learning: agents that learn from rewards.** Choose 🎮 Reinforcement in the *New
  model* dialog. No dataset needed: the agent learns by trial and error in a built-in world.
  - Two worlds: **GridWorld** (reach the goal, avoid the traps; size, traps and a fixed or random
    map are settings) and **CartPole** (balance a pole on a cart, Gym's classic physics).
  - **Deep Q-Network (DQN)** training with a replay memory, a target network and fading exploration.
    The Train screen shows the world, *Episodes*, *Discount γ* and *Explore for*, charts the reward
    per episode and the success rate, tests the agent every 10 episodes and keeps the best tested version.
  - **Use:** watch the agent play step by step with its score for every action, or play yourself with
    the arrow keys.
  - **Evaluate:** test episodes against a random agent on the same episodes: average reward with a
    confidence interval, success rate, how episodes ended, action use, best and worst episode drawn.
  - Command line: `aiforge train <agent>` (`-d gridworld|cartpole`, `--env-size`, `--traps`,
    `--layout`, `--gamma`, `--explore`), `aiforge play`, `aiforge evaluate --episodes`.
  - **Agent tour**: a guided, 6-step walk through a first agent (create, see its world, train, watch
    it play, test it), highlighting each button. `Ctrl+P` → *Take the agent tour*, and offered once on
    an untrained agent's Train screen. It follows the agent's own world (GridWorld or CartPole).
  - Guide lesson *🎮 Reinforcement: agents*; lesson 3 now points to it.
- **Guided tours** scroll the highlighted button into view when it is below the fold.
- **Unsupervised learning: models that need no labels.** The *New model* dialog now starts with
  the learning style: 🎯 Supervised, 🔍 Unsupervised (🎮 Reinforcement is shown as coming soon).
  - **k-means clustering** and **PCA**: classic algorithms, fitted in seconds on the CPU (new
    *Classic ML* layers `KMeans` and `PCA`). k-means starts from several k-means++ trials, so it
    doesn't get stuck with two centres in one group.
  - **Autoencoders** for tables and 28×28 images (new layers `Unflatten` and `ConvTranspose2d`),
    trained with epochs like the other neural networks.
  - Train on any table or image source; the label column is optional and only **kept aside** to
    check the groups. A plain folder of images works too.
  - **Use** tells an example's group or rebuild error and how unusual it is compared with the
    training data; batch prediction adds an `unusual_percentile` column.
  - **Evaluate** reports the silhouette or the variation kept against a baseline, draws a map of
    the data in the terminal, lists the most unusual examples and, with kept-aside labels, compares
    groups and labels (purity, adjusted Rand index).
  - New Guide lesson *🔍 Unsupervised: groups & oddities* and encyclopedia entries for the new layers.
  - The command line, Compare and the web API handle the new models too.
- **Guide:** new ML-basics lesson *3. Three ways to learn* explains supervised, unsupervised and
  reinforcement learning in plain words, with everyday examples and a "which one do I need?"
  checklist. The lessons after it are renumbered (4–8).

## [0.2.2] - 2026-11-03

### Added
- **Star AIForge on GitHub** from the app: `*` on the Home screen, a link in the welcome text, or
  the `Ctrl+P` palette. After your first successful training AIForge asks once, politely
  (*Maybe later* / *Don't ask again*), and never during the guided tour.

### Fixed
- **Compare:** when two runs tie for the best score, the earliest one is now starred as best
  (it used to depend on the metric whether the first or the last tied run was picked).

### Note
- There is no 0.2.1 on PyPI: its tag was pushed before the version bump was merged, so it was
  never published. 0.2.2 is that release.

## [0.2.0] - 2026-11-02

### Added
- **Pause and resume training.** A stopped, crashed or closed run continues where it left off:
  **⏯ Resume** on the Train screen, `aiforge train <model> --resume` on the command line.
  A checkpoint keeps the weights, optimizer, learning-rate schedule and random state, so the
  resumed run matches an uninterrupted one.
- **Faster training.** Mixed precision on NVIDIA GPUs (`--precision`, *Precision* on the Train
  screen) and background image loading on Linux (`--workers`, *Data workers*).
- **Training script export.** A standalone `train.py` that re-trains the model in plain PyTorch:
  the data loading, the settings of its last run and the training loop. Export → *Training script*.
- **More data sources.** Excel (`.xlsx`) and Parquet tables, images listed in a table (a path
  column + a label column), and any Hugging Face dataset by name (`-d hf:imdb`). Install the extra
  packages with `pip install "aiforge-studio[data]"`.
- **Explain a prediction.** 🔍 Explain on the Use screen and `aiforge explain`: which features or
  words pushed the model towards its answer or away from it, and a heat map of where an image
  model looks.
- **Compare runs.** 📈 Compare (`m`) and `aiforge compare`: the curves of several runs or models on
  one chart, with a table of best and last scores and only the settings that differ.
- **Guided first run.** On the first launch, a 5-minute tour of the quick start: a coach bar says
  what to do next and outlines the button to press; each step ends by itself when it is done (`F2`).
- Tests for the core modules: every template builds, exports to ONNX and TorchScript, and round-trips.
- `docs/improvement-guide.md`: the roadmap.
- Docs: CPU-only install (a ~200 MB PyTorch instead of ~2 GB).

### Fixed
- Importing a `.pt` file can no longer run code hidden in it: only supported layers are read.
- The Train screen now remembers *Max samples* with the run settings.
- Image datasets can be loaded by worker processes on macOS and Windows.
- Windows: exported training scripts work when the data path contains `\U` or `\n` (e.g. `C:\Users\...`).
- Windows: saving a training checkpoint no longer fails if the old one is being read at that moment.

## [0.1.9] - 2026-10-29

### Changed
- The Serve dialog shows the full install hint.
- Dataset hub: datasets download only from the button; server links are clickable.

## [0.1.8] - 2026-10-29

### Added
- Command line without the UI: `aiforge new / train / predict / generate / evaluate / export / serve`.
- Web API (`aiforge serve`): predictions and text generation over HTTP, with a demo page.
- CI on Linux, macOS and Windows.

### Fixed
- The app saves class names even when the model already had the right number of outputs.

### Changed
- Two-row toolbar on the home screen.

## [0.1.7] - 2026-10-23

### Added
- Language models: train a tiny GPT (or an LSTM) on any text and let it write.
- Report a problem or idea as a GitHub issue (`Ctrl+O`).
- Dark / light theme toggle.

## [0.1.6] - 2026-10-20

### Added
- **Evaluate** screen: metrics against a baseline with confidence intervals, calibration, error
  analysis, slices, robustness, learning curves and model-vs-model significance tests.

## [0.1.5] - 2026-10-20

### Added
- Choose 1D / 2D / 3D inputs in the builder.

## [0.1.4] - 2026-10-20

### Changed
- Dataset hub screenshots and refreshed README images.

## [0.1.3] - 2026-10-20

### Added
- Dataset hub: download famous datasets in one click.
- Model type picker in the New model dialog; more vision and NLP templates.
- Beginner ML lessons and model-type guides in the Guide.

## [0.1.2] - 2026-10-17

### Changed
- Badges on the PyPI page.

## [0.1.1] - 2026-10-17

### Changed
- More PyPI keywords and categories.

## [0.1.0] - 2026-10-16

### Added
- First release: design a network from a list of layers, train it, use it and export it, all
  in the terminal. Built-in Guide and resizable panels.

[Unreleased]: https://github.com/ducdinh-codehub/ai-forge/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/ducdinh-codehub/ai-forge/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/ducdinh-codehub/ai-forge/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/ducdinh-codehub/ai-forge/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/ducdinh-codehub/ai-forge/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/ducdinh-codehub/ai-forge/releases/tag/v0.1.7
