# AIForge

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat"></a>
  <a href="https://pytorch.org"><img alt="PyTorch 2.5+" src="https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=flat"></a>
  <a href="https://textual.textualize.io"><img alt="Textual 8.0+" src="https://img.shields.io/badge/Textual-8.0%2B-7B2CBF?style=flat"></a>
  <a href="https://onnx.ai"><img alt="ONNX export" src="https://img.shields.io/badge/ONNX-export-E8A0BF?style=flat"></a>
  <br>
  <a href="https://pypi.org/project/aiforge-studio/"><img alt="pypi package" src="https://img.shields.io/pypi/v/aiforge-studio?label=pypi%20package&color=4c1&style=flat"></a>
  <a href="https://pepy.tech/project/aiforge-studio"><img alt="downloads" src="https://img.shields.io/pepy/dt/aiforge-studio?label=downloads&color=0e75b6&style=flat"></a>
  <img alt="AI Powered" src="https://img.shields.io/badge/AI-Powered-ff69b4?style=flat">
  <a href="https://github.com/ducdinh-codehub/ai-forge/blob/main/LICENSE"><img alt="license MIT" src="https://img.shields.io/badge/license-MIT-D4A72C?style=flat"></a>
  <br>
  <img alt="status alpha" src="https://img.shields.io/badge/status-alpha-E05D44?style=flat">
  <a href="https://github.com/ducdinh-codehub/ai-forge/commits/main"><img alt="development active" src="https://img.shields.io/badge/development-active-44CC11?style=flat"></a>
</p>

**Build, train and use your own AI models in the terminal, without writing any code.**

![AIForge model library](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/10-home.png)

With the mouse or keyboard you can:

1. **Design** a neural network by picking layers from a list
2. **Train** it on a CSV table, text, images, or a famous public dataset
3. **Fine-tune** a trained model for a new, related task
4. **Evaluate** how good it really is
5. **Use** it to make predictions, or let a tiny GPT **write text**
6. **Export** it for Python, ONNX Runtime, mobile apps and more

No PyTorch or Python knowledge needed. The built-in **Guide** (`g`) teaches the ML basics.

## Contents

1. [Install](#1-install)
2. [Quick start: your first model in 5 minutes](#2-quick-start-your-first-model-in-5-minutes)
3. [The screens](#3-the-screens)
4. [Data](#4-data)
5. [Which template should I choose?](#5-which-template-should-i-choose)
6. [Reading the training charts](#6-reading-the-training-charts)
7. [Keyboard shortcuts](#7-keyboard-shortcuts)
8. [Troubleshooting](#8-troubleshooting)
9. [Where are my files?](#9-where-are-my-files)
10. [Command line (no UI)](#10-command-line-no-ui)
11. [Web API: use your model from other programs](#11-web-api-use-your-model-from-other-programs)
12. [For developers](#12-for-developers)

---

## 1. Install

**You need:** Python 3.10+, about 2 GB of disk space (mostly PyTorch), and a terminal window
of at least **120 × 35**. A GPU is optional: NVIDIA (CUDA) and Apple Silicon (MPS) are used
automatically.

```bash
pipx install aiforge-studio      # recommended (or: pip install aiforge-studio)
aiforge                          # start the app
```

<details>
<summary>No pipx yet? / NVIDIA GPU stats / install from source</summary>

```bash
# Install pipx once, then reopen the terminal
python3 -m pip install --user pipx && python3 -m pipx ensurepath

# Show GPU usage and temperature on NVIDIA cards
pipx inject aiforge-studio nvidia-ml-py

# Update later
pipx upgrade aiforge-studio

# From source (developers)
git clone https://github.com/ducdinh-codehub/ai-forge.git && cd ai-forge
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/aiforge                # Windows: .venv\Scripts\aiforge
```
</details>

**Moving around:** click with the mouse, or press the keys shown in the bottom bar.
`Tab` = next box · `Enter` = confirm · `Esc` = back · `q` = quit.

---

## 2. Quick start: your first model in 5 minutes

You'll teach a model to tell apart three spirals of points. The data is built in.

| Step | Do this |
|---|---|
| **1. Create** | Press `n`. Name it `spiral-classifier`, keep **Tabular data** → **MLP classifier**, click **Create & open builder**. |
| **2. Look** | The builder shows every layer, its shape and parameter count. Press `Ctrl+S`, then `Esc`. |
| **3. Train** | Press `t`. Choose *Synthetic toy data* → **spirals**, click **Load data**, then **Auto-fit model to data**. Set Epochs `40`, Learning rate `0.005`, click **▶ Start**. |
| **4. Use** | Press `p`, type `0.1, 0.45`, press Enter. The model answers **arm-0**. |
| **5. Export** | Press `x`, tick formats with the space bar, click **⇩ Export**. |

![Model builder](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/03-builder.png)

![Training](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/06-train-done.png)

Loss going **down** and accuracy going **up** means it's learning. AIForge always saves the
**best** epoch. 🎉 You built, trained, used and exported a neural network without code.

---

## 3. The screens

### Home (model library)
Your models on the left, details and training history on the right. Every action has a button
and a key (see [shortcuts](#7-keyboard-shortcuts)).

### Builder: design the model
| I want to… | Do this |
|---|---|
| Add a layer | `a`, highlight a layer, `Enter` |
| Edit a layer | Select it, change the boxes on the right |
| Change the input (1D / 2D / 3D), type or task | Select the **Input** row, pick **Input dimensions**, fill in each size (e.g. 3D image: Channels `3`, Height `64`, Width `64`) |
| Move · copy · remove a layer | `Shift+↑/↓` · `Ctrl+D` · `Del` |
| Undo · save | `Ctrl+Z` · `Ctrl+S` |
| Learn what a layer does | `Ctrl+E` (Explain tab) or `?` (full Guide) |

Mistakes show up in red right away. Layers you don't change **keep what they learned**.

### Train: teach the model
1. **① Data:** choose a source and click **Load data** (see [Data](#4-data)). A yellow warning
   means the model doesn't fit the data: click **Auto-fit model to data**. If the structure
   is wrong (e.g. a table model for images), **🔧 Fix model…** offers checked fixes.
2. **② Settings:** the defaults are fine to start. Common ones:

   | Setting | Meaning | Start with |
   |---|---|---|
   | Epochs | Passes through the data | 10–50 |
   | Batch size | Examples per update | 32–128 |
   | Learning rate | Step size: too high = unstable, too low = slow | 0.001 |
   | Early stop | Stop after N epochs with no improvement (0 = off) | 0 or 5 |
   | Device | CPU or GPU | auto |

3. Click **▶ Start** (`Ctrl+R`). Watch the charts, the log and the **💻 This computer** panel
   (CPU, RAM, GPU usage, samples/s).

Press **F1** for the **📖 Guide panel**: a live diagnosis of your run ("✅ healthy",
"⚠️ overfitting…") and every setting explained with your values.

### Fine-tune: reuse a trained model
Like Train, plus: **freeze** layers you want to keep (try **Freeze all but head**), optionally
reset the output layer, and save under a **new name** so the original stays safe.

### Use: get predictions
| Model trained on | Type |
|---|---|
| A table | Values in column order: `6, 3.5, red` |
| Text | A sentence: `what an amazing movie` |
| Images | An image path: `~/Pictures/cat.jpg` |
| Plain text (language model) | The start of a text: `Twinkle, twinkle`. The model writes what comes next; set the length and temperature (low = safe, high = creative) |

Give it a whole CSV or image folder for **batch prediction** (writes a CSV with predictions).

### Evaluate: how good is it really?
Press `v` (📊 **Evaluate**), choose the data (the validation / test data from training, or a
separate file or folder with the true answers) and click **▶ Evaluate** (`Ctrl+R`). The report:

| Section | What it shows |
|---|---|
| **Verdict + warnings** | Strong / Good / Weak / Not useful, plus e.g. "classes are imbalanced: accuracy is misleading" |
| **Metrics vs. a baseline** | Classification: accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC, log loss, calibration. Regression: MAE, RMSE, MSE, R², MAPE. Language model: perplexity, next-token accuracy, bits per token. With 95% confidence intervals. |
| **Per class · confusion matrix · calibration** | Which classes it gets wrong, mixes up, or is over-confident about |
| **Error analysis** | Its worst mistakes, with the actual example (row values, text or file name) |
| **Slices** | Results per group of each category column in a table (fairness check) |
| **Robustness · learning curves** | Score with noisy inputs; over- or underfitting from the training history |
| **Compare with another model** | Which of two models is better, with a significance test |
| **Language models** | Samples of what it writes at 3 temperatures, and the tokens it finds hardest |

![Evaluate](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/19-evaluate.png)

**💾 Save report** writes it to `exports/<model>/` as Markdown and JSON. The home screen shows
the last result. New to the metrics? Press `g` → **7. Is my model good?**

### Export
| Format | Use it to… |
|---|---|
| AIForge bundle `.zip` | Back up or share with another AIForge user |
| Weights `.pt` / `.safetensors` | Load in your own PyTorch / Hugging Face code |
| PyTorch code `.py` · JSON · summary | Read, edit or document the model |
| TorchScript · ONNX | Run in C++, mobile, ONNX Runtime, TensorRT, browsers… |

### Serve: use it from a web page or another program
`a` (or **🌐 Serve**) starts a small web server for the selected model: a demo page to try it in
the browser and a JSON API other programs can call. Only your computer can reach it, and it stops
when you close the window. Step-by-step: [Web API](#11-web-api-use-your-model-from-other-programs).

### Import
Press `i`: AIForge bundles, `.json`, `.safetensors`, or a PyTorch `nn.Sequential` saved with
`torch.save`. ⚠️ **Only import `.pt`/`.pth` files you trust**, as they can run code.

### Guide
Press `g`. Beginner lessons (ML basics, model types), a step-by-step walk-through of your
model with a design review, and an encyclopedia of every layer.

![Guide](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/13-guide-overview.png)

### Report a problem or idea
Press `Ctrl+O` on any screen (or find **Report a problem or idea** in the `Ctrl+P` palette).
Choose bug / idea / question, write a title and what happened. AIForge opens GitHub's
**New issue** page with everything filled in; check it and click **Submit** (a free GitHub
account is needed, and nothing is sent before you submit). You choose what is attached:

- versions and hardware (AIForge, Python, PyTorch, OS, GPU)
- the error messages the app showed recently
- the open model's layers (structure only, never your data or weights)
- the crash details, if AIForge closed because of an error last time; it offers this at the next start

Your home folder is shown as `~` in paths. The report is also copied to the clipboard and
saved in `exports/issue-reports/`, in case no browser opens (e.g. over SSH).

---

## 4. Data

### Option A: download a famous dataset (easiest)

Press `s` on the home screen (or **🗄 Get a famous dataset…** on the Train screen). Pick one,
press **⇩ Download**, then **▶ Train on it**. AIForge fills in all the fields and loads it.

![Dataset hub](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/17-datasets.png)

| Model type | Datasets |
|---|---|
| 📊 Tabular | Iris · Wine · Breast cancer · Adult income · Red wine quality · California housing |
| 🖼 Vision | MNIST · Fashion-MNIST · CIFAR-10 · CIFAR-100 · Imagenette · Oxford-IIIT Pets |
| 💬 Text classification | SMS Spam · SST-2 · IMDB reviews · AG News |
| 🧠 Language model | Tiny Shakespeare |

All come from their official sources (UCI, Stanford, Oxford, U. Toronto, Zalando, fast.ai).
Each one shows its publisher, license, size and the templates that suit it. Downloads show
their progress and can be cancelled:

![Downloading a dataset](https://raw.githubusercontent.com/ducdinh-codehub/ai-forge/main/docs/images/18-datasets-download.png)

### Option B: your own files

Click **📂 Browse** to pick a file; the preview shows its columns or classes.
Sample files are under **📦 Examples** in the browser.

| Kind | Format |
|---|---|
| **Table** (CSV) | One row per example. The **last column** is predicted (or choose another). Text columns and missing numbers are handled for you. |
| **Text** (CSV) | A text column and a label column: `"great movie",pos` |
| **Images** | A folder with **one sub-folder per class** (`cat/`, `dog/`). Images are resized automatically. |
| **Plain text** (.txt) | Any text for a language model to imitate: a book, poems, lyrics, code. Split into characters or words. A short sample (`nursery_rhymes.txt`) is built in; aim for 1 MB+ for real sentences. |

> **How much data?** For images, aim for 50–100+ per class. With fewer, use a
> **Transfer learning** template.

---

## 5. Which template should I choose?

| Your data | Template |
|---|---|
| 📊 Table → predict a category / a number | MLP classifier / MLP regressor |
| 🖼 Small grayscale images (28×28) | Small CNN |
| 🖼 Small colour images (32×32) | CNN (CIFAR-10) · ViT · Hybrid CNN + Transformer |
| 🖼 Your own photos, from scratch | Deep CNN (64×64) |
| 🖼 Photos, especially with few examples | Transfer learning: **MobileNetV3** (fast) · **ResNet18** · **EfficientNet-B0** (accurate) |
| 💬 Sentences / reviews | Text Transformer classifier |
| 💬 Longer or harder texts | BERT-style Transformer |
| 💬 Simpler / faster | LSTM · bidirectional GRU · Bag of words (fastest) |
| 🧠 Generate text (write like a book, poems, code…) | **Tiny GPT** (characters) · GPT-small (words, needs lots of text) · LSTM language model |
| ✎ Design it all yourself | Blank |

You can change any layer after creating the model.

---

## 6. Reading the training charts

| You see | Meaning | Do this |
|---|---|---|
| Both losses go down and level off | 👍 Learning well | Nothing |
| Train loss down, **val loss up** | Overfitting (memorizing) | Add Dropout, fewer epochs, Early stop, more data |
| Both stay high | Underfitting | More epochs, bigger model, higher learning rate |
| Lines jump around | Learning rate too high | Lower it (0.001 → 0.0003) |
| Loss is `nan` | Training broke | Much lower learning rate, start again |

New to the words (epoch, loss, overfitting…)? Press `g` → **🎓 ML basics**.

---

## 7. Keyboard shortcuts

| Home | | Builder | | Train | |
|---|---|---|---|---|---|
| `n` | New | `a` | Layer list | `Ctrl+R` | Start |
| `i` | Import | `Enter` | Add layer | `F1` | Guide panel |
| `e` | Edit | `Del` | Remove layer | `<` `>` | Panel wider / narrower |
| `t` | Train | `Shift+↑/↓` | Move layer | `Esc` | Back |
| `f` | Fine-tune | `Ctrl+D` | Duplicate layer | | |
| `p` | Use | `Ctrl+Z` | Undo | | |
| `v` | Evaluate | | | | |
| `x` | Export | `Ctrl+S` | Save | | |
| `a` | Serve (web API) | | | | |
| `s` | Datasets | `Ctrl+E` | Explain tab | | |
| `g` | Guide | `?` | Full guide | | |
| `w` `c` `r` `d` | Load weights · Duplicate · Rename · Delete | | | | |

Everywhere: `Esc` back · `Tab` next · `Ctrl+P` command palette · `Ctrl+T` dark / light · `Ctrl+O` report a problem · `Ctrl+Q` quit.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: aiforge` | Run `python3 -m pipx ensurepath` and reopen the terminal (from source: `.venv/bin/aiforge`) |
| Old models missing after an update | Start AIForge in the folder containing `workspace/`, or run `aiforge -w path/to/workspace` |
| Screen looks cut off / odd symbols | Enlarge the terminal (120×35+); use iTerm2 or Windows Terminal with a font like JetBrains Mono |
| "Model input is … but data is …" | Click **Auto-fit model to data** |
| "Final output must be a flat vector" | Click **🔧 Fix model…** and pick a fix |
| Training is slow | Check *Training on* in the 💻 panel, set Device to `auto`, or set **Max samples** to e.g. `5000` |
| GPU "none found" | NVIDIA: install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) |
| A dataset download fails | Check your internet and try again; partial files are removed automatically |
| Predictions look random | Make sure the model shows *trained* and check its validation accuracy |
| Something else is broken | Press `Ctrl+O` to report it on GitHub, with the details filled in |

---

## 9. Where are my files?

Everything lives in one workspace folder: **`~/AIForge`** by default (or `./workspace` if it
exists, or `aiforge -w <folder>` / `AIFORGE_WORKSPACE`).

```
AIForge/
├── models/<name>/     spec.json (structure) · weights.pt · history.json · evaluations.json
├── exports/           files created by Export and saved evaluation reports
├── data/              downloaded datasets (data/hub/ for the dataset hub)
└── settings.json      your preferences
```

---

## 10. Command line (no UI)

Every main step also works as a plain command, which is handy on a remote GPU machine, in
scripts, or for an overnight run. Commands use the same workspace as the app, so a model
trained here shows up there.

```bash
aiforge templates                                   # what you can start from
aiforge new spiral -t "mlp classifier"              # a template's number or part of its name
aiforge train spiral -d spirals -e 40 --lr 0.005    # toy set, mnist, a .csv/.txt file or an image folder
aiforge predict spiral "0.1, 0.45"                  # or: -f table.csv -o predictions.csv
aiforge evaluate spiral                             # metrics + a saved report
aiforge export spiral -F onnx,bundle                # or: -F all
aiforge serve spiral                                # web API + demo page (see section 11)
aiforge list
```

| I want to… | Do this |
|---|---|
| Train again on the same data | `aiforge train spiral -e 20` (leave out `-d`) |
| Continue from the saved weights | `aiforge train spiral --resume -e 20` |
| Fine-tune into a new model | `aiforge train spiral -d moons --save-as moons` |
| Train a tiny GPT and let it write | `aiforge train gpt -d poems.txt` then `aiforge generate gpt "Once upon"` |
| Pick the CSV column to predict | `--target price` (text classifiers: also `--text-column review`) |
| Stop early | `Ctrl+C`: stops after the current batch and keeps the best epoch |

`aiforge <command> -h` lists every option. Commands exit with code 1 and a one-line message
when something is wrong, so they work in scripts.

---

## 11. Web API: use your model from other programs

`aiforge serve` turns a trained model into a small web server. You can then try the model on a
web page, or call it from any program (Python, JavaScript, a spreadsheet script…).
This walk-through uses the house-price example that ships with AIForge.

### Step 1 · Install the web server parts (once)

```bash
pipx inject aiforge-studio fastapi uvicorn python-multipart
```

<sub>Installed with pip instead? Use `pip install "aiforge-studio[serve]"`.</sub>

### Step 2 · Train a model

A model that guesses a house's **price** from its number of **rooms** and its **area** (m²):

```bash
aiforge new house-price -t regressor
aiforge train house-price -d <path-to>/house_prices.csv -e 60 --lr 0.01
```

<sub>`house_prices.csv` is in [`aiforge/examples/`](aiforge/examples/house_prices.csv). You can
also do this step in the app: create the model with `n`, train it with `t`.</sub>

### Step 3 · Start the server

```bash
aiforge serve house-price
```

You should see:

```
Serving house-price at http://localhost:8000  (API docs: http://localhost:8000/docs) · Ctrl+C to stop
```

Leave this terminal open. The server runs until you press `Ctrl+C`.

<sub>In the app: select the model, press `a` (or **🌐 Serve**), then **▶ Start**. It stops when
you close that window.</sub>

### Step 4 · Try it in the browser

| Open | What you get |
|---|---|
| **http://localhost:8000** | A demo page: type `4, 110`, click **Predict**, see the price |
| **http://localhost:8000/docs** | Every endpoint, with a **Try it out** button |

### Step 5 · Call it from your own code

Open a **second** terminal (the first one is running the server).

**With curl:**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": "4, 110"}'
```

Answer:

```json
{"predictions": [{"label": "prediction", "score": 420.07}]}
```

**With Python** (no extra packages needed):

```python
import json
import urllib.request

def predict(text):
    request = urllib.request.Request(
        "http://localhost:8000/predict",
        data=json.dumps({"input": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    answer = json.load(urllib.request.urlopen(request))
    return answer["predictions"][0]["score"]

print(predict("4, 110"))   # 420.07  → a 4-room, 110 m² house
print(predict("6, 160"))   # 623.18
```

**Many inputs at once:** send `"inputs"` (a list) instead of `"input"`:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs": ["3, 70", "6, 160"]}'
```

```json
{"predictions": [[{"label": "prediction", "score": 291.96}],
                 [{"label": "prediction", "score": 623.18}]]}
```

### Other kinds of models

The steps are the same; only what you send changes. `GET /info` always tells you what a model expects.

| Model | Endpoint | Send | You get back |
|---|---|---|---|
| Numbers (CSV table) | `POST /predict` | `{"input": "4, 110"}` | A value, or each class with its probability |
| Text classifier | `POST /predict` | `{"input": "I love this movie"}` | `[{"label": "pos", "score": 0.89}, …]` |
| Images | `POST /predict/image` | The image file: `curl -F file=@cat.jpg …` | Each class with its probability |
| Language model (tiny GPT) | `POST /generate` | `{"prompt": "Once upon", "length": 200}` | `{"text": "…what it wrote…"}` |

### Good to know

| | |
|---|---|
| **Wrong input?** | You get status `422` and a plain message, e.g. `{"detail": "Expected 2 values (rooms, area), got 1"}` |
| **Port 8000 busy?** | Pick another: `aiforge serve house-price -p 8001` |
| **Who can reach it?** | Only your computer. `--host 0.0.0.0` opens it to your network, and **there is no password**, so anyone on that network can use the model |
| **Check it's running** | `curl http://localhost:8000/health` → `{"status": "ok"}` |

---

## 12. For developers

```
aiforge/
├── layers.py     layer registry: settings, building, parameter formulas
├── spec.py       model description, workspace storage, templates, code generation
├── data.py       dataset loaders and input preprocessing
├── hub.py        dataset hub: catalog of trusted datasets, download + conversion
├── trainer.py    training / fine-tuning loop
├── pipeline.py   load data → fit model → train → save, without a UI
├── cli.py        the command line (aiforge train / predict / ...)
├── serve.py      web API + demo page (FastAPI), used by `aiforge serve` and the Serve dialog
├── predict.py    single and batch predictions
├── evaluate.py   metrics, baseline, calibration, error analysis, slices, significance tests
├── model_io.py   export and import
├── guide.py      layer encyclopedia, model walk-through, design review
├── learn.py      beginner lessons
├── fixes.py      fixes for models that don't fit the data
├── sysinfo.py    hardware specs and live usage
└── tui/          the Textual interface (one module per screen)
docs/make_screenshots.py   regenerates the README screenshots
```

**Tests:** `pip install -e ".[dev]" && pytest`

**Release:** bump `__version__` in `aiforge/__init__.py`, merge to `main`, then push a matching tag
(`git tag v0.1.8 && git push origin v0.1.8`). GitHub Actions builds it and publishes it to PyPI.
Every pull request runs the tests on Linux, macOS and Windows first.

Built with [PyTorch](https://pytorch.org) and [Textual](https://textual.textualize.io) ·
[MIT License](https://github.com/ducdinh-codehub/ai-forge/blob/main/LICENSE)
