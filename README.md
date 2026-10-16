# AIForge: a no-code neural network studio in your terminal

Build, train, fine-tune, use and export PyTorch models from a full-screen terminal UI.
You never have to write code.

```
Create model  →  Train  →  Fine-tune  →  Use (predict)  →  Export
   (or Import an existing model)
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/aiforge                 # or: .venv/bin/python -m aiforge
.venv/bin/aiforge -w ~/my-models  # choose where models are stored (default ./workspace)
```

Use the mouse or the keyboard. Key hints appear in the footer at the bottom of every screen.

## Screens

### Model library (home)
Lists your models with their parameter count, training status and run history.
`n` new · `i` import · `e` edit · `t` train · `f` fine-tune · `p` use · `x` export ·
`w` load weights · `c` duplicate · `r` rename · `d` delete · `q` quit

### Create / edit a model (builder)
- **Left:** the layer palette. Highlight a layer and press Enter to insert it below the selected row.
- **Middle:** the layer stack. Every layer shows its **output shape**, its **parameter count**,
  its **share of the total** as a bar, and **how the count is calculated** (e.g. `128×32 + 32`).
  The total, trainable count and size in MB update as you edit.
- **Right:** the selected layer's settings. Select the *Input* row to set the model's input
  shape, input type, task and class names.
- Mistakes show up straight away. For example, a Conv2d placed after a Linear layer is flagged
  with "needs 3D input".
- `Shift+↑/↓` move · `Ctrl+D` duplicate · `Del` remove · `Ctrl+Z` undo · `Ctrl+S` save
- If you edit a model that has already been trained, unchanged layers **keep their trained
  weights**. Weights are matched by layer identity, not by position.

Templates: blank, MLP classifier, MLP regressor, small CNN (MNIST), CNN (CIFAR-10),
transfer learning (ResNet18), text Transformer classifier, Vision Transformer (ViT),
text classifier (Embedding + LSTM).

Layers: Linear, Flatten, Conv2d, Conv1d, MaxPool2d, AvgPool2d, GlobalAvgPool, BatchNorm,
LayerNorm, ReLU, GELU, SiLU, Tanh, Sigmoid, LeakyReLU, Dropout, Embedding, LSTM, GRU,
PatchEmbedding, PositionalEncoding, TransformerEncoder, SequencePool,
PretrainedBackbone (resnet18/34/50, mobilenet_v3_small, efficientnet_b0 with ImageNet weights).

**Transformers** are built from four layers:
- Text: `Embedding → PositionalEncoding → TransformerEncoder → SequencePool → Linear`
- Images: `PatchEmbedding → PositionalEncoding → TransformerEncoder → SequencePool → Linear`

`TransformerEncoder` lets you set depth, attention heads, feed-forward size, dropout, and a
*causal* (GPT-style) mask.

### Train
1. **Data:** choose a toy dataset, a CSV table (numeric and categorical columns), a text CSV,
   an image folder (one sub-folder per class), or MNIST / Fashion-MNIST / CIFAR-10, which
   download automatically.
2. If the model doesn't match the data, press **Auto-fit model to data**. It sets the input shape
   and resizes the output layer to the number of classes.
3. Set epochs, batch size, learning rate, optimizer, LR schedule, early stopping and device
   (CPU / CUDA / Apple MPS). Then press **Start**.
4. The screen shows live progress, loss and accuracy charts (train vs validation) and a log
   for each epoch. **Stop** halts training at any time. The best weights are kept.

### Fine-tune
Same as Train, plus:
- choose which layers to **freeze** (or click *Freeze all but head*)
- optionally **re-initialise the output layer**, e.g. when the new data has different classes
- save the result as a **new model**, so the original stays untouched

### Use
- Type one input and see the prediction with confidence bars. The input can be comma-separated
  feature values, a sentence, or the path to an image.
- **Batch:** point it at a CSV file or an image folder to get a CSV of predictions.
- Inputs are preprocessed the same way as in training: normalization, category codes,
  vocabulary, image size.

### Export
| Format | What for |
|---|---|
| AIForge bundle `.zip` | Structure + weights + history. Re-import it anywhere. |
| Weights `.pt` (state_dict) | `model.load_state_dict(torch.load(...))` |
| Weights `.safetensors` | Safe, fast format used across the Hugging Face ecosystem |
| Structure `.json` | Layers, shapes, classes, preprocessing |
| Structure `.py` | Plain PyTorch `nn.Sequential` source code |
| Summary `.txt` | Layer table with parameter counts |
| TorchScript `.pt` | `torch.jit.load()`, no Python class needed |
| ONNX `.onnx` | ONNX Runtime, TensorRT, OpenVINO, … |

### Import
AIForge bundles and folders, `.json` structures, AIForge `.safetensors` files, and any
`nn.Sequential` saved with `torch.save(model)`. Supported layers are converted into editable
builder layers and their weights are kept. A bare state_dict can be attached to a model with a
matching structure using **Load weights** (`w`).

> Only import `.pt` files you trust. Unpickling can run code.

## Example data
`examples/` has small datasets to try:
- `fruit_size.csv`: classification with a categorical column
- `house_prices.csv`: regression, target `price`
- `movie_reviews.csv`: text classification
- `color_images/`: image folder

## Project layout
```
aiforge/
  layers.py     layer registry: settings, build, shape inference, parameter formulas
  spec.py       ModelSpec (JSON model description), workspace storage, templates, code generation
  data.py       dataset loaders and input preprocessing
  trainer.py    training / fine-tuning loop (freezing, early stopping, schedulers)
  predict.py    single and batch prediction
  model_io.py   export and import
  tui/          Textual screens: home, builder, train, predict, export, modals, chart
```
Each model lives in `workspace/models/<name>/` as `spec.json`, `weights.pt` and `history.json`.
