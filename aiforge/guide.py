"""Plain-language explanations of every layer and of a whole model, as Markdown.

``LAYER_GUIDE`` is the encyclopedia (one entry per layer type). The functions at the
bottom combine it with a concrete ``ModelSpec`` so explanations use the user's real
shapes, parameter counts and settings.
"""
from __future__ import annotations

from .layers import REGISTRY
from .spec import LayerInfo, ModelSpec

ACTIVATIONS = {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU"}


def _residual_diagram() -> str:
    """Pre-norm Transformer layer with both shortcut arrows aligned under their (+)."""
    top = "x ─▶ LayerNorm ─▶ Attention ─▶ (+) ─▶ LayerNorm ─▶ FeedForward ─▶ (+) ─▶"
    a = top.index("(+)") + 1
    b = top.index("(+)", a + 2) + 1
    second = top.index("─▶ LayerNorm", a)  # where the second branch starts
    mid = [" "] * (b + 1)
    mid[0], mid[a], mid[second], mid[b] = "│", "▲", "│", "▲"
    bot = [" "] * (b + 1)
    for i in range(0, a + 1):
        bot[i] = "─"
    for i in range(second, b + 1):
        bot[i] = "─"
    bot[0], bot[a], bot[second], bot[b] = "└", "┘", "└", "┘"
    bot = "".join(bot)
    label = " shortcut "
    for start, end in ((0, a), (second, b)):
        c = (start + end - len(label)) // 2
        bot = bot[:c] + label + bot[c + len(label):]
    return "\n".join([top, "".join(mid), bot])


# Each entry: summary (one line), analogy, how (how it works), settings {param: text},
# use (when to use it), tips (list), diagram (optional, monospace).
LAYER_GUIDE: dict[str, dict] = {
    "Linear": dict(
        summary="Every output is a weighted sum of *all* inputs, plus a bias.",
        analogy="A panel of judges: each judge (output) decides how much each piece of "
                "evidence (input) matters, adds it all up, and gives a score.",
        how="For each output neuron *j*: `y_j = w_1j·x_1 + w_2j·x_2 + … + w_nj·x_n + b_j`. "
            "The weights **W** (inputs × outputs) and biases **b** (one per output) are what "
            "training adjusts. The layer works on the last dimension of its input.",
        settings={"out_features": "How many outputs (neurons) this layer produces. More neurons = "
                                  "more capacity to learn, but more parameters.",
                  "bias": "Adds a learnable constant to each output. Almost always on."},
        use="Tabular data, the end of almost every model (the *head*), and after Flatten / "
            "pooling in image models.",
        tips=["Put an **activation** (e.g. ReLU) between two Linear layers. Without one, two "
              "Linear layers are mathematically the same as a single Linear layer.",
              "The **last** Linear layer must have `out_features` = number of classes "
              "(classification) or number of values to predict (regression).",
              "Don't put Softmax/Sigmoid after the last layer - the loss function already does it."],
        diagram="x1 ──┬──▶ Σ(w·x)+b ──▶ y1\nx2 ──┼──▶ Σ(w·x)+b ──▶ y2\nx3 ──┘   (every input connects\n"
                "          to every output)",
    ),
    "Flatten": dict(
        summary="Unrolls a multi-dimensional block (e.g. feature maps) into one long list of numbers.",
        analogy="Reading a page of text line by line into one long sentence.",
        how="A tensor of shape `C×H×W` becomes a vector of length `C·H·W`. Nothing is learned - "
            "the numbers are only re-arranged.",
        settings={},
        use="Between convolution layers (which output 3D maps) and Linear layers (which need a flat list).",
        tips=["A big flattened vector makes the next Linear layer huge (e.g. 64×14×14 = 12,544 "
              "inputs × 128 outputs ≈ 1.6M parameters). **GlobalAvgPool** is a much lighter alternative."],
        diagram="┌─┬─┐\n├─┼─┤  ──▶  [ • • • • • • • • ]\n└─┴─┘",
    ),
    "Dropout": dict(
        summary="During training, randomly switches off a fraction of the values.",
        analogy="A team where random members are absent each day, so nobody can rely on one "
                "star player - everyone learns to contribute.",
        how="With probability *p* each value is set to 0; the survivors are scaled by `1/(1-p)` so "
            "the average stays the same. When you **use** the model, dropout is turned off.",
        settings={"p": "Fraction to drop. 0.1–0.3 is typical; 0.5 is strong."},
        use="When the model **overfits** (validation loss rises while training loss falls).",
        tips=["It slows learning a little - you may need more epochs.",
              "Usually placed after an activation in the dense part of the model.",
              "If the model can't even fit the training data, remove dropout."],
    ),
    "Conv2d": dict(
        summary="Slides small learnable filters over an image to detect local patterns.",
        analogy="Scanning a photo with a small stencil: wherever the picture matches the stencil "
                "(an edge, a corner, a texture), the output lights up.",
        how="Each filter is a small `k×k` grid of weights for every input channel. At each position "
            "it multiplies the patch under it by its weights and sums → one output pixel. Every filter "
            "produces one *feature map*. Output size: `(H + 2·padding − kernel) / stride + 1`.\n\n"
            "Early layers learn edges and colours; deeper layers combine them into shapes and objects.",
        settings={"out_channels": "Number of filters = number of feature maps produced.",
                  "kernel_size": "Filter size in pixels (3 is the standard choice).",
                  "stride": "Step between positions. 2 halves the image size.",
                  "padding": "Zero border added around the image. `kernel_size // 2` keeps the size the same.",
                  "bias": "Learnable constant per filter. Can be off when followed by BatchNorm."},
        use="Images and anything grid-shaped.",
        tips=["The classic block is **Conv2d → BatchNorm → ReLU**.",
              "A common pattern: each time you halve the image size (pooling), double the channels.",
              "Parameters don't depend on image size - only on channels and kernel size."],
        diagram="image          filter        feature map\n┌──────────┐   ┌───┐        ┌────────┐\n"
                "│ ░░▓▓     │ × │3×3│  ──▶   │  ██    │\n│ ░░▓▓     │   └───┘ slide  │  ██    │\n"
                "└──────────┘                └────────┘",
    ),
    "Conv1d": dict(
        summary="Like Conv2d, but slides along one dimension (time or position).",
        analogy="Running your finger along a timeline looking for a short recurring shape.",
        how="Input is `channels × length`. Each filter covers `kernel_size` consecutive steps.",
        settings={"out_channels": "Number of filters.", "kernel_size": "How many steps each filter sees.",
                  "stride": "Step size along the sequence.", "padding": "Zeros added at both ends.",
                  "bias": "Learnable constant per filter."},
        use="Audio, sensor signals, time series, DNA sequences.",
        tips=["Input must be 2D: `(channels, length)`."],
    ),
    "MaxPool2d": dict(
        summary="Shrinks feature maps by keeping only the strongest value in each window.",
        analogy="Summarising each neighbourhood by its loudest voice.",
        how="The map is split into `kernel×kernel` windows (moved by `stride`); each window becomes its "
            "maximum. With kernel 2 / stride 2 the width and height halve. No parameters.",
        settings={"kernel_size": "Window size.", "stride": "Step between windows."},
        use="After a few conv layers, to reduce size and computation and to make the model tolerate "
            "small shifts of the object.",
        tips=["Too many pooling layers make the image disappear (size 0) - the builder will warn you."],
        diagram="┌───┬───┐\n│1 3│2 0│          ┌─┬─┐\n│5 2│1 4│   ──▶    │5│4│\n├───┼───┤          ├─┼─┤\n"
                "│0 1│7 2│          │3│7│\n│3 2│0 1│          └─┴─┘\n└───┴───┘",
    ),
    "AvgPool2d": dict(
        summary="Shrinks feature maps by averaging each window.",
        analogy="Blurring then downsizing a photo.",
        how="Like MaxPool2d but takes the mean of each window instead of the maximum. No parameters.",
        settings={"kernel_size": "Window size.", "stride": "Step between windows."},
        use="When you want a smoother summary than max pooling.",
        tips=["MaxPool is more common inside CNNs; averaging is common at the very end (GlobalAvgPool)."],
    ),
    "GlobalAvgPool": dict(
        summary="Averages each whole feature map into one number: `C×H×W → C`.",
        analogy="Asking each detector one question: *on average, how much of your pattern did you see?*",
        how="For every channel, average all `H×W` values. The result is a vector with one number per "
            "channel. No parameters.",
        settings={},
        use="At the end of a CNN, instead of Flatten. It drastically reduces parameters in the next "
            "Linear layer and works for any image size.",
        tips=["Most modern CNNs (ResNet, EfficientNet) end this way."],
    ),
    "BatchNorm": dict(
        summary="Re-centres and re-scales each channel so values stay in a healthy range.",
        analogy="Adjusting every microphone in a band to the same volume before mixing.",
        how="For each channel it subtracts the batch mean and divides by the batch standard deviation, "
            "then applies a learnable scale **γ** and shift **β** (2 parameters per channel). While "
            "predicting, it uses running averages collected during training instead of batch statistics.",
        settings={},
        use="After Conv2d / Linear and before the activation. Makes training faster and more stable, "
            "and allows higher learning rates.",
        tips=["Needs reasonably large batches (≥ 8–16). With tiny batches use LayerNorm instead.",
              "During fine-tuning, frozen BatchNorm layers keep their learned statistics."],
    ),
    "LayerNorm": dict(
        summary="Normalises each example over its features (the last dimension).",
        analogy="Each student grades their own answers on the same scale - independent of the classmates.",
        how="For every single example (and every position in a sequence) it normalises the feature "
            "vector to mean 0 / std 1, then applies learnable scale and shift (2 × features parameters).",
        settings={},
        use="Transformers, RNNs, and any model with small or variable batch sizes.",
        tips=["Unlike BatchNorm it behaves the same in training and prediction."],
    ),
    "ReLU": dict(
        summary="Keeps positive values, turns negatives into 0: `max(0, x)`.",
        analogy="A one-way valve: signal passes only in one direction.",
        how="Applied to every value independently. It's the simplest *non-linearity* - without "
            "non-linear activations, any stack of layers could only learn straight-line relationships.",
        settings={},
        use="The default activation after Linear and Conv layers.",
        tips=["If many neurons get stuck at 0 (\"dying ReLU\"), try LeakyReLU or GELU."],
        diagram="  y│      ╱\n   │     ╱\n───┼────╯──── x\n   │",
    ),
    "LeakyReLU": dict(
        summary="Like ReLU, but negative values keep a small slope instead of becoming 0.",
        analogy="A valve that leaks a little backwards.",
        how="`x` if x > 0, otherwise `negative_slope · x`.",
        settings={"negative_slope": "Slope for negative inputs (0.01 typical)."},
        use="When ReLU units die (stay at 0) during training.",
        tips=[],
        diagram="  y│      ╱\n   │     ╱\n───┼───╱───── x\n  ╱│",
    ),
    "GELU": dict(
        summary="A smooth version of ReLU: `x · Φ(x)`.",
        analogy="A valve that opens gradually instead of snapping open.",
        how="Small negative values are softly reduced instead of cut to zero. Smooth curves often "
            "train slightly better in deep networks.",
        settings={},
        use="Transformers (BERT, GPT and ViT use it) and modern MLPs.",
        tips=[],
    ),
    "SiLU": dict(
        summary="`x · sigmoid(x)` (also called Swish) - smooth and self-gating.",
        analogy="The value decides for itself how much of it gets through.",
        how="Close to ReLU for large positive x, smooth around 0, slightly negative for small negatives.",
        settings={},
        use="Modern CNNs such as EfficientNet.",
        tips=[],
    ),
    "Tanh": dict(
        summary="Squashes any value into the range (−1, 1).",
        analogy="A dimmer switch with a hard maximum and minimum.",
        how="`tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`. Zero-centred output.",
        settings={},
        use="Inside RNN-style models, or when you need outputs between −1 and 1.",
        tips=["For large |x| the curve is flat, so gradients vanish - deep stacks of Tanh learn slowly."],
        diagram=" 1 ┤      ╭────\n   │     ╱\n───┼────╱────── x\n   │   ╱\n−1 ┤───╯",
    ),
    "Sigmoid": dict(
        summary="Squashes any value into (0, 1), like a probability.",
        analogy="Turning a score into a yes/no confidence.",
        how="`σ(x) = 1 / (1 + e⁻ˣ)`.",
        settings={},
        use="Gates and outputs that must lie between 0 and 1.",
        tips=["**Don't** put it at the end of a classification model here - the training loss "
              "(cross-entropy) already converts scores to probabilities."],
    ),
    "Embedding": dict(
        summary="A lookup table that turns each token id (word number) into a learned vector.",
        analogy="A dictionary where each word's definition is a list of numbers - and the "
                "definitions are written during training.",
        how="The table has `num_embeddings` rows (one per word in the vocabulary) and "
            "`embedding_dim` columns. Token id 5 simply picks row 5. During training, words used in "
            "similar ways end up with similar vectors.\n\nIn AIForge, id **0** is padding (empty "
            "positions) and id **1** is an unknown word.",
        settings={"num_embeddings": "Vocabulary size - how many different words are kept.",
                  "embedding_dim": "Length of each word vector. 32–300 is typical."},
        use="The first layer of every text model. Requires the model input type `long` (token ids).",
        tips=["Often the largest layer in text models (vocab × dim parameters). Reducing the vocabulary "
              "is the easiest way to shrink the model."],
        diagram="\"good\" → id 17 → row 17 → [0.21, -0.40, 0.88, …]",
    ),
    "LSTM": dict(
        summary="Reads a sequence step by step, carrying a memory that it learns to update and forget.",
        analogy="Reading a sentence word by word while taking notes, crossing out what no longer "
                "matters and highlighting what does.",
        how="At each step, three *gates* (forget, input, output) decide what to erase from memory, "
            "what to add and what to output. This lets it connect information that is far apart.\n\n"
            "Output: the last step's hidden state (`hidden_size`, ×2 if bidirectional), or the whole "
            "sequence when `return_sequence` is on.",
        settings={"hidden_size": "Size of the memory/state vector.",
                  "num_layers": "How many LSTMs are stacked on top of each other.",
                  "bidirectional": "Also read the sequence backwards (doubles the output size).",
                  "return_sequence": "Output every step (needed to stack another sequence layer).",
                  "dropout": "Dropout between stacked layers (only used when num_layers > 1)."},
        use="Text, time series, any ordered data - especially smaller datasets.",
        tips=["Parameters per layer ≈ 4 × (input×hidden + hidden×hidden + 2×hidden).",
              "GRU is a lighter alternative that often works as well."],
    ),
    "GRU": dict(
        summary="A simpler LSTM: reads a sequence step by step with a learned memory.",
        analogy="Taking notes while reading - with one eraser instead of LSTM's two.",
        how="Uses 2 gates (update, reset) plus a candidate state - 3 weight blocks instead of LSTM's 4, "
            "so ~25% fewer parameters and faster training.",
        settings={"hidden_size": "Size of the memory/state vector.",
                  "num_layers": "Stacked GRUs.", "bidirectional": "Also read backwards.",
                  "return_sequence": "Output every step instead of only the last.",
                  "dropout": "Dropout between stacked layers."},
        use="Same as LSTM; try both.",
        tips=[],
    ),
    "PatchEmbedding": dict(
        summary="Cuts an image into square patches and turns each patch into a vector.",
        analogy="Cutting a photo into puzzle pieces and writing a short description of each piece.",
        how="A 32×32 image with patch size 4 becomes 8×8 = 64 patches. Each patch's pixels are "
            "multiplied by a shared weight matrix → one `embed_dim` vector per patch. The result is a "
            "*sequence* the Transformer can read like a sentence of 64 words.",
        settings={"patch_size": "Side of each patch in pixels. Must divide the image height and width.",
                  "embed_dim": "Length of each patch vector (the Transformer's width)."},
        use="The first layer of a Vision Transformer (ViT).",
        tips=["Smaller patches → more patches → much more computation (attention cost grows with the "
              "square of the number of patches)."],
        diagram="┌──┬──┬──┐\n│▒▒│░░│▓▓│  ──▶  [v1, v2, v3,\n├──┼──┼──┤        v4, v5, v6, …]\n│░░│▓▓│▒▒│\n└──┴──┴──┘",
    ),
    "PositionalEncoding": dict(
        summary="Adds information about *where* each token/patch is in the sequence.",
        analogy="Numbering the pages of a shuffled book so the reader knows their order.",
        how="Attention treats its input as an unordered set - \"dog bites man\" and \"man bites dog\" "
            "would look identical. This layer adds a position vector to each element.\n\n"
            "**learned**: the position vectors are trained (`seq_len × dim` parameters).  "
            "**sinusoidal**: fixed sine/cosine waves (no parameters).",
        settings={"kind": "learned (trainable) or sinusoidal (fixed formula)."},
        use="Right before a TransformerEncoder.",
        tips=["Without it, a Transformer can't tell word order apart."],
    ),
    "TransformerEncoder": dict(
        summary="Self-attention layers: every position looks at every other position and decides "
                "what's relevant.",
        analogy="A meeting where each participant listens to everyone, pays more attention to the "
                "people relevant to them, and updates their opinion.",
        how="Each encoder layer has two parts, each with a shortcut (residual) connection and LayerNorm:\n\n"
            "1. **Multi-head self-attention** - every position creates a *query*, *key* and *value*. "
            "Attention weights = `softmax(Q·Kᵀ / √d)`: how well each query matches each key. The output "
            "is the weighted mix of values. Several *heads* do this in parallel, each free to focus on a "
            "different kind of relationship.\n"
            "2. **Feed-forward network** - a small 2-layer MLP (with GELU) applied to each position.\n\n"
            "Input and output have the same shape `(seq_len, dim)`.",
        settings={"num_layers": "How many encoder layers are stacked (depth).",
                  "num_heads": "Parallel attention heads. The vector size must be divisible by it.",
                  "ff_dim": "Hidden size of the feed-forward part (usually 2–4 × the vector size).",
                  "dropout": "Dropout inside attention and feed-forward.",
                  "causal": "Each position may only look at earlier positions (GPT-style)."},
        use="Text and images (with PatchEmbedding). Usually preceded by PositionalEncoding and "
            "followed by SequencePool for classification.",
        tips=["Transformers need more data than CNNs/LSTMs to shine - on small datasets start small "
              "(2 layers, dim 64–128).",
              "Memory grows with the **square** of the sequence length."],
        diagram=_residual_diagram(),
    ),
    "SequencePool": dict(
        summary="Turns a sequence of vectors into a single vector: `(seq_len, dim) → (dim,)`.",
        analogy="Summarising a whole discussion in one statement.",
        how="**mean**: average over positions · **max**: strongest value per feature · **first** / "
            "**last**: take one position. No parameters.",
        settings={"mode": "How to summarise: mean, max, first or last."},
        use="Between a TransformerEncoder (or sequence-returning LSTM) and the classifier head.",
        tips=["`mean` is a robust default. `last` suits causal (GPT-style) models."],
    ),
    "PretrainedBackbone": dict(
        summary="A well-known image network already trained on ImageNet, used as a feature extractor.",
        analogy="Hiring an experienced photographer instead of training a beginner - they already know "
                "what edges, textures, eyes and wheels look like.",
        how="The network (e.g. ResNet18) was trained on 1.2 million photos of 1,000 categories. Its "
            "final classifier is removed, so it outputs a feature vector (512 numbers for ResNet18) "
            "describing the image. Your own Linear layer then maps those features to your classes.",
        settings={"arch": "Which network. resnet18 / mobilenet_v3_small are fast; resnet50 / "
                          "efficientnet_b0 are stronger but slower.",
                  "pretrained": "Download the ImageNet weights (strongly recommended)."},
        use="Image classification with **little** data - often good with just dozens of images per class.",
        tips=["During fine-tuning, freeze the backbone at first and train only the head.",
              "Use colour input (3 channels), ideally 224×224. Weights download the first time you train."],
    ),
}


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _shape(s) -> str:
    return "×".join(map(str, s)) if s else "?"


def describe_input(spec: ModelSpec) -> str:
    s = spec.input_shape
    if spec.input_dtype == "long":
        return f"a sequence of **{s[0]} token ids** (words turned into numbers)"
    if len(s) == 1:
        return f"**{s[0]} numbers** per example (one row of a table)"
    if len(s) == 2:
        return f"a sequence of **{s[1]} channels × {s[0]} steps**"
    if len(s) == 3:
        kind = "grayscale" if s[0] == 1 else "colour" if s[0] == 3 else f"{s[0]}-channel"
        return f"a **{kind} image of {s[1]}×{s[2]} pixels**"
    return f"data of shape {_shape(s)}"


def describe_task(spec: ModelSpec, out) -> str:
    if spec.task == "classification":
        n = out[0] if out else "?"
        names = f" ({', '.join(spec.class_names[:6])}{'…' if len(spec.class_names) > 6 else ''})" \
            if spec.class_names else ""
        return f"chooses one of **{n} classes**{names}"
    n = out[0] if out else 1
    return f"predicts **{n} number{'s' if n != 1 else ''}** (regression)"


def role(info: LayerInfo, spec: ModelSpec) -> str:
    """One sentence describing what this layer does *in this model*."""
    t, c = info.spec.type, info.spec.config
    i, o = info.in_shape, info.out_shape
    if info.error:
        return f"⚠ {info.error}"
    last = info.index == len(spec.layers) - 1
    if t == "Linear":
        if last and spec.task == "classification":
            return f"turns {i[-1]} features into **one score per class** ({o[-1]} scores)"
        if last:
            return f"turns {i[-1]} features into the final prediction"
        return f"combines all {i[-1]} values into {o[-1]} new values"
    if t == "Flatten":
        return f"unrolls the {_shape(i)} block into a list of {o[0]:,} numbers"
    if t == "Dropout":
        return f"randomly switches off {c['p']:.0%} of the values while training"
    if t in ("Conv2d", "Conv1d"):
        return (f"scans with {c['out_channels']} filters of size {c['kernel_size']} "
                f"→ {o[0]} feature maps of {_shape(o[1:])}")
    if t in ("MaxPool2d", "AvgPool2d"):
        how = "strongest" if t == "MaxPool2d" else "average"
        return f"shrinks each map from {_shape(i[1:])} to {_shape(o[1:])} keeping the {how} signal"
    if t == "GlobalAvgPool":
        return f"averages each of the {i[0]} feature maps into one number"
    if t == "BatchNorm":
        return f"re-centres and re-scales each of the {i[0]} channels"
    if t == "LayerNorm":
        return f"normalises the {i[-1]} features of each example"
    if t in ACTIVATIONS:
        return "bends the signal (adds non-linearity) so the model can learn curves"
    if t == "Embedding":
        return f"turns each of the {i[0]} token ids into a {o[-1]}-number vector"
    if t in ("LSTM", "GRU"):
        out = "every step" if c["return_sequence"] else "a summary of the whole sequence"
        return f"reads the {i[0]} steps in order with a memory of {c['hidden_size']}, outputs {out}"
    if t == "PatchEmbedding":
        return (f"cuts the image into {o[0]} patches of {c['patch_size']}×{c['patch_size']} px, "
                f"each → a {o[1]}-number vector")
    if t == "PositionalEncoding":
        return f"stamps each of the {i[0]} positions with its location"
    if t == "TransformerEncoder":
        return (f"lets each of the {i[0]} positions look at all the others "
                f"({c['num_layers']} layers × {c['num_heads']} heads)")
    if t == "SequencePool":
        return f"summarises {i[0]} vectors into one ({c['mode']})"
    if t == "PretrainedBackbone":
        return f"a pre-trained {c['arch']} turns the image into {o[0]} features"
    return REGISTRY[t].description


def review(spec: ModelSpec) -> list[str]:
    """Design hints a beginner might miss. Each item is a Markdown sentence."""
    infos = spec.analyze()
    notes = []
    for i in infos:
        if i.error:
            notes.append(f"❌ **Layer {i.index + 1} ({i.spec.type})**: {i.error}")
    if any(i.error for i in infos):
        return notes
    types = [l.type for l in spec.layers]
    # Linear → (no activation) → Linear collapses into one linear map.
    nonlinear = ACTIVATIONS | {"LSTM", "GRU", "TransformerEncoder", "PretrainedBackbone"}
    last_linear = None
    for idx, t in enumerate(types):
        if t == "Linear":
            if last_linear is not None:
                notes.append(f"⚠️ **Layers {last_linear + 1} and {idx + 1}** are Linear layers with no "
                             "activation between them - together they can only learn what one Linear "
                             "layer could. Add a **ReLU** between them.")
            last_linear = idx
        elif t in nonlinear:
            last_linear = None
    if types and types[-1] in ACTIVATIONS:
        if spec.task == "classification":
            notes.append(f"⚠️ The model ends with **{types[-1]}**. For classification, end with a "
                         "Linear layer - the loss function turns its scores into probabilities.")
        elif types[-1] in ("ReLU", "Sigmoid", "Tanh"):
            notes.append(f"⚠️ The model ends with **{types[-1]}**, which limits the range of "
                         "predicted numbers. Regression models usually end with a plain Linear layer.")
    for i in infos:
        if i.spec.type == "Linear" and i.index > 0 and types[i.index - 1] == "Flatten" and i.params > 1_000_000:
            notes.append(f"💡 **Layer {i.index + 1}** (Linear after Flatten) has {i.params:,} parameters. "
                         "Replacing Flatten with **GlobalAvgPool** (or adding more pooling) would make the "
                         "model far smaller.")
    if "TransformerEncoder" in types:
        first_tf = types.index("TransformerEncoder")
        if "PositionalEncoding" not in types[:first_tf]:
            notes.append("⚠️ There is no **PositionalEncoding** before the TransformerEncoder, so the "
                         "model can't tell the order of words/patches.")
    total = sum(i.params for i in infos)
    has_dropout = "Dropout" in types or any(
        l.type == "TransformerEncoder" and l.config["dropout"] > 0 for l in spec.layers)
    if total > 100_000 and not has_dropout:
        notes.append("💡 No **Dropout** layer. If validation loss starts rising during training "
                     "(overfitting), adding Dropout before the last layer usually helps.")
    convs = [l for l in spec.layers if l.type == "Conv2d"]
    if len(convs) >= 2 and "BatchNorm" not in types:
        notes.append("💡 Deeper CNNs usually train faster with **BatchNorm** after each Conv2d.")
    if not notes:
        notes.append("✅ No obvious problems found.")
    return notes


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def encyclopedia_markdown(type_name: str) -> str:
    """The general article about a layer type."""
    g, lt = LAYER_GUIDE.get(type_name), REGISTRY[type_name]
    if not g:
        return f"# {type_name}\n\n{lt.description}"
    md = [f"# {type_name}", f"*{lt.category} layer*", "", f"**{g['summary']}**", "",
          f"> 💭 {g['analogy']}", "", "## How it works", g["how"]]
    if g.get("diagram"):
        md += ["", "```", g["diagram"], "```"]
    if g["settings"]:
        md += ["", "## Settings", "| Setting | Meaning |", "|---|---|"]
        md += [f"| `{k}` | {v} |" for k, v in g["settings"].items()]
    md += ["", "## When to use it", g["use"]]
    if g["tips"]:
        md += ["", "## Tips"] + [f"- {t}" for t in g["tips"]]
    return "\n".join(md)


def layer_markdown(spec: ModelSpec, idx: int) -> str:
    """Article about one layer of this model: its real numbers first, then the general guide."""
    info = spec.analyze()[idx]
    layer = info.spec
    md = [f"# Layer {idx + 1}: {layer.type}", ""]
    md += ["## In your model"]
    if info.error:
        md += [f"❌ **Problem:** {info.error}", ""]
    else:
        md += [f"This layer **{role(info, spec)}**.", "",
               "| | |", "|---|---|",
               f"| Input shape | `{_shape(info.in_shape)}` |",
               f"| Output shape | `{_shape(info.out_shape)}` |",
               f"| Parameters | **{info.params:,}**" + (f" = `{info.formula}`" if info.formula else "") + " |"]
        total = sum(i.params for i in spec.analyze()) or 1
        if info.params:
            md.append(f"| Share of model | {info.params / total:.1%} |")
        if layer.id in spec.frozen:
            md.append("| Fine-tuning | ❄ frozen (not updated) |")
        cfg = REGISTRY[layer.type].params
        if cfg:
            md += ["", "**Your settings:** " + ", ".join(f"`{p.name}={layer.config.get(p.name)}`" for p in cfg)]
    body = encyclopedia_markdown(layer.type).split("\n", 1)[1]  # drop the H1 title
    md += ["", "---", body]
    return "\n".join(md)


def layer_card_markdown(spec: ModelSpec, idx: int) -> str:
    """Short version for the builder's narrow Explain tab."""
    info = spec.analyze()[idx]
    g = LAYER_GUIDE.get(info.spec.type, {})
    md = [f"### {info.spec.type}", "", f"**{g.get('summary', REGISTRY[info.spec.type].description)}**", ""]
    if info.error:
        md += [f"❌ {info.error}", ""]
    else:
        md += [f"Here it {role(info, spec)}.", "",
               f"- in: `{_shape(info.in_shape)}`", f"- out: `{_shape(info.out_shape)}`",
               f"- params: **{info.params:,}**" + (f" = `{info.formula}`" if info.formula else ""), ""]
    if g.get("analogy"):
        md += [f"> 💭 {g['analogy']}", ""]
    if g.get("tips"):
        md += [f"💡 {g['tips'][0]}", ""]
    md.append("*Press **?** for the full guide.*")
    return "\n".join(md)


def model_markdown(spec: ModelSpec) -> str:
    """Walk-through of the whole model."""
    infos = spec.analyze()
    ok = not any(i.error for i in infos)
    out = infos[-1].out_shape if infos and ok else None
    total = sum(i.params for i in infos)
    md = [f"# Model: {spec.name}", "",
          f"**In one sentence:** this model reads {describe_input(spec)} and {describe_task(spec, out)}.", "",
          f"It has **{len(spec.layers)} layers** and **{total:,} parameters** "
          f"(≈ {total * 4 / 1024 ** 2:.2f} MB). Parameters are the numbers the model adjusts while it learns.",
          "", "## Step by step", "Data flows from top to bottom. The shape shows the size of the data "
          "after each step.", "",
          "| # | Layer | Shape after | Params | What happens |", "|---|---|---|---|---|",
          f"| | **Input** | `{_shape(spec.input_shape)}` | | {describe_input(spec)} |"]
    for i in infos:
        md.append(f"| {i.index + 1} | {i.spec.type}{' ❄' if i.spec.id in spec.frozen else ''} | "
                  f"`{_shape(i.out_shape) if i.out_shape else '✗'}` | {i.params:,} | {role(i, spec)} |")

    if ok and total:
        heavy = sorted((i for i in infos if i.params), key=lambda i: -i.params)[:3]
        md += ["", "## Where the parameters are"]
        for i in heavy:
            bar = "█" * max(1, round(i.params / total * 20))
            md.append(f"- **Layer {i.index + 1} {i.spec.type}**: {i.params:,} ({i.params / total:.0%}) `{bar}`")

    md += ["", "## How it learns"]
    if spec.task == "classification":
        md += ["The last layer outputs one **score** per class. During training, *softmax* turns the "
               "scores into probabilities and the **cross-entropy loss** measures how much probability "
               "the model gave to the correct class (`loss = −log(p_correct)`). Training nudges every "
               "parameter to make that loss smaller.", "",
               "When you **use** the model, the class with the highest probability is the answer."]
    else:
        md += ["The last layer outputs the predicted number(s). The **mean squared error (MSE) loss** "
               "measures the average squared difference between prediction and true value. Training "
               "nudges every parameter to make that error smaller. (AIForge scales the target to a "
               "standard range while training and scales predictions back for you.)"]

    md += ["", "## Design review"] + [f"- {n}" for n in review(spec)]
    md += ["", "---", "*Select a layer on the left to learn about it, or browse the "
                      "**Layer encyclopedia** below it.*"]
    return "\n".join(md)


def model_card_markdown(spec: ModelSpec) -> str:
    """Short version for the builder's Explain tab when the Input row is selected."""
    infos = spec.analyze()
    ok = not any(i.error for i in infos)
    out = infos[-1].out_shape if infos and ok else None
    notes = review(spec)
    md = ["### This model", "",
          f"Reads {describe_input(spec)} and {describe_task(spec, out)}.", "",
          f"**{len(spec.layers)} layers · {sum(i.params for i in infos):,} parameters**", "",
          "**Design review**", ""] + [f"- {n}" for n in notes[:4]]
    md += ["", "*Press **?** for a step-by-step walk-through.*"]
    return "\n".join(md)
