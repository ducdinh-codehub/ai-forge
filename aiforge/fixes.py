"""Find and verify fixes for a model whose structure doesn't fit a dataset.

Every suggested fix is applied to a copy of the model and only offered if the copy is
valid for the data, so the user never picks a fix that still fails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .data import DataInfo
from .spec import LayerSpec, ModelSpec, from_template
from .trainer import data_mismatch, fit_spec_to_data

BIG_MODEL = 5_000_000


@dataclass
class Fix:
    kind: str  # "auto" | "template" | "builder"
    title: str
    description: str
    spec: ModelSpec | None = None
    params_before: int = 0
    params_after: int = 0
    inserted: tuple[int, ...] = ()  # indices of new layers in ``spec``

    @property
    def recommended(self) -> bool:
        return self.kind == "auto"


def _shape(s) -> str:
    return "×".join(map(str, s)) if s else "?"


def problems(spec: ModelSpec, info: DataInfo | None) -> list[str]:
    probs = spec.validate()
    if info is not None and not probs:
        probs = data_mismatch(spec, info)
    return probs


def explain(spec: ModelSpec, info: DataInfo | None) -> str:
    """Plain-language explanation of what is wrong (Markdown)."""
    infos = spec.analyze()
    bad = next((i for i in infos if i.error), None)
    data = f" Your data is **{_describe(info)}**." if info else ""
    if bad is not None:
        need = {3: "an image (channels × height × width)", 2: "a sequence (steps × features)"}
        lt_ranks = {"Conv2d": 3, "MaxPool2d": 3, "AvgPool2d": 3, "GlobalAvgPool": 3, "PatchEmbedding": 3,
                    "PretrainedBackbone": 3, "Conv1d": 2, "LSTM": 2, "GRU": 2, "TransformerEncoder": 2,
                    "PositionalEncoding": 2, "SequencePool": 2}
        wants = need.get(lt_ranks.get(bad.spec.type, 0))
        if bad.spec.type == "Embedding":
            wants = "text (words turned into token ids)"
        md = (f"**Layer {bad.index + 1} ({bad.spec.type})** can't accept what it receives: "
              f"*{bad.error}*.{data}")
        if wants:
            md += (f"\n\n{bad.spec.type} is made for {wants}, so this model was designed for a "
                   "different kind of data. Choosing a template made for your data is usually the "
                   "quickest fix.")
        return md
    out = infos[-1].out_shape if infos else None
    if info is not None and info.task == "language_model":
        probs = problems(spec, info)
        return (f"{probs[0].rstrip('.') + '.' if probs else 'No problems found.'}{data}\n\n"
                "A language model needs token ids "
                "in and **one score per vocabulary token at every position** out, so it can guess the "
                "next token everywhere at once. The language-model templates are built that way.")
    if out is not None and len(out) > 1:
        first_multi = next((i for i in infos if i.spec.type == "Linear" and len(i.in_shape) > 1), None)
        md = (f"The model's last layer outputs shape **{out}**, but it must output a flat list of "
              f"**{info.n_outputs if info else '?'} values** (one score per class, or the predicted "
              f"number).{data}")
        if first_multi is not None:
            md += (f"\n\nThe reason: **layer {first_multi.index + 1} (Linear)** receives "
                   f"`{_shape(first_multi.in_shape)}`. A Linear layer only works on the *last* "
                   "dimension, so it processed each row separately instead of the whole input. "
                   "The input has to be turned into one flat list first.")
        return md
    probs = problems(spec, info)
    return probs[0] if probs else "No problems found."


def _describe(info: DataInfo) -> str:
    s = info.input_shape
    if info.task == "language_model":
        return f"plain text, read {s[0]} tokens at a time to predict the next one"
    if info.input_dtype == "long":
        return f"text ({s[0]} tokens per example)"
    if len(s) == 3:
        return f"{'grayscale' if s[0] == 1 else 'colour'} images of {s[1]}×{s[2]} pixels"
    if len(s) == 1:
        return f"a table with {s[0]} input columns"
    return f"data of shape {_shape(s)}"


def _try(spec: ModelSpec, info: DataInfo) -> ModelSpec | None:
    """Fit a candidate to the data; return it only if it's fully valid."""
    fit_spec_to_data(spec, info)
    return None if problems(spec, info) else spec


REDUCER_TEXT = {
    "Flatten": "turns the {shape} input into one list of {n:,} numbers",
    "GlobalAvgPool": "averages each of the {c} feature maps into one number ({c} values)",
    "SequencePool": "averages the {steps} steps into one vector of {d} values",
}


def suggest(spec: ModelSpec, info: DataInfo) -> list[Fix]:
    """Verified fixes, best first. Always ends with the manual 'open the builder' option."""
    base = spec.clone()
    fit_spec_to_data(base, info)
    before = base.total_params() if not any(i.error for i in base.analyze()) else spec.total_params()
    fixes: list[Fix] = []
    infos = base.analyze()

    # 1. Insert a reducer (Flatten / GlobalAvgPool / SequencePool) where shapes go wrong.
    if infos and not any(i.error for i in infos) and len(infos[-1].out_shape) > 1:
        pos = next((i.index for i in infos if i.spec.type == "Linear" and len(i.in_shape) > 1), None)
        append = pos is None
        if append:
            pos = len(base.layers)
        shape = infos[pos].in_shape if not append else infos[-1].out_shape
        has_conv = any(l.type in ("Conv2d", "PatchEmbedding") for l in base.layers[:pos])
        if len(shape) == 3:
            reducers = ["GlobalAvgPool", "Flatten"] if has_conv else ["Flatten"]
        elif len(shape) == 2:
            reducers = ["SequencePool", "Flatten"]
        else:
            reducers = ["Flatten"]
        for red in reducers:
            cand = base.clone()
            cand.layers.insert(pos, LayerSpec.new(red))
            inserted = [pos]
            if append:
                cand.layers.append(LayerSpec.new("Linear", out_features=info.n_outputs))
                inserted.append(len(cand.layers) - 1)
            cand = _try(cand, info)
            if cand is None:
                continue
            after = cand.total_params()
            what = REDUCER_TEXT[red].format(shape=_shape(shape), n=math.prod(shape), c=shape[0],
                                            steps=shape[0], d=shape[-1])
            where = f"at the end (plus an output layer)" if append else f"before layer {pos + 1}"
            desc = f"Adds **{red}** {where}: it {what}."
            if red == "Flatten" and len(shape) == 3 and not has_conv:
                desc += ("\n\n💡 This works, but for images a **CNN template** usually learns better "
                         "with far fewer parameters - it looks at small patches instead of every "
                         "pixel separately.")
            if after > BIG_MODEL:
                desc += (f"\n\n⚠️ The next Linear layer becomes very large ({after:,} parameters in "
                         "total) - training will be slow and may overfit. A template made for this "
                         "data is usually better.")
            fixes.append(Fix("auto", f"Add {red} automatically", desc, cand, before, after, tuple(inserted)))

    # 2. Replace with a template designed for this kind of data.
    for template, why in _templates_for(info):
        cand = from_template(template, spec.name)
        cand.class_names = list(info.class_names)
        cand.data = dict(spec.data)
        if template.startswith("Transfer"):
            cand.layers[0].config["pretrained"] = True
        cand = _try(cand, info)
        if cand is not None:
            fixes.append(Fix("template", f"Use the “{template}” template",
                             f"{why}\n\n⚠️ Your current layers are replaced (and any trained weights "
                             "can't be reused).", cand, before, cand.total_params()))

    # Auto fixes that create a huge model rank below a sensible template.
    fixes.sort(key=lambda f: (f.kind == "auto" and f.params_after > BIG_MODEL, ["auto", "template"].index(f.kind)))

    fixes.append(Fix("builder", "Open the builder and fix it myself",
                     "Opens the model builder with your data's input shape already set. The problem "
                     "is marked in red there - add, move or remove layers until it says **✓ ready to "
                     "train**, then save and come back here."))
    return fixes


def _templates_for(info: DataInfo) -> list[tuple[str, str]]:
    s = info.input_shape
    if info.task == "language_model":
        return [("Tiny GPT (characters, fast)", "A small GPT: a causal Transformer that learns to write."),
                ("LSTM language model (characters)", "Reads the text in order with a memory; simpler.")]
    if info.input_dtype == "long":
        return [("Text Transformer classifier", "Reads text with self-attention."),
                ("Text classifier (Embedding + LSTM)", "Reads text word by word; lighter and faster.")]
    if len(s) == 3:
        out = [("CNN (32×32 colour, e.g. CIFAR-10)",
                "A convolutional network - the standard choice for images. It works with any image "
                "size and channel count.")]
        if s[0] == 3 and min(s[1:]) >= 64:
            out.append(("Transfer learning (ResNet18 backbone)",
                        "Starts from a network pre-trained on 1.2 million photos - best when you have "
                        "few images per class. Downloads weights the first time."))
        return out
    if len(s) == 1:
        if info.task == "regression":
            return [("MLP regressor (tabular)", "A small fully-connected network for tables.")]
        return [("MLP classifier (tabular)", "A small fully-connected network for tables.")]
    return []
