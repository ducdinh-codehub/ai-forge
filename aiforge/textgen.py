"""Language models: tokenizers for plain text and text generation (next-token sampling).

A language model reads ``seq_len`` token ids and outputs, at every position, one score per
vocabulary token for what comes *next*. The vocabulary is a list: index = token id, and id 0
is ``<unk>`` (anything not in the vocabulary).
"""
from __future__ import annotations

import re
from collections import Counter

import torch
from torch import nn

UNK = "<unk>"
TOKENIZERS = {"char": "characters (small vocabulary, learns spelling)",
              "word": "words (bigger vocabulary, more fluent with more data)"}
MAX_CHARS = 512  # a character vocabulary bigger than this is almost always stray unicode

# Words, single punctuation marks and line breaks (kept so poems and dialogue keep their shape).
WORD_RE = re.compile(r"\n|[a-z0-9']+|[^\sa-z0-9]")
NO_SPACE_BEFORE = set(".,;:!?)]}'\"\n")
NO_SPACE_AFTER = set("([{\n")


def split(text: str, tokenizer: str) -> list[str]:
    if tokenizer == "word":
        return WORD_RE.findall(text.lower())
    return list(text)


def build_vocab(tokens: list[str], tokenizer: str, max_size: int) -> list[str]:
    """Most frequent tokens first after ``<unk>``; characters are kept in a readable order."""
    counts = Counter(tokens)
    limit = max(1, (max_size if tokenizer == "word" else min(max_size, MAX_CHARS)) - 1)
    keep = [t for t, _ in counts.most_common(limit)]
    if tokenizer == "char":
        keep.sort()
    return [UNK] + keep


def encode(text: str, vocab: list[str], tokenizer: str) -> list[int]:
    index = {t: i for i, t in enumerate(vocab)}
    return [index.get(t, 0) for t in split(text, tokenizer)]


def decode(ids: list[int], vocab: list[str], tokenizer: str) -> str:
    tokens = [vocab[i] if 0 <= i < len(vocab) else UNK for i in ids]
    if tokenizer != "word":
        return "".join("�" if t == UNK else t for t in tokens)
    out = ""
    for t in tokens:
        if out and out[-1] not in NO_SPACE_AFTER and t[0] not in NO_SPACE_BEFORE:
            out += " "
        out += t
    return out


@torch.no_grad()
def next_token_probs(model: nn.Module, pre: dict, prompt: str) -> torch.Tensor:
    """Probabilities for the token that follows ``prompt``."""
    model.eval()
    ids = _context(pre, encode(prompt, pre["vocab"], pre["tokenizer"]))
    dev = next(model.parameters()).device
    logits = model(torch.tensor([ids], device=dev))[0, -1].float()
    logits[0] = float("-inf")  # never suggest <unk>
    return logits.softmax(-1).cpu()


@torch.no_grad()
def generate(model: nn.Module, pre: dict, prompt: str = "", length: int = 200,
             temperature: float = 0.8, top_k: int = 0, seed: int | None = None) -> str:
    """Continue ``prompt`` by sampling ``length`` tokens; returns only the new text.

    temperature < 1 picks likely tokens (safer, more repetitive), > 1 takes more risks.
    top_k > 0 only samples among the k most likely tokens."""
    model.eval()
    vocab, tok, seq_len = pre["vocab"], pre["tokenizer"], int(pre["seq_len"])
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    ids = _context(pre, encode(prompt, vocab, tok))
    new: list[int] = []
    for _ in range(length):
        x = torch.tensor([ids[-seq_len:]], device=dev)
        logits = model(x)[0, -1].float().cpu()
        logits[0] = float("-inf")
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if 0 < top_k < len(logits):
                cut = logits.topk(top_k).values[-1]
                logits[logits < cut] = float("-inf")
            nxt = int(torch.multinomial(logits.softmax(-1), 1, generator=gen))
        ids.append(nxt)
        new.append(nxt)
    text = decode(new, vocab, tok)
    if tok == "word" and prompt and not prompt[-1].isspace() and text and text[0] not in NO_SPACE_BEFORE:
        text = " " + text
    return text


def _context(pre: dict, ids: list[int]) -> list[int]:
    """The model needs at least one token to look at: start from a line break (or the most
    common token) when the prompt is empty."""
    ids = ids[-int(pre["seq_len"]):]
    if ids:
        return ids
    vocab = pre["vocab"]
    return [vocab.index("\n")] if "\n" in vocab else [1 if len(vocab) > 1 else 0]
