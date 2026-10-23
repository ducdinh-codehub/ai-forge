"""Report a problem or idea as a GitHub issue.

The app never holds a GitHub token: it builds the issue text and opens GitHub's "new issue"
page with the title and body filled in, so the user reviews it and submits it with their own
account. Nothing is sent anywhere until they click Submit on GitHub.
"""
from __future__ import annotations

import platform
import sys
import traceback
import urllib.parse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from . import __version__

REPO = "ducdinh-codehub/ai-forge"
NEW_ISSUE = f"https://github.com/{REPO}/issues/new"
# Browsers and GitHub reject very long URLs; past this the body is shortened in the link and the
# full report goes to the clipboard instead.
MAX_URL = 7000
CRASH_FILE = "last-crash.txt"  # in the workspace; read (and removed) at the next start

KINDS = {
    "bug": ("🐞 Something doesn't work", "Bug"),
    "idea": ("💡 An idea / feature request", "Idea"),
    "question": ("❓ A question", "Question"),
}
PROMPTS = {
    "bug": "What did you do, what happened, and what did you expect instead?",
    "idea": "What would you like AIForge to do, and why would it help you?",
    "question": "What would you like to know?",
}


@dataclass
class Report:
    kind: str
    title: str
    description: str
    system: str = ""  # Markdown, empty = not included
    errors: str = ""
    model: str = ""
    crash: str = ""  # traceback of the last crash

    @property
    def full_title(self) -> str:
        return f"[{KINDS[self.kind][1]}] {self.title.strip()}"

    def body(self) -> str:
        parts = [self.description.strip() or "_(no description)_"]
        crash = f"```text\n{self.crash.strip()}\n```" if self.crash else ""
        for heading, text in (("Crash", crash), ("Recent errors", self.errors), ("Model", self.model),
                              ("System", self.system)):
            if text:
                parts += ["", f"### {heading}", "", text]
        parts += ["", "<sub>Sent from AIForge's 🐞 Report a problem dialog.</sub>"]
        return "\n".join(parts)


class ErrorLog:
    """The last few error messages the app showed, to attach to bug reports."""

    def __init__(self, size: int = 8):
        self.items: deque[tuple[str, str]] = deque(maxlen=size)

    def add(self, message: str) -> None:
        import time

        self.items.append((time.strftime("%H:%M:%S"), scrub(" ".join(str(message).split()))[:400]))

    def markdown(self) -> str:
        return "\n".join(f"- `{t}` {m}" for t, m in self.items)


def scrub(text: str) -> str:
    """Hide the user's home folder (and so their user name) in paths."""
    home = str(Path.home())
    return text.replace(home, "~") if len(home) > 1 else text


def save_crash(root: Path, error: BaseException) -> None:
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    try:
        (Path(root) / CRASH_FILE).write_text(scrub(text)[-6000:], encoding="utf-8")
    except OSError:
        pass  # never let crash reporting hide the original crash


def take_crash(root: Path) -> str | None:
    """The traceback saved by the last crash, if any (removed so it is offered only once)."""
    f = Path(root) / CRASH_FILE
    try:
        text = f.read_text(encoding="utf-8")
        f.unlink()
        return text
    except OSError:
        return None


def system_info() -> str:
    """Versions and hardware that help reproduce a bug. No paths, user names or data."""
    lines = [f"- AIForge {__version__}",
             f"- Python {platform.python_version()} on {platform.system()} {platform.release()} "
             f"({platform.machine()})"]
    try:
        import torch

        dev = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else \
            "Apple MPS" if torch.backends.mps.is_available() else "CPU only"
        lines.append(f"- PyTorch {torch.__version__} · {dev}")
    except Exception:  # noqa: BLE001 - a report must never fail because of this
        pass
    try:
        import textual

        lines.append(f"- Textual {textual.__version__}")
    except Exception:  # noqa: BLE001
        pass
    if sys.platform != "win32":
        import os

        lines.append(f"- Terminal: {os.environ.get('TERM_PROGRAM') or os.environ.get('TERM', '?')}")
    return "\n".join(lines)


def model_summary(spec) -> str:
    """The model's structure (layer types, shapes, settings) - never its data or weights."""
    infos = spec.analyze()
    lines = [f"- Task: {spec.task} · input {tuple(spec.input_shape)} {spec.input_dtype} · "
             f"{sum(i.params for i in infos):,} parameters"]
    for i in infos:
        cfg = ", ".join(f"{k}={v}" for k, v in i.spec.config.items())
        shape = i.out_shape if i.out_shape else f"✗ {i.error}"
        lines.append(f"  {i.index + 1}. {i.spec.type}({cfg}) → {shape}")
    return "\n".join(lines[:1]) + "\n\n```\n" + "\n".join(l.strip() for l in lines[1:]) + "\n```"


def issue_url(report: Report) -> tuple[str, bool]:
    """GitHub's new-issue link with the report filled in, and whether the body had to be cut
    to fit (then the user pastes the full text from the clipboard)."""
    def make(body: str) -> str:
        return NEW_ISSUE + "?" + urllib.parse.urlencode({"title": report.full_title, "body": body})

    body = report.body()
    url = make(body)
    if len(url) <= MAX_URL:
        return url, False
    note = ("\n\n---\n⚠️ **The report was too long for a link, so it is cut here.** AIForge copied the "
            "full report to your clipboard and saved it to a file: replace this text with it.")
    lo, hi = 0, len(body)
    while lo < hi:  # longest prefix that fits, measured after URL encoding
        mid = (lo + hi + 1) // 2
        if len(make(body[:mid] + note)) <= MAX_URL:
            lo = mid
        else:
            hi = mid - 1
    return make(body[:lo] + note), True
