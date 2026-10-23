"""Gradient ASCII-art banner for the home screen."""
from __future__ import annotations

from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.widget import Widget

LOGO = [
    " █████╗ ██╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗",
    "██╔══██╗██║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝",
    "███████║██║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  ",
    "██╔══██║██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ",
    "██║  ██║██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗",
    "╚═╝  ╚═╝╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
]
TAGLINE = "build · train · fine-tune · export neural networks - no code required"

# Gradient stops: pink → violet → blue → cyan → mint
STOPS = [(255, 78, 205), (155, 93, 255), (79, 140, 255), (0, 212, 255), (0, 255, 163)]


def _gradient(t: float) -> tuple[int, int, int]:
    t = t % 1.0
    pos = t * (len(STOPS) - 1)
    i = min(int(pos), len(STOPS) - 2)
    f = pos - i
    a, b = STOPS[i], STOPS[i + 1]
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


class Banner(Widget):
    """Logo with a diagonal colour gradient that slowly drifts."""

    DEFAULT_CSS = """
    Banner { height: 8; width: 100%; }
    """

    def __init__(self, animate: bool = True, **kw):
        super().__init__(**kw)
        self.phase = 0.0
        self._animate = animate

    def on_mount(self) -> None:
        if self._animate:
            self.set_interval(1 / 12, self._tick)

    def _tick(self) -> None:
        self.phase = (self.phase + 0.006) % 1.0
        self.refresh()

    def render(self) -> Text:
        # The neon stops wash out on a light background: darken them there, and
        # fade the "shadow" towards white instead of black.
        light = not self.app.current_theme.dark
        main = 0.7 if light else 1.0
        width = len(LOGO[0])
        pad = " " * max(0, (self.size.width - width) // 2)
        out = Text()
        for row, line in enumerate(LOGO):
            out.append(pad)
            for col, ch in enumerate(line):
                if ch == " ":
                    out.append(" ")
                    continue
                # Diagonal sweep: colour depends on column and row.
                r, g, b = _gradient(col / (width * 1.6) + row * 0.02 + self.phase)
                if ch == "█":
                    r, g, b = (round(c * main) for c in (r, g, b))
                elif light:  # the box-drawing "shadow" is drawn fainter
                    r, g, b = (round(c + (255 - c) * 0.5) for c in (r, g, b))
                else:
                    r, g, b = (round(c * 0.45) for c in (r, g, b))
                out.append(ch, Style(color=Color.from_rgb(r, g, b), bold=ch == "█"))
            out.append("\n")
        tag_pad = " " * max(0, (self.size.width - len(TAGLINE)) // 2)
        out.append(tag_pad)
        for i, ch in enumerate(TAGLINE):
            r, g, b = (round(c * main) for c in _gradient(i / (len(TAGLINE) * 1.6) + self.phase + 0.15))
            out.append(ch, Style(color=Color.from_rgb(r, g, b), italic=True))
        return out
