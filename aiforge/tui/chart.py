"""Tiny terminal line chart: plots up to two series on shared axes."""
from __future__ import annotations

from rich.text import Text
from textual.widget import Widget

# Braille dots give 2×4 sub-cell resolution per character.
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


class LineChart(Widget):
    DEFAULT_CSS = "LineChart { height: 8; }"

    def __init__(self, title: str, labels=("train", "val"), colors=("cyan", "magenta"),
                 fmt: str = "{:.3f}", **kw):
        super().__init__(**kw)
        self.title, self.labels, self.palette, self.fmt = title, labels, colors, fmt
        self.series: list[list[float]] = [[], []]

    def set_data(self, *series: list[float]) -> None:
        self.series = [list(s) for s in series]
        self.refresh()

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        out = Text()
        legend = "  ".join(f"[{c}]━[/] {l}" for l, c, s in zip(self.labels, self.palette, self.series) if s)
        out.append_text(Text.from_markup(f"[b]{self.title}[/b]  {legend}\n"))
        rows = h - 2
        values = [v for s in self.series for v in s]
        if rows < 2 or not values:
            out.append("  waiting for the first epoch…", style="dim")
            return out
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        axis = max(len(self.fmt.format(hi)), len(self.fmt.format(lo))) + 1
        cols = max(4, w - axis - 1)
        px_w, px_h = cols * 2, rows * 4
        grid = [[0] * cols for _ in range(rows)]
        owner = [[-1] * cols for _ in range(rows)]
        for si, s in enumerate(self.series):
            if not s:
                continue
            pts = []
            n = len(s)
            for i, v in enumerate(s):
                x = 0 if n == 1 else round(i * (px_w - 1) / (n - 1))
                y = round((hi - v) / (hi - lo) * (px_h - 1))
                pts.append((x, y))
            # Connect consecutive points so the curve reads as a line.
            for (x0, y0), (x1, y1) in zip(pts, pts[1:] or pts):
                steps = max(abs(x1 - x0), abs(y1 - y0), 1)
                for k in range(steps + 1):
                    x = x0 + (x1 - x0) * k // steps
                    y = y0 + (y1 - y0) * k // steps
                    r, c = y // 4, x // 2
                    grid[r][c] |= _DOTS[y % 4][x % 2]
                    owner[r][c] = si if owner[r][c] in (-1, si) else 2
        for r in range(rows):
            label = self.fmt.format(hi) if r == 0 else self.fmt.format(lo) if r == rows - 1 else ""
            out.append(f"{label:>{axis - 1}} ┤" if label else " " * (axis - 1) + " │", style="dim")
            for c in range(cols):
                if grid[r][c]:
                    o = owner[r][c]
                    out.append(chr(0x2800 + grid[r][c]), style=self.palette[o] if o < 2 else "white")
                else:
                    out.append(" ")
            if r < rows - 1:
                out.append("\n")
        n = max(len(s) for s in self.series)
        out.append(f"\n{' ' * axis}epoch 1{('epoch ' + str(n)).rjust(max(0, cols - 7))}", style="dim")
        return out
