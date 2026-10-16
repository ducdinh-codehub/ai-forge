"""'This computer' panel: hardware specs plus live CPU / RAM / GPU gauges and training speed."""
from __future__ import annotations

import time
from collections import deque

from rich.text import Text
from textual.widget import Widget

from ..sysinfo import GB, Sample, Specs, fmt_bytes

SPARK = "▁▂▃▄▅▆▇█"
HISTORY = 40


def _level_color(pct: float) -> str:
    return "green" if pct < 60 else "yellow" if pct < 85 else "red"


def _bar(pct: float, width: int) -> Text:
    pct = max(0.0, min(100.0, pct))
    full = round(pct / 100 * width)
    t = Text()
    t.append("█" * full, style=_level_color(pct))
    t.append("░" * (width - full), style="grey35")
    return t


def _spark(values, top: float = 100.0) -> Text:
    t = Text()
    for v in values:
        i = min(len(SPARK) - 1, max(0, round(v / top * (len(SPARK) - 1))))
        t.append(SPARK[i], style=_level_color(v / top * 100))
    return t


class SystemPanel(Widget):
    """Renders specs once and gauges every time ``update_sample`` is called."""

    DEFAULT_CSS = """
    SystemPanel { height: 8; border: round $panel; padding: 0 1; margin-top: 1; }
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.border_title = "💻 This computer"
        self.specs: Specs | None = None
        self.sample: Sample | None = None
        self.device_choice = "auto"
        self.cpu_hist: deque[float] = deque(maxlen=HISTORY)
        self.ram_hist: deque[float] = deque(maxlen=HISTORY)
        self.gpu_hist: deque[float] = deque(maxlen=HISTORY)
        self.training: dict | None = None  # {"rate": samples/s, "device": str}
        self._rate_ref: tuple[float, int] | None = None

    # -- data in -------------------------------------------------------------

    def set_specs(self, specs: Specs) -> None:
        self.specs = specs
        self.refresh()

    def set_device_choice(self, choice: str) -> None:
        self.device_choice = choice
        self.refresh()

    def update_sample(self, s: Sample) -> None:
        self.sample = s
        self.cpu_hist.append(s.cpu)
        self.ram_hist.append(s.ram_used / s.ram_total * 100)
        if s.gpus and s.gpus[0].util is not None:
            self.gpu_hist.append(s.gpus[0].util)
        self.refresh()

    def training_progress(self, samples_done: int, device: str) -> None:
        """Feed the number of samples processed so far; computes throughput."""
        now = time.monotonic()
        if self._rate_ref is None or samples_done < self._rate_ref[1]:
            self._rate_ref = (now, samples_done)
            self.training = {"rate": None, "device": device}
            return
        t0, n0 = self._rate_ref
        if now - t0 >= 1.0:
            rate = (samples_done - n0) / (now - t0)
            prev = (self.training or {}).get("rate")
            self.training = {"rate": rate if prev is None else 0.5 * prev + 0.5 * rate, "device": device}
            self._rate_ref = (now, samples_done)
            self.refresh()

    def training_stopped(self) -> None:
        self.training = None
        self._rate_ref = None
        self.refresh()

    # -- rendering ------------------------------------------------------------

    def _device_label(self) -> str:
        sp = self.specs
        # While a run is going, show where it is actually running, not the dropdown's new value.
        dev = self.training["device"] if self.training else self.device_choice
        if dev == "auto" and sp:
            dev = sp.best_device
        return {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (MPS)", "cpu": "CPU"}.get(dev, dev)

    def render(self) -> Text:
        sp, s = self.specs, self.sample
        out = Text()
        if sp is None:
            out.append("Detecting hardware…", style="dim")
            return self._crop(out)
        width = max(40, self.size.width)
        bar_w = max(8, min(24, width // 6))

        # Line 1: specs
        layout = sp.core_layout.replace(" performance", "P").replace(" efficiency", "E").replace(" + ", "+")
        cores = f"{sp.cores_physical} cores" + (f" ({layout})" if layout else "")
        out.append("CPU ", style="bold")
        out.append(f"{sp.cpu_name} · {cores}   ")
        out.append("RAM ", style="bold")
        out.append(f"{fmt_bytes(sp.ram_total)}   ")
        out.append("GPU ", style="bold")
        if sp.gpus:
            g = sp.gpus[0]
            extra = [f"{g.cores} cores" if g.cores and g.backend == "mps" else "",
                     fmt_bytes(g.memory_total) if g.memory_total else ""]
            out.append(f"{g.name} · {' · '.join(e for e in extra if e)}"
                       + (f" (+{len(sp.gpus) - 1} more)" if len(sp.gpus) > 1 else ""))
        else:
            out.append("none found", style="yellow")
        out.append("\n")
        dev_label = self._device_label()
        out.append("Training on: " if self.training else "Will train on: ", style="dim")
        out.append(dev_label, style="bold green" if "GPU" in dev_label else "bold yellow")
        if "GPU" not in dev_label and sp.gpus and not self.training:
            out.append("  ← a GPU is available: set Device to auto", style="yellow")
        elif not sp.gpus:
            out.append("  (no supported GPU - CPU training is fine for small models)", style="dim")
        out.append(f"   · {sp.os} · PyTorch {sp.torch_version}", style="dim")
        out.append("\n")

        if s is None:
            out.append("Measuring usage…", style="dim")
            return self._crop(out)

        # Line 3: CPU
        out.append("CPU  ", style="bold")
        out.append_text(_bar(s.cpu, bar_w))
        out.append(f" {s.cpu:3.0f}%  ")
        out.append_text(_spark(self.cpu_hist))
        out.append("  cores ", style="dim")
        out.append_text(_spark(s.per_cpu))
        out.append("\n")

        # Line 4: RAM
        ram_pct = s.ram_used / s.ram_total * 100
        out.append("RAM  ", style="bold")
        out.append_text(_bar(ram_pct, bar_w))
        out.append(f" {ram_pct:3.0f}%  ")
        out.append_text(_spark(self.ram_hist))
        out.append(f"  {s.ram_used / GB:.1f} / {s.ram_total / GB:.1f} GB used", style="dim")
        out.append("\n")

        # Line 5: GPU
        out.append("GPU  ", style="bold")
        g = s.gpus[0] if s.gpus else None
        if g is None:
            out.append("—", style="dim")
        else:
            if g.util is not None:
                out.append_text(_bar(g.util, bar_w))
                out.append(f" {g.util:3.0f}%  ")
                out.append_text(_spark(self.gpu_hist))
            else:
                out.append("utilisation n/a (pip install nvidia-ml-py)", style="dim")
            parts = []
            shared = not (self.specs and self.specs.gpus and self.specs.gpus[0].memory_total)
            if g.mem_used is not None and g.mem_total:
                parts.append(f"{'shared ' if shared else ''}memory {fmt_bytes(g.mem_used)} / "
                             f"{fmt_bytes(g.mem_total)}")
            elif g.mem_used is not None:
                parts.append(f"memory {fmt_bytes(g.mem_used)} in use")
            if g.mem_app:
                parts.append(f"AIForge {fmt_bytes(g.mem_app)}")
            if g.temp is not None:
                parts.append(f"{g.temp:.0f}°C")
            if g.power is not None:
                parts.append(f"{g.power:.0f} W")
            if parts:
                out.append("  " + " · ".join(parts), style="dim")
        out.append("\n")

        # Line 6: this app + training speed
        out.append("App  ", style="bold")
        out.append(f"AIForge uses CPU {s.proc_cpu:.0f}% · RAM {fmt_bytes(s.proc_ram)}", style="dim")
        if self.training:
            rate = self.training.get("rate")
            out.append("   │  ", style="dim")
            out.append("Speed ", style="bold")
            out.append(f"{rate:,.0f} samples/s" if rate else "measuring…", style="bold cyan")
        return self._crop(out)

    def _crop(self, text: Text) -> Text:
        """One row per line, cut to the panel width so nothing wraps out of view."""
        width = max(10, self.size.width)
        lines = text.split("\n")
        for line in lines:
            line.truncate(width, overflow="ellipsis")
        return Text("\n").join(lines)
