"""Hardware specs and live usage (CPU, RAM, GPU) for the training screen.

Everything degrades gracefully: unknown values are ``None`` and the UI hides them.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

import psutil
import torch

GB = 1024 ** 3


@dataclass
class GPUSpec:
    name: str
    backend: str  # "cuda" | "mps"
    memory_total: int | None = None  # bytes; None = shared with system RAM
    cores: int | None = None
    index: int = 0


@dataclass
class Specs:
    os: str
    cpu_name: str
    cores_physical: int | None
    cores_logical: int | None
    core_layout: str  # e.g. "5 performance + 6 efficiency"
    ram_total: int
    gpus: list[GPUSpec] = field(default_factory=list)
    torch_version: str = torch.__version__

    @property
    def best_device(self) -> str:
        if any(g.backend == "cuda" for g in self.gpus):
            return "cuda"
        if any(g.backend == "mps" for g in self.gpus):
            return "mps"
        return "cpu"


@dataclass
class GPUSample:
    util: float | None = None  # %
    mem_used: int | None = None  # bytes in use on the device (all processes)
    mem_total: int | None = None
    mem_app: int | None = None  # bytes allocated by this process (AIForge)
    temp: float | None = None  # °C
    power: float | None = None  # W


@dataclass
class Sample:
    cpu: float
    per_cpu: list[float]
    ram_used: int
    ram_total: int
    proc_cpu: float
    proc_ram: int
    gpus: list[GPUSample]


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            return name
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Windows":
        out = _run(["wmic", "cpu", "get", "name"]).splitlines()
        names = [l.strip() for l in out[1:] if l.strip()]
        if names:
            return names[0]
    return platform.processor() or platform.machine() or "Unknown CPU"


def _core_layout() -> str:
    if platform.system() == "Darwin":
        p = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()
        e = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]).strip()
        if p.isdigit() and e.isdigit():
            return f"{p} performance + {e} efficiency"
    return ""


def _apple_gpu() -> GPUSpec | None:
    out = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
    model = re.search(r'"model" = "([^"]+)"', out)
    cores = re.search(r'"gpu-core-count" = (\d+)', out)
    name = model.group(1) if model else _cpu_name()
    return GPUSpec(name=name, backend="mps", cores=int(cores.group(1)) if cores else None)


@lru_cache(maxsize=1)
def specs() -> Specs:
    gpus: list[GPUSpec] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(GPUSpec(p.name, "cuda", p.total_memory, getattr(p, "multi_processor_count", None), i))
    elif torch.backends.mps.is_available():
        g = _apple_gpu()
        if g:
            gpus.append(g)
    os_name = {"Darwin": "macOS " + platform.mac_ver()[0]}.get(platform.system(),
                                                             f"{platform.system()} {platform.release()}")
    return Specs(os=os_name, cpu_name=_cpu_name(), cores_physical=psutil.cpu_count(logical=False),
                 cores_logical=psutil.cpu_count(logical=True), core_layout=_core_layout(),
                 ram_total=psutil.virtual_memory().total, gpus=gpus)


# ---------------------------------------------------------------------------
# Live sampling
# ---------------------------------------------------------------------------


class Monitor:
    """Call ``sample()`` periodically (e.g. once a second) from a worker thread."""

    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.proc.cpu_percent(None)  # first call primes the counters
        psutil.cpu_percent(None, percpu=True)
        self._nvml = None
        if any(g.backend == "cuda" for g in specs().gpus):
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
            except Exception:  # noqa: BLE001 - optional dependency
                self._nvml = None

    def sample(self) -> Sample:
        per = psutil.cpu_percent(None, percpu=True)
        vm = psutil.virtual_memory()
        try:
            proc_cpu, proc_ram = self.proc.cpu_percent(None), self.proc.memory_info().rss
        except psutil.Error:
            proc_cpu, proc_ram = 0.0, 0
        return Sample(cpu=sum(per) / max(1, len(per)), per_cpu=per, ram_used=vm.total - vm.available,
                      ram_total=vm.total, proc_cpu=proc_cpu, proc_ram=proc_ram,
                      gpus=[self._gpu(g) for g in specs().gpus])

    def _gpu(self, g: GPUSpec) -> GPUSample:
        s = GPUSample()
        if g.backend == "cuda":
            try:
                free, total = torch.cuda.mem_get_info(g.index)
                s.mem_used, s.mem_total = total - free, total
                s.mem_app = torch.cuda.memory_allocated(g.index)
            except Exception:  # noqa: BLE001
                pass
            if self._nvml:
                try:
                    h = self._nvml.nvmlDeviceGetHandleByIndex(g.index)
                    s.util = float(self._nvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    s.temp = float(self._nvml.nvmlDeviceGetTemperature(h, self._nvml.NVML_TEMPERATURE_GPU))
                    s.power = self._nvml.nvmlDeviceGetPowerUsage(h) / 1000
                except Exception:  # noqa: BLE001
                    pass
        elif g.backend == "mps":
            # Apple GPUs share system memory; ioreg reports utilisation without admin rights.
            out = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], timeout=1.5)
            m = re.search(r'"Device Utilization %"=(\d+)', out)
            if m:
                s.util = float(m.group(1))
            m = re.search(r'"In use system memory"=(\d+)', out)
            if m:
                s.mem_used = int(m.group(1))
            try:
                s.mem_app = torch.mps.driver_allocated_memory()
                s.mem_total = int(torch.mps.recommended_max_memory())
            except Exception:  # noqa: BLE001
                pass
        return s


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= GB:
        return f"{n / GB:.1f} GB"
    return f"{n / 1024 ** 2:.0f} MB"


def specs_summary() -> str:
    """One-paragraph plain text summary (also used by the README/tests)."""
    s = specs()
    gpu = ", ".join(f"{g.name} ({g.backend.upper()})" for g in s.gpus) or "none - training runs on the CPU"
    return (f"{s.os} · CPU {s.cpu_name}, {s.cores_physical} cores · RAM {fmt_bytes(s.ram_total)} · "
            f"GPU {gpu} · PyTorch {s.torch_version}")
