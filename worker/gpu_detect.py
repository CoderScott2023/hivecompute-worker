"""
Hardware detection — figures out the best available compute device
and reports specs back to the coordinator at registration time.
"""
from __future__ import annotations
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch

from shared.protocol import DeviceType


@dataclass
class HardwareProfile:
    device_type: DeviceType
    device_name: str
    vram_gb: float          # 0.0 for CPU
    ram_gb: float
    num_cpus: int
    torch_device: str       # "cuda:0", "mps", "cpu"
    dtype: str              # best supported dtype for training


def detect() -> HardwareProfile:
    """Auto-detect the best available device and return its profile."""
    ram_gb = _get_ram_gb()
    num_cpus = torch.get_num_threads()

    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        # bfloat16 is preferred on Ampere+ (compute capability >= 8.0)
        dtype = "bfloat16" if props.major >= 8 else "float16"
        return HardwareProfile(
            device_type=DeviceType.CUDA,
            device_name=props.name,
            vram_gb=round(vram_gb, 2),
            ram_gb=round(ram_gb, 2),
            num_cpus=num_cpus,
            torch_device="cuda:0",
            dtype=dtype,
        )

    # ROCm (AMD — also reports as cuda in PyTorch)
    if _is_rocm():
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        return HardwareProfile(
            device_type=DeviceType.ROCM,
            device_name=props.name,
            vram_gb=round(vram_gb, 2),
            ram_gb=round(ram_gb, 2),
            num_cpus=num_cpus,
            torch_device="cuda:0",
            dtype="float16",
        )

    # Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = _get_mac_gpu_name()
        return HardwareProfile(
            device_type=DeviceType.MPS,
            device_name=device_name,
            vram_gb=0.0,         # unified memory — not separately reported
            ram_gb=round(ram_gb, 2),
            num_cpus=num_cpus,
            torch_device="mps",
            dtype="float32",     # MPS bfloat16 support is incomplete
        )

    # CPU fallback
    return HardwareProfile(
        device_type=DeviceType.CPU,
        device_name=platform.processor() or "CPU",
        vram_gb=0.0,
        ram_gb=round(ram_gb, 2),
        num_cpus=num_cpus,
        torch_device="cpu",
        dtype="float32",
    )


def to_torch_dtype(dtype_str: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_str, torch.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_rocm() -> bool:
    try:
        out = subprocess.check_output(["rocm-smi", "--version"], stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 0.0


def _get_mac_gpu_name() -> str:
    try:
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in result.splitlines():
            if "Chipset Model" in line or "Chip" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "Apple Silicon GPU"
