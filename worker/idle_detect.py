"""
Cross-platform idle detection.

"Idle" means: user hasn't touched keyboard/mouse AND GPU utilization is low.
We only run training jobs when both conditions are true so we don't
disrupt the owner's experience.
"""
from __future__ import annotations
import platform
import subprocess
import time
from typing import Optional


SYSTEM = platform.system()   # "Windows" | "Darwin" | "Linux"

# Training starts after this many seconds of combined user + GPU idle.
# Lowered from 120 → 30 so the system is more responsive on machines that
# briefly fire input events from background apps (mouse drift, Discord, etc).
# Bailing mid-training is cheap now: the coordinator instantly releases the
# shard when the worker's next heartbeat reports IDLE status, so any other
# worker can pick it up.
USER_IDLE_THRESHOLD = int(__import__("os").environ.get("IDLE_THRESHOLD_SECS", "30"))
GPU_UTIL_MAX = int(__import__("os").environ.get("GPU_UTIL_MAX_PCT", "10"))


# ── User-input idle time ──────────────────────────────────────────────────────

def get_user_idle_seconds() -> float:
    """Returns seconds since the last keyboard/mouse input."""
    if SYSTEM == "Windows":
        return _windows_idle()
    elif SYSTEM == "Darwin":
        return _macos_idle()
    else:
        return _linux_idle()


def _windows_idle() -> float:
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
    return millis / 1000.0


def _macos_idle() -> float:
    try:
        result = subprocess.check_output(
            ["ioreg", "-c", "IOHIDSystem"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in result.splitlines():
            if "HIDIdleTime" in line:
                ns = int(line.split("=")[-1].strip())
                return ns / 1_000_000_000.0
    except Exception:
        pass
    return 0.0


def _linux_idle() -> float:
    # xprintidle returns milliseconds; falls back to 0 if not installed
    try:
        ms = int(subprocess.check_output(["xprintidle"], stderr=subprocess.DEVNULL))
        return ms / 1000.0
    except Exception:
        pass
    # Fallback: check /proc/uptime vs last X event (unreliable but better than nothing)
    return 0.0


# ── GPU utilization ───────────────────────────────────────────────────────────

def get_gpu_utilization_pct() -> Optional[float]:
    """Returns current GPU utilization 0–100, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        pass

    # ROCm
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "GPU use" in line:
                return float(line.split(":")[1].strip().replace("%", ""))
    except Exception:
        pass

    return None   # no GPU or query failed — assume idle


# ── Combined check ────────────────────────────────────────────────────────────

def is_idle() -> bool:
    """True when the machine is idle enough to start training."""
    user_idle = get_user_idle_seconds()
    if user_idle < USER_IDLE_THRESHOLD:
        return False

    gpu_util = get_gpu_utilization_pct()
    if gpu_util is not None and gpu_util > GPU_UTIL_MAX:
        return False

    return True


def wait_until_idle(poll_interval: float = 5.0) -> None:
    """Block until is_idle() returns True."""
    while not is_idle():
        user_idle = get_user_idle_seconds()
        gpu_util = get_gpu_utilization_pct()
        gpu_str = f"{gpu_util:.0f}%" if gpu_util is not None else "n/a"
        print(
            f"[idle] Waiting... user_idle={user_idle:.0f}s "
            f"(need {USER_IDLE_THRESHOLD}s)  "
            f"gpu_util={gpu_str}"
        )
        time.sleep(poll_interval)
