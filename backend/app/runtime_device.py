"""
Centralised Apple-Metal / device runtime configuration.

Single source of truth for: which torch device to use (Metal ``mps`` on Apple
Silicon), how to configure the MPS runtime for unified-memory use, and how big
to make model batches given the detected unified memory. CPU / CUDA paths
degrade cleanly.

The M-series "unified memory" advantage is twofold and this module leans into
both: (1) there is no CPU↔GPU copy, so keeping a model *resident* on the GPU is
cheap and avoids reload latency (see the RAM manager); (2) the GPU can address
most of system RAM, so batch sizes should scale with it rather than being pinned
to a hard-coded 16 GB assumption.
"""
from __future__ import annotations

import logging
import os
import platform
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def total_memory_gb() -> float:
    """Total unified/system memory in GiB (best-effort; defaults to 16)."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 16.0


def _macos_supports_mps() -> bool:
    if platform.system() != "Darwin":
        return False
    version = platform.mac_ver()[0]
    try:
        return int(version.split(".", 1)[0]) >= 14
    except (ValueError, IndexError):
        return False


def _mps_usable(torch: Any) -> bool:
    if not _macos_supports_mps():
        return False
    try:
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            return False
        torch.ones(1, device="mps").cpu()  # real round-trip
        return True
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.info("MPS not usable: %s", exc)
        return False


@lru_cache(maxsize=1)
def resolve_device(requested: Optional[str] = None) -> str:
    """Return the best torch device: ``mps`` > ``cuda`` > ``cpu``."""
    norm = (requested or os.getenv("SURYA_DEVICE") or os.getenv("TORCH_DEVICE") or "").strip().lower()
    if norm == "cpu":
        return "cpu"
    try:
        import torch

        if norm == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if norm == "mps":
            return "mps" if _mps_usable(torch) else "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if _mps_usable(torch):
            return "mps"
    except Exception:
        pass
    return "cpu"


def configure_mps_runtime(device: Optional[str] = None) -> str:
    """Configure the MPS runtime once (idempotent); return the resolved device.

    - ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so any op without an MPS kernel falls
      back to CPU instead of crashing the request (robustness on Apple GPUs).
    - Exposes the device via ``SURYA_DEVICE`` / ``TORCH_DEVICE`` for libraries
      that read it.
    """
    device = device or resolve_device()
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("SURYA_DEVICE", device)
    os.environ.setdefault("TORCH_DEVICE", device)
    return device


def surya_batch_sizes(device: Optional[str] = None) -> dict[str, int]:
    """Recognition / detector / layout batch sizes scaled to unified memory.

    Tuned on a 16 GB M4 (8-core GPU); scales up with more unified memory so an
    M-series Pro/Max isn't left underutilised, and never exceeds values that
    are safe for the Apple GPU's working set.
    """
    device = device or resolve_device()
    if device == "cpu":
        # CPU: modest batches keep latency reasonable without thrashing cores.
        return {"recognition": 32, "detector": 6, "layout": 6}
    if device == "cuda":
        return {"recognition": 256, "detector": 36, "layout": 36}

    # MPS: keep the proven-safe 16 GB baseline (32 / 4 / 12) and scale up ONLY
    # when more unified memory is available. On a 16 GB M-series, larger batches
    # thrash the GPU working set (it swaps), so bigger is *slower*, not faster;
    # an M4 Pro/Max with more memory does get larger batches.
    gb = total_memory_gb()
    return {
        "recognition": max(32, min(192, int(gb * 2))),
        "detector": max(4, min(32, int(gb // 4))),
        "layout": max(12, min(32, int(gb // 4) + 8)),
    }


def device_summary() -> dict[str, Any]:
    """Human/health-readable summary of the active acceleration."""
    device = resolve_device()
    summary: dict[str, Any] = {
        "device": device,
        "unified_memory_gb": round(total_memory_gb(), 1),
        "batch_sizes": surya_batch_sizes(device),
    }
    if device == "mps":
        try:
            import torch

            summary["mps_recommended_working_set_gb"] = round(
                torch.mps.recommended_max_memory() / (1024 ** 3), 1
            )
        except Exception:
            pass
    return summary
