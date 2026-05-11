"""
RAM Manager - Sequential model swap manager for 16GB Unified Memory constraint.

Ensures that only one model (VLM or LLM) is loaded into RAM at any given
time.  Uses context-manager patterns for safe load/unload cycles and
monitors memory pressure via ``psutil`` to trigger proactive unloading.

Key guarantees:
  - ``with_vlm`` / ``with_llm`` are the *only* ways to access models.
  - Models are never loaded simultaneously.
  - If memory pressure exceeds the threshold the model is unloaded after use.
  - Context-manager support allows ``async with ram`` for full cleanup.

Spec: Section 8.1 - RAM Manager
"""

from __future__ import annotations

import asyncio
import gc
import logging
from contextlib import AbstractAsyncContextManager
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Attempt to import psutil; if unavailable, stub mode is activated.
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False
    logger.warning(
        "psutil not available - RAMManager running in stub mode. "
        "Install with: pip install psutil"
    )

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Model imports - deferred to allow mock/stub usage
# ---------------------------------------------------------------------------

# Use the existing managers from the vision and llm packages.
# These are imported here to keep the module self-contained.


class RAMManager(AbstractAsyncContextManager):
    """Sequential model swap manager for 16GB Unified Memory constraint.

    The ``RAMManager`` owns exactly two model handles - a VLM manager and
    an LLM manager - but guarantees that **at most one** is resident in
    memory at any time.  Callers request temporary access via
    :meth:`with_vlm` or :meth:`with_llm`; the manager handles loading,
    callback execution, and conditional unloading.

    Attributes:
        MEMORY_PRESSURE_THRESHOLD: Fraction of total RAM (0-1) above
            which a model is unloaded after use.  Default: 80 %.
    """

    MEMORY_PRESSURE_THRESHOLD: float = 0.80
    """80 %% of 16 GB = 12.8 GB."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise the RAMManager with empty model handles.

        Models are **not** loaded at construction time.  The first call to
        :meth:`with_vlm` or :meth:`with_llm` triggers the initial load.
        """
        self._vlm: Any = None
        self._llm: Any = None
        self._current_model: Optional[str] = None
        self._lock = asyncio.Lock()
        logger.info("RAMManager initialised (no models loaded).")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vlm(self) -> Any:
        """Lazy-create the VLM manager on first access."""
        if self._vlm is None:
            from app.vision.vlm_loader import VLMManager

            self._vlm = VLMManager()
        return self._vlm

    @property
    def llm(self) -> Any:
        """Lazy-create the LLM manager on first access."""
        if self._llm is None:
            from app.llm.llm_loader import LLMManager

            self._llm = LLMManager()
        return self._llm

    @property
    def current_model(self) -> Optional[str]:
        """Which model is currently resident (``'vlm'`` | ``'llm'`` | ``None``)."""
        return self._current_model

    # ------------------------------------------------------------------
    # Async context-manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RAMManager":
        """Enter the async context (no-op; models are loaded on demand)."""
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit the async context, unconditionally unloading all models."""
        await self.unload_all()

    # ------------------------------------------------------------------
    # Public API: model access
    # ------------------------------------------------------------------

    async def with_vlm(self, callback: Callable[[Any], Awaitable[T]]) -> T:
        """Load VLM, run *callback*, and unload if memory pressure is high.

        This is the **only** supported way to access the VLM.  It
        guarantees sequential loading: if the LLM is currently loaded it
        is unloaded first.

        Args:
            callback: Async callable that receives the loaded VLM manager
                and returns an arbitrary value.

        Returns:
            Whatever *callback* returns.

        Raises:
            Any exception raised by *callback* is propagated after the
            finally-block runs.
        """
        async with self._lock:
            await self._swap_to("vlm")
            try:
                return await callback(self.vlm)
            finally:
                if self._check_memory_pressure():
                    await self._do_unload("vlm")

    async def with_llm(self, callback: Callable[[Any], Awaitable[T]]) -> T:
        """Load LLM, run *callback*, and unload if memory pressure is high.

        This is the **only** supported way to access the LLM.  It
        guarantees sequential loading: if the VLM is currently loaded it
        is unloaded first.

        Args:
            callback: Async callable that receives the loaded LLM manager
                and returns an arbitrary value.

        Returns:
            Whatever *callback* returns.
        """
        async with self._lock:
            await self._swap_to("llm")
            try:
                return await callback(self.llm)
            finally:
                if self._check_memory_pressure():
                    await self._do_unload("llm")

    # ------------------------------------------------------------------
    # Public API: force unload
    # ------------------------------------------------------------------

    async def unload_all(self) -> None:
        """Unload **both** models unconditionally.

        Safe to call at any time - used by the context-manager exit
        and during graceful shutdown.
        """
        async with self._lock:
            if self._current_model == "vlm":
                await self._do_unload("vlm")
            elif self._current_model == "llm":
                await self._do_unload("llm")
        gc.collect()
        logger.info("RAMManager: all models unloaded.")

    # ------------------------------------------------------------------
    # Memory diagnostics
    # ------------------------------------------------------------------

    def check_memory_pressure(self) -> bool:
        """Return ``True`` if unified memory usage exceeds the threshold.

        Uses ``psutil.virtual_memory().percent`` divided by 100 and
        compares against :attr:`MEMORY_PRESSURE_THRESHOLD`.

        In stub mode (psutil unavailable) always returns ``True`` to be
        conservative and trigger unloading.
        """
        return self._check_memory_pressure()

    def get_ram_usage_mb(self) -> float:
        """Return current system RAM usage in megabytes.

        Returns:
            RAM used in MB, or ``0.0`` if psutil is unavailable.
        """
        if not _PSUTIL_AVAILABLE or psutil is None:
            return 0.0
        return float(psutil.virtual_memory().used) / (1024 * 1024)

    def get_ram_usage_percent(self) -> float:
        """Return current system RAM usage as a fraction (0.0-1.0).

        Returns:
            Fraction of RAM used, or ``0.0`` if psutil is unavailable.
        """
        if not _PSUTIL_AVAILABLE or psutil is None:
            return 0.0
        return psutil.virtual_memory().percent / 100.0

    # ------------------------------------------------------------------
    # Internal: model swapping
    # ------------------------------------------------------------------

    async def _swap_to(self, model: str) -> None:
        """Unload the current model (if any) and load *model*.

        Args:
            model: One of ``'vlm'`` or ``'llm'``.
        """
        if self._current_model == model:
            logger.debug("Model '%s' already loaded - no swap needed.", model)
            return

        # Unload current model first
        if self._current_model == "vlm":
            await self._do_unload("vlm")
        elif self._current_model == "llm":
            await self._do_unload("llm")

        # Load the requested model
        if model == "vlm":
            logger.info("RAMManager: loading VLM ...")
            self.vlm.load()
            self._current_model = "vlm"
        elif model == "llm":
            logger.info("RAMManager: loading LLM ...")
            self.llm.load()
            self._current_model = "llm"
        else:
            raise ValueError(f"Unknown model '{model}' - expected 'vlm' or 'llm'.")

        logger.info("RAMManager: now loaded -> '%s'.", self._current_model)

    async def _do_unload(self, model: str) -> None:
        """Unload a specific model and clear the current_model pointer."""
        if model == "vlm" and self._vlm is not None:
            self._vlm.unload()
            logger.info("RAMManager: VLM unloaded.")
        elif model == "llm" and self._llm is not None:
            self._llm.unload()
            logger.info("RAMManager: LLM unloaded.")
        self._current_model = None

    # ------------------------------------------------------------------
    # Internal: memory pressure check
    # ------------------------------------------------------------------

    def _check_memory_pressure(self) -> bool:
        """Return ``True`` if memory pressure exceeds threshold."""
        if not _PSUTIL_AVAILABLE or psutil is None:
            # In stub mode be conservative: always unload
            logger.debug("psutil unavailable - assuming memory pressure.")
            return True
        usage_percent = psutil.virtual_memory().percent / 100
        exceeds = usage_percent > self.MEMORY_PRESSURE_THRESHOLD
        if exceeds:
            logger.info(
                "Memory pressure detected: %.1f%% > threshold %.1f%%",
                usage_percent * 100,
                self.MEMORY_PRESSURE_THRESHOLD * 100,
            )
        return exceeds
