"""
VLM Loader — MLX VLM integration with graceful fallback.

Manages the Llama-3.2-11B-Vision-Instruct-4bit model via mlx-vlm,
with load/unload cycles for RAM management on Apple Silicon (M4).
All mlx_vlm imports are wrapped in try/except for environments without
MLX (testing, CI, non-macOS). Falls back to a MockVLMManager stub.

Spec: Section 6.1 — VLM Loader
"""
from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
from typing import Any, Optional, Protocol, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import mlx_vlm; if unavailable, mark as unavailable so we can
# fall back to stub mode.
# ---------------------------------------------------------------------------
try:
    from mlx_vlm import load as _mlx_load
    from mlx_vlm.utils import generate as _mlx_generate

    _MLX_AVAILABLE = True
except Exception:
    _MLX_AVAILABLE = False
    _mlx_load = None
    _mlx_generate = None
    logger.warning("mlx_vlm not available — falling back to stub mode")


# ---------------------------------------------------------------------------
# VLMManager
# ---------------------------------------------------------------------------

class VLMManager:
    """MLX VLM with load/unload for Unified Memory RAM management.

    Loads the quantized Llama-3.2-11B-Vision-Instruct-4bit model via
    ``mlx_vlm`` on macOS with Apple Silicon.  In environments where
    ``mlx_vlm`` is unavailable the manager transparently falls back to
    *stub mode* (mock responses).

    Attributes:
        MODEL_ID: Hugging Face model identifier used by ``mlx_vlm.load``.
    """

    MODEL_ID: str = "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and processor into Unified Memory / GPU.

        Raises:
            RuntimeError: If MLX is not available on this platform.
        """
        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "MLX VLM is not available on this platform. "
                "Use MockVLMManager for testing / development."
            )
        logger.info("Loading VLM model %s ...", self.MODEL_ID)
        self.model, self.processor = _mlx_load(self.MODEL_ID)
        self._loaded = True
        logger.info("VLM loaded successfully.")

    def unload(self) -> None:
        """Unload model / processor and force garbage collection.

        This frees GPU-resident memory on Apple Silicon Unified Memory
        architectures, which is critical when running sequentially with
        the LLM (Section 8.1 RAM Manager).
        """
        logger.info("Unloading VLM ...")
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        logger.info("VLM unloaded and GC completed.")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is currently resident in memory."""
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Run VLM inference on *image_path* with *prompt*.

        Args:
            image_path: Absolute or relative path to the image file.
            prompt: Text prompt forwarded to the vision model.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            Raw generated text string from the VLM.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("VLM not loaded. Call load() before generate().")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not _MLX_AVAILABLE:
            raise RuntimeError("MLX VLM is not available.")

        logger.debug("VLM generate — image=%s prompt_len=%d", image_path, len(prompt))
        output: str = _mlx_generate(
            self.model,
            self.processor,
            image_path,
            prompt,
            verbose=False,
            max_tokens=max_tokens,
        )
        return output


# ---------------------------------------------------------------------------
# MockVLMManager — drop-in replacement for testing / dev without the model
# ---------------------------------------------------------------------------

class MockVLMManager(VLMManager):
    """Stub VLM manager that returns realistic mock responses.

    Useful for:
    *   CI pipelines where MLX is not installed.
    *   Frontend development without downloading 5.2 GB of weights.
    *   Unit testing the Dual-Pass extractor and Quality Gate.

    Inherits the same interface as :class:`VLMManager` but never touches
    GPU memory.  ``load()`` / ``unload()`` are no-ops.
    """

    # Realistic mock responses keyed by pass type ---------------
    _MOCK_STRUCTURAL_RESPONSE: str = json.dumps(
        {
            "familienname": {
                "label_text": "Familienname",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.12, "y1": 0.25, "x2": 0.45, "y2": 0.30},
            },
            "vorname": {
                "label_text": "Vorname",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.12, "y1": 0.32, "x2": 0.45, "y2": 0.37},
            },
            "geburtsdatum": {
                "label_text": "Geburtsdatum",
                "field_type": "date",
                "estimated_bbox": {"x1": 0.12, "y1": 0.40, "x2": 0.35, "y2": 0.45},
            },
            "geburtsort": {
                "label_text": "Geburtsort",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.40, "y1": 0.40, "x2": 0.70, "y2": 0.45},
            },
            "staatsangehoerigkeit": {
                "label_text": "Staatsangeh\u00f6rigkeit",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.12, "y1": 0.48, "x2": 0.50, "y2": 0.53},
            },
            "strasse": {
                "label_text": "Stra\u00dfe",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.12, "y1": 0.56, "x2": 0.50, "y2": 0.61},
            },
            "hausnummer": {
                "label_text": "Hausnummer",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.52, "y1": 0.56, "x2": 0.62, "y2": 0.61},
            },
            "postleitzahl": {
                "label_text": "PLZ",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.12, "y1": 0.64, "x2": 0.25, "y2": 0.69},
            },
            "wohnort": {
                "label_text": "Wohnort",
                "field_type": "text",
                "estimated_bbox": {"x1": 0.30, "y1": 0.64, "x2": 0.60, "y2": 0.69},
            },
        },
        indent=2,
        ensure_ascii=False,
    )

    _MOCK_VALUE_RESPONSE: str = json.dumps(
        {
            "familienname": {"raw_value": "M\u00fcller", "confidence_0_to_1": 0.95},
            "vorname": {"raw_value": "Hans", "confidence_0_to_1": 0.93},
            "geburtsdatum": {"raw_value": "15.03.1985", "confidence_0_to_1": 0.91},
            "geburtsort": {"raw_value": "Berlin", "confidence_0_to_1": 0.88},
            "staatsangehoerigkeit": {"raw_value": "Deutsch", "confidence_0_to_1": 0.92},
            "strasse": {"raw_value": "Hauptstra\u00dfe", "confidence_0_to_1": 0.85},
            "hausnummer": {"raw_value": "42", "confidence_0_to_1": 0.97},
            "postleitzahl": {"raw_value": "10115", "confidence_0_to_1": 0.96},
            "wohnort": {"raw_value": "Berlin", "confidence_0_to_1": 0.94},
        },
        indent=2,
        ensure_ascii=False,
    )

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    # -- overrides -------------------------------------------------------

    def load(self) -> None:
        """No-op: mock model is always \"loaded\"."""
        self._loaded = True
        logger.info("MockVLMManager load() — no-op (stub mode)")

    def unload(self) -> None:
        """No-op: no GPU memory to release."""
        self._loaded = False
        self.model = None
        self.processor = None
        logger.info("MockVLMManager unload() — no-op (stub mode)")

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Return a pre-canned mock response based on *prompt* content.

        Detects structural vs value pass by keyword matching in the
        prompt and returns the corresponding mock JSON.
        """
        if not self._loaded:
            raise RuntimeError("Mock VLM not loaded. Call load() before generate().")

        # Simulate inference latency for realism
        time.sleep(0.01)

        prompt_lower = prompt.lower()
        if "structural" in prompt_lower or "pass a" in prompt_lower:
            logger.debug("MockVLMManager → returning structural map")
            return self._MOCK_STRUCTURAL_RESPONSE
        elif "value" in prompt_lower or "pass b" in prompt_lower:
            logger.debug("MockVLMManager → returning value map")
            return self._MOCK_VALUE_RESPONSE
        else:
            logger.debug("MockVLMManager → returning generic response")
            return json.dumps(
                {"status": "mock_response", "prompt_length": len(prompt)},
                indent=2,
            )
