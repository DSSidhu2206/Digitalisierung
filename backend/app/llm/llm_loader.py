"""GGUF LLM loader with load/unload cycle for RAM management.

Provides ``LLMManager`` for loading Llama-3.1-8B-Instruct via llama-cpp-python,
and ``MockLLMManager`` for testing without the heavy model dependency.
Both share the same interface so they can be swapped transparently.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import llama_cpp; fall back to MockLLMManager on failure.
# ---------------------------------------------------------------------------
try:
    from llama_cpp import Llama

    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    _LLAMA_CPP_AVAILABLE = False
    logger.warning(
        "llama_cpp not available — LLMManager will fall back to MockLLMManager. "
        "Install llama-cpp-python for local inference."
    )


class LLMManager:
    """GGUF LLM manager with explicit load/unload for 16 GB RAM budget.

    Loads ``models/llama-3.1-8b-instruct-Q4_K_M.gguf`` via llama-cpp-python.
    All generation is **deterministic** (temperature defaults to 0.0).

    Attributes:
        MODEL_PATH: Relative path to the GGUF model file.
    """

    MODEL_PATH: str = "models/llama-3.1-8b-instruct-Q4_K_M.gguf"

    def __init__(self) -> None:
        self.llm: Optional[Any] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the GGUF model into RAM.

        Apple-Silicon-optimised settings:
        - n_ctx=8192       … context window
        - n_threads=8      … Apple Silicon performance cores
        - n_batch=512      … prompt processing batch size
        - n_gpu_layers=-1  … offload ALL transformer layers to the Metal GPU
        - verbose=False    … suppress llama.cpp stderr spam

        Raises:
            RuntimeError: If llama_cpp is not installed or model file is missing.
        """
        if not _LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama_cpp is not installed. Install with: "
                "`pip install llama-cpp-python`"
            )

        # Prefer the configured model path; fall back to the legacy constant.
        n_gpu_layers = -1
        configured_path: Optional[str] = None
        try:
            from config import get_settings

            settings = get_settings()
            n_gpu_layers = int(getattr(settings, "LLM_N_GPU_LAYERS", -1))
            configured_path = getattr(settings, "LLM_MODEL_PATH", None)
        except Exception:
            pass

        resolved = None
        for candidate in (configured_path, self.MODEL_PATH):
            if not candidate:
                continue
            path = candidate if os.path.isabs(candidate) else os.path.join(os.getcwd(), candidate)
            if os.path.isfile(path):
                resolved = path
                break
        if resolved is None:
            raise RuntimeError(
                f"GGUF model not found (looked for {configured_path!r}, {self.MODEL_PATH!r})"
            )

        logger.info("Loading GGUF model from %s (n_gpu_layers=%s, Metal) …", resolved, n_gpu_layers)
        self.llm = Llama(
            model_path=resolved,
            n_ctx=8192,
            n_threads=8,
            n_batch=512,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._loaded = True
        logger.info("GGUF model loaded successfully.")

    def unload(self) -> None:
        """Release the model from RAM and force garbage collection."""
        if self.llm is not None:
            del self.llm
        self.llm = None
        self._loaded = False
        gc.collect()
        logger.info("GGUF model unloaded.")

    def is_loaded(self) -> bool:
        """Return whether the model is currently resident in memory."""
        return self._loaded and self.llm is not None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Run deterministic text generation.

        Args:
            prompt: The full prompt text (including system / user tokens).
            max_tokens: Maximum number of output tokens.
            temperature: Sampling temperature. **Must** be 0.0 for deterministic
                extraction. Non-zero values are logged as warnings.

        Returns:
            Generated text string (may contain JSON).
        """
        if not self.is_loaded():
            raise RuntimeError("LLM is not loaded. Call load() first.")

        if temperature != 0.0:
            logger.warning(
                "Non-zero temperature (%.2f) requested — "
                "extraction should always use 0.0 for determinism.",
                temperature,
            )

        logger.debug(
            "Generating with max_tokens=%d, temperature=%.1f",
            max_tokens,
            temperature,
        )
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
        text: str = output["choices"][0]["text"].strip()
        return text


# ---------------------------------------------------------------------------
# Mock implementation for CI / testing environments without llama_cpp
# ---------------------------------------------------------------------------

class MockLLMManager(LLMManager):
    """Drop-in replacement for ``LLMManager`` that returns structured JSON.

    Used when llama_cpp is not installed or when writing unit tests.
    Generates deterministic JSON responses based on the prompt content.
    """

    def __init__(self) -> None:
        # Intentionally skip super().__init__ to avoid any Llama references.
        self.llm = None  # type: ignore[assignment]
        self._loaded = False

    def load(self) -> None:
        """Pretend to load a model."""
        self._loaded = True
        logger.info("MockLLMManager: model 'loaded'.")

    def unload(self) -> None:
        """Pretend to unload a model."""
        self._loaded = False
        gc.collect()
        logger.info("MockLLMManager: model 'unloaded'.")

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Return a deterministic JSON response based on prompt content.

        Analyses the prompt for document-type and field-name cues to produce
        a plausible but recognisably-mock JSON payload.
        """
        if not self._loaded:
            raise RuntimeError("MockLLMManager is not loaded. Call load() first.")

        if temperature != 0.0:
            logger.warning(
                "Non-zero temperature (%.2f) requested — "
                "extraction should always use 0.0 for determinism.",
                temperature,
            )

        # Determine document type from prompt
        doc_type = "Unknown"
        if "Meldebescheinigung" in prompt:
            doc_type = "Meldebescheinigung"
        elif "Steuerbescheid" in prompt:
            doc_type = "Steuerbescheid"
        elif "Gehaltsausweis" in prompt:
            doc_type = "Gehaltsausweis"
        elif "Personalausweis" in prompt:
            doc_type = "Personalausweis"

        # Build a mock structured JSON response
        mock_data: dict[str, Any] = {"document_type": doc_type, "_mock": True}

        field_cues = [
            ("familienname", "Muster"),
            ("vorname", "Max"),
            ("geburtsdatum", "15.03.1985"),
            ("geburtsort", "Berlin"),
            ("staatsangehoerigkeit", "deutsch"),
            ("strasse", "Hauptstraße"),
            ("hausnummer", "42"),
            ("postleitzahl", "10115"),
            ("wohnort", "Berlin"),
            ("einzugsdatum", "01.01.2024"),
            ("steuer_id", "12345678901"),
            ("arbeitgeber", "Muster GmbH"),
            ("arbeitnehmer", "Max Muster"),
            ("brutto_lohn", 5000.00),
            ("netto_lohn", 3200.00),
        ]

        for field_name, default_value in field_cues:
            if field_name in prompt.lower():
                mock_data[field_name] = default_value

        # Inject few-shot learned values if present in prompt
        for line in prompt.split("\n"):
            if "was corrected to" in line.lower():
                parts = line.split("was corrected to")
                if len(parts) == 2:
                    original = parts[0].strip().strip("- '").strip()
                    corrected_part = parts[1].strip()
                    corrected = corrected_part.split("'")[0].strip()
                    if corrected:
                        # Find which field this correction applies to
                        for field_name, _ in field_cues:
                            if field_name in original.lower():
                                mock_data[field_name] = corrected
                                mock_data["_few_shot_applied"] = True
                                break

        logger.debug("MockLLMManager: returning synthetic response for %s", doc_type)
        return json.dumps(mock_data, indent=2, ensure_ascii=False)
