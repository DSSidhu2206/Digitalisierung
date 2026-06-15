"""
Vision loader backed by Surya OCR.

Historically this module wrapped an MLX VLM.  The public interface is kept so
the RAM manager and dual-pass pipeline do not need a broad rewrite, but the
real extractor is now Surya OCR/layout analysis.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from app.vision.surya_extractor import SuryaDocumentExtractor, SuryaExtraction

logger = logging.getLogger(__name__)


class VLMManager:
    """Surya-backed vision manager with the legacy VLMManager interface."""

    MODEL_ID: str = SuryaDocumentExtractor.MODEL_ID
    supports_direct_extraction: bool = True

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self._loaded: bool = False
        self._engine: str = self._resolve_engine()
        self._extractor = self._build_extractor()
        self._fallback: Any = None  # lazy Surya for "tiered" mode
        self._last_image: str | None = None
        self._last_extraction: SuryaExtraction | None = None

    @staticmethod
    def _resolve_engine() -> str:
        try:
            from config import get_settings

            return (getattr(get_settings(), "OCR_ENGINE", "surya") or "surya").strip().lower()
        except Exception:
            return "surya"

    def _build_extractor(self) -> Any:
        """Pick the primary OCR engine from config (Surya by default)."""
        if self._engine in ("apple-vision", "apple", "vision", "tiered"):
            try:
                from app.vision.apple_vision_extractor import AppleVisionExtractor

                return AppleVisionExtractor()
            except Exception as exc:
                logger.warning("Apple Vision unavailable (%s); using Surya.", exc)
                self._engine = "surya"
        # Layout blocks are not consumed by the field mapper, so the layout model
        # is skipped: ~30% faster per document and less unified memory.
        return SuryaDocumentExtractor(enable_layout=False, allow_fallback=False)

    @property
    def is_loaded(self) -> bool:
        """Return whether Surya predictors are loaded."""
        return self._loaded

    def load(self) -> None:
        """Load Surya OCR/layout predictors."""
        self._extractor.load()
        self.model = self._extractor
        self.processor = self.MODEL_ID
        self._loaded = True
        logger.info("Surya vision manager loaded with engine=%s", self._extractor.engine)

    def unload(self) -> None:
        """Unload Surya predictors and clear cached extraction state."""
        self._extractor.unload()
        self.model = None
        self.processor = None
        self._last_image = None
        self._last_extraction = None
        self._loaded = False
        logger.info("Surya vision manager unloaded.")

    def extract_document(
        self,
        image_path: str,
        *,
        with_layout: bool = True,
    ) -> SuryaExtraction:
        """Run Surya document extraction once for the image."""
        if not self._loaded:
            raise RuntimeError("VLM not loaded. Call load() before extract_document().")
        path = str(Path(image_path))
        if self._last_image == path and self._last_extraction is not None:
            return self._last_extraction
        extraction = self._extractor.extract(path, with_layout=with_layout)
        if self._engine == "tiered" and self._needs_surya_fallback(extraction):
            extraction = self._surya_fallback(path, extraction)
        self._last_image = path
        self._last_extraction = extraction
        return extraction

    @staticmethod
    def _needs_surya_fallback(extraction: SuryaExtraction) -> bool:
        """Tiered: fall back to Surya when the fast engine looks unreliable."""
        lines = list(getattr(extraction, "lines", []) or [])
        if len(lines) < 3:
            return True
        mean_conf = sum(float(ln.confidence) for ln in lines) / max(len(lines), 1)
        return mean_conf < 0.55

    def _surya_fallback(self, path: str, primary: SuryaExtraction) -> SuryaExtraction:
        if self._fallback is None:
            self._fallback = SuryaDocumentExtractor(enable_layout=False, allow_fallback=True)
        try:
            if not self._fallback.is_loaded:
                self._fallback.load()
            surya = self._fallback.extract(path)
            logger.info(
                "Tiered fallback to Surya (%d vs %d lines)", len(surya.lines), len(primary.lines)
            )
            return surya if len(surya.lines) >= len(primary.lines) else primary
        except Exception as exc:
            logger.warning("Surya fallback failed (%s); keeping primary result.", exc)
            return primary

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Return JSON/text derived from one Surya extraction.

        This keeps compatibility with older two-prompt code paths while avoiding
        a heavyweight VLM generation dependency.
        """
        if not self._loaded:
            raise RuntimeError("VLM not loaded. Call load() before generate().")
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        extraction = self.extract_document(image_path, with_layout=True)
        prompt_lower = prompt.lower()
        if "structural" in prompt_lower or "pass a" in prompt_lower:
            return json.dumps(
                self._structural_map_from_extraction(extraction),
                ensure_ascii=False,
                indent=2,
            )
        if "value" in prompt_lower or "pass b" in prompt_lower:
            return json.dumps(
                self._value_map_from_extraction(extraction),
                ensure_ascii=False,
                indent=2,
            )
        return extraction.text[:max_tokens]

    @classmethod
    def _structural_map_from_extraction(
        cls,
        extraction: SuryaExtraction,
    ) -> dict[str, dict[str, Any]]:
        structural: dict[str, dict[str, Any]] = {}
        for index, line in enumerate(extraction.lines, start=1):
            key, _ = cls._split_line(line.text)
            field_name = cls._unique_field_name(
                structural,
                cls._normalise_field_name(key or f"line_{index:04d}"),
            )
            structural[field_name] = {
                "label_text": key or f"OCR line {index}",
                "field_type": cls._infer_field_type(line.text),
                "estimated_bbox": line.bbox,
                "engine": extraction.engine,
            }
        return structural

    @classmethod
    def _value_map_from_extraction(
        cls,
        extraction: SuryaExtraction,
    ) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for index, line in enumerate(extraction.lines, start=1):
            key, value = cls._split_line(line.text)
            field_name = cls._unique_field_name(
                values,
                cls._normalise_field_name(key or f"line_{index:04d}"),
            )
            values[field_name] = {
                "raw_value": value or line.text,
                "confidence_0_to_1": line.confidence,
                "engine": extraction.engine,
            }
        return values

    @staticmethod
    def _split_line(text: str) -> tuple[str, str]:
        cleaned = " ".join(text.strip().split())
        for separator in (":", "=", "|"):
            if separator in cleaned:
                left, right = cleaned.split(separator, 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return left, right
        return "", cleaned

    @staticmethod
    def _normalise_field_name(value: str) -> str:
        field = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        field = "_".join(part for part in field.split("_") if part)
        if not field:
            return "field"
        if field[0].isdigit():
            field = f"field_{field}"
        return field[:80]

    @staticmethod
    def _unique_field_name(existing: dict[str, Any], base: str) -> str:
        candidate = base or "field"
        suffix = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _infer_field_type(text: str) -> str:
        stripped = text.strip()
        digits = sum(ch.isdigit() for ch in stripped)
        if digits >= 6 and any(ch in stripped for ch in (".", "/", "-")):
            return "date"
        if digits and digits >= max(1, len(stripped) // 2):
            return "number"
        return "text"


class MockVLMManager(VLMManager):
    """Drop-in vision manager for tests and development without model weights."""

    supports_direct_extraction = False

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
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False
        self._rng = random.Random(seed)

    def load(self) -> None:
        self._loaded = True
        logger.info("MockVLMManager load() - no-op")

    def unload(self) -> None:
        self._loaded = False
        self.model = None
        self.processor = None
        logger.info("MockVLMManager unload() - no-op")

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("Mock VLM not loaded. Call load() before generate().")
        time.sleep(0.01)
        prompt_lower = prompt.lower()
        if "structural" in prompt_lower or "pass a" in prompt_lower:
            return self._MOCK_STRUCTURAL_RESPONSE
        if "value" in prompt_lower or "pass b" in prompt_lower:
            return self._MOCK_VALUE_RESPONSE
        return json.dumps({"status": "mock_response", "prompt_length": len(prompt)}, indent=2)
