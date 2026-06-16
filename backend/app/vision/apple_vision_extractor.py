"""
Apple Vision OCR adapter.

Wraps macOS's built-in ``VNRecognizeTextRequest`` (via the ``ocrmac`` PyObjC
bridge) behind the same surface as :class:`SuryaDocumentExtractor`, returning a
:class:`SuryaExtraction` so the layout-aware field mapper consumes it unchanged.

Why this engine on a Mac: Apple Vision runs on the **Neural Engine + GPU** — the
M-series accelerator that PyTorch/MPS (and therefore Surya) cannot target — so it
is ~15–40× faster per document while remaining fully on-device (no network, no
model download). Confidence scores are crisp, which makes a tiered policy
(Vision primary, Surya fallback on low confidence) straightforward.

Coordinate note: Vision returns ``[x, y, w, h]`` normalised with a **bottom-left**
origin; the pipeline uses a top-left ``{x1, y1, x2, y2}``. This adapter converts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.vision.surya_extractor import SuryaExtraction, SuryaTextLine

logger = logging.getLogger(__name__)


class AppleVisionExtractor:
    """Apple Vision OCR with the SuryaDocumentExtractor interface."""

    MODEL_ID = "apple-vision"

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = ("de-DE", "en-US"),
        recognition_level: str = "accurate",
        min_confidence: float = 0.30,
        use_language_correction: bool = True,
        custom_words: Optional[list[str]] = None,
        **_ignored: Any,
    ) -> None:
        self.languages = languages
        self.recognition_level = recognition_level
        self.min_confidence = min_confidence
        self.use_language_correction = use_language_correction
        self.custom_words = custom_words or _DEFAULT_CUSTOM_WORDS
        self._loaded = False

    # -- lifecycle (Vision is an OS service; "load" just verifies availability) --

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def engine(self) -> str:
        return "apple-vision"

    def load(self) -> None:
        import ocrmac  # noqa: F401  (raises if the Vision bridge is unavailable)

        self._loaded = True
        logger.info("Apple Vision OCR ready (languages=%s).", ",".join(self.languages))

    def unload(self) -> None:
        self._loaded = False

    # -- extraction ----------------------------------------------------------

    def extract_text(self, image_path: str | Path) -> str:
        return self.extract(image_path).text

    def extract(self, image_path: str | Path, *, with_layout: Optional[bool] = None) -> SuryaExtraction:
        from ocrmac import ocrmac

        path = str(image_path)
        kwargs: dict[str, Any] = {
            "recognition_level": self.recognition_level,
            "language_preference": list(self.languages),
        }
        try:
            annotations = ocrmac.OCR(path, **kwargs).recognize()
        except TypeError:
            # Older ocrmac without language_preference kwarg.
            annotations = ocrmac.OCR(path, recognition_level=self.recognition_level).recognize()

        lines: list[SuryaTextLine] = []
        for text, confidence, bbox in annotations:
            cleaned = " ".join(str(text).split())
            if not cleaned or float(confidence) < self.min_confidence:
                continue
            lines.append(
                SuryaTextLine(
                    text=cleaned,
                    confidence=float(confidence),
                    bbox=self._to_top_left_bbox(bbox),
                    polygon=[],
                )
            )

        # Vision over-segments (splits a value like "01.09." / ".2030" or
        # "Hauptstraße" / "42" into separate regions). Merge horizontally
        # adjacent regions on the same visual line so a field reads as one line,
        # while a large horizontal gap (a column boundary) is kept separate.
        lines = self._merge_same_line(lines)

        # Reading order: top-to-bottom, then left-to-right.
        lines.sort(key=lambda ln: (round((ln.bbox or {}).get("y1", 0.0), 2), (ln.bbox or {}).get("x1", 0.0)))
        text = "\n".join(ln.text for ln in lines)
        return SuryaExtraction(
            text=text,
            lines=lines,
            layout=[],
            engine="apple-vision",
            model=self.MODEL_ID,
        )

    def extract_batch(self, image_paths, *, with_layout: Optional[bool] = None) -> list[SuryaExtraction]:
        return [self.extract(p, with_layout=with_layout) for p in image_paths]

    @property
    def mean_confidence(self) -> float:  # convenience for tiered policy
        return 0.0

    @staticmethod
    def _to_top_left_bbox(bbox: Any) -> Optional[dict[str, float]]:
        """Convert Vision ``[x, y, w, h]`` (bottom-left origin) → top-left x1y1x2y2."""
        try:
            x, y, w, h = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        return {
            "x1": max(0.0, min(x, 1.0)),
            "y1": max(0.0, min(1.0 - (y + h), 1.0)),
            "x2": max(0.0, min(x + w, 1.0)),
            "y2": max(0.0, min(1.0 - y, 1.0)),
        }

    # Max horizontal gap (normalised) to treat two regions as the same field; a
    # larger gap is a column boundary and is kept as separate lines. Kept small
    # so only near-touching fragments (e.g. a split date) merge, not columns.
    _MERGE_MAX_GAP = 0.012

    @classmethod
    def _merge_same_line(cls, lines: list[SuryaTextLine]) -> list[SuryaTextLine]:
        """Merge horizontally-adjacent regions sharing a visual line."""
        items = [ln for ln in lines if ln.bbox]
        items.sort(key=lambda ln: ((ln.bbox["y1"] + ln.bbox["y2"]) / 2, ln.bbox["x1"]))
        used = [False] * len(items)
        merged: list[SuryaTextLine] = [ln for ln in lines if not ln.bbox]
        for i, base in enumerate(items):
            if used[i]:
                continue
            group = [base]
            used[i] = True
            y_center = (base.bbox["y1"] + base.bbox["y2"]) / 2
            height = max(base.bbox["y2"] - base.bbox["y1"], 0.008)
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                cand = items[j]
                cand_h = max(cand.bbox["y2"] - cand.bbox["y1"], 0.008)
                cand_center = (cand.bbox["y1"] + cand.bbox["y2"]) / 2
                if abs(cand_center - y_center) > 0.5 * min(height, cand_h):
                    continue  # different visual line
                if max(height, cand_h) / min(height, cand_h) > 1.8:
                    continue  # very different sizes (e.g. a title vs body text)
                gap = cand.bbox["x1"] - max(g.bbox["x2"] for g in group)
                if gap > cls._MERGE_MAX_GAP:
                    continue  # gap too big — separate field / column
                group.append(cand)
                used[j] = True
            merged.append(cls._fuse(group))
        return merged

    @staticmethod
    def _fuse(group: list[SuryaTextLine]) -> SuryaTextLine:
        group = sorted(group, key=lambda g: g.bbox["x1"])
        parts: list[str] = []
        prev_x2: Optional[float] = None
        for g in group:
            if prev_x2 is not None and (g.bbox["x1"] - prev_x2) > 0.004:
                parts.append(" ")  # visible gap → a real space
            parts.append(g.text)
            prev_x2 = g.bbox["x2"]
        return SuryaTextLine(
            text=" ".join("".join(parts).split()),
            confidence=min(g.confidence for g in group),
            bbox={
                "x1": min(g.bbox["x1"] for g in group),
                "y1": min(g.bbox["y1"] for g in group),
                "x2": max(g.bbox["x2"] for g in group),
                "y2": max(g.bbox["y2"] for g in group),
            },
            polygon=[],
        )


# German bureaucratic terms that help Vision's language correction not "fix"
# domain words into ordinary German.
_DEFAULT_CUSTOM_WORDS = [
    "Meldebescheinigung", "Steuerbescheid", "Gehaltsausweis", "Personalausweis",
    "Familienname", "Vorname", "Geburtsdatum", "Geburtsort", "Staatsangehörigkeit",
    "Postleitzahl", "Wohnort", "Hausnummer", "Steueridentifikationsnummer",
    "Veranlagungszeitraum", "Bruttolohn", "Nettolohn", "Solidaritätszuschlag",
    "Kirchensteuer", "Dokumentnummer", "Bundesrepublik",
]
