"""
Quality Gate — Visual Admissibility Gate (Stage 1).

Determines whether an uploaded image is of sufficient quality for
deterministic bureaucratic document extraction.  Runs heuristics for:

*   Text sharpness (Laplacian variance)
*   Contrast ratio
*   Minimum dimension checks
*   Document orientation detection
*   Document type classification (basic keyword heuristic)

If the combined legibility score falls below LEGIBILITY_THRESHOLD (0.7)
the gate refuses the image with a formatted rejection reason.

Spec: Section 6.2 — Quality Gate
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple DocumentType enum (mirror of models.enums — local to avoid import
# cycle since the cold-pipeline branch may not yet be merged).
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    MELDEBESCHEINIGUNG = "Meldebescheinigung"
    STEUERBESCHEID = "Steuerbescheid"
    GEHALTSAUSWEIS = "Gehaltsausweis"
    PERSONALAUSWEIS = "Personalausweis"
    UNBEKANNT = "Unbekannt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityAssessment:
    """Result of quality gate analysis.

    Attributes:
        legibility_score: Combined 0-1 score (higher = better).
        orientation: Detected orientation string.
        form_type: Detected or guessed document type.
        is_admissible: Whether the image passes the gate.
        rejection_reason: Human-readable reason if refused, else None.
    """

    legibility_score: float
    orientation: str
    form_type: DocumentType
    is_admissible: bool
    rejection_reason: Optional[str] = None
    rotation_degrees: int = 0


@dataclass(frozen=True)
class RefusalResult:
    """Formatted refusal returned when an image is rejected.

    Attributes:
        refused: Always True.
        reason: Detailed rejection explanation.
    """

    refused: bool
    reason: str


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------

class QualityGate:
    """Stage 1: Visual Admissibility Gate.

    Analyses an image and decides whether it is good enough for the
    dual-pass extraction pipeline.  The gate combines multiple
    low-level computer-vision heuristics into a single *legibility
    score*; anything below :attr:`LEGIBILITY_THRESHOLD` is refused.

    Attributes:
        LEGIBILITY_THRESHOLD: Minimum score (0.0-1.0) for admittance.
        MIN_DIMENSION: Smallest acceptable width or height in pixels.
        SHARPNESS_WEIGHT: Weight for Laplacian-variance component.
        CONTRAST_WEIGHT: Weight for Michelson contrast component.
        DIMENSION_WEIGHT: Weight for minimum-dimension component.
    """

    LEGIBILITY_THRESHOLD: float = 0.7
    MIN_DIMENSION: int = 600
    SHARPNESS_WEIGHT: float = 0.4
    CONTRAST_WEIGHT: float = 0.35
    DIMENSION_WEIGHT: float = 0.25

    def __init__(self, use_tesseract: Optional[bool] = None) -> None:
        """Create the quality gate.

        Args:
            use_tesseract: If ``True``, run a tesseract OCR pass for document-type
                detection and OSD-based 90/180/270° deskew. If ``None`` (default),
                read ``QUALITY_GATE_FAST`` from config — fast mode skips that
                ~2-3 s pass and lets the orchestrator classify the document type
                from the engine's own OCR text instead.
        """
        if use_tesseract is None:
            try:
                from config import get_settings

                use_tesseract = not bool(getattr(get_settings(), "QUALITY_GATE_FAST", True))
            except Exception:
                use_tesseract = False
        self.use_tesseract = use_tesseract

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, image_path: str) -> QualityAssessment:
        """Analyse *image_path* and return a :class:`QualityAssessment`.

        The analysis pipeline:

        1. Load image (PIL fallback to OpenCV).
        2. Detect orientation (heuristic or Tesseract).
        3. Compute sharpness (Laplacian variance).
        4. Compute contrast ratio (Michelson).
        5. Check minimum dimensions.
        6. Detect document type (keyword heuristic on a small preview).
        7. Combine into legibility score.
        8. Decide admissibility.

        Args:
            image_path: Path to the image file.

        Returns:
            :class:`QualityAssessment` with all fields populated.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("QualityGate assessing %s", image_path)

        img = self._load_image(image_path)
        h, w = img.shape[:2]

        # 1. Orientation detection ---------------------------------------
        orientation, rotation_degrees = self._detect_orientation(img)

        # 2. Sharpness (Laplacian variance) ------------------------------
        sharpness_score = self._compute_sharpness(img)

        # 3. Contrast ratio (Michelson) ----------------------------------
        contrast_score = self._compute_contrast(img)

        # 4. Minimum dimension check -------------------------------------
        dim_score = self._compute_dimension_score(w, h)

        # 5. Document type — tesseract keyword pass only when explicitly enabled;
        # otherwise the orchestrator classifies it from the OCR text (faster).
        form_type = (
            self._detect_document_type(img)
            if self.use_tesseract
            else DocumentType.UNBEKANNT
        )

        # 6. Combine into legibility score (weighted average) ------------
        legibility_score = (
            self.SHARPNESS_WEIGHT * sharpness_score
            + self.CONTRAST_WEIGHT * contrast_score
            + self.DIMENSION_WEIGHT * dim_score
        )
        legibility_score = float(np.clip(legibility_score, 0.0, 1.0))

        # 7. Decide admissibility ----------------------------------------
        is_admissible = legibility_score >= self.LEGIBILITY_THRESHOLD
        rejection_reason: Optional[str] = None
        if not is_admissible:
            reasons = []
            if sharpness_score < 0.5:
                reasons.append(
                    f"text too blurry (sharpness={sharpness_score:.2f})"
                )
            if contrast_score < 0.5:
                reasons.append(
                    f"insufficient contrast ({contrast_score:.2f})"
                )
            if dim_score < 0.5:
                reasons.append(
                    f"resolution too low ({w}x{h}, min {self.MIN_DIMENSION}px)"
                )
            if legibility_score < self.LEGIBILITY_THRESHOLD:
                reasons.append(
                    f"overall legibility {legibility_score:.2f} < threshold "
                    f"{self.LEGIBILITY_THRESHOLD}"
                )
            rejection_reason = "; ".join(reasons)
            logger.warning(
                "QualityGate REFUSED %s — %s", image_path, rejection_reason
            )

        logger.info(
            "QualityGate result: score=%.3f admissible=%s type=%s orient=%s",
            legibility_score,
            is_admissible,
            form_type.value,
            orientation,
        )

        return QualityAssessment(
            legibility_score=legibility_score,
            orientation=orientation,
            form_type=form_type,
            is_admissible=is_admissible,
            rejection_reason=rejection_reason,
            rotation_degrees=rotation_degrees,
        )

    def deskew_to_temp(
        self, image_path: str, assessment: "QualityAssessment"
    ) -> tuple[str, Optional[str]]:
        """Return ``(working_path, temp_path_or_None)`` uprighting the page.

        Rotates by the counter-clockwise correction the gate detected (only
        set when Tesseract OSD was confident). Writes a temp file and returns
        its path so the caller can delete it; returns the original path and
        ``None`` when no rotation is needed or rotation fails.
        """
        degrees = int(getattr(assessment, "rotation_degrees", 0)) % 360
        if degrees == 0:
            return image_path, None
        try:
            import tempfile
            from PIL import Image

            suffix = os.path.splitext(image_path)[1] or ".png"
            with Image.open(image_path) as im:
                # PIL rotates CCW for positive angles; OSD 'rotate' is the CW
                # offset from upright, so a CCW rotation by that amount restores it.
                corrected = im.rotate(degrees, expand=True)
                if corrected.mode not in ("RGB", "L"):
                    corrected = corrected.convert("RGB")
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                corrected.save(tmp_path)
            logger.info("Auto-oriented image by %d° CCW -> %s", degrees, tmp_path)
            return tmp_path, tmp_path
        except Exception as exc:
            logger.warning("Auto-orientation failed (%s); using original image", exc)
            return image_path, None

    def refuse(self, reason: str) -> RefusalResult:
        """Create a formatted refusal result.

        Args:
            reason: Short technical reason for refusal.

        Returns:
            :class:`RefusalResult` with a human-readable message.
        """
        formatted = (
            f"Image quality insufficient for deterministic extraction: {reason}"
        )
        return RefusalResult(refused=True, reason=formatted)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image as grayscale NumPy array.

        Tries OpenCV first, falls back to Pillow.
        """
        try:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"cv2.imread returned None for {image_path}")
            return img
        except Exception:
            from PIL import Image
            pil_img = Image.open(image_path).convert("L")
            return np.array(pil_img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Orientation detection
    # ------------------------------------------------------------------

    def _detect_orientation(self, img: np.ndarray) -> tuple[str, int]:
        """Detect orientation, returning ``(orientation, rotation_degrees)``.

        *rotation_degrees* is the counter-clockwise correction to apply
        (0/90/180/270) and is **only** non-zero when Tesseract OSD is
        confident.  The projection-profile fallback is display-only and never
        triggers auto-rotation — acting on a guess can corrupt an already
        upright page.
        """
        # Tesseract OSD (the only confident rotation signal) — only when enabled.
        if self.use_tesseract:
            try:
                import pytesseract
                osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
                angle = int(osd.get("rotate", 0)) % 360
                mapping = {
                    0: "correct",
                    90: "rotated_90",
                    180: "rotated_180",
                    270: "rotated_270",
                }
                if angle in mapping:
                    return mapping[angle], angle
            except Exception:
                pass  # fall through to heuristic

        # Fallback: horizontal vs vertical projection variance (display only).
        h_var = float(np.var(np.sum(img, axis=1)))
        v_var = float(np.var(np.sum(img, axis=0)))
        if v_var > h_var * 1.5:
            return "rotated_90", 0  # looks rotated, but not confident → no rotate
        return "correct", 0

    # ------------------------------------------------------------------
    # Sharpness (Laplacian variance)
    # ------------------------------------------------------------------

    def _compute_sharpness(self, img: np.ndarray) -> float:
        """Compute sharpness score from Laplacian variance.

        Score is normalised against an empirical ceiling so that typical
        scanned documents score ~0.7-0.95.

        Returns:
            Float in [0.0, 1.0].
        """
        img_f = img.astype(np.float64)
        try:
            import cv2
            variance = float(cv2.Laplacian(img, cv2.CV_64F).var())
        except Exception:
            # Pure NumPy fallback (3x3 Laplacian kernel)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
            try:
                from scipy import ndimage
                convolved = ndimage.convolve(img_f, kernel)
            except ImportError:
                # scipy not available — use pure NumPy sliding window
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(img_f, kernel.shape)
                convolved = (windows * kernel).sum(axis=(-2, -1))
            variance = float(convolved.var())

        # Normalise — recalibrated for full-resolution scans.
        score = min(variance / 800.0, 1.0)

        # Sensor/scan noise also inflates Laplacian variance, so a grainy image
        # can masquerade as "sharp". Estimate high-frequency noise from the
        # residual after a light 3x3 blur and damp the score when it dominates.
        noise = float(np.abs(img_f - self._box_blur3(img_f)).mean())
        if noise > 18.0:
            score *= max(0.4, 1.0 - (noise - 18.0) / 60.0)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _box_blur3(a: np.ndarray) -> np.ndarray:
        """3x3 mean filter with edge replication (pure NumPy)."""
        p = np.pad(a, 1, mode="edge")
        return (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0

    # ------------------------------------------------------------------
    # Contrast (Michelson)
    # ------------------------------------------------------------------

    def _compute_contrast(self, img: np.ndarray) -> float:
        """Robust contrast via the 2nd–98th percentile intensity spread.

        Michelson ``(max-min)/(max+min)`` saturated to ~1.0 for almost any
        scan — a single near-black and near-white pixel was enough — so it
        never flagged genuinely low-contrast (faded/washed-out) documents. The
        percentile spread ignores a handful of outlier pixels and actually
        tracks readability.

        Returns:
            Float in [0.0, 1.0].
        """
        p2 = float(np.percentile(img, 2))
        p98 = float(np.percentile(img, 98))
        spread = (p98 - p2) / 255.0
        # spread >= 0.60 → excellent; <= 0.15 → unreadable.
        return float(np.clip((spread - 0.15) / (0.60 - 0.15), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Dimension score
    # ------------------------------------------------------------------

    def _compute_dimension_score(self, width: int, height: int) -> float:
        """Score based on whether dimensions exceed the minimum.

        Returns:
            Float in [0.0, 1.0]. 1.0 if both dimensions >= MIN_DIMENSION.
        """
        min_dim = min(width, height)
        if min_dim >= self.MIN_DIMENSION:
            return 1.0
        return min_dim / self.MIN_DIMENSION

    # ------------------------------------------------------------------
    # Document type detection
    # ------------------------------------------------------------------

    def _detect_document_type(self, img: np.ndarray) -> DocumentType:
        """Basic heuristic document type detection.

        Attempts a quick OCR preview and keyword-matches against known
        document types.  Falls back to UNBEKANNT if no keywords match.
        """
        keywords: dict[DocumentType, list[str]] = {
            DocumentType.MELDEBESCHEINIGUNG: [
                "meldebescheinigung",
                "meldeamt",
                "meldebehörde",
                "einzugsdatum",
                "auszugsdatum",
                "familienstand",
                "bürgeramt",
            ],
            DocumentType.STEUERBESCHEID: [
                "steuerbescheid",
                "finanzamt",
                "steueridentifikationsnummer",
                "einkommensteuer",
                "veranlagungszeitraum",
            ],
            DocumentType.GEHALTSAUSWEIS: [
                "gehaltsausweis",
                "lohnabrechnung",
                "brutto",
                "nettolohn",
                "arbeitgeber",
                "sozialversicherung",
            ],
            DocumentType.PERSONALAUSWEIS: [
                "personalausweis",
                "reisepass",
                "dokumentnummer",
                "gültig bis",
                "ausweisnummer",
            ],
        }

        # Try OCR a small preview for keyword matching
        preview_text = ""
        try:
            import pytesseract
            from PIL import Image as PILImage
            # Resize large images before OCR to avoid excessive memory/speed costs
            max_preview_dim = 1200
            h, w = img.shape[:2]
            if max(h, w) > max_preview_dim:
                pil_preview = PILImage.fromarray(img)
                ratio = max_preview_dim / max(h, w)
                new_size = (int(w * ratio), int(h * ratio))
                pil_preview = pil_preview.resize(new_size, PILImage.Resampling.LANCZOS)
                ocr_img = np.array(pil_preview)
            else:
                ocr_img = img
            preview_text = pytesseract.image_to_string(ocr_img, lang="deu").lower()
        except Exception:
            # If Tesseract unavailable, use a tiny mock preview
            preview_text = ""

        # Keyword matching
        best_type = DocumentType.UNBEKANNT
        best_count = 0
        for doc_type, words in keywords.items():
            count = sum(1 for word in words if word in preview_text)
            if count > best_count:
                best_count = count
                best_type = doc_type

        if best_count == 0:
            logger.debug("No document type keywords matched — UNBEKANNT")
        else:
            logger.debug("Detected document type: %s (%d keywords)", best_type.value, best_count)

        return best_type
