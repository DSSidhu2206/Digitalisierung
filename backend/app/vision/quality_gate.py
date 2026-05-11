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
        orientation = self._detect_orientation(img)

        # 2. Sharpness (Laplacian variance) ------------------------------
        sharpness_score = self._compute_sharpness(img)

        # 3. Contrast ratio (Michelson) ----------------------------------
        contrast_score = self._compute_contrast(img)

        # 4. Minimum dimension check -------------------------------------
        dim_score = self._compute_dimension_score(w, h)

        # 5. Document type detection (keyword heuristic) -----------------
        form_type = self._detect_document_type(img)

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
        )

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

    def _detect_orientation(self, img: np.ndarray) -> str:
        """Detect image orientation.

        Strategy (in order of preference):

        1. If ``pytesseract`` is available, use OSD (Orientation and
           Script Detection).
        2. Fallback: compare horizontal vs vertical projection profiles.
           Bureaucratic documents have dominant horizontal text lines.
        3. If all else fails, return ``"correct"``.

        Returns:
            One of ``"correct"``, ``"rotated_90"``, ``"rotated_180"``,
            ``"rotated_270"``.
        """
        # Try Tesseract OSD first
        try:
            import pytesseract
            osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
            angle = int(osd.get("rotate", 0))
            if angle == 0:
                return "correct"
            elif angle == 90:
                return "rotated_90"
            elif angle == 180:
                return "rotated_180"
            elif angle == 270:
                return "rotated_270"
        except Exception:
            pass  # fall through to heuristic

        # Fallback: compare horizontal vs vertical projection variance
        # Horizontal text → stronger horizontal projection peaks
        h_proj = np.sum(img, axis=1)
        v_proj = np.sum(img, axis=0)

        h_var = float(np.var(h_proj))
        v_var = float(np.var(v_proj))

        if h_var > v_var * 1.5:
            return "correct"
        elif v_var > h_var * 1.5:
            # 90 or 270 — we can't distinguish without OCR, guess 90
            return "rotated_90"
        else:
            # Ambiguous — assume correct
            return "correct"

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
        try:
            import cv2
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            variance = float(laplacian.var())
        except Exception:
            # Pure NumPy fallback (3x3 Laplacian kernel)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
            try:
                from scipy import ndimage
                convolved = ndimage.convolve(img.astype(np.float64), kernel)
            except ImportError:
                # scipy not available — use pure NumPy sliding window
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(img.astype(np.float64), kernel.shape)
                convolved = (windows * kernel).sum(axis=(-2, -1))
            variance = float(convolved.var())

        # Normalise — 500 is a typical threshold for readable text
        score = min(variance / 500.0, 1.0)
        return score

    # ------------------------------------------------------------------
    # Contrast (Michelson)
    # ------------------------------------------------------------------

    def _compute_contrast(self, img: np.ndarray) -> float:
        """Compute Michelson contrast score.

        Returns:
            Float in [0.0, 1.0].
        """
        i_max = float(np.max(img))
        i_min = float(np.min(img))
        if i_max + i_min == 0:
            return 0.0
        contrast = (i_max - i_min) / (i_max + i_min)
        return contrast

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
