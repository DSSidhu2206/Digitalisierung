"""
Provenance Tracker — Bounding box tracking for every extraction.

Converts VLM-estimated coordinates into normalised :class:`BoundingBox`
objects, provides highlight-drawing utilities for the Source Inspector,
and supports overlap-detection for field disambiguation.

Spec: Section 6.4 — Provenance
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BoundingBox (standalone to avoid cross-branch import)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """Normalised bounding box (0-1 range) for provenance tracking.

    Attributes:
        x1: Left edge (normalised 0-1).
        y1: Top edge (normalised 0-1).
        x2: Right edge (normalised 0-1).
        y2: Bottom edge (normalised 0-1).
        page: Zero-based page index (for multi-page TIFF/PDF).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 0

    # -- helpers ----------------------------------------------------------

    @property
    def width(self) -> float:
        """Normalised width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Normalised height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Normalised area."""
        return self.width * self.height

    @property
    def is_valid(self) -> bool:
        """True when 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1."""
        return (
            0.0 <= self.x1 < self.x2 <= 1.0
            and 0.0 <= self.y1 < self.y2 <= 1.0
        )

    def to_pixel_coords(self, image_dimensions: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Scale normalised coordinates to pixel values.

        Args:
            image_dimensions: ``(width, height)`` in pixels.

        Returns:
            ``(x1_px, y1_px, x2_px, y2_px)`` as integers.
        """
        img_w, img_h = image_dimensions
        return (
            int(round(self.x1 * img_w)),
            int(round(self.y1 * img_h)),
            int(round(self.x2 * img_w)),
            int(round(self.y2 * img_h)),
        )


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Track bounding box coordinates for every extracted field.

    Converts the free-form JSON that a VLM returns for each field into
    a strict, validated :class:`BoundingBox`.  Also provides helpers to
    draw highlight overlays (for the Source Inspector) and to detect
    overlapping regions (for field disambiguation).
    """

    def track(
        self,
        field_name: str,
        vlm_response: dict[str, Any],
        image_dimensions: Tuple[int, int],
    ) -> BoundingBox:
        """Convert a VLM response dict into a normalised :class:`BoundingBox`.

        The VLM response dict is expected to contain either:

        *   ``estimated_bbox``: ``{"x1": float, "y1": float, "x2": float, "y2": float}``
            in normalised (0-1) coordinates.
        *   Or the keys ``x1``, ``y1``, ``x2``, ``y2`` directly at the
            top level of the dict.

        If the coordinates are given in pixel values (any component > 1)
they are automatically normalised against *image_dimensions*.

        Args:
            field_name: Name of the field being tracked (for logging).
            vlm_response: Raw dict returned by the VLM for this field.
            image_dimensions: ``(width, height)`` of the original image.

        Returns:
            A validated, normalised :class:`BoundingBox`.
        """
        img_w, img_h = image_dimensions

        # Extract bbox sub-dict if present, else use top-level keys
        bbox_data = vlm_response.get("estimated_bbox", vlm_response)

        if not isinstance(bbox_data, dict):
            logger.warning(
                "No bounding box data for field '%s', returning default", field_name
            )
            return BoundingBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0)

        x1 = float(bbox_data.get("x1", 0.0))
        y1 = float(bbox_data.get("y1", 0.0))
        x2 = float(bbox_data.get("x2", 0.0))
        y2 = float(bbox_data.get("y2", 0.0))

        # Auto-normalise if coordinates appear to be in pixel space.
        # Negative coordinates indicate normalised values that are out of
        # range — in that case we skip normalisation and clamp only.
        has_negative = any(v < 0 for v in (x1, y1, x2, y2))
        if not has_negative:
            if x1 > 1.0:
                x1 = min(x1 / img_w, 1.0)
            if y1 > 1.0:
                y1 = min(y1 / img_h, 1.0)
            if x2 > 1.0:
                x2 = min(x2 / img_w, 1.0)
            if y2 > 1.0:
                y2 = min(y2 / img_h, 1.0)

        # Ensure proper ordering (x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Clamp to [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        if not bbox.is_valid:
            logger.warning(
                "Invalid bounding box for '%s': %s — clamping to zero-area",
                field_name,
                bbox,
            )
            return BoundingBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0)

        logger.debug(
            "Tracked '%s' -> bbox(%.3f, %.3f, %.3f, %.3f)",
            field_name,
            x1,
            y1,
            x2,
            y2,
        )
        return bbox

    def highlight_region(
        self,
        bbox: BoundingBox,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Draw a semi-transparent highlight rectangle on *image*.

        Used by the Source Inspector widget to visually indicate which
        region of the document a given field was extracted from.

        Args:
            bbox: Normalised :class:`BoundingBox` to highlight.
            image: OpenCV BGR image (``np.ndarray``).
            color: BGR tuple for the rectangle colour (default green).
            thickness: Border thickness in pixels.
            alpha: Fill opacity (0 = invisible, 1 = solid).

        Returns:
            A new image array with the highlight drawn.
        """
        h, w = image.shape[:2]
        x1_px, y1_px, x2_px, y2_px = bbox.to_pixel_coords((w, h))

        # Ensure we don't draw outside the image
        x1_px = max(0, x1_px)
        y1_px = max(0, y1_px)
        x2_px = min(w, x2_px)
        y2_px = min(h, y2_px)

        if x1_px >= x2_px or y1_px >= y2_px:
            logger.warning("highlight_region: degenerate bbox — nothing drawn")
            return image.copy()

        overlay = image.copy()

        try:
            import cv2
        except ImportError:
            # Fallback: draw with PIL if OpenCV unavailable
            return self._highlight_region_pil(
                bbox, image, color, thickness, alpha
            )

        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1_px, y1_px), (x2_px, y2_px), color, -1)

        # Blend overlay with original
        highlighted = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw border
        cv2.rectangle(highlighted, (x1_px, y1_px), (x2_px, y2_px), color, thickness)

        return highlighted

    def _highlight_region_pil(
        self,
        bbox: BoundingBox,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """PIL-based fallback for highlight_region when OpenCV is unavailable."""
        from PIL import Image, ImageDraw

        h, w = image.shape[:2]
        x1_px, y1_px, x2_px, y2_px = bbox.to_pixel_coords((w, h))
        x1_px = max(0, x1_px)
        y1_px = max(0, y1_px)
        x2_px = min(w, x2_px)
        y2_px = min(h, y2_px)

        if x1_px >= x2_px or y1_px >= y2_px:
            return image.copy()

        pil_img = Image.fromarray(image).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [x1_px, y1_px, x2_px, y2_px],
            fill=(*color, int(255 * alpha)),
            outline=color,
            width=thickness,
        )
        # Composite overlay onto original image
        result = Image.alpha_composite(pil_img, overlay)
        return np.array(result.convert("RGB"))

    def scale_bbox(
        self,
        bbox: BoundingBox,
        target_dimensions: Tuple[int, int],
    ) -> BoundingBox:
        """Scale a normalised :class:`BoundingBox` to *target_dimensions*.

        This is a convenience wrapper around :meth:`BoundingBox.to_pixel_coords`
        that returns a new :class:`BoundingBox` whose coordinates are
        expressed in the target pixel space (not normalised).

        Args:
            bbox: Source normalised bounding box.
            target_dimensions: ``(width, height)`` of the target image.

        Returns:
            A :class:`BoundingBox` with pixel coordinates (may exceed 1.0).
        """
        x1_px, y1_px, x2_px, y2_px = bbox.to_pixel_coords(target_dimensions)
        return BoundingBox(
            x1=float(x1_px),
            y1=float(y1_px),
            x2=float(x2_px),
            y2=float(y2_px),
            page=bbox.page,
        )

    def intersection_area(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
    ) -> float:
        """Compute the normalised intersection area of two bounding boxes.

        Used for overlap detection — if two fields' bounding boxes
        overlap significantly they may need disambiguation.

        Args:
            bbox1: First normalised bounding box.
            bbox2: Second normalised bounding box.

        Returns:
            Intersection area in normalised coordinates [0.0, 1.0].
        """
        x_left = max(bbox1.x1, bbox2.x1)
        y_top = max(bbox1.y1, bbox2.y1)
        x_right = min(bbox1.x2, bbox2.x2)
        y_bottom = min(bbox1.y2, bbox2.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        return float((x_right - x_left) * (y_bottom - y_top))

    def iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute Intersection-over-Union (IoU) of two bounding boxes.

        Args:
            bbox1: First normalised bounding box.
            bbox2: Second normalised bounding box.

        Returns:
            IoU in [0.0, 1.0].
        """
        intersection = self.intersection_area(bbox1, bbox2)
        union = bbox1.area + bbox2.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union
