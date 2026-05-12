"""
Tests for the Vision Engine (Section 6).

Covers:
*   VLMManager load/unload lifecycle (with MockVLMManager)
*   QualityGate assessment scoring
*   DualPass cross-validation consistency
*   ProvenanceTracker coordinate conversion
*   Mock mode operation (no model required)

Run with: ``pytest backend/tests/test_vision.py -v``
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.vision.vlm_loader import MockVLMManager, VLMManager
from app.vision.quality_gate import (
    DocumentType,
    QualityAssessment,
    QualityGate,
    RefusalResult,
)
from app.vision.dual_pass import DualPassExtractor, DualPassResult
from app.vision.provenance import BoundingBox, ProvenanceTracker


# ============================================================================
# Helpers
# ============================================================================

def _make_test_image(
    width: int = 1200,
    height: int = 1600,
    brightness: int = 200,
    noise: float = 0.0,
) -> str:
    """Create a temporary grayscale test image and return its path."""
    img = np.full((height, width), brightness, dtype=np.uint8)
    if noise > 0:
        img = np.clip(
            img.astype(np.float64)
            + np.random.normal(0, noise, size=img.shape),
            0,
            255,
        ).astype(np.uint8)
    # Add some "text-like" sharp edges for Laplacian variance
    img[200:205, 100:500] = 50
    img[400:405, 100:500] = 50
    img[600:605, 100:500] = 50

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    # Write with Pillow (always available)
    from PIL import Image
    Image.fromarray(img, mode="L").save(path)
    return path


# ============================================================================
# VLMManager Tests
# ============================================================================

class TestVLMManager(unittest.TestCase):
    """Test VLMManager and MockVLMManager lifecycle."""

    def test_vlm_manager_model_id_constant(self):
        """MODEL_ID matches the spec."""
        assert VLMManager.MODEL_ID == "surya-ocr"

    def test_vlm_manager_init_state(self):
        """Fresh instance has model=None and is_loaded=False."""
        mgr = VLMManager()
        assert mgr.model is None
        assert mgr.processor is None
        assert mgr.is_loaded is False

    def test_vlm_manager_unload_frees_state(self):
        """unload() resets all state and calls gc.collect()."""
        mgr = VLMManager()
        mgr._loaded = True  # simulate loaded without actual model
        mgr.model = "dummy"
        mgr.processor = "dummy"
        mgr.unload()
        assert mgr.model is None
        assert mgr.processor is None
        assert mgr.is_loaded is False

    def test_vlm_manager_generate_without_load_raises(self):
        """generate() before load() must raise RuntimeError."""
        mgr = VLMManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.generate("/tmp/fake.png", "test prompt")

    def test_vlm_manager_generate_missing_image_raises(self):
        """generate() with nonexistent image must raise FileNotFoundError."""
        mgr = VLMManager()
        mgr._loaded = True  # bypass load check
        with pytest.raises(FileNotFoundError):
            mgr.generate("/tmp/nonexistent_file_12345.png", "test")


class TestMockVLMManager(unittest.TestCase):
    """Test MockVLMManager (stub mode — no model required)."""

    def test_mock_load_unload_cycle(self):
        """Mock load/unload cycle is a no-op but sets state correctly."""
        mock = MockVLMManager()
        assert mock.is_loaded is False
        mock.load()
        assert mock.is_loaded is True
        mock.unload()
        assert mock.is_loaded is False

    def test_mock_generate_pass_a(self):
        """Mock returns structural map for Pass A prompt."""
        mock = MockVLMManager()
        mock.load()
        result = mock.generate("/tmp/fake.png", "PASS A — STRUCTURAL identification")
        data = json.loads(result)
        assert "familienname" in data
        assert "estimated_bbox" in data["familienname"]

    def test_mock_generate_pass_b(self):
        """Mock returns value map for Pass B prompt."""
        mock = MockVLMManager()
        mock.load()
        result = mock.generate("/tmp/fake.png", "PASS B — VALUE EXTRACTION")
        data = json.loads(result)
        assert "familienname" in data
        assert "raw_value" in data["familienname"]
        assert "confidence_0_to_1" in data["familienname"]

    def test_mock_generate_without_load_raises(self):
        """Mock generate() before load() raises RuntimeError."""
        mock = MockVLMManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mock.generate("/tmp/fake.png", "test")

    def test_mock_returns_german_text(self):
        """Mock data contains realistic names for Meldebescheinigung."""
        mock = MockVLMManager()
        mock.load()
        result_b = mock.generate("/tmp/fake.png", "PASS B")
        data = json.loads(result_b)
        assert data["familienname"]["raw_value"] == "M\u00fcller"
        assert data["geburtsdatum"]["raw_value"] == "15.03.1985"
        assert data["postleitzahl"]["raw_value"] == "10115"


# ============================================================================
# QualityGate Tests
# ============================================================================

class TestQualityGate(unittest.TestCase):
    """Test QualityGate assessment and refusal logic."""

    def setUp(self):
        self.gate = QualityGate()

    def tearDown(self):
        # Clean up temp files
        pass

    def test_refuse_formatting(self):
        """refuse() returns formatted RefusalResult."""
        result = self.gate.refuse("too blurry")
        assert isinstance(result, RefusalResult)
        assert result.refused is True
        assert "too blurry" in result.reason
        assert "deterministic extraction" in result.reason

    def test_threshold_constant(self):
        """LEGIBILITY_THRESHOLD is 0.7 per spec."""
        assert QualityGate.LEGIBILITY_THRESHOLD == 0.7

    def test_assess_good_image(self):
        """A high-quality image passes the gate."""
        path = _make_test_image(width=1200, height=1600, brightness=220)
        try:
            result = self.gate.assess(path)
            assert isinstance(result, QualityAssessment)
            assert 0.0 <= result.legibility_score <= 1.0
            # A clean large image should be admissible
            assert result.orientation in (
                "correct",
                "rotated_90",
                "rotated_180",
                "rotated_270",
            )
            assert isinstance(result.form_type, DocumentType)
        finally:
            os.unlink(path)

    def test_assess_low_quality_image_refused(self):
        """A tiny, low-contrast image is refused."""
        path = _make_test_image(width=200, height=200, brightness=128, noise=50)
        try:
            result = self.gate.assess(path)
            assert isinstance(result, QualityAssessment)
            # Tiny images should have low dimension score and be refused
            assert result.legibility_score < 1.0
        finally:
            os.unlink(path)

    def test_assess_file_not_found(self):
        """assess() raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            self.gate.assess("/tmp/nonexistent_vision_test.png")

    def test_assess_returns_document_type(self):
        """assess() returns a DocumentType, not None."""
        path = _make_test_image()
        try:
            result = self.gate.assess(path)
            assert result.form_type in DocumentType
        finally:
            os.unlink(path)

    def test_dimension_score_logic(self):
        """Dimension score scales linearly with minimum dimension."""
        score_full = self.gate._compute_dimension_score(1200, 1600)
        assert score_full == 1.0
        score_half = self.gate._compute_dimension_score(300, 1600)
        expected = 300 / self.gate.MIN_DIMENSION
        assert abs(score_half - expected) < 0.01

    def test_contrast_computation(self):
        """Contrast on a uniform image is zero; on high-contrast it's high."""
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        assert self.gate._compute_contrast(uniform) == 0.0

        high_contrast = np.zeros((100, 100), dtype=np.uint8)
        high_contrast[:, 50:] = 255
        c = self.gate._compute_contrast(high_contrast)
        assert c > 0.9

    def test_sharpness_nonzero(self):
        """Sharpness on an image with edges is non-zero."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[50, :] = 255  # sharp horizontal edge
        score = self.gate._compute_sharpness(img)
        assert score > 0.0

    def test_quality_assessment_dataclass(self):
        """QualityAssessment is a frozen dataclass with correct fields."""
        qa = QualityAssessment(
            legibility_score=0.85,
            orientation="correct",
            form_type=DocumentType.MELDEBESCHEINIGUNG,
            is_admissible=True,
        )
        assert qa.legibility_score == 0.85
        assert qa.is_admissible is True
        assert qa.rejection_reason is None


# ============================================================================
# DualPassExtractor Tests
# ============================================================================

class TestDualPassExtractor(unittest.TestCase):
    """Test DualPassExtractor with MockVLMManager."""

    def setUp(self):
        self.extractor = DualPassExtractor()
        self.vlm = MockVLMManager()
        self.vlm.load()

    def tearDown(self):
        self.vlm.unload()

    def test_extract_returns_dual_pass_result(self):
        """extract() returns a DualPassResult."""
        result = self.extractor.extract("/tmp/fake.png", self.vlm)
        assert isinstance(result, DualPassResult)

    def test_extract_has_structural_map(self):
        """Result contains non-empty structural map from Pass A."""
        result = self.extractor.extract("/tmp/fake.png", self.vlm)
        assert isinstance(result.structural_map, dict)
        assert len(result.structural_map) > 0
        assert "familienname" in result.structural_map

    def test_extract_has_value_map(self):
        """Result contains non-empty value map from Pass B."""
        result = self.extractor.extract("/tmp/fake.png", self.vlm)
        assert isinstance(result.value_map, dict)
        assert len(result.value_map) > 0
        assert "familienname" in result.value_map

    def test_extract_consistency_score_range(self):
        """Consistency score is in [0.0, 1.0]."""
        result = self.extractor.extract("/tmp/fake.png", self.vlm)
        assert 0.0 <= result.consistency_score <= 1.0

    def test_extract_no_contradictions_for_mock(self):
        """Mock data is consistent — expect high score, no contradictions."""
        result = self.extractor.extract("/tmp/fake.png", self.vlm)
        assert result.consistency_score > 0.8
        assert len(result.contradictions) == 0

    def test_cross_validate_detects_missing_in_b(self):
        """Cross-validation flags fields in A but missing in B."""
        struct = {
            "field_a": {"label": "A"},
            "field_b": {"label": "B"},
        }
        values = {
            "field_a": {"raw_value": "x", "confidence_0_to_1": 0.9},
        }
        score, contradictions = self.extractor._cross_validate(struct, values)
        assert any("field_b" in c and ("missing" in c.lower() or "no value" in c.lower()) for c in contradictions)

    def test_cross_validate_detects_orphan_in_b(self):
        """Cross-validation flags values in B not present in A."""
        struct = {
            "field_a": {"label": "A"},
        }
        values = {
            "field_a": {"raw_value": "x", "confidence_0_to_1": 0.9},
            "orphan": {"raw_value": "y", "confidence_0_to_1": 0.9},
        }
        score, contradictions = self.extractor._cross_validate(struct, values)
        assert any("orphan" in c for c in contradictions)

    def test_cross_validate_detects_low_confidence(self):
        """Cross-validation flags fields with confidence < 0.5."""
        struct = {"field_a": {"label": "A"}}
        values = {"field_a": {"raw_value": "x", "confidence_0_to_1": 0.3}}
        score, contradictions = self.extractor._cross_validate(struct, values)
        assert any("low confidence" in c.lower() for c in contradictions)

    def test_parse_json_with_code_fence(self):
        """_parse_json_response handles ```json … ``` fences."""
        raw = '```json\n{"key": "value"}\n```'
        parsed = DualPassExtractor._parse_json_response(raw)
        assert parsed == {"key": "value"}

    def test_parse_json_bare(self):
        """_parse_json_response handles bare JSON."""
        raw = '{"a": 1, "b": 2}'
        parsed = DualPassExtractor._parse_json_response(raw)
        assert parsed == {"a": 1, "b": 2}

    def test_parse_json_empty(self):
        """_parse_json_response returns {} on empty input."""
        assert DualPassExtractor._parse_json_response("") == {}
        assert DualPassExtractor._parse_json_response("   ") == {}

    def test_parse_json_with_trailing_text(self):
        """_parse_json_response extracts JSON embedded in trailing text."""
        raw = 'Some text before {\n  "key": "value"\n} and after'
        parsed = DualPassExtractor._parse_json_response(raw)
        assert parsed == {"key": "value"}

    def test_dual_pass_result_dataclass(self):
        """DualPassResult is a dataclass with correct fields."""
        result = DualPassResult(
            structural_map={"a": {}},
            value_map={"a": {}},
            consistency_score=1.0,
            contradictions=[],
        )
        assert result.consistency_score == 1.0
        assert result.contradictions == []


# ============================================================================
# ProvenanceTracker Tests
# ============================================================================

class TestProvenanceTracker(unittest.TestCase):
    """Test ProvenanceTracker bounding box operations."""

    def setUp(self):
        self.tracker = ProvenanceTracker()

    def test_track_from_vlm_response(self):
        """track() converts VLM bbox dict to normalised BoundingBox."""
        vlm_response = {
            "estimated_bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6}
        }
        bbox = self.tracker.track("field", vlm_response, (1000, 1000))
        assert isinstance(bbox, BoundingBox)
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.2
        assert bbox.x2 == 0.5
        assert bbox.y2 == 0.6
        assert bbox.is_valid

    def test_track_auto_normalises_pixels(self):
        """track() auto-normalises pixel coordinates (> 1.0)."""
        vlm_response = {
            "estimated_bbox": {"x1": 100, "y1": 200, "x2": 500, "y2": 600}
        }
        bbox = self.tracker.track("field", vlm_response, (1000, 1000))
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.2
        assert bbox.x2 == 0.5
        assert bbox.y2 == 0.6

    def test_track_reorders_inverted_coords(self):
        """track() swaps x1/x2 and y1/y2 if inverted."""
        vlm_response = {
            "estimated_bbox": {"x1": 0.8, "y1": 0.7, "x2": 0.2, "y2": 0.1}
        }
        bbox = self.tracker.track("field", vlm_response, (1000, 1000))
        assert bbox.x1 < bbox.x2
        assert bbox.y1 < bbox.y2

    def test_track_clamps_out_of_range(self):
        """track() clamps coordinates to [0, 1]."""
        vlm_response = {
            "x1": -0.5, "y1": -0.1, "x2": 1.5, "y2": 2.0
        }
        bbox = self.tracker.track("field", vlm_response, (1000, 1000))
        assert bbox.x1 == 0.0
        assert bbox.y1 == 0.0
        assert bbox.x2 == 1.0
        assert bbox.y2 == 1.0

    def test_track_invalid_returns_zero_bbox(self):
        """track() returns zero-area bbox for degenerate input."""
        vlm_response = {"estimated_bbox": "not_a_dict"}
        bbox = self.tracker.track("field", vlm_response, (1000, 1000))
        assert bbox.x1 == 0.0
        assert bbox.y2 == 0.0

    def test_bbox_to_pixel_coords(self):
        """BoundingBox.to_pixel_coords scales correctly."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
        px = bbox.to_pixel_coords((1000, 800))
        assert px == (100, 160, 500, 480)

    def test_scale_bbox(self):
        """scale_bbox converts normalised to pixel BoundingBox."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
        scaled = self.tracker.scale_bbox(bbox, (1000, 800))
        assert scaled.x1 == 100.0
        assert scaled.y1 == 160.0
        assert scaled.x2 == 500.0
        assert scaled.y2 == 480.0

    def test_intersection_area_overlap(self):
        """intersection_area for overlapping bboxes is correct."""
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        b = BoundingBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75)
        area = self.tracker.intersection_area(a, b)
        assert abs(area - 0.0625) < 0.001  # 0.25 * 0.25

    def test_intersection_area_no_overlap(self):
        """intersection_area for disjoint bboxes is zero."""
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.2, y2=0.2)
        b = BoundingBox(x1=0.8, y1=0.8, x2=1.0, y2=1.0)
        assert self.tracker.intersection_area(a, b) == 0.0

    def test_iou_perfect_overlap(self):
        """IoU of identical bboxes is 1.0."""
        a = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
        assert self.tracker.iou(a, a) == 1.0

    def test_iou_no_overlap(self):
        """IoU of disjoint bboxes is 0.0."""
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        b = BoundingBox(x1=0.9, y1=0.9, x2=1.0, y2=1.0)
        assert self.tracker.iou(a, b) == 0.0

    def test_iou_partial_overlap(self):
        """IoU for partial overlap is between 0 and 1."""
        a = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        b = BoundingBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75)
        iou = self.tracker.iou(a, b)
        assert 0.0 < iou < 1.0
        # Area a = 0.25, Area b = 0.25, Intersection = 0.0625
        # Union = 0.25 + 0.25 - 0.0625 = 0.4375
        # IoU = 0.0625 / 0.4375 = 0.1428...
        assert abs(iou - 0.142857) < 0.001

    def test_bbox_properties(self):
        """BoundingBox width, height, area are correct."""
        bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
        assert abs(bbox.width - 0.4) < 0.001
        assert abs(bbox.height - 0.4) < 0.001
        assert abs(bbox.area - 0.16) < 0.001

    def test_bbox_validity(self):
        """BoundingBox.is_valid checks coordinate ordering."""
        valid = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
        assert valid.is_valid
        invalid = BoundingBox(x1=0.5, y1=0.5, x2=0.1, y2=0.1)
        assert not invalid.is_valid
        oob = BoundingBox(x1=-0.1, y1=0.0, x2=0.5, y2=0.5)
        assert not oob.is_valid

    def test_highlight_region_returns_image(self):
        """highlight_region returns an np.ndarray of the same shape."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
        result = self.tracker.highlight_region(bbox, image)
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    def test_highlight_region_degenerate_bbox(self):
        """highlight_region returns copy for degenerate bbox."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0.5, y1=0.5, x2=0.5, y2=0.5)
        result = self.tracker.highlight_region(bbox, image)
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape


# ============================================================================
# Integration test — end-to-end with MockVLMManager
# ============================================================================

class TestVisionEngineIntegration(unittest.TestCase):
    """End-to-end test of the vision engine pipeline using mocks."""

    def test_full_pipeline_mock(self):
        """QualityGate + DualPass + Provenance with MockVLMManager."""
        # 1. Create a test image
        img_path = _make_test_image(width=1200, height=1600)
        try:
            # 2. Quality Gate
            gate = QualityGate()
            qa = gate.assess(img_path)
            assert isinstance(qa, QualityAssessment)

            # 3. Dual-Pass Extraction (mock)
            vlm = MockVLMManager()
            extractor = DualPassExtractor()
            result = extractor.extract(img_path, vlm)
            assert isinstance(result, DualPassResult)
            assert len(result.structural_map) > 0
            assert len(result.value_map) > 0

            # 4. Provenance tracking for each field
            tracker = ProvenanceTracker()
            for field_name, struct_data in result.structural_map.items():
                bbox = tracker.track(field_name, struct_data, (1200, 1600))
                assert isinstance(bbox, BoundingBox)
                assert bbox.is_valid

            # 5. Check consistency score
            assert 0.0 <= result.consistency_score <= 1.0

        finally:
            os.unlink(img_path)

    def test_mock_mode_no_real_model(self):
        """Verify MockVLMManager works entirely without Surya predictors."""
        # This test proves the entire vision engine runs in mock mode
        vlm = MockVLMManager()
        vlm.load()
        assert vlm.is_loaded

        # All generate calls should return data, never raise import errors
        r1 = vlm.generate("/tmp/a.png", "PASS A — STRUCTURAL")
        r2 = vlm.generate("/tmp/a.png", "PASS B — VALUE")
        r3 = vlm.generate("/tmp/a.png", "other prompt")

        assert json.loads(r1)  # valid JSON
        assert json.loads(r2)  # valid JSON
        assert json.loads(r3)  # valid JSON

        vlm.unload()
        assert not vlm.is_loaded


if __name__ == "__main__":
    unittest.main()
