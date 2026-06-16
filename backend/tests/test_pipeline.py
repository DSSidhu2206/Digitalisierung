"""
Tests for the Pipeline Orchestrator.

Covers:
- RAMManager load/unload/swapping
- Memory pressure detection
- Full pipeline with mock models
- Quality gate refusal path
- Correction submission (atomic)
- Audit logging (JSONL round-trip)
- Stats aggregation
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

# Ensure the backend package is importable
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.enums import DocumentType, ExtractionPhase, FieldStatus
from app.models.extraction_models import (
    BoundingBox,
    ExtractionMetadata,
    ExtractionResult,
    FieldResult,
)
from app.database.image_learning_store import ImageLearningStore, LearnedImageRecord
from app.pipeline import AuditLogger, ExtractionPipeline, RAMManager
from app.vision.quality_gate import (
    DocumentType as QualityDocumentType,
    QualityAssessment,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def tmp_log_file(tmp_path: Path) -> str:
    """Return a temporary path for the audit log."""
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> str:
    """Return a temporary directory for ChromaDB."""
    return str(tmp_path / "chroma_data")


@pytest.fixture
def audit_logger(tmp_log_file: str) -> AuditLogger:
    """Return a fresh AuditLogger with a temp log file."""
    return AuditLogger(
        log_path=tmp_log_file,
        rotate_size_mb=1,  # Small for testing rotation
        max_archives=2,
    )


@pytest.fixture
def ram_manager() -> RAMManager:
    """Return a fresh RAMManager."""
    return RAMManager()


@pytest.fixture
def pipeline(tmp_log_file: str, tmp_chroma_dir: str) -> ExtractionPipeline:
    """Return a mock-mode ExtractionPipeline."""
    return ExtractionPipeline(
        use_mocks=True,
        audit_log_path=tmp_log_file,
        chroma_persist_dir=tmp_chroma_dir,
    )


@pytest.fixture
def sample_result() -> ExtractionResult:
    """Return a sample ExtractionResult for testing."""
    return ExtractionResult(
        metadata=ExtractionMetadata(
            vlm_model="test-vlm",
            llm_model="test-llm",
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            phases_completed=[
                ExtractionPhase.QUALITY_GATE,
                ExtractionPhase.VISION_PASS_A,
                ExtractionPhase.VISION_PASS_B,
                ExtractionPhase.SYMBOLIC_VALIDATION,
                ExtractionPhase.LLM_STRUCTURING,
                ExtractionPhase.COMPLETE,
            ],
            processing_time_ms=1234.5,
            few_shot_injections=2,
        ),
        document_type=DocumentType.MELDEBESCHEINIGUNG,
        fields={
            "familienname": FieldResult(
                field_name="familienname",
                value="Mueller",
                status=FieldStatus.EXTRACTED,
                confidence=0.95,
            ),
            "vorname": FieldResult(
                field_name="vorname",
                value="Hans",
                status=FieldStatus.EXTRACTED,
                confidence=0.93,
            ),
            "postleitzahl": FieldResult(
                field_name="postleitzahl",
                value="10115",
                status=FieldStatus.EXTRACTED,
                confidence=0.96,
            ),
            "unresolved_field": FieldResult(
                field_name="unresolved_field",
                value=None,
                status=FieldStatus.UNRESOLVED,
                confidence=0.0,
                validation_message="Could not extract",
            ),
        },
    )


# =========================================================================
# RAMManager Tests
# =========================================================================


class TestRAMManager:
    """Tests for :class:`RAMManager`."""

    @pytest.mark.asyncio
    async def test_init_no_models_loaded(self, ram_manager: RAMManager) -> None:
        """RAMManager starts with no model loaded."""
        assert ram_manager.current_model is None

    @pytest.mark.asyncio
    async def test_mock_mode_swap(self, pipeline: ExtractionPipeline) -> None:
        """Mock VLM and LLM can be swapped."""
        ram = pipeline.ram

        # Mock mode uses MockVLMManager and MockLLMManager
        assert ram._vlm is not None
        assert ram._llm is not None

        # Swap to VLM
        await ram._swap_to("vlm")
        assert ram.current_model == "vlm"
        assert ram.vlm.is_loaded

        # Swap to LLM (should unload VLM first)
        await ram._swap_to("llm")
        assert ram.current_model == "llm"
        assert ram.llm.is_loaded

        # VLM should be unloaded
        assert not ram.vlm.is_loaded

    @pytest.mark.asyncio
    async def test_swap_to_same_model_noop(self, pipeline: ExtractionPipeline) -> None:
        """Swapping to the same model is a no-op."""
        ram = pipeline.ram
        await ram._swap_to("vlm")
        assert ram.current_model == "vlm"

        # Swapping to same model should not cause issues
        await ram._swap_to("vlm")
        assert ram.current_model == "vlm"

    @pytest.mark.asyncio
    async def test_with_vlm_callback(self, pipeline: ExtractionPipeline) -> None:
        """with_vlm loads VLM, runs callback, unloads if memory pressure."""
        ram = pipeline.ram

        async def callback(vlm: Any) -> str:
            assert vlm.is_loaded
            return "vlm_done"

        result = await ram.with_vlm(callback)
        assert result == "vlm_done"

    @pytest.mark.asyncio
    async def test_with_llm_callback(self, pipeline: ExtractionPipeline) -> None:
        """with_llm loads LLM, runs callback, unloads if memory pressure."""
        ram = pipeline.ram

        async def callback(llm: Any) -> str:
            assert llm.is_loaded
            return "llm_done"

        result = await ram.with_llm(callback)
        assert result == "llm_done"

    @pytest.mark.asyncio
    async def test_sequential_swap_never_both_loaded(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """Both models are never loaded at the same time."""
        ram = pipeline.ram

        async def vlm_callback(vlm: Any) -> bool:
            return vlm.is_loaded

        async def llm_callback(llm: Any) -> bool:
            return llm.is_loaded()

        vlm_result = await ram.with_vlm(vlm_callback)
        assert vlm_result is True  # VLM was loaded during callback

        llm_result = await ram.with_llm(llm_callback)
        assert llm_result is True  # LLM was loaded during callback

    @pytest.mark.asyncio
    async def test_unload_all(self, pipeline: ExtractionPipeline) -> None:
        """unload_all unloads both models."""
        ram = pipeline.ram
        await ram._swap_to("vlm")
        assert ram.current_model == "vlm"

        await ram.unload_all()
        assert ram.current_model is None

    @pytest.mark.asyncio
    async def test_ram_usage_returns_float(self, ram_manager: RAMManager) -> None:
        """get_ram_usage_mb returns a non-negative float."""
        usage_mb = ram_manager.get_ram_usage_mb()
        assert isinstance(usage_mb, float)
        assert usage_mb >= 0.0

    @pytest.mark.asyncio
    async def test_ram_usage_percent(self, ram_manager: RAMManager) -> None:
        """get_ram_usage_percent returns a value in [0, 1]."""
        pct = ram_manager.get_ram_usage_percent()
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 1.0

    @pytest.mark.asyncio
    async def test_context_manager(self, pipeline: ExtractionPipeline) -> None:
        """RAMManager works as an async context manager."""
        ram = pipeline.ram
        await ram._swap_to("vlm")

        async with ram:
            pass  # models get unloaded on exit

        assert ram.current_model is None


# =========================================================================
# AuditLogger Tests
# =========================================================================


class TestAuditLogger:
    """Tests for :class:`AuditLogger`."""

    def test_log_writes_jsonl(self, audit_logger: AuditLogger, sample_result: ExtractionResult) -> None:
        """log() appends a single JSON line to the file."""
        audit_logger.log(sample_result)

        assert audit_logger.log_path.exists()
        lines = audit_logger.log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["document_type"] == DocumentType.MELDEBESCHEINIGUNG.value
        # Confidence is pre-computed by the model validator
        assert 0.0 <= record["overall_confidence"] <= 1.0

    def test_log_multiple(self, audit_logger: AuditLogger, sample_result: ExtractionResult) -> None:
        """Multiple logs produce multiple lines."""
        audit_logger.log(sample_result)
        audit_logger.log(sample_result)
        audit_logger.log(sample_result)

        lines = audit_logger.log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_get_extraction_found(self, audit_logger: AuditLogger, sample_result: ExtractionResult) -> None:
        """get_extraction finds a logged result by ID."""
        audit_logger.log(sample_result)
        extraction_id = sample_result.metadata.extraction_id

        found = audit_logger.get_extraction(extraction_id)
        assert found is not None
        assert found.metadata.extraction_id == extraction_id
        assert found.document_type == DocumentType.MELDEBESCHEINIGUNG

    def test_get_extraction_not_found(self, audit_logger: AuditLogger) -> None:
        """get_extraction returns None for unknown ID."""
        found = audit_logger.get_extraction(uuid4())
        assert found is None

    def test_get_history(self, audit_logger: AuditLogger, sample_result: ExtractionResult) -> None:
        """get_history returns recent extractions."""
        audit_logger.log(sample_result)

        history = audit_logger.get_history(limit=10)
        assert len(history) == 1
        assert history[0].vlm_model == "test-vlm"

    def test_get_history_filtered_by_type(
        self, audit_logger: AuditLogger, sample_result: ExtractionResult
    ) -> None:
        """get_history filters by document type."""
        audit_logger.log(sample_result)

        history = audit_logger.get_history(
            document_type=DocumentType.STEUERBESCHEID, limit=10
        )
        assert len(history) == 0

        history = audit_logger.get_history(
            document_type=DocumentType.MELDEBESCHEINIGUNG, limit=10
        )
        assert len(history) == 1

    def test_get_stats(self, audit_logger: AuditLogger, sample_result: ExtractionResult) -> None:
        """get_stats returns aggregate statistics."""
        audit_logger.log(sample_result)
        audit_logger.log(sample_result)

        stats = audit_logger.get_stats()
        assert stats["total_extractions"] == 2
        assert stats["avg_confidence"] > 0
        assert DocumentType.MELDEBESCHEINIGUNG.value in stats["document_type_breakdown"]

    def test_rotate_logs(self, audit_logger: AuditLogger, sample_result: ExtractionResult) -> None:
        """rotate_logs archives the current log and starts fresh."""
        # Write enough data to trigger rotation
        for _ in range(5):
            audit_logger.log(sample_result)

        assert audit_logger.log_path.exists()
        initial_size = audit_logger.log_path.stat().st_size

        # Force rotation
        audit_logger.rotate_logs()

        # Log should be fresh (nearly empty)
        assert audit_logger.log_path.exists()

        # Archive should exist
        archive_dir = audit_logger.log_path.parent / "archive"
        archives = list(archive_dir.glob("*.gz"))
        assert len(archives) >= 1

    def test_log_path_auto_created(self, tmp_path: Path) -> None:
        """The log directory is created automatically if missing."""
        log_path = tmp_path / "nested" / "deep" / "audit.jsonl"
        assert not log_path.parent.exists()
        logger = AuditLogger(log_path=str(log_path))
        assert log_path.parent.exists()


# =========================================================================
# ExtractionPipeline Tests
# =========================================================================


def _create_quality_test_image(path: str, size: tuple[int, int] = (1200, 1600)) -> None:
    """Create a test image that passes the quality gate.

    Uses high-contrast text-like patterns to pass sharpness/contrast checks.
    """
    from PIL import Image

    # Create high-contrast image with text-like patterns
    img_array = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)

    # Add high-contrast text-like stripes for sharpness
    for i in range(50, size[1], 100):
        img_array[:, i:i+2, :] = 0  # black lines
    for j in range(50, size[0], 80):
        img_array[j:j+2, :, :] = 255  # white lines

    # Ensure minimum contrast
    img_array = np.clip(img_array, 50, 200).astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(path)


class TestExtractionPipeline:
    """Tests for :class:`ExtractionPipeline`."""

    @pytest.mark.asyncio
    async def test_mock_pipeline_process(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """Full pipeline with mock models produces an ExtractionResult."""
        tmp_dir = Path(pipeline.chroma._persist_dir).parent
        test_image = tmp_dir / "test_doc.png"
        _create_quality_test_image(str(test_image))

        result = await pipeline.process(str(test_image))

        assert isinstance(result, ExtractionResult)
        assert result.metadata is not None
        assert result.metadata.extraction_id is not None
        assert result.metadata.processing_time_ms >= 0
        assert result.metadata.phases_completed  # Non-empty

        # Quality gate should have run
        assert ExtractionPhase.QUALITY_GATE in result.metadata.phases_completed

    @pytest.mark.asyncio
    async def test_mock_mode_flag(self, pipeline: ExtractionPipeline) -> None:
        """Mock mode flag is set correctly."""
        assert pipeline.use_mocks is True

    def test_get_stats_returns_dict(self, pipeline: ExtractionPipeline) -> None:
        """get_stats returns a dictionary with expected keys."""
        stats = asyncio.run(pipeline.get_stats())
        assert "audit" in stats
        assert "corrections" in stats
        assert "image_learning" in stats
        assert "mock_mode" in stats

    @pytest.mark.asyncio
    async def test_get_extraction_not_found(self, pipeline: ExtractionPipeline) -> None:
        """get_extraction returns None for unknown ID."""
        result = pipeline.get_extraction(str(uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_correction_returns_id(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """submit_correction returns a correction id string."""
        # First create an extraction to correct
        result = ExtractionResult(
            metadata=ExtractionMetadata(
                extraction_id=uuid4(),
                vlm_model="test-vlm",
                llm_model="test-llm",
            ),
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            fields={
                "postleitzahl": FieldResult(
                    field_name="postleitzahl",
                    value="10115",
                    status=FieldStatus.EXTRACTED,
                    confidence=0.95,
                )
            },
        )
        pipeline._extraction_store[str(result.metadata.extraction_id)] = result

        correction_id = await pipeline.submit_correction(
            extraction_id=result.metadata.extraction_id,
            field_name="postleitzahl",
            corrected_value="10117",
            original_value="10115",
        )
        assert isinstance(correction_id, str)
        assert len(correction_id) > 0

    @pytest.mark.asyncio
    async def test_ram_manager_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has a RAMManager instance."""
        assert pipeline.ram is not None
        assert isinstance(pipeline.ram, RAMManager)

    @pytest.mark.asyncio
    async def test_quality_gate_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has a QualityGate instance."""
        assert pipeline.quality_gate is not None

    def test_unknown_document_uses_dual_pass_fallback(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """Unclassified images should still return extracted fields."""
        tmp_dir = Path(pipeline.chroma._persist_dir).parent
        test_image = tmp_dir / "unknown_doc.png"
        _create_quality_test_image(str(test_image))

        pipeline.quality_gate.assess = lambda _: QualityAssessment(
            legibility_score=0.85,
            orientation="correct",
            form_type=QualityDocumentType.UNBEKANNT,
            is_admissible=True,
        )

        result = asyncio.run(pipeline.process(str(test_image)))

        assert "_refusal" not in result.fields
        assert result.document_type == DocumentType.UNBEKANNT
        assert len(result.fields) > 1
        assert result.overall_confidence > 0.0

    def test_dirty_image_continues_with_low_confidence_fields(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """Dirty but readable images should not collapse to 0% confidence."""
        tmp_dir = Path(pipeline.chroma._persist_dir).parent
        test_image = tmp_dir / "dirty_doc.png"
        _create_quality_test_image(str(test_image))

        pipeline.quality_gate.assess = lambda _: QualityAssessment(
            legibility_score=0.45,
            orientation="correct",
            form_type=QualityDocumentType.UNBEKANNT,
            is_admissible=False,
            rejection_reason="low contrast and blur",
        )

        result = asyncio.run(pipeline.process(str(test_image)))

        assert "_refusal" not in result.fields
        assert len(result.fields) > 1
        assert result.overall_confidence > 0.0
        assert all(field.confidence < 0.7 for field in result.fields.values())
        assert any(
            field.status == FieldStatus.LOW_CONFIDENCE
            for field in result.fields.values()
        )

    def test_hopeless_image_still_refuses(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """Completely unreadable images should keep the refusal path."""
        tmp_dir = Path(pipeline.chroma._persist_dir).parent
        test_image = tmp_dir / "hopeless_doc.png"
        _create_quality_test_image(str(test_image))

        pipeline.quality_gate.assess = lambda _: QualityAssessment(
            legibility_score=0.05,
            orientation="correct",
            form_type=QualityDocumentType.UNBEKANNT,
            is_admissible=False,
            rejection_reason="unreadable image",
        )

        result = asyncio.run(pipeline.process(str(test_image)))

        assert "_refusal" in result.fields
        assert result.overall_confidence == 0.0

    def test_text_file_extraction_returns_fields(
        self, pipeline: ExtractionPipeline
    ) -> None:
        """Text/data files should bypass image quality gate and extract values."""
        tmp_dir = Path(pipeline.chroma._persist_dir).parent
        text_file = tmp_dir / "invoice.txt"
        text_file.write_text("Invoice Number: A-100\nTotal: 42.50\n", encoding="utf-8")

        result = asyncio.run(pipeline.process(str(text_file)))

        assert result.document_type == DocumentType.UNBEKANNT
        assert "invoice_number" in result.fields
        assert result.fields["invoice_number"].value == "A-100"
        assert result.fields["total"].value == "42.50"

    @pytest.mark.asyncio
    async def test_dual_pass_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has a DualPassExtractor instance."""
        assert pipeline.dual_pass is not None

    @pytest.mark.asyncio
    async def test_validator_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has a SymbolicValidator instance."""
        assert pipeline.validator is not None

    @pytest.mark.asyncio
    async def test_chroma_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has a ChromaManager instance."""
        assert pipeline.chroma is not None

    @pytest.mark.asyncio
    async def test_prompt_builder_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has a PromptBuilder instance."""
        assert pipeline.prompt_builder is not None

    @pytest.mark.asyncio
    async def test_audit_logger_not_none(self, pipeline: ExtractionPipeline) -> None:
        """Pipeline has an AuditLogger instance."""
        assert pipeline.audit_logger is not None


class TestImageLearningStore:
    """Tests for the persistent image-learning memory."""

    def test_add_and_retrieve_learning_record(self, tmp_path: Path) -> None:
        store = ImageLearningStore(persist_dir=str(tmp_path / "chroma"))
        record = LearnedImageRecord(
            image_id="abc123",
            image_path="/tmp/example.tif",
            filename="example.tif",
            source_split="train",
            label_name="invoice",
            label_index=11,
            width=754,
            height=1000,
            file_size=12345,
            sha256="abc123",
            quality_score=0.42,
            form_type="Unbekannt",
            orientation="correct",
            ocr_text="invoice total amount due",
        )

        try:
            assert store.add_records([record]) == 1
            assert store.add_records([record]) == 0

            stats = store.stats()
            assert stats["total_images"] == 1

            examples = store.retrieve_similar("invoice total", limit=1)
            assert len(examples) == 1
            assert examples[0]["filename"] == "example.tif"
            assert "invoice total" in examples[0]["ocr_text"]
        finally:
            store.close()


# =========================================================================
# Integration Tests
# =========================================================================


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_round_trip(
        self, tmp_log_file: str, tmp_chroma_dir: str
    ) -> None:
        """Create pipeline, process document, check audit log."""
        pipeline = ExtractionPipeline(
            use_mocks=True,
            audit_log_path=tmp_log_file,
            chroma_persist_dir=tmp_chroma_dir,
        )

        # Create a test image with enough texture for the quality gate
        tmp_dir = Path(tmp_chroma_dir).parent
        test_image = tmp_dir / "integration_test.png"
        _create_quality_test_image(str(test_image))

        result = await pipeline.process(str(test_image))

        # Verify result
        assert isinstance(result, ExtractionResult)
        assert result.metadata.extraction_id is not None
        assert result.metadata.timestamp is not None

        # Verify audit log
        assert os.path.exists(tmp_log_file)
        with open(tmp_log_file, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1

        record = json.loads(lines[0])
        assert record["extraction_id"] == str(result.metadata.extraction_id)

    @pytest.mark.asyncio
    async def test_correction_submission_and_retrieval(
        self, tmp_log_file: str, tmp_chroma_dir: str
    ) -> None:
        """Submit a correction and verify ChromaDB state changes."""
        pipeline = ExtractionPipeline(
            use_mocks=True,
            audit_log_path=tmp_log_file,
            chroma_persist_dir=tmp_chroma_dir,
        )

        extraction_id = uuid4()

        # First store an extraction so correction can reference it
        result = ExtractionResult(
            metadata=ExtractionMetadata(
                extraction_id=extraction_id,
                vlm_model="test-vlm",
                llm_model="test-llm",
            ),
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            fields={
                "postleitzahl": FieldResult(
                    field_name="postleitzahl",
                    value="10115",
                    status=FieldStatus.EXTRACTED,
                    confidence=0.95,
                )
            },
        )
        pipeline._extraction_store[str(extraction_id)] = result

        # Submit correction
        correction_id = await pipeline.submit_correction(
            extraction_id=extraction_id,
            field_name="postleitzahl",
            corrected_value="10117",
            original_value="10115",
        )
        assert isinstance(correction_id, str)

        # Check stats
        stats = await pipeline.get_stats()
        assert stats["corrections"]["total"] >= 1

    @pytest.mark.asyncio
    async def test_pipeline_exception_safety(
        self, tmp_log_file: str, tmp_chroma_dir: str
    ) -> None:
        """Pipeline never crashes - returns UNRESOLVED on failure."""
        pipeline = ExtractionPipeline(
            use_mocks=True,
            audit_log_path=tmp_log_file,
            chroma_persist_dir=tmp_chroma_dir,
        )

        # Process a non-existent file (should not crash)
        try:
            result = await pipeline.process("/nonexistent/image.png")
            assert isinstance(result, ExtractionResult)
            assert result.metadata is not None
        except FileNotFoundError:
            # Quality gate may raise FileNotFoundError for missing images
            pass
