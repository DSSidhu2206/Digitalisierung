"""
Pipeline Orchestrator — main multi-stage extraction pipeline.

Coordinates the full document extraction workflow:
1. Quality Gate — refuse inadmissible images
2. Dual-Pass VLM extraction (load → process → unload)
3. Retrieve few-shot corrections from ChromaDB
4. LLM structuring (load → process → unload)
5. Symbolic validation
6. Audit logging
7. Return ExtractionResult

The orchestrator is designed as a singleton that is created once at
application startup and reused for every request.

Spec: Section 8.2 — Pipeline Orchestrator
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from app.database.chroma_manager import ChromaManager
from app.database.image_learning_store import ImageLearningStore
from app.ingestion.file_text_extractor import FileTextExtractor
from app.models import (
    BoundingBox,
    DocumentType,
    ExtractionMetadata,
    ExtractionPhase,
    ExtractionResult,
    FieldResult,
    FieldStatus,
)
from app.pipeline.audit_logger import AuditLogger
from app.pipeline.local_training import LocalTrainingManager
from app.vision.quality_gate import QualityGate
from app.vision.vlm_loader import VLMManager

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Main multi-stage pipeline orchestrator.

    Manages the full document extraction lifecycle from image upload
    to structured result.  Integrates the vision engine, LLM engine,
    validation layer, ChromaDB, and audit logging.

    The pipeline is designed to be created once (singleton) and
    reused across all requests.  Model loading/unloading is handled
    by the RAM manager for sequential processing.
    """

    DEGRADED_LEGIBILITY_THRESHOLD = 0.20
    LOW_CONFIDENCE_THRESHOLD = 0.70

    def __init__(
        self,
        use_mocks: bool = False,
        audit_log_path: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
    ) -> None:
        """Initialise all pipeline components.

        Args:
            use_mocks: If True, use mock model managers (no real inference).
            audit_log_path: Override path for the audit JSONL log.
            chroma_persist_dir: Override directory for ChromaDB persistence.
        """
        project_root = Path(__file__).resolve().parents[3]
        image_learning_dir = chroma_persist_dir or str(project_root / "chroma_data")
        self.quality_gate = QualityGate()
        self.chroma = ChromaManager(persist_dir=chroma_persist_dir or "./chroma_data")
        self.image_learning = ImageLearningStore(persist_dir=image_learning_dir)
        self.local_training = LocalTrainingManager(project_root, self.image_learning)
        self.file_text_extractor = FileTextExtractor()
        self.audit_logger = AuditLogger(
            log_path=audit_log_path or "./logs/audit.jsonl"
        )
        # In-memory store for extraction results (keyed by extraction_id)
        self._extraction_store: Dict[str, ExtractionResult] = {}
        self._use_mocks = use_mocks
        logger.info("ExtractionPipeline initialised (mock=%s)", use_mocks)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def process(self, image_path: str) -> ExtractionResult:
        """Run the full extraction pipeline on *image_path*.

        Pipeline stages:
            1. Quality Gate → refuse if inadmissible
            2. (Async) Dual-Pass VLM extraction
            3. Retrieve few-shot corrections from ChromaDB
            4. LLM structuring with schema forcing
            5. Symbolic validation
            6. Audit logging
            7. Store result for later retrieval

        Args:
            image_path: Absolute path to the uploaded image.

        Returns:
            :class:`ExtractionResult` with full metadata and fields.
            Even on pipeline errors, a partial result with
            ``UNRESOLVED`` fields is returned (never crashes).
        """
        start_time: float = time.perf_counter()
        phases: list[ExtractionPhase] = []

        if not self.file_text_extractor.is_image(image_path):
            return self._text_file_extraction_result(
                image_path,
                start_time,
                phases,
            )

        # --- Stage 1: Quality Gate --------------------------------------
        try:
            assessment = self.quality_gate.assess(image_path)
            phases.append(ExtractionPhase.QUALITY_GATE)

            if not assessment.is_admissible:
                if assessment.legibility_score < self.DEGRADED_LEGIBILITY_THRESHOLD:
                    logger.warning(
                        "Quality gate refused: %s", assessment.rejection_reason
                    )
                    return self._refusal_result(
                        reason=assessment.rejection_reason or "Image refused by quality gate",
                        vlm_model="stub",
                        llm_model="stub",
                        start_time=start_time,
                        phases=phases,
                    )
                logger.warning(
                    "Quality gate continuing in degraded mode: %s",
                    assessment.rejection_reason,
                )

            detected_doc_type = DocumentType(assessment.form_type.value)
        except Exception as exc:
            logger.error("Quality gate error: %s", exc)
            return self._refusal_result(
                reason=f"Quality gate error: {exc}",
                vlm_model="stub",
                llm_model="stub",
                start_time=start_time,
                phases=phases,
            )

        # --- Stage 2-5: Run Surya extraction and validation -------------
        try:
            phases.extend([
                ExtractionPhase.VISION_PASS_A,
                ExtractionPhase.VISION_PASS_B,
                ExtractionPhase.CONSISTENCY_CHECK,
                ExtractionPhase.SYMBOLIC_VALIDATION,
                ExtractionPhase.LLM_STRUCTURING,
            ])

            result = await self._dual_pass_extraction_result(
                document_type=detected_doc_type,
                image_path=image_path,
                assessment=assessment,
                start_time=start_time,
                phases=phases,
            )

        except Exception as exc:
            logger.exception("Pipeline processing error: %s", exc)
            # Return partial result with UNRESOLVED fields
            result = self._partial_error_result(
                document_type=detected_doc_type,
                error=str(exc),
                start_time=start_time,
                phases=phases,
            )

        # --- Stage 6: Audit logging -------------------------------------
        try:
            self.audit_logger.log(result)
        except Exception as exc:
            logger.error("Audit logging error (non-fatal): %s", exc)

        # --- Stage 7: Store for retrieval -------------------------------
        self._extraction_store[str(result.metadata.extraction_id)] = result

        phases.append(ExtractionPhase.COMPLETE)
        result.metadata.phases_completed = phases
        result.metadata.processing_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return result

    # ------------------------------------------------------------------
    # Correction (learning loop)
    # ------------------------------------------------------------------

    async def submit_correction(
        self,
        extraction_id: UUID,
        field_name: str,
        corrected_value: Any,
        original_value: Any,
    ) -> str:
        """Submit a user correction and store in ChromaDB (learning loop).

        The operation is atomic: the correction is embedded and stored
        as a single ChromaDB transaction.

        Args:
            extraction_id: UUID of the original extraction.
            field_name: Schema field name being corrected.
            corrected_value: New corrected value.
            original_value: Original (wrong) value before correction.

        Returns:
            The UUID string of the stored correction record.

        Raises:
            ValueError: If the extraction_id is not found.
            RuntimeError: If ChromaDB storage fails.
        """
        extraction_key: str = str(extraction_id)
        if extraction_key not in self._extraction_store:
            raise ValueError(f"Extraction not found: {extraction_id}")

        extraction: ExtractionResult = self._extraction_store[extraction_key]
        doc_type_value: str = extraction.document_type.value

        # Store in ChromaDB atomically
        correction_id: str = self.chroma.add_correction(
            original_text=str(original_value) if original_value is not None else "",
            corrected_value=str(corrected_value) if corrected_value is not None else "",
            field_name=field_name,
            document_type=doc_type_value,
            reason=f"User correction on extraction {extraction_id}",
        )

        # Update the in-memory extraction result
        if field_name in extraction.fields:
            field = extraction.fields[field_name]
            field.value = corrected_value
            field.status = FieldStatus.CORRECTED
            extraction.corrected_count += 1

        logger.info(
            "Correction stored: extraction=%s field=%s correction_id=%s",
            extraction_id,
            field_name,
            correction_id,
        )
        return correction_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_extraction(self, extraction_id: str) -> Optional[ExtractionResult]:
        """Retrieve a previously stored extraction by ID.

        Args:
            extraction_id: UUID string of the extraction.

        Returns:
            The :class:`ExtractionResult` if found, else ``None``.
        """
        return self._extraction_store.get(extraction_id)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Return pipeline statistics.

        Combines audit log statistics with correction DB stats.

        Returns:
            Dictionary with keys: ``total_extractions``,
            ``avg_confidence``, ``unresolved_rate``,
            ``avg_processing_time_ms``, ``document_type_breakdown``,
            ``model_versions``.
        """
        audit_stats = self.audit_logger.get_stats()
        chroma_stats = self.chroma.get_stats()
        return {
            "audit": audit_stats,
            "corrections": chroma_stats,
            "image_learning": self.image_learning.stats(),
            "mock_mode": self._use_mocks,
        }

    async def get_extraction(self, extraction_id: "UUID") -> Optional[ExtractionResult]:
        """Async wrapper for retrieving an extraction by ID."""
        return self._extraction_store.get(str(extraction_id))

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def ram(self) -> Any:
        """Return the RAMManager instance (lazy-created)."""
        if not hasattr(self, "_ram"):
            from app.pipeline.ram_manager import RAMManager

            self._ram = RAMManager()
            if self._use_mocks:
                from app.vision.vlm_loader import MockVLMManager
                from app.llm.llm_loader import MockLLMManager

                self._ram._vlm = MockVLMManager()
                self._ram._llm = MockLLMManager()
        return self._ram

    @property
    def dual_pass(self) -> Any:
        """Return the DualPassExtractor instance (lazy-created)."""
        if not hasattr(self, "_dual_pass"):
            from app.vision.dual_pass import DualPassExtractor

            self._dual_pass = DualPassExtractor()
        return self._dual_pass

    @property
    def validator(self) -> Any:
        """Return the SymbolicValidator instance (lazy-created)."""
        if not hasattr(self, "_validator"):
            from app.validators.symbolic_rules import SymbolicValidator

            self._validator = SymbolicValidator()
        return self._validator

    @property
    def prompt_builder(self) -> Any:
        """Return the PromptBuilder instance (lazy-created)."""
        if not hasattr(self, "_prompt_builder"):
            from app.llm.prompt_builder import PromptBuilder

            self._prompt_builder = PromptBuilder()
        return self._prompt_builder

    @property
    def use_mocks(self) -> bool:
        """Whether the pipeline is running in mock mode."""
        return self._use_mocks

    def health_check(self) -> dict[str, Any]:
        """Return health status of pipeline components.

        Returns:
            Dictionary with ``vlm_loaded``, ``llm_loaded``,
            ``ram_usage_mb``, and ``version``.
        """
        import psutil

        mem = psutil.virtual_memory()
        return {
            "vlm_loaded": False,  # TODO: wire to VLMManager
            "llm_loaded": False,  # TODO: wire to LLMManager
            "ram_usage_mb": round(mem.used / (1024 * 1024), 2),
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all resources during application shutdown."""
        logger.info("ExtractionPipeline shutting down")
        # Clean up in-memory store
        self._extraction_store.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refusal_result(
        self,
        reason: str,
        vlm_model: str,
        llm_model: str,
        start_time: float,
        phases: list[ExtractionPhase],
    ) -> ExtractionResult:
        """Build an ExtractionResult for a refused image."""
        metadata = ExtractionMetadata(
            vlm_model=vlm_model,
            llm_model=llm_model,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            phases_completed=phases,
        )
        return ExtractionResult(
            metadata=metadata,
            document_type=DocumentType.UNBEKANNT,
            fields={
                "_refusal": FieldResult(
                    field_name="_refusal",
                    value=reason,
                    status=FieldStatus.UNRESOLVED,
                    confidence=0.0,
                    validation_message=reason,
                )
            },
        )

    def _mock_extraction_result(
        self,
        document_type: DocumentType,
        image_path: str,
        start_time: float,
        phases: list[ExtractionPhase],
    ) -> ExtractionResult:
        """Build a realistic mock extraction result for development.

        TODO: Replace with real VLM + LLM integration.
        """
        metadata = ExtractionMetadata(
            vlm_model=VLMManager.MODEL_ID,
            llm_model="llama-3.1-8b-instruct",
            document_type=document_type,
            phases_completed=phases,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

        # Return mock fields appropriate to the document type
        if document_type == DocumentType.MELDEBESCHEINIGUNG:
            fields = self._mock_meldebescheinigung_fields()
        elif document_type == DocumentType.STEUERBESCHEID:
            fields = self._mock_steuerbescheid_fields()
        elif document_type == DocumentType.GEHALTSAUSWEIS:
            fields = self._mock_gehaltsausweis_fields()
        elif document_type == DocumentType.PERSONALAUSWEIS:
            fields = self._mock_personalausweis_fields()
        else:
            fields = {
                "unbekannt": FieldResult(
                    field_name="unbekannt",
                    value=None,
                    status=FieldStatus.UNRESOLVED,
                    confidence=0.0,
                    validation_message="Document type not recognised",
                )
            }

        return ExtractionResult(
            metadata=metadata,
            document_type=document_type,
            fields=fields,
        )

    def _text_file_extraction_result(
        self,
        file_path: str,
        start_time: float,
        phases: list[ExtractionPhase],
    ) -> ExtractionResult:
        """Extract structured key/value candidates from a non-image file."""
        extracted = self.file_text_extractor.extract(file_path)
        phases.extend([
            ExtractionPhase.QUALITY_GATE,
            ExtractionPhase.LLM_STRUCTURING,
            ExtractionPhase.SYMBOLIC_VALIDATION,
        ])
        fields = self._fields_from_text(extracted.text)
        if not fields:
            fields = {
                "_text": FieldResult(
                    field_name="_text",
                    value=extracted.text[:5000] or None,
                    status=FieldStatus.LOW_CONFIDENCE if extracted.text else FieldStatus.UNRESOLVED,
                    confidence=0.35 if extracted.text else 0.0,
                    raw_text=extracted.text[:5000] or None,
                    validation_message=(
                        f"Parsed with {extracted.parser}; no stable key/value fields found"
                    ),
                )
            }

        metadata = ExtractionMetadata(
            vlm_model="file-text-extractor",
            llm_model="rule-based-key-value",
            document_type=DocumentType.UNBEKANNT,
            phases_completed=phases,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        result = ExtractionResult(
            metadata=metadata,
            document_type=DocumentType.UNBEKANNT,
            fields=fields,
        )
        try:
            self.audit_logger.log(result)
        except Exception as exc:
            logger.error("Audit logging error (non-fatal): %s", exc)
        self._extraction_store[str(result.metadata.extraction_id)] = result
        return result

    def _fields_from_text(self, text: str) -> dict[str, FieldResult]:
        """Extract conservative key/value candidates from text."""
        fields: dict[str, FieldResult] = {}
        for line in text.splitlines()[:5000]:
            cleaned = " ".join(line.strip().split())
            if not cleaned or len(cleaned) > 500:
                continue
            key = ""
            value = ""
            if ":" in cleaned:
                key, value = cleaned.split(":", 1)
            elif "|" in cleaned:
                parts = [part.strip() for part in cleaned.split("|") if part.strip()]
                if len(parts) == 2:
                    key, value = parts
            if not key or not value:
                continue
            field_name = self._normalise_text_field_name(key)
            if not field_name or field_name in fields:
                continue
            confidence = 0.65 if len(value.strip()) >= 2 else 0.3
            fields[field_name] = FieldResult(
                field_name=field_name,
                value=value.strip()[:1000],
                status=(
                    FieldStatus.LOW_CONFIDENCE
                    if confidence < self.LOW_CONFIDENCE_THRESHOLD
                    else FieldStatus.EXTRACTED
                ),
                confidence=confidence,
                raw_text=cleaned,
                validation_message="Rule-based file text extraction",
            )
            if len(fields) >= 200:
                break
        return fields

    @staticmethod
    def _normalise_text_field_name(key: str) -> str:
        field = "".join(ch.lower() if ch.isalnum() else "_" for ch in key.strip())
        field = "_".join(part for part in field.split("_") if part)
        if not field or field[0].isdigit():
            field = f"field_{field}" if field else ""
        return field[:80]

    async def _dual_pass_extraction_result(
        self,
        document_type: DocumentType,
        image_path: str,
        assessment: Any,
        start_time: float,
        phases: list[ExtractionPhase],
    ) -> ExtractionResult:
        """Build an extraction result from the dual-pass vision output.

        This path is intentionally used for degraded or unclassified images.
        It preserves partially readable values as low-confidence review items
        instead of collapsing the whole request to a single 0-confidence
        refusal field.
        """
        metadata = ExtractionMetadata(
            vlm_model=VLMManager.MODEL_ID,
            llm_model="dual-pass-only",
            document_type=document_type,
            phases_completed=phases,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        learned_examples = self.image_learning.retrieve_similar(
            query_text=(
                f"document_type={document_type.value}\n"
                f"quality={float(assessment.legibility_score):.2f}\n"
                f"orientation={assessment.orientation}"
            ),
            document_type=document_type.value,
        )
        metadata.few_shot_injections = len(learned_examples)
        learned_context = self.image_learning.build_prompt_section(learned_examples)

        async def run_dual_pass(vlm: Any) -> Any:
            return self.dual_pass.extract(
                image_path,
                vlm,
                learned_context=learned_context,
            )

        dual_result = await self.ram.with_vlm(run_dual_pass)
        fields = self._fields_from_dual_pass(dual_result, assessment)
        if not fields:
            reason = "No readable fields found after degraded dual-pass extraction"
            fields = {
                "_unresolved": FieldResult(
                    field_name="_unresolved",
                    value=None,
                    status=FieldStatus.UNRESOLVED,
                    confidence=max(0.0, min(float(assessment.legibility_score), 0.25)),
                    validation_message=reason,
                )
            }

        return ExtractionResult(
            metadata=metadata,
            document_type=document_type,
            fields=fields,
        )

    def _fields_from_dual_pass(
        self,
        dual_result: Any,
        assessment: Any,
    ) -> dict[str, FieldResult]:
        """Convert dual-pass values into reviewable field results."""
        fields: dict[str, FieldResult] = {}
        quality_factor = max(0.25, min(float(assessment.legibility_score), 1.0))
        if not assessment.is_admissible:
            quality_factor = min(quality_factor, 0.49)

        warning = assessment.rejection_reason
        for field_name, value_entry in dual_result.value_map.items():
            if not isinstance(value_entry, dict):
                continue

            raw_confidence = self._coerce_confidence(
                value_entry.get("confidence_0_to_1", 0.0)
            )
            confidence = max(0.0, min(raw_confidence * quality_factor, 1.0))
            raw_value = value_entry.get("raw_value")
            has_value = raw_value not in (None, "", "null", "None")

            if not has_value:
                status = FieldStatus.UNRESOLVED
            elif confidence < self.LOW_CONFIDENCE_THRESHOLD:
                status = FieldStatus.LOW_CONFIDENCE
            else:
                status = FieldStatus.EXTRACTED

            fields[field_name] = FieldResult(
                field_name=field_name,
                value=raw_value if has_value else None,
                status=status,
                confidence=confidence,
                source_bbox=self._bbox_from_structure(
                    dual_result.structural_map.get(field_name, {})
                ),
                raw_text=str(raw_value) if raw_value is not None else None,
                validation_message=warning if status != FieldStatus.EXTRACTED else None,
            )

        for field_name in set(dual_result.structural_map) - set(fields):
            fields[field_name] = FieldResult(
                field_name=field_name,
                value=None,
                status=FieldStatus.UNRESOLVED,
                confidence=0.0,
                source_bbox=self._bbox_from_structure(
                    dual_result.structural_map.get(field_name, {})
                ),
                validation_message="Field detected but no value was readable",
            )

        return fields

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Return *value* as a bounded confidence float."""
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bbox_from_structure(structure_entry: Any) -> Optional[BoundingBox]:
        """Build a BoundingBox from dual-pass structural metadata."""
        if not isinstance(structure_entry, dict):
            return None
        bbox_data = structure_entry.get("estimated_bbox")
        if not isinstance(bbox_data, dict):
            return None
        try:
            bbox = BoundingBox(**bbox_data)
            return bbox if bbox.is_valid() else None
        except Exception:
            return None

    def _partial_error_result(
        self,
        document_type: DocumentType,
        error: str,
        start_time: float,
        phases: list[ExtractionPhase],
    ) -> ExtractionResult:
        """Build a partial result when the pipeline errors."""
        metadata = ExtractionMetadata(
            vlm_model="stub",
            llm_model="stub",
            document_type=document_type,
            phases_completed=phases,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        return ExtractionResult(
            metadata=metadata,
            document_type=document_type,
            fields={
                "_error": FieldResult(
                    field_name="_error",
                    value=None,
                    status=FieldStatus.UNRESOLVED,
                    confidence=0.0,
                    validation_message=f"Pipeline error: {error}",
                )
            },
        )

    # -- Mock field generators -----------------------------------------

    @staticmethod
    def _mock_meldebescheinigung_fields() -> dict[str, FieldResult]:
        """Return realistic mock fields for a Meldebescheinigung."""
        return {
            "familienname": FieldResult(
                field_name="familienname",
                value="M\u00fcller",
                status=FieldStatus.EXTRACTED,
                confidence=0.95,
                source_bbox=BoundingBox(x1=0.12, y1=0.25, x2=0.45, y2=0.30),
                raw_text="M\u00fcller",
            ),
            "vorname": FieldResult(
                field_name="vorname",
                value="Hans",
                status=FieldStatus.EXTRACTED,
                confidence=0.93,
                source_bbox=BoundingBox(x1=0.12, y1=0.32, x2=0.45, y2=0.37),
                raw_text="Hans",
            ),
            "geburtsdatum": FieldResult(
                field_name="geburtsdatum",
                value="1985-03-15",
                status=FieldStatus.EXTRACTED,
                confidence=0.91,
                source_bbox=BoundingBox(x1=0.12, y1=0.40, x2=0.35, y2=0.45),
                raw_text="15.03.1985",
            ),
            "geburtsort": FieldResult(
                field_name="geburtsort",
                value="Berlin",
                status=FieldStatus.EXTRACTED,
                confidence=0.88,
                source_bbox=BoundingBox(x1=0.40, y1=0.40, x2=0.70, y2=0.45),
                raw_text="Berlin",
            ),
            "staatsangehoerigkeit": FieldResult(
                field_name="staatsangehoerigkeit",
                value="Deutsch",
                status=FieldStatus.EXTRACTED,
                confidence=0.92,
                source_bbox=BoundingBox(x1=0.12, y1=0.48, x2=0.50, y2=0.53),
                raw_text="Deutsch",
            ),
            "strasse": FieldResult(
                field_name="strasse",
                value="Hauptstra\u00dfe",
                status=FieldStatus.EXTRACTED,
                confidence=0.85,
                source_bbox=BoundingBox(x1=0.12, y1=0.56, x2=0.50, y2=0.61),
                raw_text="Hauptstra\u00dfe",
            ),
            "hausnummer": FieldResult(
                field_name="hausnummer",
                value="42",
                status=FieldStatus.EXTRACTED,
                confidence=0.97,
                source_bbox=BoundingBox(x1=0.52, y1=0.56, x2=0.62, y2=0.61),
                raw_text="42",
            ),
            "postleitzahl": FieldResult(
                field_name="postleitzahl",
                value="10115",
                status=FieldStatus.EXTRACTED,
                confidence=0.96,
                source_bbox=BoundingBox(x1=0.12, y1=0.64, x2=0.25, y2=0.69),
                raw_text="10115",
            ),
            "wohnort": FieldResult(
                field_name="wohnort",
                value="Berlin",
                status=FieldStatus.EXTRACTED,
                confidence=0.94,
                source_bbox=BoundingBox(x1=0.30, y1=0.64, x2=0.60, y2=0.69),
                raw_text="Berlin",
            ),
        }

    @staticmethod
    def _mock_steuerbescheid_fields() -> dict[str, FieldResult]:
        """Return realistic mock fields for a Steuerbescheid."""
        return {
            "steueridentifikationsnummer": FieldResult(
                field_name="steueridentifikationsnummer",
                value="12345678901",
                status=FieldStatus.EXTRACTED,
                confidence=0.92,
            ),
            "veranlagungszeitraum": FieldResult(
                field_name="veranlagungszeitraum",
                value="2023",
                status=FieldStatus.EXTRACTED,
                confidence=0.95,
            ),
            "zu_versteuerndes_einkommen": FieldResult(
                field_name="zu_versteuerndes_einkommen",
                value=45000.00,
                status=FieldStatus.EXTRACTED,
                confidence=0.88,
            ),
            "festgesetzte_steuer": FieldResult(
                field_name="festgesetzte_steuer",
                value=8500.00,
                status=FieldStatus.EXTRACTED,
                confidence=0.90,
            ),
        }

    @staticmethod
    def _mock_gehaltsausweis_fields() -> dict[str, FieldResult]:
        """Return realistic mock fields for a Gehaltsausweis."""
        return {
            "arbeitgeber": FieldResult(
                field_name="arbeitgeber",
                value="Beispiel GmbH",
                status=FieldStatus.EXTRACTED,
                confidence=0.93,
            ),
            "brutto_lohn": FieldResult(
                field_name="brutto_lohn",
                value=5200.00,
                status=FieldStatus.EXTRACTED,
                confidence=0.91,
            ),
            "netto_lohn": FieldResult(
                field_name="netto_lohn",
                value=3200.00,
                status=FieldStatus.EXTRACTED,
                confidence=0.91,
            ),
            "abrechnungszeitraum": FieldResult(
                field_name="abrechnungszeitraum",
                value="01.2024",
                status=FieldStatus.EXTRACTED,
                confidence=0.94,
            ),
        }

    @staticmethod
    def _mock_personalausweis_fields() -> dict[str, FieldResult]:
        """Return realistic mock fields for a Personalausweis."""
        return {
            "dokumentnummer": FieldResult(
                field_name="dokumentnummer",
                value="T22000129",
                status=FieldStatus.EXTRACTED,
                confidence=0.96,
            ),
            "familienname": FieldResult(
                field_name="familienname",
                value="M\u00fcller",
                status=FieldStatus.EXTRACTED,
                confidence=0.94,
            ),
            "vorname": FieldResult(
                field_name="vorname",
                value="Hans",
                status=FieldStatus.EXTRACTED,
                confidence=0.93,
            ),
            "geburtsdatum": FieldResult(
                field_name="geburtsdatum",
                value="1985-03-15",
                status=FieldStatus.EXTRACTED,
                confidence=0.92,
            ),
            "geburtsort": FieldResult(
                field_name="geburtsort",
                value="Berlin",
                status=FieldStatus.EXTRACTED,
                confidence=0.89,
            ),
            "staatsangehoerigkeit": FieldResult(
                field_name="staatsangehoerigkeit",
                value="Deutsch",
                status=FieldStatus.EXTRACTED,
                confidence=0.94,
            ),
            "gueltig_bis": FieldResult(
                field_name="gueltig_bis",
                value="2030-06-30",
                status=FieldStatus.EXTRACTED,
                confidence=0.90,
            ),
        }
