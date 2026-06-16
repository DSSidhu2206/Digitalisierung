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
from app.vision.field_mapper import MappedField, SchemaFieldMapper
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
        self.field_mapper = SchemaFieldMapper()
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
        self._store_dir = Path(__file__).resolve().parents[2] / "extraction_store"
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

        # --- Stage 2-5: Surya extraction → schema mapping → validation --
        # Upright the page first if the quality gate confidently detected a
        # 90/180/270° rotation (OCR accuracy collapses on rotated input).
        working_path, oriented_temp = self.quality_gate.deskew_to_temp(
            image_path, assessment
        )
        try:
            result = await self._dual_pass_extraction_result(
                document_type=detected_doc_type,
                image_path=working_path,
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
        finally:
            if oriented_temp and oriented_temp != image_path:
                try:
                    os.remove(oriented_temp)
                except OSError:
                    pass

        # --- Stage 6: Audit logging -------------------------------------
        try:
            self.audit_logger.log(result)
        except Exception as exc:
            logger.error("Audit logging error (non-fatal): %s", exc)

        # --- Stage 7: Store for retrieval (memory + disk) ---------------
        self._extraction_store[str(result.metadata.extraction_id)] = result
        self._persist_result(result)

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

        Checks the in-memory store first, then falls back to the on-disk store
        so results survive a server restart.

        Args:
            extraction_id: UUID string of the extraction.

        Returns:
            The :class:`ExtractionResult` if found, else ``None``.
        """
        cached = self._extraction_store.get(extraction_id)
        if cached is not None:
            return cached
        return self._load_result(extraction_id)

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
        vlm_loaded = False
        llm_loaded = False
        ram = getattr(self, "_ram", None)
        if ram is not None:
            try:
                vlm = getattr(ram, "_vlm", None)
                vlm_loaded = bool(vlm is not None and vlm.is_loaded)
            except Exception:
                pass
            try:
                llm = getattr(ram, "_llm", None)
                is_loaded_attr = getattr(llm, "is_loaded", False)
                llm_loaded = bool(is_loaded_attr() if callable(is_loaded_attr) else is_loaded_attr)
            except Exception:
                pass
        device = "cpu"
        try:
            from app.database.embedding_model import _resolve_device

            device = _resolve_device()
        except Exception:
            pass
        return {
            "vlm_loaded": vlm_loaded,
            "llm_loaded": llm_loaded,
            "ram_usage_mb": round(mem.used / (1024 * 1024), 2),
            "device": device,
        }

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all resources during application shutdown."""
        logger.info("ExtractionPipeline shutting down")
        # Clean up in-memory store
        self._extraction_store.clear()

    def _persist_result(self, result: ExtractionResult) -> None:
        """Persist an extraction to disk so /extractions survives a restart."""
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            path = self._store_dir / f"{result.metadata.extraction_id}.json"
            path.write_text(result.model_dump_json(), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not persist extraction result: %s", exc)

    def _load_result(self, extraction_id: str) -> Optional[ExtractionResult]:
        """Load a persisted extraction from disk, if present."""
        try:
            path = self._store_dir / f"{extraction_id}.json"
            if path.is_file():
                return ExtractionResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
        except Exception as exc:
            logger.debug("Could not load persisted extraction %s: %s", extraction_id, exc)
        return None

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
        self._persist_result(result)
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
        """Run vision extraction → schema mapping → symbolic validation.

        Real path (Surya): a single OCR/layout pass feeds the layout-aware
        :class:`SchemaFieldMapper`, then correction memory and the symbolic
        validator. Mock path: the canned dual-pass value map. The phases
        recorded reflect only what actually executed — no cosmetic labels.
        """
        # Recognition hints for noisy scans (few-shot image learning).
        learned_examples = self.image_learning.retrieve_similar(
            query_text=(
                f"document_type={document_type.value}\n"
                f"quality={float(assessment.legibility_score):.2f}\n"
                f"orientation={assessment.orientation}"
            ),
            document_type=document_type.value,
        )
        learned_context = self.image_learning.build_prompt_section(learned_examples)

        async def run_vlm(vlm: Any) -> Any:
            if not vlm.is_loaded:
                vlm.load()
            if getattr(vlm, "supports_direct_extraction", False):
                return ("surya", vlm.extract_document(image_path, with_layout=False))
            return ("dual", self.dual_pass.extract(
                image_path, vlm, learned_context=learned_context))

        kind, payload = await self.ram.with_vlm(run_vlm)

        if kind == "surya":
            # Re-classify from the full Surya OCR text when the quality-gate
            # preview was inconclusive. This is more reliable than the gate's
            # quick down-sampled tesseract pass and works even without tesseract.
            if document_type == DocumentType.UNBEKANNT:
                reclassified = self._classify_from_text(getattr(payload, "text", "") or "")
                if reclassified is not None:
                    document_type = reclassified
            mapped = self.field_mapper.map(payload, document_type)
        else:
            mapped = self._mapped_from_value_map(payload)

        # Scale confidence by assessed image quality.
        quality_factor = max(0.5, min(float(assessment.legibility_score), 1.0))
        if not assessment.is_admissible:
            quality_factor = min(quality_factor, 0.6)
        for mapped_field in mapped.values():
            mapped_field.confidence = max(
                0.0, min(mapped_field.confidence * quality_factor, 1.0)
            )

        fields, llm_used = self._build_validated_fields(
            mapped, document_type, image_path
        )

        if not fields:
            fields = {
                "_unresolved": FieldResult(
                    field_name="_unresolved",
                    value=None,
                    status=FieldStatus.UNRESOLVED,
                    confidence=max(0.0, min(float(assessment.legibility_score), 0.25)),
                    validation_message="No readable fields found",
                )
            }

        # Record only phases that truly ran.
        phases.append(ExtractionPhase.VISION_PASS_A)
        if kind == "dual":
            phases.extend(
                [ExtractionPhase.VISION_PASS_B, ExtractionPhase.CONSISTENCY_CHECK]
            )
        phases.append(ExtractionPhase.SYMBOLIC_VALIDATION)
        if llm_used:
            phases.append(ExtractionPhase.LLM_STRUCTURING)

        vlm_model_name = (
            getattr(payload, "engine", VLMManager.MODEL_ID)
            if kind == "surya"
            else VLMManager.MODEL_ID
        )
        metadata = ExtractionMetadata(
            vlm_model=vlm_model_name,
            llm_model="llama-cpp" if llm_used else "symbolic-validated",
            document_type=document_type,
            phases_completed=list(phases),
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
        metadata.few_shot_injections = len(learned_examples)

        return ExtractionResult(
            metadata=metadata,
            document_type=document_type,
            fields=fields,
        )

    _DOCTYPE_KEYWORDS: dict[DocumentType, list[str]] = {
        DocumentType.MELDEBESCHEINIGUNG: [
            "meldebescheinigung", "meldebehörde", "meldeamt", "bürgeramt", "einzugsdatum",
        ],
        DocumentType.STEUERBESCHEID: [
            "steuerbescheid", "finanzamt", "steueridentifikationsnummer",
            "einkommensteuer", "veranlagungszeitraum",
        ],
        DocumentType.GEHALTSAUSWEIS: [
            "gehaltsausweis", "lohnabrechnung", "gehaltsabrechnung", "nettolohn",
            "gesamt-brutto", "sozialversicherung",
        ],
        DocumentType.PERSONALAUSWEIS: [
            "personalausweis", "bundesrepublik deutschland", "dokumentnummer",
            "gültig bis", "ausweisnummer",
        ],
    }

    @classmethod
    def _classify_from_text(cls, text: str) -> Optional[DocumentType]:
        """Keyword-classify a document type from OCR text (None if no match)."""
        lower = text.lower()
        best: Optional[DocumentType] = None
        best_count = 0
        for doc_type, words in cls._DOCTYPE_KEYWORDS.items():
            count = sum(1 for word in words if word in lower)
            if count > best_count:
                best, best_count = doc_type, count
        return best if best_count > 0 else None

    def _mapped_from_value_map(self, dual_result: Any) -> dict[str, MappedField]:
        """Adapt a (mock) DualPassResult value/structural map to MappedFields."""
        mapped: dict[str, MappedField] = {}
        value_map = getattr(dual_result, "value_map", {}) or {}
        structural = getattr(dual_result, "structural_map", {}) or {}
        for name, entry in value_map.items():
            if not isinstance(entry, dict):
                continue
            raw_value = entry.get("raw_value")
            confidence = self._coerce_confidence(entry.get("confidence_0_to_1", 0.0))
            bbox = None
            struct = structural.get(name)
            if isinstance(struct, dict):
                bbox = struct.get("estimated_bbox")
            has_value = raw_value not in (None, "", "null", "None")
            mapped[name] = MappedField(
                field_name=name,
                value=raw_value if has_value else None,
                confidence=confidence,
                raw_text=str(raw_value) if raw_value is not None else "",
                bbox=bbox if isinstance(bbox, dict) else None,
                source="inline",
            )
        return mapped

    def _apply_corrections(
        self, canonical: dict[str, MappedField], document_type: DocumentType
    ) -> set[str]:
        """Auto-apply exact-match corrections from memory (closes the loop).

        Only an *exact* (case-insensitive) match between a stored correction's
        original text and the current raw value triggers a substitution — this
        is deterministic correction recall, never fabrication.
        """
        applied: set[str] = set()
        try:
            if self.chroma.get_stats().get("total", 0) <= 0:
                return applied
        except Exception:
            return applied

        for name, mapped_field in canonical.items():
            raw = (
                mapped_field.raw_text
                or ("" if mapped_field.value is None else str(mapped_field.value))
            ).strip()
            if not raw:
                continue
            try:
                hits = self.chroma.retrieve_relevant(name, raw, document_type.value)
            except Exception as exc:
                logger.debug("Correction retrieval failed for %s: %s", name, exc)
                continue
            for hit in hits:
                original = str(hit.get("original", "")).strip()
                corrected = hit.get("corrected")
                if original and corrected is not None and original.lower() == raw.lower():
                    mapped_field.value = corrected
                    applied.add(name)
                    logger.info("Applied correction memory to field '%s'", name)
                    break
        return applied

    def _build_validated_fields(
        self,
        mapped: dict[str, MappedField],
        document_type: DocumentType,
        image_path: str,
    ) -> tuple[dict[str, FieldResult], bool]:
        """Correction recall + symbolic validation → FieldResults.

        Returns ``(fields, llm_used)``.
        """
        canonical = {n: mf for n, mf in mapped.items() if not n.startswith("_line_")}

        # Optional LLM structuring refinement (no-op unless weights present).
        llm_used = self._maybe_llm_structuring(canonical, document_type, image_path)

        # Correction memory recall — the learning loop, now closed.
        corrected = self._apply_corrections(canonical, document_type)

        # Deterministic symbolic validation (regex + checksum + business rules).
        valued = {n: mf.value for n, mf in canonical.items()}
        validated = self.validator.validate_document(document_type, valued)

        fields: dict[str, FieldResult] = {}
        for name, mapped_field in canonical.items():
            result = validated.get(name)
            status = result.status if result is not None else FieldStatus.EXTRACTED
            message = result.validation_message if result is not None else None
            if name in corrected:
                status = FieldStatus.CORRECTED
                message = "Auto-applied from correction memory"
            elif (
                status == FieldStatus.EXTRACTED
                and mapped_field.confidence < self.LOW_CONFIDENCE_THRESHOLD
            ):
                status = FieldStatus.LOW_CONFIDENCE
                message = "Below confidence threshold — review recommended"
            fields[name] = FieldResult(
                field_name=name,
                value=mapped_field.value,
                status=status,
                confidence=mapped_field.confidence,
                source_bbox=self._bbox_to_model(mapped_field.bbox),
                raw_text=mapped_field.raw_text or None,
                validation_message=message,
            )

        # Preserve unmapped OCR lines as low-confidence review items.
        for name, mapped_field in mapped.items():
            if not name.startswith("_line_"):
                continue
            fields[name] = FieldResult(
                field_name=name,
                value=mapped_field.value,
                status=FieldStatus.LOW_CONFIDENCE,
                confidence=mapped_field.confidence,
                source_bbox=self._bbox_to_model(mapped_field.bbox),
                raw_text=mapped_field.raw_text or None,
                validation_message="Unmapped OCR line — not a recognised field",
            )
        return fields, llm_used

    def _maybe_llm_structuring(
        self,
        canonical: dict[str, MappedField],
        document_type: DocumentType,
        image_path: str,
    ) -> bool:
        """Optional local-LLM structuring hook. Returns whether it ran.

        Disabled (returns ``False``) unless a GGUF model and llama.cpp are
        available, so metadata never claims an LLM ran when it did not. The
        real implementation lives in :meth:`_run_llm_structuring`.
        """
        try:
            return self._run_llm_structuring(canonical, document_type, image_path)
        except Exception as exc:  # never let optional refinement break extraction
            logger.debug("LLM structuring skipped: %s", exc)
            return False

    def _run_llm_structuring(
        self,
        canonical: dict[str, MappedField],
        document_type: DocumentType,
        image_path: str,
    ) -> bool:
        """Fill OCR-missed fields with a local GGUF model (Metal). Guarded.

        Returns ``True`` only if the model actually ran and contributed at least
        one value. Disabled unless ``ENABLE_LLM_STRUCTURING`` is set, the GGUF
        model exists on disk, and ``llama-cpp-python`` is importable. LLM-filled
        values are given sub-threshold confidence so they remain flagged for
        review and are still subject to symbolic validation downstream.
        """
        from config import get_settings

        settings = get_settings()
        if not getattr(settings, "ENABLE_LLM_STRUCTURING", False):
            return False

        model_path = Path(settings.LLM_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[3] / model_path
        if not model_path.exists():
            logger.info("LLM structuring enabled but model not found: %s", model_path)
            return False
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            logger.info("LLM structuring enabled but llama-cpp-python is unavailable")
            return False

        missing = [
            name
            for name, mapped_field in canonical.items()
            if mapped_field.value is None
            or mapped_field.confidence < self.LOW_CONFIDENCE_THRESHOLD
        ]
        if not missing:
            return False

        from app.llm.llm_structurer import LLMStructurer

        structurer = LLMStructurer(
            str(model_path),
            n_ctx=settings.LLM_N_CTX,
            n_threads=settings.LLM_N_THREADS,
            n_gpu_layers=settings.LLM_N_GPU_LAYERS,
            temperature=settings.LLM_TEMPERATURE,
        )
        try:
            filled = structurer.fill_fields(document_type, canonical, missing)
        finally:
            structurer.close()

        applied = False
        for name, (value, confidence) in (filled or {}).items():
            mapped_field = canonical.get(name)
            if mapped_field is None or value in (None, ""):
                continue
            if confidence > mapped_field.confidence:
                mapped_field.value = value
                mapped_field.confidence = confidence
                mapped_field.source = "llm"
                applied = True
        return applied

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """Return *value* as a bounded confidence float."""
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _bbox_to_model(bbox: Any) -> Optional[BoundingBox]:
        """Build a validated BoundingBox from a normalised bbox dict."""
        if not isinstance(bbox, dict):
            return None
        try:
            coords = {k: float(bbox[k]) for k in ("x1", "y1", "x2", "y2") if k in bbox}
            if len(coords) != 4:
                return None
            model = BoundingBox(**coords)
            return model if model.is_valid() else None
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
