"""
Core extraction data models for the Digitalisierung ABE pipeline.

These models capture the full lifecycle of a document extraction:
provenance (bounding boxes), per-field results with confidence scoring,
metadata audit trails, and the aggregate extraction result.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import DocumentType, ExtractionPhase, FieldStatus


class BoundingBox(BaseModel):
    """Normalised axis-aligned bounding-box coordinates.

    All coordinates are in the range ``[0.0, 1.0]`` relative to the
    original page dimensions so that the box is resolution-independent.
    """

    model_config = ConfigDict(frozen=True)

    x1: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalised)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalised)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right edge (normalised)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (normalised)")
    page: int = Field(default=0, ge=0, description="0-based page index")

    def area(self) -> float:
        """Return the normalised area of the bounding box."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def is_valid(self) -> bool:
        """Return ``True`` if the box has positive width and height."""
        return self.x1 < self.x2 and self.y1 < self.y2


class FieldResult(BaseModel):
    """Result for a single extracted field.

    Every field in every document is wrapped in this container so that
    the frontend can uniformly display provenance, confidence, and
    correction status regardless of the underlying data type.
    """

    model_config = ConfigDict(validate_assignment=True)

    field_name: str = Field(..., description="Schema key for the field")
    value: Union[str, int, float, date, None] = Field(
        None, description="Extracted value (None when UNRESOLVED)"
    )
    status: FieldStatus = Field(..., description="Lifecycle status of this field")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    source_bbox: Optional[BoundingBox] = Field(
        default=None, description="Provenance: where the value was found in the image"
    )
    raw_text: Optional[str] = Field(
        default=None, description="Raw text exactly as the VLM saw it"
    )
    validation_message: Optional[str] = Field(
        default=None,
        description="Human-readable explanation when validation failed",
    )

    def is_resolved(self) -> bool:
        """Return ``True`` if this field has a usable value."""
        return self.status == FieldStatus.EXTRACTED or self.status == FieldStatus.CORRECTED

    def needs_review(self) -> bool:
        """Return ``True`` if the field requires human review."""
        return self.status in (
            FieldStatus.UNRESOLVED,
            FieldStatus.VALIDATION_FAILURE,
            FieldStatus.LOW_CONFIDENCE,
        )


class ExtractionMetadata(BaseModel):
    """Immutable audit trail recorded for every extraction request.

    This block is attached to every :class:`ExtractionResult` so that
    downstream consumers can trace model versions, performance
    characteristics, and few-shot learning influences.
    """

    model_config = ConfigDict(validate_assignment=True)

    extraction_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this extraction run",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when extraction started",
    )
    vlm_model: str = Field(
        ..., description="Vision model identifier, e.g. llama-3.2-11b-vision"
    )
    llm_model: str = Field(
        ..., description="LLM model identifier, e.g. llama-3.1-8b-instruct"
    )
    vlm_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature for VLM"
    )
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Sampling temperature for LLM"
    )
    document_type: Optional[DocumentType] = Field(
        default=None, description="Detected document type"
    )
    phases_completed: list[ExtractionPhase] = Field(
        default_factory=list, description="Pipeline phases that finished successfully"
    )
    ram_peak_mb: Optional[float] = Field(
        default=None, description="Peak RAM usage during extraction (MB)"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, description="Wall-clock processing time (milliseconds)"
    )
    few_shot_injections: int = Field(
        default=0, ge=0, description="Number of few-shot examples injected into the prompt"
    )


class ExtractionResult(BaseModel):
    """Complete output of a single document extraction pipeline run.

    Aggregates metadata, typed schema fields, and automatically
    computes summary statistics (unresolved count, confidence average)
    via a Pydantic ``model_validator``.
    """

    # Note: validate_assignment is disabled because the model_validator
    # modifies fields, which would trigger infinite recursion.
    model_config = ConfigDict(validate_assignment=False)

    metadata: ExtractionMetadata = Field(..., description="Audit trail for this run")
    document_type: DocumentType = Field(..., description="Classified document type")
    fields: dict[str, FieldResult] = Field(
        default_factory=dict, description="Map of field name → FieldResult"
    )
    unresolved_count: int = Field(
        default=0, ge=0, description="Number of fields with status UNRESOLVED"
    )
    validation_failure_count: int = Field(
        default=0, ge=0, description="Number of fields with status VALIDATION_FAILURE"
    )
    corrected_count: int = Field(
        default=0, ge=0, description="Number of fields with status CORRECTED"
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Arithmetic mean of all field confidence scores",
    )

    @model_validator(mode="after")
    def compute_counts(self) -> "ExtractionResult":
        """Derive aggregate statistics from the individual field results.

        Counts the occurrences of each non-extracted status and computes
        the arithmetic mean of all field confidence values.
        """
        self.unresolved_count = sum(
            1 for f in self.fields.values() if f.status == FieldStatus.UNRESOLVED
        )
        self.validation_failure_count = sum(
            1 for f in self.fields.values() if f.status == FieldStatus.VALIDATION_FAILURE
        )
        self.corrected_count = sum(
            1 for f in self.fields.values() if f.status == FieldStatus.CORRECTED
        )
        confidences = [
            f.confidence for f in self.fields.values() if f.confidence is not None
        ]
        self.overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return self

    def get_fields_needing_review(self) -> dict[str, FieldResult]:
        """Return all fields that require human review."""
        return {
            name: field
            for name, field in self.fields.items()
            if field.needs_review()
        }
