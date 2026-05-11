"""
Request and response Pydantic models for the Digitalisierung ABE API.

These models define the JSON schema for every endpoint and are
automatically rendered into OpenAPI documentation by FastAPI.

Spec: Section 10 — API Layer
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models import ExtractionResult


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractResponse(BaseModel):
    """Response wrapper for a successful document extraction.

    Wraps the core :class:`ExtractionResult` with a request-scoped
    tracing ID so that individual HTTP requests can be correlated
    with their extraction runs.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "request_id": "req_abc123",
            "result": {
                "metadata": {
                    "extraction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "timestamp": "2024-01-15T10:30:00+00:00",
                    "vlm_model": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
                    "llm_model": "llama-3.1-8b-instruct",
                    "processing_time_ms": 1234.5,
                },
                "document_type": "Meldebescheinigung",
                "fields": {},
                "unresolved_count": 0,
                "overall_confidence": 0.92,
            },
        }
    })

    request_id: str = Field(
        ..., description="Unique request ID for tracing (X-Request-ID header value)"
    )
    result: ExtractionResult = Field(
        ..., description="Full extraction result from the pipeline"
    )


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


class CorrectionRequest(BaseModel):
    """Request body for submitting a user correction.

    A correction records the user's fix for a single field and
    triggers the ChromaDB learning loop so the system improves over
    time via few-shot injection.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "extraction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "field_name": "postleitzahl",
            "corrected_value": "10117",
            "original_value": "10115",
        }
    })

    extraction_id: UUID = Field(
        ..., description="UUID of the extraction being corrected"
    )
    field_name: str = Field(
        ..., min_length=1, description="Schema field name (e.g. 'postleitzahl')"
    )
    corrected_value: Any = Field(
        ..., description="The corrected value"
    )
    original_value: Optional[Any] = Field(
        default=None, description="The original (wrong) value before correction"
    )


class CorrectionResponse(BaseModel):
    """Response for a successful correction submission."""

    success: bool = Field(
        ..., description="Whether the correction was stored successfully"
    )
    correction_id: str = Field(
        ..., description="UUID of the stored correction record in ChromaDB"
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class StatsResponse(BaseModel):
    """Pipeline-wide statistics response.

    Aggregated across the audit log and the ChromaDB correction
    collection to give a complete picture of system health and
    performance.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "total_extractions": 42,
            "avg_confidence": 0.8912,
            "unresolved_rate": 0.05,
            "avg_processing_time_ms": 2345.6,
            "document_type_breakdown": {
                "Meldebescheinigung": 20,
                "Steuerbescheid": 12,
                "Gehaltsausweis": 10,
            },
            "model_versions": {
                "vlm": ["mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"],
                "llm": ["llama-3.1-8b-instruct"],
            },
            "total_corrections": 5,
        }
    })

    total_extractions: int = Field(
        default=0, ge=0, description="Total number of extractions processed"
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average overall confidence across all extractions"
    )
    unresolved_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Ratio of unresolved fields to total fields"
    )
    avg_processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average wall-clock processing time in milliseconds"
    )
    document_type_breakdown: Dict[str, int] = Field(
        default_factory=dict, description="Count per document type"
    )
    model_versions: Dict[str, List[str]] = Field(
        default_factory=dict, description="Lists of used VLM and LLM model identifiers"
    )
    total_corrections: int = Field(
        default=0, ge=0, description="Total user corrections stored in ChromaDB"
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response.

    Provides a lightweight endpoint for load balancers and monitoring
    tools to verify that the service and its backing models are
    operational.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "ok",
            "vlm_loaded": False,
            "llm_loaded": False,
            "ram_usage_mb": 6144.5,
            "version": "1.0.0",
        }
    })

    status: str = Field(
        default="ok", description="Service health status: 'ok' or 'degraded'"
    )
    vlm_loaded: bool = Field(
        ..., description="Whether the VLM is currently loaded in memory"
    )
    llm_loaded: bool = Field(
        ..., description="Whether the LLM is currently loaded in memory"
    )
    ram_usage_mb: float = Field(
        ..., ge=0.0, description="Current RAM usage in megabytes"
    )
    version: str = Field(
        ..., description="Application version string"
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Structured error response returned for every failed request.

    Guarantees that the client always receives valid JSON with a
    predictable schema, even when an unhandled exception occurs.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error_code": "IMAGE_REFUSED",
            "message": "Image quality insufficient for deterministic extraction",
            "details": {"legibility_score": 0.45},
            "request_id": "req_abc123",
        }
    })

    error_code: str = Field(
        ..., description="Machine-readable error code (e.g. 'VALIDATION_ERROR')"
    )
    message: str = Field(
        ..., description="Human-readable error description"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional additional context (field-level errors, etc.)"
    )
    request_id: str = Field(
        ..., description="Request ID for correlating the error with server logs"
    )


# ---------------------------------------------------------------------------
# Refusal response (returned when quality gate rejects the image)
# ---------------------------------------------------------------------------


class RefusalResponse(BaseModel):
    """Response when the quality gate refuses an image.

    Contains the refusal reason so the frontend can display it to
    the user and suggest corrective action (retake photo, etc.).
    """

    refused: bool = Field(default=True, description="Always true for refusals")
    reason: str = Field(..., description="Human-readable refusal reason")
    request_id: str = Field(..., description="Request ID for tracing")
    suggestion: Optional[str] = Field(
        default=None, description="Suggested action to resolve the issue"
    )
