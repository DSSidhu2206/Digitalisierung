"""
FastAPI router for the Digitalisierung ABE REST API.

Implements all endpoints defined in Spec Section 10.1:
*   POST /api/v1/extract — document extraction from image
*   POST /api/v1/corrections — user correction (learning loop)
*   GET /api/v1/extractions/{extraction_id} — retrieve extraction
*   GET /api/v1/stats — pipeline statistics
*   GET /api/v1/health — health check

Every endpoint uses Pydantic request/response models for automatic
OpenAPI documentation and strict validation.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_audit_logger,
    get_chroma_manager,
    get_pipeline,
    verify_file_size,
    verify_file_type,
)
from app.api.middleware import get_request_id
from app.api.models import (
    CorrectionRequest,
    CorrectionResponse,
    ErrorResponse,
    ExtractResponse,
    HealthResponse,
    LocalTrainingStartRequest,
    LocalTrainingStatusResponse,
    RefusalResponse,
    StatsResponse,
)
from config import APP_VERSION, get_settings
from app.models import DocumentType, ExtractionResult, FieldResult, FieldStatus
from app.pipeline.audit_logger import AuditLogger
from app.pipeline.local_training import TrainingConfig
from app.pipeline.orchestrator import ExtractionPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helper: store uploaded file to temp directory
# ---------------------------------------------------------------------------


async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temporary directory.

    The caller is responsible for cleaning up the file after use.

    Args:
        file: The FastAPI UploadFile.

    Returns:
        Absolute path to the saved file.
    """
    settings = get_settings()
    upload_dir: str = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename with microsecond-precision timestamp to avoid collisions
    safe_name: str = os.path.basename(file.filename or "upload")
    dest_path: str = os.path.join(upload_dir, f"{int(time.time() * 1_000_000)}_{safe_name}")

    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    logger.debug("Uploaded file saved to %s", dest_path)
    return dest_path


# ---------------------------------------------------------------------------
# POST /api/v1/extract
# ---------------------------------------------------------------------------


@router.post(
    "/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": RefusalResponse, "description": "Image refused by quality gate"},
        422: {"model": ErrorResponse, "description": "Validation error (bad file type, etc.)"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ExtractResponse, "description": "Pipeline error — returns partial result"},
    },
    summary="Extract structured data from an uploaded document image",
    description=(
        "Upload an image (PNG, JPG, JPEG, TIFF, BMP, or PDF) and run the full "
        "extraction pipeline. Returns structured data with per-field confidence, "
        "provenance bounding boxes, and an audit trail. If the pipeline encounters "
        "an error, a partial result with UNRESOLVED fields is still returned."
    ),
)
async def extract_document(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Document image file (PNG, JPG, JPEG, TIFF, BMP, PDF). Max 50 MB.",
    ),
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> ExtractResponse:
    """Upload image → run full pipeline → return structured extraction.

    **Request:** multipart/form-data with a single ``file`` field.

    **Response:** JSON containing:
    - ``request_id`` — tracing ID (same as X-Request-ID header)
    - ``result`` — full :class:`ExtractionResult` with metadata and fields

    **Error handling:**
    - 400 — Image refused by quality gate (blur, low contrast, etc.)
    - 413 — File exceeds 50 MB limit
    - 422 — Invalid file type or missing file
    - 500 — Pipeline error (still returns partial result with UNRESOLVED fields)
    """
    request_id: str = get_request_id(request)
    file_path: Optional[str] = None

    # --- Validate file type and size --------------------------------
    # FastAPI runs Depends() validators before entering the endpoint
    # body, but we also do explicit validation here for clarity.

    # Validate MIME type
    validated_file: UploadFile = await verify_file_type(file)
    # Validate file size
    validated_file = await verify_file_size(validated_file)

    try:
        # --- Save to temp directory ----------------------------------
        file_path = await _save_upload(validated_file)

        # --- Run extraction pipeline ---------------------------------
        logger.info(
            "Starting extraction: request_id=%s file=%s",
            request_id,
            validated_file.filename,
        )
        result: ExtractionResult = await pipeline.process(file_path)

        # Check for quality-gate refusal (encoded in the result)
        if result.document_type == DocumentType.UNBEKANNT and "_refusal" in result.fields:
            refusal_field: FieldResult = result.fields["_refusal"]
            refusal_msg: str = str(refusal_field.value) if refusal_field.value else "Image refused"
            logger.info("Extraction refused: request_id=%s reason=%s", request_id, refusal_msg)

            # Return the extraction result anyway (with refusal info)
            return ExtractResponse(request_id=request_id, result=result)

        logger.info(
            "Extraction complete: request_id=%s extraction_id=%s doc_type=%s confidence=%.3f",
            request_id,
            result.metadata.extraction_id,
            result.document_type.value,
            result.overall_confidence,
        )

        return ExtractResponse(request_id=request_id, result=result)

    except HTTPException:
        raise  # Re-raise FastAPI HTTPExceptions (validation errors, etc.)

    except Exception as exc:
        logger.exception(
            "Pipeline error: request_id=%s error=%s", request_id, str(exc)
        )
        # Build a partial result with UNRESOLVED fields
        from datetime import datetime, timezone
        from uuid import uuid4

        from app.models import BoundingBox, ExtractionMetadata

        partial_result = ExtractionResult(
            metadata=ExtractionMetadata(
                extraction_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                vlm_model="stub",
                llm_model="stub",
                processing_time_ms=0.0,
            ),
            document_type=DocumentType.UNBEKANNT,
            fields={
                "_error": FieldResult(
                    field_name="_error",
                    value=None,
                    status=FieldStatus.UNRESOLVED,
                    confidence=0.0,
                    validation_message=f"Pipeline error: {str(exc)}",
                )
            },
        )
        return ExtractResponse(request_id=request_id, result=partial_result)

    finally:
        # --- Clean up temp file ---------------------------------------
        if file_path and os.path.isfile(file_path):
            try:
                os.remove(file_path)
                logger.debug("Cleaned up temp file: %s", file_path)
            except OSError as exc:
                logger.warning("Failed to clean up temp file %s: %s", file_path, exc)


# ---------------------------------------------------------------------------
# POST /api/v1/corrections
# ---------------------------------------------------------------------------


@router.post(
    "/corrections",
    response_model=CorrectionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Extraction not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
    summary="Submit a user correction for the learning loop",
    description=(
        "Submit a corrected value for a single field. The correction is stored "
        "atomically in ChromaDB with an embedding so it can be retrieved as a "
        "few-shot example for future extractions."
    ),
)
async def submit_correction(
    request: Request,
    body: CorrectionRequest,
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> CorrectionResponse:
    """Submit user correction → update ChromaDB learning loop.

    **Request body:**
    - ``extraction_id`` — UUID of the extraction being corrected
    - ``field_name`` — schema field name (e.g. "postleitzahl")
    - ``corrected_value`` — the new correct value
    - ``original_value`` — the original (wrong) value (optional)

    **Response:**
    - ``success`` — whether the correction was stored
    - ``correction_id`` — UUID of the stored correction

    **Error handling:**
    - 404 — Extraction ID not found
    - 422 — Validation error (missing field, etc.)
    - 500 — ChromaDB storage failure
    """
    request_id: str = get_request_id(request)

    try:
        correction_id: str = await pipeline.submit_correction(
            extraction_id=body.extraction_id,
            field_name=body.field_name,
            corrected_value=body.corrected_value,
            original_value=body.original_value,
        )

        logger.info(
            "Correction submitted: request_id=%s extraction_id=%s field=%s correction_id=%s",
            request_id,
            body.extraction_id,
            body.field_name,
            correction_id,
        )

        return CorrectionResponse(success=True, correction_id=correction_id)

    except ValueError as exc:
        logger.warning(
            "Correction failed (not found): request_id=%s error=%s",
            request_id,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Correction storage error: request_id=%s error=%s",
            request_id,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store correction: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /api/v1/extractions/{extraction_id}
# ---------------------------------------------------------------------------


@router.get(
    "/extractions/{extraction_id}",
    response_model=ExtractionResult,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Extraction not found"},
        422: {"model": ErrorResponse, "description": "Invalid UUID format"},
    },
    summary="Retrieve a previously stored extraction by ID",
    description=(
        "Look up an extraction result by its UUID. The extraction must have been "
        "processed during the current server session (in-memory store)."
    ),
)
async def get_extraction(
    extraction_id: UUID,
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> ExtractionResult:
    """Retrieve extraction by ID.

    **Path parameter:** ``extraction_id`` — UUID string.

    **Response:** full :class:`ExtractionResult` JSON.

    **Error handling:**
    - 404 — Extraction not found
    - 422 — Invalid UUID format
    """
    result: Optional[ExtractionResult] = pipeline.get_extraction(str(extraction_id))

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction not found: {extraction_id}",
        )

    return result


# ---------------------------------------------------------------------------
# GET /api/v1/stats
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get pipeline statistics",
    description=(
        "Returns aggregate statistics: total extractions, average confidence, "
        "unresolved rate, average processing time, document type breakdown, "
        "model versions, and total corrections stored."
    ),
)
async def get_stats(
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> StatsResponse:
    """Pipeline statistics: counts, confidence averages, model versions.

    **Response:**
    - ``total_extractions`` — total number processed
    - ``avg_confidence`` — mean overall confidence
    - ``unresolved_rate`` — ratio of unresolved fields
    - ``avg_processing_time_ms`` — mean processing time
    - ``document_type_breakdown`` — count per document type
    - ``model_versions`` — lists of VLM / LLM models used
    - ``total_corrections`` — corrections in ChromaDB
    """
    stats_result = pipeline.get_stats()
    if hasattr(stats_result, "__await__"):
        stats_result = await stats_result
    stats: dict[str, Any] = stats_result
    audit_stats = stats.get("audit", stats)
    correction_stats = stats.get("corrections", {})
    image_learning_stats = stats.get("image_learning", {})

    return StatsResponse(
        total_extractions=audit_stats.get("total_extractions", 0),
        avg_confidence=audit_stats.get("avg_confidence", 0.0),
        unresolved_rate=audit_stats.get("unresolved_rate", 0.0),
        avg_processing_time_ms=audit_stats.get("avg_processing_time_ms", 0.0),
        document_type_breakdown=audit_stats.get("document_type_breakdown", {}),
        model_versions=audit_stats.get("model_versions", {"vlm": [], "llm": []}),
        total_corrections=correction_stats.get("total", stats.get("total_corrections", 0)),
        total_learned_images=image_learning_stats.get("total_images", 0),
    )


# ---------------------------------------------------------------------------
# Local training controls
# ---------------------------------------------------------------------------


@router.get(
    "/local-training/status",
    response_model=LocalTrainingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get local image-learning status",
)
async def get_local_training_status(
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> LocalTrainingStatusResponse:
    """Return current local RVL image-learning status."""
    return LocalTrainingStatusResponse(**pipeline.local_training.status())


@router.post(
    "/local-training/start",
    response_model=LocalTrainingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Start or resume local image learning",
)
async def start_local_training(
    body: LocalTrainingStartRequest,
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> LocalTrainingStatusResponse:
    """Start a resumable local learning chunk for the RVL corpus."""
    training_status = pipeline.local_training.start(
        TrainingConfig(
            batch_size=body.batch_size,
            max_seconds=body.max_seconds,
            plateau_window=body.plateau_window,
            heartbeat_seconds=body.heartbeat_seconds,
        )
    )
    return LocalTrainingStatusResponse(**training_status)


@router.post(
    "/local-training/stop",
    response_model=LocalTrainingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Stop local image learning",
)
async def stop_local_training(
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> LocalTrainingStatusResponse:
    """Stop the currently managed local learning process."""
    return LocalTrainingStatusResponse(**pipeline.local_training.stop())


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description=(
        "Lightweight health check for load balancers and monitoring. Returns "
        "service status, model load state, RAM usage, and version."
    ),
)
async def health_check(
    pipeline: ExtractionPipeline = Depends(get_pipeline),
) -> HealthResponse:
    """Health check.

    **Response:**
    - ``status`` — ``"ok"`` or ``"degraded"``
    - ``vlm_loaded`` — whether VLM is in memory
    - ``llm_loaded`` — whether LLM is in memory
    - ``ram_usage_mb`` — current RAM usage in MB
    - ``version`` — application version string
    """
    health: dict[str, Any] = pipeline.health_check()

    # Determine overall status
    status_str: str = "ok"
    if health.get("ram_usage_mb", 0) > 14000:  # > 14 GB on 16 GB system
        status_str = "degraded"

    return HealthResponse(
        status=status_str,
        vlm_loaded=health.get("vlm_loaded", False),
        llm_loaded=health.get("llm_loaded", False),
        ram_usage_mb=health.get("ram_usage_mb", 0.0),
        version=APP_VERSION,
    )
