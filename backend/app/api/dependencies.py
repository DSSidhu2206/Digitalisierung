"""
FastAPI dependency-injection providers for the Digitalisierung ABE API.

All heavy-weight objects (pipeline, audit logger, ChromaDB manager) are
created lazily and cached so that subsequent requests reuse the same
instances.  This minimises startup latency and keeps model memory
resident across calls.

Spec: Section 10 — API Layer (Dependencies)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends, File, HTTPException, UploadFile, status

from config import ALLOWED_IMAGE_TYPES, MAX_FILE_SIZE_BYTES
from app.database.chroma_manager import ChromaManager
from app.pipeline.audit_logger import AuditLogger
from app.pipeline.orchestrator import ExtractionPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton instances (module-level cache)
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[ExtractionPipeline] = None
_audit_logger_instance: Optional[AuditLogger] = None
_chroma_manager_instance: Optional[ChromaManager] = None


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_pipeline() -> ExtractionPipeline:
    """Return the singleton :class:`ExtractionPipeline` instance.

    The pipeline is created on the first call and cached for the
    lifetime of the process.  This ensures that the VLM / LLM
    managers, ChromaDB client, and audit logger are shared across
    all requests.

    Returns:
        The shared :class:`ExtractionPipeline`.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Creating ExtractionPipeline singleton ...")
        _pipeline_instance = ExtractionPipeline()
        logger.info("ExtractionPipeline singleton created.")
    return _pipeline_instance


@lru_cache(maxsize=1)
def get_audit_logger() -> AuditLogger:
    """Return the singleton :class:`AuditLogger` instance.

    Returns:
        The shared :class:`AuditLogger`.
    """
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance


@lru_cache(maxsize=1)
def get_chroma_manager() -> ChromaManager:
    """Return the singleton :class:`ChromaManager` instance.

    Returns:
        The shared :class:`ChromaManager`.
    """
    global _chroma_manager_instance
    if _chroma_manager_instance is None:
        _chroma_manager_instance = ChromaManager()
    return _chroma_manager_instance


# ---------------------------------------------------------------------------
# File validation dependencies
# ---------------------------------------------------------------------------


async def verify_file_type(file: UploadFile = File(...)) -> UploadFile:
    """Validate that the uploaded file has an allowed MIME type.

    Allowed types: PNG, JPG, JPEG, TIFF, BMP, PDF.

    Args:
        file: The uploaded file from the multipart request.

    Returns:
        The same *file* if validation passes.

    Raises:
        HTTPException: 400 if the file type is not supported.
    """
    content_type: Optional[str] = file.content_type
    if content_type is None:
        # Try to infer from filename
        ext = os.path.splitext(file.filename or "")[1].lower()
        ext_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
            ".pdf": "application/pdf",
        }
        content_type = ext_map.get(ext)

    if content_type not in ALLOWED_IMAGE_TYPES:
        logger.warning(
            "Rejected upload: unsupported content_type=%s filename=%s",
            content_type,
            file.filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed types: PNG, JPG, JPEG, TIFF, BMP, PDF."
            ),
        )

    return file


async def verify_file_size(file: UploadFile = File(...)) -> UploadFile:
    """Validate that the uploaded file does not exceed the size limit.

    Max file size: 50 MB.

    Args:
        file: The uploaded file from the multipart request.

    Returns:
        The same *file* if validation passes.

    Raises:
        HTTPException: 413 if the file exceeds the size limit.
    """
    # Read first chunk to check size without loading entire file
    chunk_size = 8192
    total_size = 0

    # Read chunks without consuming the SpooledTemporaryFile
    # We need to seek back after reading
    spool = file.file
    try:
        while True:
            chunk = spool.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE_BYTES:
                logger.warning(
                    "Rejected upload: file too large (%d bytes > %d bytes)",
                    total_size,
                    MAX_FILE_SIZE_BYTES,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File too large: maximum allowed size is "
                        f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                    ),
                )
        # Reset file pointer for downstream consumers
        spool.seek(0)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error checking file size: %s", exc)
        # If we can't determine size, allow it through — the
        # actual save will fail if it's truly too large
        try:
            spool.seek(0)
        except Exception:
            pass

    return file


# ---------------------------------------------------------------------------
# Convenience combined dependency
# ---------------------------------------------------------------------------


async def validate_upload(
    file: UploadFile = Depends(verify_file_type),
    _: UploadFile = Depends(verify_file_size),
) -> UploadFile:
    """Combined validation: file type + file size.

    This is a convenience dependency that chains both validators.

    Args:
        file: The validated upload file.

    Returns:
        The fully validated :class:`UploadFile`.
    """
    # verify_file_size is applied via Depends, but we receive the
    # type-verified file here.  Both validators have run by now.
    return file
