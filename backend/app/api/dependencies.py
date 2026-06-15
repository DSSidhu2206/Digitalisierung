"""
FastAPI dependency-injection providers for the Digitalisierung ABE API.

All heavy-weight objects (pipeline, audit logger, ChromaDB manager) are
created lazily and cached so that subsequent requests reuse the same
instances.  This minimises startup latency and keeps model memory
resident across calls.

Spec: Section 10 — API Layer (Dependencies)
"""

from __future__ import annotations

import hmac
import logging
import os
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends, File, Header, HTTPException, UploadFile, status

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
# Authentication
# ---------------------------------------------------------------------------


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Require a valid API key on mutating / expensive endpoints.

    No-op when ``API_KEY`` is unset — the server binds to localhost by
    default, so this is defense-in-depth, not the primary control. When a key
    is configured it is compared in constant time.
    """
    from config import get_settings

    expected = (get_settings().API_KEY or "").strip()
    if not expected:
        return
    provided = (x_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# File validation dependencies
# ---------------------------------------------------------------------------

# Magic-byte signatures for the binary types we accept. Text types
# (txt/csv/tsv/json/xml/html/md) have no reliable signature and are handled
# downstream as size-bounded text.
_BINARY_MAGIC: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/jpg": (b"\xff\xd8\xff",),
    "image/tiff": (b"II*\x00", b"MM\x00*"),
    "image/bmp": (b"BM",),
    "application/pdf": (b"%PDF",),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        b"PK\x03\x04",
        b"PK\x05\x06",
        b"PK\x07\x08",
    ),
}


def _magic_matches(content_type: Optional[str], header: bytes) -> bool:
    """Whether *header* bytes are consistent with the declared *content_type*."""
    signatures = _BINARY_MAGIC.get(content_type or "")
    if signatures is None:
        return True  # text type — nothing to sniff
    return any(header.startswith(sig) for sig in signatures)


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
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".md": "text/markdown",
            ".json": "application/json",
            ".xml": "application/xml",
            ".html": "text/html",
            ".htm": "text/html",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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
                "Allowed types: PNG, JPG, JPEG, TIFF, BMP, PDF, TXT, CSV, "
                "TSV, Markdown, JSON, XML, HTML, and DOCX."
            ),
        )

    # Content-sniff: for binary types the magic bytes must match the declared
    # type. Defends against a .png-named text/zip bomb or polyglot upload.
    try:
        header = file.file.read(8)
        file.file.seek(0)
    except Exception:
        header = b""
    if not _magic_matches(content_type, header):
        logger.warning(
            "Rejected upload: magic bytes do not match declared type %s", content_type
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match its declared type.",
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
        try:
            spool.seek(0)
        except Exception:
            pass
        # Fail closed: if size cannot be validated, reject rather than risk an
        # unbounded write to disk.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not validate file size; upload rejected.",
        )

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
