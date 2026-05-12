"""
API tests for the Digitalisierung FastAPI backend.

Uses FastAPI's TestClient for end-to-end endpoint testing with mocked
pipeline instances.  No real model inference is performed — all VLM / LLM
calls are replaced with deterministic stubs.

Tests cover:
* File upload → extraction
* Correction submission
* Extraction retrieval by ID
* Stats endpoint
* Health check
* Error handling (invalid file type, file too large)
* Refusal path (low quality image)
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import create_app
from app.models import (
    BoundingBox,
    DocumentType,
    ExtractionMetadata,
    ExtractionPhase,
    ExtractionResult,
    FieldResult,
    FieldStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline() -> Generator[MagicMock, None, None]:
    """Provide a mocked ExtractionPipeline."""
    mock = MagicMock()
    mock.process = AsyncMock()
    mock.submit_correction = AsyncMock()
    mock.get_extraction = MagicMock(return_value=None)
    mock.get_stats = MagicMock(return_value={
        "total_extractions": 0,
        "avg_confidence": 0.0,
        "unresolved_rate": 0.0,
        "avg_processing_time_ms": 0.0,
        "document_type_breakdown": {},
        "model_versions": {"vlm": [], "llm": []},
        "total_corrections": 0,
    })
    mock.health_check = MagicMock(return_value={
        "vlm_loaded": False,
        "llm_loaded": False,
        "ram_usage_mb": 4096.0,
    })
    mock.shutdown = MagicMock()
    yield mock


@pytest.fixture
def client(mock_pipeline: MagicMock) -> Generator[TestClient, None, None]:
    """Create a TestClient with a mocked pipeline.

    Uses FastAPI's ``dependency_overrides`` to inject the mock pipeline
    into every endpoint that depends on ``get_pipeline``.
    """
    from app.api.dependencies import get_pipeline as _get_pipeline_dep

    app = create_app()
    app.dependency_overrides[_get_pipeline_dep] = lambda: mock_pipeline

    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def sample_extraction_result() -> ExtractionResult:
    """Return a realistic sample ExtractionResult."""
    extraction_id = uuid4()
    return ExtractionResult(
        metadata=ExtractionMetadata(
            extraction_id=extraction_id,
            vlm_model="surya-ocr",
            llm_model="llama-3.1-8b-instruct",
            processing_time_ms=1234.5,
            phases_completed=[
                ExtractionPhase.QUALITY_GATE,
                ExtractionPhase.VISION_PASS_A,
                ExtractionPhase.VISION_PASS_B,
                ExtractionPhase.SYMBOLIC_VALIDATION,
                ExtractionPhase.LLM_STRUCTURING,
                ExtractionPhase.COMPLETE,
            ],
        ),
        document_type=DocumentType.MELDEBESCHEINIGUNG,
        fields={
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
            "postleitzahl": FieldResult(
                field_name="postleitzahl",
                value="10115",
                status=FieldStatus.EXTRACTED,
                confidence=0.96,
                source_bbox=BoundingBox(x1=0.12, y1=0.64, x2=0.25, y2=0.69),
                raw_text="10115",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_file(
    filename: str = "test.png",
    content_type: str = "image/png",
    size_bytes: int = 1024,
) -> tuple[bytes, str]:
    """Create a fake image file for upload testing.

    Args:
        filename: Name of the file.
        content_type: MIME type.
        size_bytes: Size of the fake image data.

    Returns:
        (file_bytes, filename)
    """
    # Minimal PNG header (89 50 4E 47 0D 0A 1A 0A) + padding
    header = b"\x89PNG\r\n\x1a\n"
    padding = b"\x00" * max(0, size_bytes - len(header))
    return header + padding, filename


# ---------------------------------------------------------------------------
# Tests: POST /api/v1/extract
# ---------------------------------------------------------------------------


class TestExtractDocument:
    """Tests for the POST /api/v1/extract endpoint."""

    def test_extract_success(
        self,
        client: TestClient,
        mock_pipeline: MagicMock,
        sample_extraction_result: ExtractionResult,
    ) -> None:
        """Happy path: upload valid PNG → receive extraction result."""
        mock_pipeline.process.return_value = sample_extraction_result

        file_data, filename = _make_image_file("test.png", "image/png")

        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "request_id" in data
        assert "result" in data
        assert data["result"]["document_type"] == "Meldebescheinigung"
        assert data["result"]["overall_confidence"] > 0

        # Verify pipeline was called
        mock_pipeline.process.assert_called_once()

    def test_extract_unsupported_file_type(self, client: TestClient) -> None:
        """Uploading a non-image file should return 400."""
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.exe", io.BytesIO(b"not an image"), "application/x-msdownload")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        # FastAPI HTTPException uses 'detail'; our middleware converts to ErrorResponse
        assert "detail" in data or "error_code" in data
        error_msg = data.get("detail", data.get("message", ""))
        assert "Unsupported file type" in error_msg

    def test_extract_no_file(self, client: TestClient) -> None:
        """Request without a file should return 422."""
        response = client.post("/api/v1/extract")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_extract_pipeline_error_returns_partial(
        self,
        client: TestClient,
        mock_pipeline: MagicMock,
    ) -> None:
        """Pipeline errors should still return 200 with partial UNRESOLVED result."""
        mock_pipeline.process.side_effect = RuntimeError("VLM crashed")

        file_data, filename = _make_image_file("test.png", "image/png")

        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should return a partial result with UNRESOLVED fields
        assert data["result"]["document_type"] == "Unbekannt"
        assert "_error" in data["result"]["fields"]
        assert data["result"]["fields"]["_error"]["status"] == "UNRESOLVED"

    def test_extract_refusal(
        self,
        client: TestClient,
        mock_pipeline: MagicMock,
    ) -> None:
        """Quality gate refusal should return extraction with refusal info."""
        from datetime import datetime, timezone

        refusal_result = ExtractionResult(
            metadata=ExtractionMetadata(
                vlm_model="stub",
                llm_model="stub",
                processing_time_ms=100.0,
            ),
            document_type=DocumentType.UNBEKANNT,
            fields={
                "_refusal": FieldResult(
                    field_name="_refusal",
                    value="Image too blurry",
                    status=FieldStatus.UNRESOLVED,
                    confidence=0.0,
                    validation_message="Image too blurry",
                )
            },
        )
        mock_pipeline.process.return_value = refusal_result

        file_data, filename = _make_image_file("test.png", "image/png")

        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["result"]["document_type"] == "Unbekannt"
        assert "_refusal" in data["result"]["fields"]

    def test_extract_request_id_header(self, client: TestClient) -> None:
        """Response should include X-Request-ID header."""
        file_data, filename = _make_image_file("test.png", "image/png")

        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
        )

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_extract_with_client_request_id(self, client: TestClient) -> None:
        """Client-provided X-Request-ID should be echoed back."""
        client_id = "my-custom-request-id-123"
        file_data, filename = _make_image_file("test.png", "image/png")

        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
            headers={"X-Request-ID": client_id},
        )

        assert response.headers["X-Request-ID"] == client_id
        data = response.json()
        assert data["request_id"] == client_id

    def test_extract_response_time_header(self, client: TestClient) -> None:
        """Response should include X-Process-Time header."""
        file_data, filename = _make_image_file("test.png", "image/png")

        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
        )

        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0


# ---------------------------------------------------------------------------
# Tests: POST /api/v1/corrections
# ---------------------------------------------------------------------------


class TestSubmitCorrection:
    """Tests for the POST /api/v1/corrections endpoint."""

    def test_correction_success(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        """Submit a correction → receive success response."""
        mock_pipeline.submit_correction.return_value = "correction-uuid-123"

        payload = {
            "extraction_id": str(uuid4()),
            "field_name": "postleitzahl",
            "corrected_value": "10117",
            "original_value": "10115",
        }

        response = client.post("/api/v1/corrections", json=payload)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["correction_id"] == "correction-uuid-123"

        mock_pipeline.submit_correction.assert_called_once()

    def test_correction_not_found(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        """Submit correction for non-existent extraction → 404."""
        mock_pipeline.submit_correction.side_effect = ValueError("Extraction not found")

        payload = {
            "extraction_id": str(uuid4()),
            "field_name": "postleitzahl",
            "corrected_value": "10117",
        }

        response = client.post("/api/v1/corrections", json=payload)

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error_code" in data

    def test_correction_missing_field_name(self, client: TestClient) -> None:
        """Submit correction without field_name → 422."""
        payload = {
            "extraction_id": str(uuid4()),
            "corrected_value": "10117",
        }

        response = client.post("/api/v1/corrections", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/extractions/{extraction_id}
# ---------------------------------------------------------------------------


class TestGetExtraction:
    """Tests for the GET /api/v1/extractions/{extraction_id} endpoint."""

    def test_get_extraction_success(
        self,
        client: TestClient,
        mock_pipeline: MagicMock,
        sample_extraction_result: ExtractionResult,
    ) -> None:
        """Retrieve an existing extraction by ID."""
        extraction_id = str(sample_extraction_result.metadata.extraction_id)
        mock_pipeline.get_extraction.return_value = sample_extraction_result

        response = client.get(f"/api/v1/extractions/{extraction_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["document_type"] == "Meldebescheinigung"
        assert "metadata" in data

    def test_get_extraction_not_found(self, client: TestClient) -> None:
        """Request non-existent extraction → 404."""
        response = client.get(f"/api/v1/extractions/{uuid4()}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error_code" in data

    def test_get_extraction_invalid_uuid(self, client: TestClient) -> None:
        """Request with invalid UUID format → 422."""
        response = client.get("/api/v1/extractions/not-a-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """Tests for the GET /api/v1/stats endpoint."""

    def test_stats_response(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        """Stats endpoint returns structured statistics."""
        mock_pipeline.get_stats.return_value = {
            "total_extractions": 42,
            "avg_confidence": 0.89,
            "unresolved_rate": 0.05,
            "avg_processing_time_ms": 2345.6,
            "document_type_breakdown": {
                "Meldebescheinigung": 20,
                "Steuerbescheid": 12,
                "Gehaltsausweis": 10,
            },
            "model_versions": {
                "vlm": ["surya-ocr"],
                "llm": ["llama-3.1-8b-instruct"],
            },
            "total_corrections": 5,
        }

        response = client.get("/api/v1/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_extractions"] == 42
        assert data["avg_confidence"] == 0.89
        assert data["unresolved_rate"] == 0.05
        assert data["total_corrections"] == 5
        assert "Meldebescheinigung" in data["document_type_breakdown"]

    def test_stats_empty(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        """Stats endpoint returns zeros when no data exists."""
        mock_pipeline.get_stats.return_value = {
            "total_extractions": 0,
            "avg_confidence": 0.0,
            "unresolved_rate": 0.0,
            "avg_processing_time_ms": 0.0,
            "document_type_breakdown": {},
            "model_versions": {"vlm": [], "llm": []},
            "total_corrections": 0,
        }

        response = client.get("/api/v1/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_extractions"] == 0
        assert data["avg_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Tests: GET /api/v1/health
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for the GET /api/v1/health endpoint."""

    def test_health_ok(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        """Health endpoint returns ok status."""
        mock_pipeline.health_check.return_value = {
            "vlm_loaded": False,
            "llm_loaded": False,
            "ram_usage_mb": 4096.0,
        }

        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["vlm_loaded"] is False
        assert data["llm_loaded"] is False
        assert data["ram_usage_mb"] == 4096.0
        assert data["version"] == "1.0.0"

    def test_health_degraded(self, client: TestClient, mock_pipeline: MagicMock) -> None:
        """Health endpoint returns degraded when RAM is critically high."""
        mock_pipeline.health_check.return_value = {
            "vlm_loaded": True,
            "llm_loaded": True,
            "ram_usage_mb": 15000.0,  # Critical on 16GB system
        }

        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for global error handling middleware."""

    def test_request_id_on_error(self, client: TestClient) -> None:
        """Even error responses include X-Request-ID header."""
        response = client.get("/api/v1/extractions/invalid-uuid")

        assert "X-Request-ID" in response.headers
        data = response.json()
        assert "request_id" in data

    def test_error_response_structure(self, client: TestClient) -> None:
        """Error responses have the expected ErrorResponse structure."""
        response = client.get("/api/v1/extractions/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "error_code" in data
        assert "message" in data
        assert "request_id" in data

    def test_404_not_found(self, client: TestClient) -> None:
        """Non-existent route returns 404 with structured error."""
        response = client.get("/api/v1/nonexistent-endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error_code" in data


# ---------------------------------------------------------------------------
# Tests: Response headers
# ---------------------------------------------------------------------------


class TestResponseHeaders:
    """Tests for middleware-injected response headers."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """CORS headers are present on responses."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost"},
        )

        assert "access-control-allow-origin" in response.headers

    def test_process_time_header(self, client: TestClient) -> None:
        """X-Process-Time header is present on all responses."""
        response = client.get("/api/v1/health")

        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0


# ---------------------------------------------------------------------------
# Tests: File type validation
# ---------------------------------------------------------------------------


class TestFileValidation:
    """Tests for file type and size validation."""

    def test_valid_png(self, client: TestClient) -> None:
        """PNG file passes validation."""
        file_data, filename = _make_image_file("test.png", "image/png")
        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/png")},
        )
        # Should reach the pipeline (may return 200 or mock result)
        assert response.status_code == status.HTTP_200_OK

    def test_valid_jpg(self, client: TestClient) -> None:
        """JPEG file passes validation."""
        file_data, filename = _make_image_file("test.jpg", "image/jpeg")
        response = client.post(
            "/api/v1/extract",
            files={"file": (filename, io.BytesIO(file_data), "image/jpeg")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_valid_pdf(self, client: TestClient) -> None:
        """PDF file passes validation."""
        pdf_data = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\n%%EOF"
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.pdf", io.BytesIO(pdf_data), "application/pdf")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_valid_txt(self, client: TestClient) -> None:
        """Plain text file passes validation for file-text extraction."""
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_invalid_gif_rejected(self, client: TestClient) -> None:
        """GIF file is rejected (not in allowed list)."""
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.gif", io.BytesIO(b"GIF89a"), "image/gif")},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST


# ---------------------------------------------------------------------------
# Tests: Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root(self, client: TestClient) -> None:
        """Root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_openapi_docs(self, client: TestClient) -> None:
        """OpenAPI docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK

    def test_openapi_schema(self, client: TestClient) -> None:
        """OpenAPI JSON schema is accessible and valid."""
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK

        schema = response.json()
        assert schema["info"]["title"] == "Digitalisierung"
        assert schema["info"]["version"] == "1.0.0"

        # Verify all endpoints are documented
        paths = schema["paths"]
        assert "/api/v1/extract" in paths
        assert "/api/v1/corrections" in paths
        assert "/api/v1/extractions/{extraction_id}" in paths
        assert "/api/v1/stats" in paths
        assert "/api/v1/health" in paths
