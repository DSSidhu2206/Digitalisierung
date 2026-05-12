"""
FastAPI middleware for the Digitalisierung ABE API.

Provides:
*   Error handling — catches all exceptions and returns structured JSON
*   CORS — configured for Flutter desktop origin
*   Request logging — structured JSON logs for every request
*   Request ID injection — unique ID per request for distributed tracing
*   Response time header — ``X-Process-Time`` on every response

Spec: Section 10 — API Layer (Middleware)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Coroutine, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.models import ErrorResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_ID_HEADER: str = "X-Request-ID"
PROCESS_TIME_HEADER: str = "X-Process-Time"


# ---------------------------------------------------------------------------
# 1. Error Handling Middleware
# ---------------------------------------------------------------------------


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch all unhandled exceptions and return structured JSON.

    Guarantees that the client always receives a valid
    :class:`ErrorResponse` body with an appropriate HTTP status code,
    even for unexpected server errors.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request with full exception wrapping."""
        request_id: str = _get_request_id(request)

        try:
            return await call_next(request)

        except Exception as exc:
            logger.exception(
                "Unhandled exception: request_id=%s method=%s path=%s error=%s",
                request_id,
                request.method,
                request.url.path,
                str(exc),
            )

            status_code = 500
            error_code = "INTERNAL_ERROR"
            message = "An unexpected error occurred. Please try again later."
            details: Optional[dict[str, Any]] = None

            # Map known exception types to meaningful error codes
            if hasattr(exc, "status_code"):
                status_code = int(exc.status_code)  # type: ignore[attr-defined]
            if hasattr(exc, "detail"):
                detail_val = exc.detail  # type: ignore[attr-defined]
                if isinstance(detail_val, str):
                    message = detail_val
                elif isinstance(detail_val, dict):
                    details = detail_val
                    message = detail_val.get("message", message)

            if status_code == 404:
                error_code = "NOT_FOUND"
            elif status_code == 422:
                error_code = "VALIDATION_ERROR"
                message = message or "Request validation failed"
            elif status_code == 400:
                error_code = "BAD_REQUEST"
            elif status_code == 413:
                error_code = "FILE_TOO_LARGE"
            elif status_code == 429:
                error_code = "RATE_LIMITED"

            error_body = ErrorResponse(
                error_code=error_code,
                message=message,
                details=details,
                request_id=request_id,
            )

            return JSONResponse(
                status_code=status_code,
                content=error_body.model_dump(),
            )


# ---------------------------------------------------------------------------
# 2. Request Logging Middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit a structured JSON log line for every request.

    The log contains the HTTP method, path, status code, response
    time, client host, and the request ID.  This makes it trivial to
    grep, filter, and aggregate logs in Loki / CloudWatch / etc.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request start, process, and completion."""
        request_id: str = _get_request_id(request)
        start_time: float = time.perf_counter()

        logger.info(
            "Request started: request_id=%s method=%s path=%s client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
        )

        try:
            response: Response = await call_next(request)
        except Exception:
            raise  # Let ErrorHandlingMiddleware deal with it
        finally:
            elapsed_ms: float = (time.perf_counter() - start_time) * 1000

        status_code: int = response.status_code
        logger.info(
            "Request completed: request_id=%s method=%s path=%s status=%d elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )

        return response


# ---------------------------------------------------------------------------
# 3. Request ID Injection Middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request.

    The ID is taken from the ``X-Request-ID`` header if the client
    already provided one; otherwise a fresh UUID is generated.  The
    ID is attached to both the request state (``request.state.request_id``)
    and the response headers.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Inject or propagate the request ID."""
        request_id: str = _get_request_id(request)
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


# ---------------------------------------------------------------------------
# 4. Response Time Header Middleware
# ---------------------------------------------------------------------------


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """Add ``X-Process-Time`` header to every response.

    The value is the wall-clock time (in milliseconds) spent
    processing the request inside the application.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Measure and attach processing time."""
        start_time: float = time.perf_counter()

        response: Response = await call_next(request)

        elapsed_ms: float = (time.perf_counter() - start_time) * 1000
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.3f}"

        return response


# ---------------------------------------------------------------------------
# CORS setup
# ---------------------------------------------------------------------------


def add_cors_middleware(app: FastAPI) -> None:
    """Configure CORS for the dashboard frontend.

    Allows the dashboard (opened via ``file://`` or served from
    ``localhost``) to make cross-origin requests to the API.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "app://localhost",   # Flutter desktop (macOS)
            "null",              # file:// origin (dashboard opened locally)
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER, PROCESS_TIME_HEADER],
    )


# ---------------------------------------------------------------------------
# Convenience: register all middleware on an app
# ---------------------------------------------------------------------------


def add_all_middleware(app: FastAPI) -> None:
    """Register every middleware on *app* in the correct order.

    Order matters: error handling must be innermost so it can catch
    exceptions from all other layers.
    """
    # CORS first (outermost)
    add_cors_middleware(app)

    # Custom middleware — order is reverse of registration
    # (last added = outermost in Starlette)
    app.add_middleware(ResponseTimeMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)  # innermost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_request_id(request: Request) -> str:
    """Extract or generate a request ID.

    Returns the ``X-Request-ID`` header value if present and
    non-empty; otherwise a fresh UUID4 string.
    """
    header_value: Optional[str] = request.headers.get(REQUEST_ID_HEADER)
    if header_value:
        return header_value
    return str(uuid.uuid4())


def get_request_id(request: Request) -> str:
    """Public helper: get the request ID from request state.

    Falls back to extracting from headers or generating a new one.
    """
    state_id: Optional[str] = getattr(request.state, "request_id", None)
    if state_id:
        return state_id
    return _get_request_id(request)
