"""
FastAPI application entry point for Digitalisierung.

Creates the FastAPI app, registers all middleware, includes the API
router, and handles startup / shutdown lifecycle events for model
resource management.

Usage::

    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

Spec: Section 10 — API Layer (Main Application)
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# Ensure the project root is on sys.path so ``backend`` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import router
from app.api.middleware import add_all_middleware, get_request_id
from app.api.models import ErrorResponse
from config import settings, APP_VERSION, UPLOAD_DIR
from app.pipeline.orchestrator import ExtractionPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton reference to the pipeline for lifecycle management
# ---------------------------------------------------------------------------

_pipeline_singleton: ExtractionPipeline | None = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → serve requests → shutdown.

    **Startup:**
    1. Create the ExtractionPipeline singleton.
    2. Log configuration and model paths.
    3. Ensure upload / log directories exist.

    **Shutdown:**
    1. Unload models and free GPU/Unified Memory.
    2. Close database connections.
    3. Log shutdown complete.
    """
    global _pipeline_singleton

    # --- Startup ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Digitalisierung — starting up")
    logger.info("Version: %s", APP_VERSION)
    logger.info("App name: %s", settings.APP_NAME)
    logger.info("Debug: %s", settings.DEBUG)
    logger.info("VLM model: %s", settings.VLM_MODEL_ID)
    logger.info("LLM model: %s", settings.LLM_MODEL_PATH)
    logger.info("ChromaDB persist dir: %s", settings.CHROMA_PERSIST_DIR)
    logger.info("Upload dir: %s", UPLOAD_DIR)
    logger.info("Audit log: %s", settings.AUDIT_LOG_PATH)
    logger.info("RAM threshold: %.0f%%", settings.RAM_THRESHOLD * 100)
    logger.info("=" * 60)

    # Ensure directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.AUDIT_LOG_PATH), exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

    # Create pipeline singleton (this loads ChromaDB, etc.)
    try:
        _pipeline_singleton = ExtractionPipeline()
        logger.info("ExtractionPipeline created successfully.")
    except Exception as exc:
        logger.error("Failed to create ExtractionPipeline: %s", exc)
        # Continue anyway — the pipeline will operate in stub mode

    logger.info("Startup complete. Serving requests.")

    yield  # --- Application serves requests here ---

    # --- Shutdown ---------------------------------------------------------
    logger.info("Digitalisierung — shutting down ...")

    if _pipeline_singleton is not None:
        try:
            _pipeline_singleton.shutdown()
            logger.info("Pipeline shutdown complete.")
        except Exception as exc:
            logger.error("Error during pipeline shutdown: %s", exc)
        finally:
            _pipeline_singleton = None

    logger.info("Shutdown complete. Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`FastAPI` instance ready to be served by
        uvicorn.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Neuro-symbolic, privacy-first document intelligence engine "
            "for German bureaucratic documents. "
            "Transforms unstructured imagery into structured databases "
            "with zero-tolerance for fabrication."
        ),
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Register all middleware (CORS, error handling, logging, request ID,
    # response time)
    add_all_middleware(app)

    frontend_out = PROJECT_ROOT / "frontend" / "out"
    frontend_next_static = frontend_out / "_next" / "static"
    frontend_index = frontend_out / "index.html"
    frontend_dashboard = frontend_out / "dashboard" / "index.html"

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        """Add browser hardening headers for the dashboard surface."""
        response = await call_next(request)

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")

        frontend_paths = (
            request.url.path == "/"
            or request.url.path == "/dashboard"
            or request.url.path.startswith("/_next/")
        )
        if frontend_paths:
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' blob: data:; "
                "connect-src 'self' http://localhost:* http://127.0.0.1:*; "
                "frame-src 'self' blob:; "
                "font-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'",
            )

        return response

    # ------------------------------------------------------------------
    # Exception handlers — convert FastAPI defaults → ErrorResponse
    # ------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Convert HTTPException → structured ErrorResponse."""
        request_id = get_request_id(request)
        status_code = exc.status_code

        if status_code == 404:
            error_code = "NOT_FOUND"
        elif status_code == 422:
            error_code = "VALIDATION_ERROR"
        elif status_code == 413:
            error_code = "FILE_TOO_LARGE"
        elif status_code == 429:
            error_code = "RATE_LIMITED"
        elif status_code == 400:
            error_code = "BAD_REQUEST"
        else:
            error_code = "HTTP_ERROR"

        detail = exc.detail
        message = detail if isinstance(detail, str) else str(detail)

        error_body = ErrorResponse(
            error_code=error_code,
            message=message,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_body.model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Convert RequestValidationError → structured ErrorResponse."""
        request_id = get_request_id(request)

        # Extract the first error message for the human-readable message
        errors = exc.errors()
        if errors:
            first_error = errors[0]
            loc = ".".join(str(loc) for loc in first_error.get("loc", []))
            msg = first_error.get("msg", "Validation failed")
            message = f"{loc}: {msg}" if loc else msg
        else:
            message = "Request validation failed"

        error_body = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=message,
            details={"errors": errors},
            request_id=request_id,
        )
        return JSONResponse(
            status_code=422,
            content=error_body.model_dump(),
        )

    # Include API router with /api/v1 prefix
    app.include_router(router, prefix="/api/v1")

    # NOTE: uploaded files are deliberately NOT served statically. They contain
    # PII (German registration / tax / ID documents), are deleted immediately
    # after processing, and the dashboard previews them via client-side blob
    # URLs — so there is no reason to expose them unauthenticated over HTTP.

    if frontend_next_static.is_dir():
        app.mount(
            "/_next/static",
            StaticFiles(directory=str(frontend_next_static)),
            name="next-static",
        )

    # Serve the built dashboard at /dashboard so the frontend runs from the
    # same origin as the API in production.
    _dashboard_cache: dict[str, str | None] = {"html": None}

    def _load_dashboard_html() -> str:
        if settings.DEBUG or _dashboard_cache["html"] is None:
            dashboard_file = frontend_dashboard if frontend_dashboard.is_file() else frontend_index
            if dashboard_file.is_file():
                _dashboard_cache["html"] = dashboard_file.read_text(encoding="utf-8")
                logger.info("Dashboard loaded from %s (%d chars)", dashboard_file, len(_dashboard_cache["html"]))
            else:
                _dashboard_cache["html"] = (
                    "<!doctype html><html><head><title>Dashboard not built</title>"
                    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head>"
                    "<body style=\"font-family:system-ui;background:#0b1018;color:#dce6f2;padding:32px\">"
                    "<h1>Dashboard not built</h1>"
                    "<p>Run <code>npm install</code> and <code>npm run build</code> in <code>frontend/</code>, "
                    "then restart the FastAPI server.</p>"
                    "</body></html>"
                )
        return _dashboard_cache["html"] or ""

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        """Serve the dashboard shell from the built TypeScript frontend."""
        return HTMLResponse(content=_load_dashboard_html())

    # Root endpoint — redirect browsers to dashboard, JSON for API clients
    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        """Redirect browsers to the dashboard; return JSON for API clients."""
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return HTMLResponse(content=_load_dashboard_html(), status_code=200)
        return {
            "message": "Digitalisierung API",
            "version": APP_VERSION,
            "docs": "/docs",
            "api": "/api/v1",
            "dashboard": "/dashboard",
        }

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str, request: Request) -> HTMLResponse:
        """Allow refreshing frontend routes while preserving API 404 behavior."""
        reserved_prefixes = ("api/", "uploads/", "_next/")
        reserved_exact = {"docs", "redoc", "openapi.json", "favicon.ico"}
        if full_path in reserved_exact or full_path.startswith(reserved_prefixes):
            raise HTTPException(status_code=404, detail="Not found")

        accept = request.headers.get("accept", "")
        if "text/html" not in accept:
            raise HTTPException(status_code=404, detail="Not found")
        return HTMLResponse(content=_load_dashboard_html())

    return app


# ---------------------------------------------------------------------------
# Uvicorn entry point
# ---------------------------------------------------------------------------

app: FastAPI = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
