"""
API package for Digitalisierung ABE.

Exports the main API router so that :mod:`app.main` can
include it with a single import.
"""

from app.api.routes import router

__all__ = ["router"]
