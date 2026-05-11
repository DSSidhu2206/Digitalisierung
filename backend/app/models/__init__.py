"""Models package for Digitalisierung ABE."""

from app.models.enums import DocumentType, ExtractionPhase, FieldStatus
from app.models.extraction_models import (
    BoundingBox,
    ExtractionMetadata,
    ExtractionResult,
    FieldResult,
)
from app.models.document_schemas import (
    DOCUMENT_SCHEMAS,
    GehaltsausweisSchema,
    MeldebescheinigungSchema,
    PersonalausweisSchema,
    SteuerbescheidSchema,
    get_schema_for_document_type,
)

__all__ = [
    "DocumentType",
    "ExtractionPhase",
    "FieldStatus",
    "BoundingBox",
    "ExtractionMetadata",
    "ExtractionResult",
    "FieldResult",
    "DOCUMENT_SCHEMAS",
    "GehaltsausweisSchema",
    "MeldebescheinigungSchema",
    "PersonalausweisSchema",
    "SteuerbescheidSchema",
    "get_schema_for_document_type",
]
