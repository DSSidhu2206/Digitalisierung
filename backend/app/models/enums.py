"""
Enumeration types for the Digitalisierung ABE pipeline.

Defines the core taxonomies used throughout extraction, validation,
and correction workflows. All enums are string-backed for easy JSON
serialization.
"""

from enum import Enum


class FieldStatus(str, Enum):
    """Lifecycle status of an extracted field.

    Every field moves through a subset of these states during extraction:
      1. VLM extracts raw text
      2. Symbolic validation is applied
      3. If validation fails → VALIDATION_FAILURE
      4. If confidence is too low → LOW_CONFIDENCE
      5. If extraction is impossible → UNRESOLVED
      6. User may correct → CORRECTED
    """

    EXTRACTED = "EXTRACTED"
    """High-confidence extraction that passed all symbolic checks."""

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    """Value was extracted but confidence fell below the 0.7 threshold."""

    UNRESOLVED = "UNRESOLVED"
    """Extraction failed — value is unreadable, missing, or ambiguous."""

    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    """Extracted value failed a deterministic symbolic validation rule."""

    CORRECTED = "CORRECTED"
    """User manually corrected the field after initial extraction."""


class DocumentType(str, Enum):
    """Supported German bureaucratic document types.

    The pipeline attempts to classify every uploaded image into one of
    these categories during the Quality Gate phase. If classification
    fails, the document is typed as ``UNBEKANNT`` (unknown).
    """

    MELDEBESCHEINIGUNG = "Meldebescheinigung"
    """Meldebescheinigung / Anmeldebestaetigung — residence registration."""

    STEUERBESCHEID = "Steuerbescheid"
    """Steuerbescheid — tax assessment notice from Finanzamt."""

    GEHALTSAUSWEIS = "Gehaltsausweis"
    """Gehaltsausweis / Lohnabrechnung — salary statement."""

    PERSONALAUSWEIS = "Personalausweis"
    """Personalausweis — German national identity card."""

    UNBEKANNT = "Unbekannt"
    """Unknown / unclassifiable document type."""


class ExtractionPhase(str, Enum):
    """Phases of the extraction pipeline execution.

    These phases are recorded in :class:`ExtractionMetadata` to provide
    a full audit trail of which pipeline stages completed successfully.
    """

    QUALITY_GATE = "QUALITY_GATE"
    """Phase 1: Visual admissibility check (blur, orientation, legibility)."""

    VISION_PASS_A = "VISION_PASS_A"
    """Phase 2a: Structural pass — identify all form fields and labels."""

    VISION_PASS_B = "VISION_PASS_B"
    """Phase 2b: Value pass — extract raw values for each detected field."""

    CONSISTENCY_CHECK = "CONSISTENCY_CHECK"
    """Phase 3: Cross-validate Pass A structure against Pass B values."""

    SYMBOLIC_VALIDATION = "SYMBOLIC_VALIDATION"
    """Phase 4: Apply deterministic regex, checksum, and business rules."""

    LLM_STRUCTURING = "LLM_STRUCTURING"
    """Phase 5: Structured schema forcing via Instructor + LLM."""

    COMPLETE = "COMPLETE"
    """Pipeline finished — all applicable phases completed."""
