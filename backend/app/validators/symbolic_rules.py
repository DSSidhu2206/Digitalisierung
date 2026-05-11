"""
Symbolic validation rules for German bureaucratic document fields.

This module provides:
  - A curated dictionary of regex patterns per field type.
  - Business rules (e.g. birthdate cannot be in the future).
  - :class:`SymbolicValidator` — the main entry point for validating
    individual fields and entire documents.

All rules are deterministic, explainable, and side-effect free.  When
a rule fails it produces a human-readable ``validation_message`` that
is stored in the :class:`FieldResult`.

Anti-Fabrication Constraints (Section 12):
  - Every rule rejects rather than corrects.
  - No extrapolation or auto-completion is performed.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, ClassVar, Optional

from app.models.enums import DocumentType, FieldStatus
from app.models.extraction_models import FieldResult
from app.validators.checksums import (
    validate_iban,
    validate_postleitzahl,
    validate_steuer_id,
)
from app.validators.german_validators import GermanDateValidator, GermanStreetValidator


# ---------------------------------------------------------------------------
# Regex pattern catalogue per field type
# ---------------------------------------------------------------------------

regex_patterns: dict[str, str] = {
    # Personal data
    "familienname": r"^[A-Za-zÄÖÜäöüß\-\\s']{2,100}$",
    "vorname": r"^[A-Za-zÄÖÜäöüß\-\\s']{2,100}$",
    "geburtsort": r"^[A-Za-zÄÖÜäöüß\-\\s().]{2,100}$",
    "staatsangehoerigkeit": r"^[A-Za-zÄÖÜäöüß\-\\s]{2,100}$",
    # Address fields
    "strasse": r"^[A-Za-zÄÖÜäöüß0-9\-\\s./]{2,100}$",
    "hausnummer": r"^[0-9]{1,4}[a-zA-Z]?$",
    "postleitzahl": r"^\d{5}$",
    "wohnort": r"^[A-Za-zÄÖÜäöüß\-\\s]{2,100}$",
    "vorherige_anschrift": r"^[A-Za-zÄÖÜäöüß0-9\-\\s.,/]{5,200}$",
    # Document-specific identifiers
    "steueridentifikationsnummer": r"^\d{11}$",
    "dokumentnummer": r"^[A-Z0-9]{9}$",
    # Financial
    "veranlagungszeitraum": r"^\d{4}$",
    "abrechnungszeitraum": r"^(0[1-9]|1[0-2])\.\d{4}$",
    "steuerklasse": r"^(I|II|III|IV|V|VI)$",
    "sozialversicherungsnummer": r"^\d{12}$",
    # Generic
    "arbeitgeber": r"^[A-Za-zÄÖÜäöüß0-9\-\\s&.,()]{2,200}$",
    "ausstellende_behoerde": r"^[A-Za-zÄÖÜäöüß0-9\-\\s.,()]{2,200}$",
}
"""Mapping of field-name → regex pattern.

Patterns cover the four supported document types.  Field names match
the attribute names in the Pydantic schema classes so that
:class:`SymbolicValidator` can look them up dynamically.
"""


# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------

def business_rules(
    field_name: str, value: Any, document_type: Optional[DocumentType] = None
) -> tuple[bool, Optional[str]]:
    """Apply high-level business rules that go beyond regex matching.

    Business rules enforce semantic constraints such as "a birthdate
    cannot be in the future" or "a salary must be non-negative".

    Parameters
    ----------
    field_name:
        Schema attribute name (e.g. ``"geburtsdatum"``).
    value:
        The extracted value.  May be ``None``.
    document_type:
        Optional document type for context-specific rules.

    Returns
    -------
    tuple[bool, Optional[str]]
        ``(True, None)`` if the value passes all business rules, or
        ``(False, message)`` if a rule is violated.
    """
    if value is None:
        return True, None  # UNRESOLVED fields have no value to validate

    # ------------------------------------------------------------------
    # Birthdate rules
    # ------------------------------------------------------------------
    if field_name == "geburtsdatum":
        return _rule_birthdate(value)

    # ------------------------------------------------------------------
    # Document expiry rules
    # ------------------------------------------------------------------
    if field_name == "gueltig_bis":
        return _rule_expiry_date(value)

    # ------------------------------------------------------------------
    # Financial amount rules
    # ------------------------------------------------------------------
    if field_name in ("zu_versteuerndes_einkommen", "festgesetzte_steuer"):
        return _rule_non_negative_amount(value, field_name)

    if field_name in ("brutto_lohn", "netto_lohn"):
        return _rule_salary(value, field_name)

    # ------------------------------------------------------------------
    # Steuer-ID checksum rule
    # ------------------------------------------------------------------
    if field_name == "steueridentifikationsnummer":
        return _rule_steuer_id_checksum(value)

    # ------------------------------------------------------------------
    # PLZ structural rule
    # ------------------------------------------------------------------
    if field_name == "postleitzahl":
        return _rule_plz(value)

    # ------------------------------------------------------------------
    # Street name rule
    # ------------------------------------------------------------------
    if field_name == "strasse":
        return _rule_street(value)

    # ------------------------------------------------------------------
    # Move-in date cannot be in the future
    # ------------------------------------------------------------------
    if field_name == "einzugsdatum":
        return _rule_move_in_date(value)

    # ------------------------------------------------------------------
    # Net salary must not exceed gross salary
    # ------------------------------------------------------------------
    if field_name == "netto_lohn":
        return _rule_net_leq_gross(value, document_type)

    # No specific business rule for this field
    return True, None


def _rule_birthdate(value: Any) -> tuple[bool, Optional[str]]:
    """Birthdate cannot be in the future and must be reasonable."""
    if isinstance(value, date):
        birth = value
    elif isinstance(value, str):
        parsed = GermanDateValidator.parse(value)
        if parsed is None:
            return False, "Ungueltiges Geburtsdatum (nicht parsierbar)."
        birth = parsed
    else:
        return False, f"Geburtsdatum hat unerwarteten Typ: {type(value).__name__}"

    today = datetime.now(timezone.utc).date()

    if birth > today:
        return False, "Geburtsdatum liegt in der Zukunft."

    # Reasonable lower bound: nobody older than 150 years
    from datetime import timedelta

    if (today - birth).days > 150 * 365:
        return False, "Geburtsdatum unplausibel (aelter als 150 Jahre)."

    return True, None


def _rule_expiry_date(value: Any) -> tuple[bool, Optional[str]]:
    """Document expiry date must be in the future (or today)."""
    if isinstance(value, date):
        expiry = value
    elif isinstance(value, str):
        parsed = GermanDateValidator.parse(value, allow_future=True)
        if parsed is None:
            return False, "Ungueltiges Ablaufdatum (nicht parsierbar)."
        expiry = parsed
    else:
        return False, f"Ablaufdatum hat unerwarteten Typ: {type(value).__name__}"

    today = datetime.now(timezone.utc).date()

    if expiry < today:
        return False, "Dokument ist abgelaufen."

    return True, None


def _rule_non_negative_amount(value: Any, field_name: str) -> tuple[bool, Optional[str]]:
    """Financial amounts must be non-negative."""
    if isinstance(value, (int, float)):
        if value < 0:
            return False, f"{field_name} darf nicht negativ sein."
        return True, None
    return False, f"{field_name} muss eine Zahl sein."


def _rule_salary(value: Any, field_name: str) -> tuple[bool, Optional[str]]:
    """Salary amounts must be positive."""
    if isinstance(value, (int, float)):
        if value <= 0:
            return False, f"{field_name} muss groesser als 0 sein."
        if value > 10_000_000:
            return False, f"{field_name} unplausibel hoch."
        return True, None
    return False, f"{field_name} muss eine Zahl sein."


def _rule_steuer_id_checksum(value: Any) -> tuple[bool, Optional[str]]:
    """Steuer-ID must pass the 11-digit checksum algorithm."""
    if not isinstance(value, str):
        return False, "Steuer-ID muss ein String sein."
    if not validate_steuer_id(value):
        return False, "Steuer-ID hat ungueltige Pruefsumme."
    return True, None


def _rule_plz(value: Any) -> tuple[bool, Optional[str]]:
    """Postleitzahl must be a valid 5-digit German PLZ."""
    if not isinstance(value, str):
        return False, "PLZ muss ein String sein."
    if not validate_postleitzahl(value):
        return False, "PLZ muss genau 5 Ziffern enthalten."
    return True, None


def _rule_street(value: Any) -> tuple[bool, Optional[str]]:
    """Street name must pass the German street validator."""
    if not isinstance(value, str):
        return False, "Strassenname muss ein String sein."
    if not GermanStreetValidator.validate(value):
        return False, "Strassenname ist ungueltig oder unvollstaendig."
    return True, None


def _rule_move_in_date(value: Any) -> tuple[bool, Optional[str]]:
    """Move-in date cannot be in the future."""
    if isinstance(value, date):
        move_in = value
    elif isinstance(value, str):
        parsed = GermanDateValidator.parse(value)
        if parsed is None:
            return False, "Ungueltiges Einzugsdatum."
        move_in = parsed
    else:
        return False, "Einzugsdatum hat unerwarteten Typ."

    today = datetime.now(timezone.utc).date()
    if move_in > today:
        return False, "Einzugsdatum liegt in der Zukunft."

    return True, None


def _rule_net_leq_gross(
    value: Any, document_type: Optional[DocumentType] = None
) -> tuple[bool, Optional[str]]:
    """Net salary must not exceed gross salary.

    This rule is a stub for cross-field validation.  The full
    implementation requires access to both fields simultaneously via
    :meth:`SymbolicValidator.validate_document`.
    """
    # Single-field validation is always a pass; cross-field check
    # happens at the document level.
    return True, None


# ---------------------------------------------------------------------------
# SymbolicValidator
# ---------------------------------------------------------------------------

class SymbolicValidator:
    """Main entry point for symbolic validation of extracted fields.

    Orchestrates regex matching, business-rule evaluation, and
    checksum validation.  Produces a :class:`FieldResult` with an
    updated status and validation message.

    Example usage::

        validator = SymbolicValidator()
        field = validator.validate_field(
            field_name="postleitzahl",
            value="12345",
            document_type=DocumentType.MELDEBESCHEINIGUNG,
        )
        if field.status == FieldStatus.VALIDATION_FAILURE:
            print(field.validation_message)
    """

    def validate_field(
        self,
        field_name: str,
        value: Any,
        document_type: Optional[DocumentType] = None,
        raw_text: Optional[str] = None,
        confidence: float = 1.0,
    ) -> FieldResult:
        """Validate a single extracted field.

        The validation pipeline runs in three stages:
          1. **Regex check** — does the value match the expected pattern?
          2. **Business rules** — do semantic constraints hold?
          3. **Checksum validation** — for identifier fields (Steuer-ID).

        If all stages pass, the field status is ``EXTRACTED``.  If any
        stage fails, the status is ``VALIDATION_FAILURE`` and the
        ``validation_message`` explains why.

        Parameters
        ----------
        field_name:
            Schema attribute name (e.g. ``"postleitzahl"``).
        value:
            The extracted value.  May be ``None`` (UNRESOLVED).
        document_type:
            Document type for context-specific rule selection.
        raw_text:
            Raw text exactly as seen by the VLM.
        confidence:
            Confidence score from the extraction engine.

        Returns
        -------
        FieldResult
            A fully populated :class:`FieldResult` with the appropriate
            ``status`` and ``validation_message``.
        """
        # Stage 0: None value → UNRESOLVED
        if value is None:
            return FieldResult(
                field_name=field_name,
                value=None,
                status=FieldStatus.UNRESOLVED,
                confidence=confidence,
                raw_text=raw_text,
                validation_message="Feld konnte nicht extrahiert werden.",
            )

        # Stage 1: Regex pattern check
        pattern = regex_patterns.get(field_name)
        if pattern is not None:
            value_str = str(value) if not isinstance(value, str) else value
            if not re.match(pattern, value_str):
                return FieldResult(
                    field_name=field_name,
                    value=value,
                    status=FieldStatus.VALIDATION_FAILURE,
                    confidence=confidence,
                    raw_text=raw_text,
                    validation_message=(
                        f"Wert '{value_str}' entspricht nicht dem erwarteten "
                        f"Format fuer '{field_name}'."
                    ),
                )

        # Stage 2: Business rules
        passed, message = business_rules(field_name, value, document_type)
        if not passed:
            return FieldResult(
                field_name=field_name,
                value=value,
                status=FieldStatus.VALIDATION_FAILURE,
                confidence=confidence,
                raw_text=raw_text,
                validation_message=message,
            )

        # All checks passed
        return FieldResult(
            field_name=field_name,
            value=value,
            status=FieldStatus.EXTRACTED,
            confidence=confidence,
            raw_text=raw_text,
            validation_message=None,
        )

    def validate_document(
        self,
        document_type: DocumentType,
        fields: dict[str, Any],
    ) -> dict[str, FieldResult]:
        """Validate all fields of a document.

        In addition to per-field validation, this method runs
        cross-field business rules such as "net salary must not exceed
        gross salary".

        Parameters
        ----------
        document_type:
            The classified document type.
        fields:
            Mapping of field name → extracted value.

        Returns
        -------
        dict[str, FieldResult]
            Mapping of field name → validated :class:`FieldResult`.
        """
        results: dict[str, FieldResult] = {}

        for field_name, value in fields.items():
            results[field_name] = self.validate_field(
                field_name=field_name,
                value=value,
                document_type=document_type,
            )

        # Cross-field validation: net salary ≤ gross salary
        if document_type == DocumentType.GEHALTSAUSWEIS:
            results = self._cross_validate_salary(results)

        # Cross-field validation: PLZ matches city (stub)
        if document_type == DocumentType.MELDEBESCHEINIGUNG:
            results = self._cross_validate_address(results)

        return results

    def _cross_validate_salary(
        self, results: dict[str, FieldResult]
    ) -> dict[str, FieldResult]:
        """Ensure net salary does not exceed gross salary."""
        brutto = results.get("brutto_lohn")
        netto = results.get("netto_lohn")

        if brutto is None or netto is None:
            return results

        b_val = brutto.value
        n_val = netto.value

        if b_val is None or n_val is None:
            return results

        if isinstance(n_val, (int, float)) and isinstance(b_val, (int, float)):
            if n_val > b_val:
                results["netto_lohn"] = FieldResult(
                    field_name="netto_lohn",
                    value=n_val,
                    status=FieldStatus.VALIDATION_FAILURE,
                    confidence=netto.confidence,
                    raw_text=netto.raw_text,
                    validation_message=(
                        f"Nettolohn ({n_val}) uebersteigt Bruttolohn ({b_val})."
                    ),
                )

        return results

    def _cross_validate_address(
        self, results: dict[str, FieldResult]
    ) -> dict[str, FieldResult]:
        """Cross-validate that PLZ and city are consistent.

        This is currently a stub — a full implementation would require
        a PLZ→city lookup database.  The stub documents the expected
        interface.
        """
        plz_field = results.get("postleitzahl")
        city_field = results.get("wohnort")

        if plz_field is None or city_field is None:
            return results

        plz_val = plz_field.value
        city_val = city_field.value

        if plz_val is None or city_val is None:
            return results

        # TODO: Implement PLZ → city cross-reference lookup.
        # For now, if both fields are EXTRACTED, we accept them.
        # A production system would query a PLZ database here.

        return results
