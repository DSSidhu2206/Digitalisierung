"""
Tests for the layout-aware field mapper and the remediated validators.

These exercise the *new* extraction core — turning OCR lines into canonical
schema fields — and the correctness fixes, without requiring the ML stack
(Surya / llama.cpp / numpy). They run on the synthetic layouts below, which
mimic how German forms place labels and values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from app.models.enums import DocumentType, FieldStatus
from app.vision.field_mapper import SchemaFieldMapper, parse_german_amount
from app.validators.checksums import validate_steuer_id
from app.validators.german_validators import GermanStreetValidator
from app.validators.symbolic_rules import SymbolicValidator


@dataclass
class _Line:
    text: str
    confidence: float = 0.9
    bbox: Optional[dict] = None


@dataclass
class _Extraction:
    lines: list


def _bb(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


@pytest.fixture
def mapper():
    return SchemaFieldMapper()


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------

def test_inline_label_value(mapper):
    ext = _Extraction([_Line("Familienname: Müller", 0.95, _bb(0.1, 0.2, 0.5, 0.24))])
    out = mapper.map(ext, DocumentType.MELDEBESCHEINIGUNG)
    assert out["familienname"].value == "Müller"
    assert out["familienname"].source == "inline"


def test_spatial_right_association(mapper):
    ext = _Extraction([
        _Line("Vorname", 0.93, _bb(0.10, 0.30, 0.28, 0.34)),
        _Line("Hans", 0.92, _bb(0.35, 0.30, 0.50, 0.34)),
    ])
    out = mapper.map(ext, DocumentType.MELDEBESCHEINIGUNG)
    assert out["vorname"].value == "Hans"
    assert out["vorname"].source == "spatial-right"


def test_spatial_below_association(mapper):
    ext = _Extraction([
        _Line("Geburtsdatum", 0.90, _bb(0.10, 0.40, 0.34, 0.44)),
        _Line("15.03.1985", 0.91, _bb(0.10, 0.46, 0.34, 0.50)),
    ])
    out = mapper.map(ext, DocumentType.MELDEBESCHEINIGUNG)
    assert out["geburtsdatum"].value == "15.03.1985"
    assert out["geburtsdatum"].source == "spatial-below"


def test_plz_pattern_fallback(mapper):
    ext = _Extraction([_Line("10115 Berlin", 0.88, _bb(0.10, 0.66, 0.40, 0.70))])
    out = mapper.map(ext, DocumentType.MELDEBESCHEINIGUNG)
    assert out["postleitzahl"].value == "10115"
    assert out["postleitzahl"].source == "pattern"


def test_steuer_id_pattern_with_spaces(mapper):
    ext = _Extraction([
        _Line("Steueridentifikationsnummer", 0.9, _bb(0.10, 0.10, 0.45, 0.14)),
        _Line("86 095 742 719", 0.9, _bb(0.50, 0.10, 0.85, 0.14)),
    ])
    out = mapper.map(ext, DocumentType.STEUERBESCHEID)
    assert out["steueridentifikationsnummer"].value == "86095742719"


def test_unmapped_lines_preserved_low_confidence(mapper):
    ext = _Extraction([_Line("Some unrelated footer text", 0.9, _bb(0.1, 0.9, 0.9, 0.94))])
    out = mapper.map(ext, DocumentType.MELDEBESCHEINIGUNG)
    assert any(k.startswith("_line_") for k in out)


@pytest.mark.parametrize("text,expected", [
    ("5.200,00 €", 5200.0),
    ("45.200,00 EUR", 45200.0),
    ("3200,00", 3200.0),
    ("1.234.567,89", 1234567.89),
    ("42", 42.0),
    ("5.200", 5200.0),
])
def test_parse_german_amount(text, expected):
    assert parse_german_amount(text) == expected


# ---------------------------------------------------------------------------
# Mapping → symbolic validation integration
# ---------------------------------------------------------------------------

def test_valid_steuerbescheid_validates(mapper):
    ext = _Extraction([
        _Line("Steueridentifikationsnummer:  86095742719", 0.9, _bb(0.1, 0.1, 0.8, 0.14)),
        _Line("zu versteuerndes Einkommen:  45.200,00 EUR", 0.9, _bb(0.1, 0.3, 0.8, 0.34)),
    ])
    out = mapper.map(ext, DocumentType.STEUERBESCHEID)
    valued = {n: f.value for n, f in out.items() if not n.startswith("_line_")}
    validated = SymbolicValidator().validate_document(DocumentType.STEUERBESCHEID, valued)
    assert validated["steueridentifikationsnummer"].status == FieldStatus.EXTRACTED


def test_invalid_steuer_id_fails_validation(mapper):
    ext = _Extraction([_Line("Steueridentifikationsnummer:  12345678901", 0.9, _bb(0.1, 0.1, 0.8, 0.14))])
    out = mapper.map(ext, DocumentType.STEUERBESCHEID)
    valued = {n: f.value for n, f in out.items() if not n.startswith("_line_")}
    validated = SymbolicValidator().validate_document(DocumentType.STEUERBESCHEID, valued)
    assert validated["steueridentifikationsnummer"].status == FieldStatus.VALIDATION_FAILURE


# ---------------------------------------------------------------------------
# Remediated validators
# ---------------------------------------------------------------------------

def test_steuer_id_iso7064_known_valid():
    # 86095742719 is a well-known structurally + checksum valid IdNr.
    assert validate_steuer_id("86095742719") is True


def test_steuer_id_rejects_corrupted_check_digit():
    assert validate_steuer_id("86095742718") is False


def test_steuer_id_rejects_leading_zero():
    assert validate_steuer_id("06095742719") is False


@pytest.mark.parametrize("street,ok", [
    ("Hauptstraße", True),
    ("Am See", True),                 # multi-word: the \\s regex-bug fix
    ("Straße des 17. Juni", True),    # digits allowed inside a street name
    ("von-Bülow-Straße", True),
    ("12345", False),                 # pure digits: not a street
    ("....", False),
])
def test_street_validator(street, ok):
    assert GermanStreetValidator.validate(street) is ok
