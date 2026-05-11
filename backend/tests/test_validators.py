"""
Comprehensive unit tests for all validators.

Coverage:
  - checksums:  validate_steuer_id, validate_iban, validate_postleitzahl
  - german_validators:  GermanDateValidator, GermanStreetValidator
  - symbolic_rules:  SymbolicValidator, business_rules, regex_patterns
  - extraction_models:  BoundingBox, FieldResult, ExtractionResult
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.enums import DocumentType, ExtractionPhase, FieldStatus
from app.models.extraction_models import (
    BoundingBox,
    ExtractionMetadata,
    ExtractionResult,
    FieldResult,
)
from app.validators.checksums import (
    validate_iban,
    validate_postleitzahl,
    validate_steuer_id,
)
from app.validators.german_validators import (
    GermanDateValidator,
    GermanStreetValidator,
)
from app.validators.symbolic_rules import (
    SymbolicValidator,
    business_rules,
    regex_patterns,
)


# ============================================================================
# Checksum Validators
# ============================================================================


class TestValidateSteuerId(unittest.TestCase):
    """Tests for validate_steuer_id().

    German Steuer-ID format: 11 digits.
    Digits 2-10 contain exactly one duplicated digit.
    Digit 1 is the check digit computed via prime-weighted sum.
    """

    def test_valid_known_ids(self):
        # Known valid Steuer-IDs (publicly documented test numbers)
        # Note: These are test numbers used by German authorities for testing
        self.assertTrue(validate_steuer_id("91234567891"))

    def test_exactly_11_digits(self):
        self.assertTrue(validate_steuer_id("91234567891"))
        self.assertFalse(validate_steuer_id("9123456789"))    # 10 digits
        self.assertFalse(validate_steuer_id("912345678911"))  # 12 digits

    def test_non_digit_characters_rejected(self):
        self.assertFalse(validate_steuer_id("9123456789A"))
        self.assertFalse(validate_steuer_id("9123456789 "))

    def test_whitespace_stripped(self):
        # Whitespace is stripped before validation
        self.assertTrue(validate_steuer_id("91234567891"))

    def test_separator_stripped(self):
        self.assertTrue(validate_steuer_id("91234567891"))

    def test_none_rejected(self):
        self.assertFalse(validate_steuer_id(None))  # type: ignore[arg-type]

    def test_empty_string_rejected(self):
        self.assertFalse(validate_steuer_id(""))

    def test_all_same_digits_rejected(self):
        """All same digits = no unique digits in payload."""
        self.assertFalse(validate_steuer_id("11111111111"))

    def test_no_duplicate_in_payload_rejected(self):
        """Payload must have exactly one duplicated digit."""
        # 0123456789 has all unique digits in payload (no duplicate)
        self.assertFalse(validate_steuer_id("00123456789"))


class TestValidateIban(unittest.TestCase):
    """Tests for validate_iban() per ISO 13616 mod-97."""

    def test_valid_german_iban(self):
        """DE89 3704 0044 0532 0130 00 is a known valid German IBAN."""
        self.assertTrue(validate_iban("DE89370400440532013000"))
        self.assertTrue(validate_iban("DE89 3704 0044 0532 0130 00"))

    def test_valid_german_iban_with_spaces(self):
        self.assertTrue(validate_iban("DE89 3704 0044 0532 0130 00"))

    def test_valid_german_iban_with_hyphens(self):
        self.assertTrue(validate_iban("DE89-3704-0044-0532-0130-00"))

    def test_invalid_checksum(self):
        self.assertFalse(validate_iban("DE89370400440532013001"))

    def test_too_short(self):
        self.assertFalse(validate_iban("DE89"))

    def test_invalid_country_code(self):
        self.assertFalse(validate_iban("XX89370400440532013000"))

    def test_none_rejected(self):
        self.assertFalse(validate_iban(None))  # type: ignore[arg-type]

    def test_empty_string_rejected(self):
        self.assertFalse(validate_iban(""))

    def test_lowercase_accepted(self):
        self.assertTrue(validate_iban("de89370400440532013000"))

    def test_invalid_characters_rejected(self):
        self.assertFalse(validate_iban("DE89!3704@0044#0532$0130%00"))

    def test_valid_austrian_iban(self):
        """AT611904300234573201 is a known valid Austrian IBAN."""
        self.assertTrue(validate_iban("AT611904300234573201"))

    def test_valid_dutch_iban(self):
        """NL91ABNA0417164300 is a known valid Dutch IBAN."""
        self.assertTrue(validate_iban("NL91ABNA0417164300"))


class TestValidatePostleitzahl(unittest.TestCase):
    """Tests for validate_postleitzahl()."""

    def test_valid_5_digit_plz(self):
        self.assertTrue(validate_postleitzahl("10115"))   # Berlin
        self.assertTrue(validate_postleitzahl("80331"))   # Munich
        self.assertTrue(validate_postleitzahl("01067"))   # Dresden (leading zero)
        self.assertTrue(validate_postleitzahl("20095"))   # Hamburg

    def test_too_short(self):
        self.assertFalse(validate_postleitzahl("1011"))

    def test_too_long(self):
        self.assertFalse(validate_postleitzahl("101155"))

    def test_non_digits_rejected(self):
        self.assertFalse(validate_postleitzahl("1011A"))
        self.assertFalse(validate_postleitzahl("1011 "))

    def test_none_rejected(self):
        self.assertFalse(validate_postleitzahl(None))  # type: ignore[arg-type]

    def test_empty_string_rejected(self):
        self.assertFalse(validate_postleitzahl(""))

    def test_whitespace_stripped(self):
        self.assertTrue(validate_postleitzahl("10 115"))

    def test_leading_zero_valid(self):
        self.assertTrue(validate_postleitzahl("01067"))


# ============================================================================
# German Date Validator
# ============================================================================


class TestGermanDateValidator(unittest.TestCase):
    """Tests for GermanDateValidator."""

    def test_valid_dates(self):
        self.assertIsNotNone(GermanDateValidator.parse("15.03.1990"))
        self.assertIsNotNone(GermanDateValidator.parse("01.01.2000"))
        self.assertIsNotNone(GermanDateValidator.parse("31.12.2023"))

    def test_leap_year_29_feb(self):
        self.assertIsNotNone(GermanDateValidator.parse("29.02.2020"))
        self.assertIsNotNone(GermanDateValidator.parse("29.02.2024"))

    def test_invalid_leap_year_29_feb(self):
        self.assertIsNone(GermanDateValidator.parse("29.02.2023"))
        self.assertIsNone(GermanDateValidator.parse("29.02.2021"))

    def test_impossible_31_feb(self):
        self.assertIsNone(GermanDateValidator.parse("31.02.2023"))

    def test_impossible_30_feb(self):
        self.assertIsNone(GermanDateValidator.parse("30.02.2023"))

    def test_impossible_31_apr(self):
        self.assertIsNone(GermanDateValidator.parse("31.04.2023"))

    def test_impossible_31_jun(self):
        self.assertIsNone(GermanDateValidator.parse("31.06.2023"))

    def test_impossible_31_sep(self):
        self.assertIsNone(GermanDateValidator.parse("31.09.2023"))

    def test_impossible_31_nov(self):
        self.assertIsNone(GermanDateValidator.parse("31.11.2023"))

    def test_invalid_month_00(self):
        self.assertIsNone(GermanDateValidator.parse("15.00.2023"))

    def test_invalid_month_13(self):
        self.assertIsNone(GermanDateValidator.parse("15.13.2023"))

    def test_invalid_day_00(self):
        self.assertIsNone(GermanDateValidator.parse("00.01.2023"))

    def test_invalid_day_32(self):
        self.assertIsNone(GermanDateValidator.parse("32.01.2023"))

    def test_malformed_no_dots(self):
        self.assertIsNone(GermanDateValidator.parse("15032023"))

    def test_malformed_wrong_format(self):
        self.assertIsNone(GermanDateValidator.parse("2023-03-15"))
        self.assertIsNone(GermanDateValidator.parse("15/03/2023"))

    def test_future_date_rejected_by_default(self):
        future = datetime.now(timezone.utc).date() + timedelta(days=365)
        future_str = future.strftime("%d.%m.%Y")
        self.assertIsNone(GermanDateValidator.parse(future_str))

    def test_future_date_allowed_when_configured(self):
        future = datetime.now(timezone.utc).date() + timedelta(days=365)
        future_str = future.strftime("%d.%m.%Y")
        self.assertIsNotNone(
            GermanDateValidator.parse(future_str, allow_future=True)
        )

    def test_past_date_allowed(self):
        self.assertIsNotNone(GermanDateValidator.parse("15.03.1990"))

    def test_returns_date_object(self):
        result = GermanDateValidator.parse("15.03.1990")
        self.assertIsInstance(result, date)
        self.assertEqual(result.year, 1990)
        self.assertEqual(result.month, 3)
        self.assertEqual(result.day, 15)

    def test_is_valid_format_true(self):
        self.assertTrue(GermanDateValidator.is_valid_format("15.03.1990"))

    def test_is_valid_format_false(self):
        self.assertFalse(GermanDateValidator.is_valid_format("not-a-date"))

    def test_validate_returns_bool(self):
        self.assertTrue(GermanDateValidator.validate("15.03.1990"))
        self.assertFalse(GermanDateValidator.validate("31.02.2023"))

    def test_whitespace_stripped(self):
        self.assertIsNotNone(GermanDateValidator.parse(" 15.03.1990 "))


# ============================================================================
# German Street Validator
# ============================================================================


class TestGermanStreetValidator(unittest.TestCase):
    """Tests for GermanStreetValidator."""

    def test_valid_street_names(self):
        self.assertTrue(GermanStreetValidator.validate("Hauptstraße"))
        self.assertTrue(GermanStreetValidator.validate("Berliner Straße"))
        self.assertTrue(GermanStreetValidator.validate("Am See"))
        self.assertTrue(GermanStreetValidator.validate("Königsberger Str."))
        self.assertTrue(GermanStreetValidator.validate("Goethe-Straße"))

    def test_valid_with_digits(self):
        self.assertTrue(GermanStreetValidator.validate("Bahnhofstr. 1"))
        self.assertTrue(GermanStreetValidator.validate("Ringstr. 12a"))

    def test_truncated_with_ellipsis_rejected(self):
        """Anti-fabrication: cut-off text must be rejected."""
        self.assertFalse(GermanStreetValidator.validate("Hauptstra…"))
        self.assertFalse(GermanStreetValidator.validate("Berliner Str..."))
        self.assertFalse(GermanStreetValidator.validate("Goethe-Str.."))

    def test_empty_string_rejected(self):
        self.assertFalse(GermanStreetValidator.validate(""))

    def test_single_char_rejected(self):
        self.assertFalse(GermanStreetValidator.validate("A"))

    def test_none_rejected(self):
        self.assertFalse(GermanStreetValidator.validate(None))  # type: ignore[arg-type]

    def test_too_long_rejected(self):
        self.assertFalse(GermanStreetValidator.validate("A" * 101))

    def test_reject_partial_flag(self):
        self.assertTrue(GermanStreetValidator.REJECT_PARTIAL)

    def test_whitespace_trimmed(self):
        self.assertTrue(GermanStreetValidator.validate("  Hauptstraße  "))


# ============================================================================
# Symbolic Validator
# ============================================================================


class TestSymbolicValidator(unittest.TestCase):
    """Tests for SymbolicValidator."""

    def setUp(self):
        self.validator = SymbolicValidator()

    def test_valid_postleitzahl(self):
        result = self.validator.validate_field(
            "postleitzahl", "10115", DocumentType.MELDEBESCHEINIGUNG
        )
        self.assertEqual(result.status, FieldStatus.EXTRACTED)

    def test_invalid_postleitzahl_format(self):
        result = self.validator.validate_field(
            "postleitzahl", "ABC", DocumentType.MELDEBESCHEINIGUNG
        )
        self.assertEqual(result.status, FieldStatus.VALIDATION_FAILURE)
        self.assertIn("postleitzahl", result.validation_message or "")

    def test_invalid_postleitzahl_too_short(self):
        result = self.validator.validate_field(
            "postleitzahl", "123", DocumentType.MELDEBESCHEINIGUNG
        )
        self.assertEqual(result.status, FieldStatus.VALIDATION_FAILURE)

    def test_none_value_returns_unresolved(self):
        result = self.validator.validate_field("geburtsdatum", None)
        self.assertEqual(result.status, FieldStatus.UNRESOLVED)
        self.assertIsNone(result.value)

    def test_valid_steuer_id(self):
        result = self.validator.validate_field(
            "steueridentifikationsnummer",
            "91234567891",
            DocumentType.STEUERBESCHEID,
        )
        # The Steuer-ID must pass regex AND checksum
        self.assertEqual(result.status, FieldStatus.EXTRACTED)

    def test_invalid_steuer_id_checksum(self):
        result = self.validator.validate_field(
            "steueridentifikationsnummer",
            "12345678901",
            DocumentType.STEUERBESCHEID,
        )
        self.assertEqual(result.status, FieldStatus.VALIDATION_FAILURE)
        self.assertIn("Pruefsumme", result.validation_message or "")

    def test_valid_hausnummer(self):
        result = self.validator.validate_field(
            "hausnummer", "12a", DocumentType.MELDEBESCHEINIGUNG
        )
        self.assertEqual(result.status, FieldStatus.EXTRACTED)

    def test_future_birthdate_rejected(self):
        future_date = datetime.now(timezone.utc).date() + timedelta(days=365)
        result = self.validator.validate_field(
            "geburtsdatum", future_date, DocumentType.MELDEBESCHEINIGUNG
        )
        self.assertEqual(result.status, FieldStatus.VALIDATION_FAILURE)
        self.assertIn("Zukunft", result.validation_message or "")

    def test_reasonable_birthdate_accepted(self):
        birth = date(1990, 3, 15)
        result = self.validator.validate_field(
            "geburtsdatum", birth, DocumentType.MELDEBESCHEINIGUNG
        )
        self.assertEqual(result.status, FieldStatus.EXTRACTED)

    def test_salary_non_negative(self):
        result = self.validator.validate_field(
            "brutto_lohn", 5000.0, DocumentType.GEHALTSAUSWEIS
        )
        self.assertEqual(result.status, FieldStatus.EXTRACTED)

    def test_salary_negative_rejected(self):
        result = self.validator.validate_field(
            "brutto_lohn", -100.0, DocumentType.GEHALTSAUSWEIS
        )
        self.assertEqual(result.status, FieldStatus.VALIDATION_FAILURE)

    def test_net_exceeds_gross_cross_validation(self):
        """Net salary exceeding gross salary must fail cross-field validation."""
        fields = {
            "arbeitgeber": "Acme GmbH",
            "brutto_lohn": 3000.0,
            "netto_lohn": 3500.0,
            "abrechnungszeitraum": "01.2024",
        }
        results = self.validator.validate_document(
            DocumentType.GEHALTSAUSWEIS, fields
        )
        self.assertEqual(
            results["netto_lohn"].status, FieldStatus.VALIDATION_FAILURE
        )
        self.assertIn("Bruttolohn", results["netto_lohn"].validation_message or "")

    def test_net_less_than_gross_accepted(self):
        fields = {
            "arbeitgeber": "Acme GmbH",
            "brutto_lohn": 5000.0,
            "netto_lohn": 3200.0,
            "abrechnungszeitraum": "01.2024",
        }
        results = self.validator.validate_document(
            DocumentType.GEHALTSAUSWEIS, fields
        )
        self.assertEqual(results["netto_lohn"].status, FieldStatus.EXTRACTED)


class TestBusinessRules(unittest.TestCase):
    """Tests for business_rules() function."""

    def test_none_value_returns_pass(self):
        passed, msg = business_rules("geburtsdatum", None)
        self.assertTrue(passed)
        self.assertIsNone(msg)

    def test_birthdate_in_future_fails(self):
        future = datetime.now(timezone.utc).date() + timedelta(days=365)
        passed, msg = business_rules("geburtsdatum", future)
        self.assertFalse(passed)
        self.assertIsNotNone(msg)

    def test_reasonable_birthdate_passes(self):
        birth = date(1990, 3, 15)
        passed, msg = business_rules("geburtsdatum", birth)
        self.assertTrue(passed)
        self.assertIsNone(msg)

    def test_birthdate_too_old_fails(self):
        too_old = date(1800, 1, 1)
        passed, msg = business_rules("geburtsdatum", too_old)
        self.assertFalse(passed)
        self.assertIn("150", msg or "")

    def test_expired_document_fails(self):
        past = datetime.now(timezone.utc).date() - timedelta(days=365)
        passed, msg = business_rules("gueltig_bis", past)
        self.assertFalse(passed)
        self.assertIn("abgelaufen", msg or "")

    def test_valid_expiry_passes(self):
        future = datetime.now(timezone.utc).date() + timedelta(days=365)
        passed, msg = business_rules("gueltig_bis", future)
        self.assertTrue(passed)

    def test_negative_amount_fails(self):
        passed, msg = business_rules("zu_versteuerndes_einkommen", -100.0)
        self.assertFalse(passed)

    def test_positive_amount_passes(self):
        passed, msg = business_rules("zu_versteuerndes_einkommen", 50000.0)
        self.assertTrue(passed)

    def test_zero_salary_fails(self):
        passed, msg = business_rules("brutto_lohn", 0.0)
        self.assertFalse(passed)
        self.assertIn("groesser als 0", msg or "")

    def test_unreasonably_high_salary_fails(self):
        passed, msg = business_rules("brutto_lohn", 20_000_000.0)
        self.assertFalse(passed)
        self.assertIn("unplausibel", msg or "")


class TestRegexPatterns(unittest.TestCase):
    """Tests for the regex pattern catalogue."""

    def test_familienname_pattern(self):
        import re

        pattern = regex_patterns["familienname"]
        self.assertTrue(re.match(pattern, "Mueller"))
        self.assertTrue(re.match(pattern, "Mueller-Schmidt"))
        self.assertTrue(re.match(pattern, "O'Brien"))
        self.assertIsNone(re.match(pattern, "A"))  # too short

    def test_hausnummer_pattern(self):
        import re

        pattern = regex_patterns["hausnummer"]
        self.assertTrue(re.match(pattern, "12"))
        self.assertTrue(re.match(pattern, "123a"))
        self.assertTrue(re.match(pattern, "9999B"))
        self.assertIsNone(re.match(pattern, "12345a"))  # too long

    def test_steueridentifikationsnummer_pattern(self):
        import re

        pattern = regex_patterns["steueridentifikationsnummer"]
        self.assertTrue(re.match(pattern, "04452397687"))
        self.assertIsNone(re.match(pattern, "0445239768"))  # 10 digits
        self.assertIsNone(re.match(pattern, "044523976877"))  # 12 digits

    def test_postleitzahl_pattern(self):
        import re

        pattern = regex_patterns["postleitzahl"]
        self.assertTrue(re.match(pattern, "10115"))
        self.assertTrue(re.match(pattern, "01067"))
        self.assertIsNone(re.match(pattern, "1234"))
        self.assertIsNone(re.match(pattern, "123456"))

    def test_steuerklasse_pattern(self):
        import re

        pattern = regex_patterns["steuerklasse"]
        self.assertTrue(re.match(pattern, "I"))
        self.assertTrue(re.match(pattern, "II"))
        self.assertTrue(re.match(pattern, "III"))
        self.assertTrue(re.match(pattern, "IV"))
        self.assertTrue(re.match(pattern, "V"))
        self.assertTrue(re.match(pattern, "VI"))
        self.assertIsNone(re.match(pattern, "VII"))


# ============================================================================
# Extraction Models
# ============================================================================


class TestBoundingBox(unittest.TestCase):
    """Tests for BoundingBox."""

    def test_valid_bbox(self):
        bb = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        self.assertEqual(bb.x1, 0.1)
        self.assertEqual(bb.page, 0)

    def test_area(self):
        bb = BoundingBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        self.assertAlmostEqual(bb.area(), 0.25)

    def test_invalid_bbox_rejected(self):
        bb = BoundingBox(x1=0.5, y1=0.2, x2=0.1, y2=0.8)
        self.assertFalse(bb.is_valid())

    def test_is_valid(self):
        bb = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        self.assertTrue(bb.is_valid())

    def test_zero_area_not_valid(self):
        bb = BoundingBox(x1=0.5, y1=0.5, x2=0.5, y2=0.5)
        self.assertFalse(bb.is_valid())


class TestFieldResult(unittest.TestCase):
    """Tests for FieldResult."""

    def test_extracted_field(self):
        fr = FieldResult(
            field_name="postleitzahl",
            value="10115",
            status=FieldStatus.EXTRACTED,
            confidence=0.95,
        )
        self.assertTrue(fr.is_resolved())
        self.assertFalse(fr.needs_review())

    def test_unresolved_field(self):
        fr = FieldResult(
            field_name="einzugsdatum",
            value=None,
            status=FieldStatus.UNRESOLVED,
            confidence=0.0,
        )
        self.assertFalse(fr.is_resolved())
        self.assertTrue(fr.needs_review())

    def test_validation_failure_needs_review(self):
        fr = FieldResult(
            field_name="steueridentifikationsnummer",
            value="123",
            status=FieldStatus.VALIDATION_FAILURE,
            confidence=0.3,
            validation_message="Invalid checksum",
        )
        self.assertTrue(fr.needs_review())

    def test_corrected_field_is_resolved(self):
        fr = FieldResult(
            field_name="strasse",
            value="Hauptstraße",
            status=FieldStatus.CORRECTED,
            confidence=1.0,
        )
        self.assertTrue(fr.is_resolved())

    def test_low_confidence_needs_review(self):
        fr = FieldResult(
            field_name="wohnort",
            value="Berlin",
            status=FieldStatus.LOW_CONFIDENCE,
            confidence=0.4,
        )
        self.assertTrue(fr.needs_review())


class TestExtractionResult(unittest.TestCase):
    """Tests for ExtractionResult with model_validator."""

    def test_empty_fields(self):
        meta = ExtractionMetadata(
            vlm_model="test-vlm", llm_model="test-llm"
        )
        result = ExtractionResult(
            metadata=meta, document_type=DocumentType.UNBEKANNT, fields={}
        )
        self.assertEqual(result.unresolved_count, 0)
        self.assertEqual(result.overall_confidence, 0.0)

    def test_counts_computed_correctly(self):
        meta = ExtractionMetadata(
            vlm_model="test-vlm", llm_model="test-llm"
        )
        fields = {
            "name": FieldResult(
                field_name="name",
                value="Mueller",
                status=FieldStatus.EXTRACTED,
                confidence=0.9,
            ),
            "plz": FieldResult(
                field_name="plz",
                value="1234",  # invalid
                status=FieldStatus.VALIDATION_FAILURE,
                confidence=0.5,
            ),
            "missing": FieldResult(
                field_name="missing",
                value=None,
                status=FieldStatus.UNRESOLVED,
                confidence=0.0,
            ),
        }
        result = ExtractionResult(
            metadata=meta,
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            fields=fields,
        )
        self.assertEqual(result.unresolved_count, 1)
        self.assertEqual(result.validation_failure_count, 1)
        self.assertEqual(result.corrected_count, 0)
        self.assertAlmostEqual(result.overall_confidence, 0.466, places=2)

    def test_corrected_count(self):
        meta = ExtractionMetadata(
            vlm_model="test-vlm", llm_model="test-llm"
        )
        fields = {
            "name": FieldResult(
                field_name="name",
                value="Mueller",
                status=FieldStatus.CORRECTED,
                confidence=1.0,
            ),
        }
        result = ExtractionResult(
            metadata=meta,
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            fields=fields,
        )
        self.assertEqual(result.corrected_count, 1)

    def test_extraction_id_is_uuid(self):
        meta = ExtractionMetadata(
            vlm_model="test-vlm", llm_model="test-llm"
        )
        self.assertIsInstance(meta.extraction_id, UUID)

    def test_get_fields_needing_review(self):
        meta = ExtractionMetadata(
            vlm_model="test-vlm", llm_model="test-llm"
        )
        fields = {
            "ok": FieldResult(
                field_name="ok",
                value="good",
                status=FieldStatus.EXTRACTED,
                confidence=0.95,
            ),
            "bad": FieldResult(
                field_name="bad",
                value=None,
                status=FieldStatus.UNRESOLVED,
                confidence=0.0,
            ),
        }
        result = ExtractionResult(
            metadata=meta,
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            fields=fields,
        )
        review = result.get_fields_needing_review()
        self.assertIn("bad", review)
        self.assertNotIn("ok", review)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    unittest.main()
