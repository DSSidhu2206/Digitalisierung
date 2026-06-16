"""
Layout-aware field mapper — turns raw OCR lines into canonical schema fields.

The previous pipeline dumped OCR lines as ``line_0001`` / naive colon-splits and
never populated the strict German document schemas.  This module closes that gap:
given a normalised OCR/layout extraction (Surya text lines with bounding boxes)
and a :class:`DocumentType`, it produces canonical schema field names
(``familienname``, ``postleitzahl``, ``steueridentifikationsnummer`` …) using:

1. **Inline ``Label: Value``** detection on a single line.
2. **Spatial association** — for a recognised label with no inline value, find the
   value to its right (same row) or directly below it, using bounding boxes.
   This is how real German forms lay out label → value.
3. **Value-pattern fallback** — PLZ (5 digits), dates (DD.MM.YYYY), Steuer-ID
   (11 digits), IBAN and currency amounts are recognised by shape and attached to
   the most plausible field even when the label OCR is weak.

Anti-fabrication: nothing is auto-completed or guessed.  A field is only emitted
when a value is actually read from the image.  Unmapped lines are preserved as
low-confidence ``_line_*`` review items so no text is silently dropped.

The module is intentionally dependency-light (stdlib + the OCR dataclasses via
duck typing) so it can be unit-tested without the ML stack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from app.models.enums import DocumentType


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MappedField:
    """One mapped value, ready to be wrapped in a FieldResult."""

    field_name: str
    value: Any
    confidence: float
    raw_text: str
    bbox: Optional[dict[str, float]] = None
    source: str = "inline"  # inline | spatial-right | spatial-below | pattern | unmapped


# ---------------------------------------------------------------------------
# Canonical label dictionaries per document type
# ---------------------------------------------------------------------------
# Keys are canonical schema field names; values are lowercase label synonyms.
# Matching is substring-based on alphanumeric-normalised text, longest-first.

_LABELS: dict[DocumentType, dict[str, list[str]]] = {
    DocumentType.MELDEBESCHEINIGUNG: {
        "familienname": ["familienname", "nachname", "name"],
        "vorname": ["vorname", "vornamen", "rufname"],
        "geburtsdatum": ["geburtsdatum", "geborenam", "gebam", "geburtstag"],
        "geburtsort": ["geburtsort", "geboreninin", "geborenin"],
        "staatsangehoerigkeit": ["staatsangehoerigkeit", "staatsangehörigkeit", "nationalitaet", "nationalität"],
        "strasse": ["strasse", "straße", "anschriftstrasse", "wohnungsanschrift"],
        "hausnummer": ["hausnummer", "hausnr", "nr"],
        "postleitzahl": ["postleitzahl", "plz"],
        "wohnort": ["wohnort", "ort", "stadt", "gemeinde", "wohnortstadt"],
        "einzugsdatum": ["einzugsdatum", "einzug", "eingezogenam", "zuzugsdatum", "zuzug"],
        "vorherige_anschrift": ["vorherigeanschrift", "vorigeanschrift", "fruehereanschrift", "letzteanschrift"],
    },
    DocumentType.STEUERBESCHEID: {
        "steueridentifikationsnummer": ["steueridentifikationsnummer", "steuerid", "idnr", "identifikationsnummer", "steuernummer"],
        "veranlagungszeitraum": ["veranlagungszeitraum", "veranlagung", "zeitraum", "steuerjahr", "jahr"],
        "zu_versteuerndes_einkommen": ["zuversteuerndeseinkommen", "zveinkommen", "versteuerndeseinkommen", "einkommen"],
        "festgesetzte_steuer": ["festgesetztesteuer", "festgesetzteeinkommensteuer", "festgesetzt", "einkommensteuer"],
        "ermässigung": ["ermaessigung", "ermäßigung", "steuerermaessigung"],
        "vorauszahlungen": ["vorauszahlungen", "vorauszahlung", "geleistetevorauszahlungen"],
        "kirchensteuer": ["kirchensteuer", "kist"],
        "solidaritätszuschlag": ["solidaritaetszuschlag", "solidaritätszuschlag", "soli", "solz"],
    },
    DocumentType.GEHALTSAUSWEIS: {
        "arbeitgeber": ["arbeitgeber", "firma", "unternehmen", "ag"],
        "brutto_lohn": ["bruttolohn", "bruttogehalt", "brutto", "gesamtbrutto", "bruttobezuege"],
        "netto_lohn": ["nettolohn", "nettogehalt", "netto", "auszahlungsbetrag", "nettoverdienst"],
        "abrechnungszeitraum": ["abrechnungszeitraum", "abrechnungsmonat", "lohnzeitraum", "zeitraum", "monat"],
        "steuerklasse": ["steuerklasse", "stkl", "lohnsteuerklasse", "klasse"],
        "steueridentifikationsnummer": ["steueridentifikationsnummer", "steuerid", "idnr", "identifikationsnummer"],
        "sozialversicherungsnummer": ["sozialversicherungsnummer", "svnummer", "rentenversicherungsnummer", "svnr"],
        "rentenversicherung": ["rentenversicherung", "rv"],
        "krankenversicherung": ["krankenversicherung", "kv"],
        "arbeitslosenversicherung": ["arbeitslosenversicherung", "av"],
        "pflegeversicherung": ["pflegeversicherung", "pv"],
    },
    DocumentType.PERSONALAUSWEIS: {
        "dokumentnummer": ["dokumentnummer", "ausweisnummer", "documentno", "dokumentnr", "cardno"],
        "familienname": ["familienname", "name", "surname", "nachname"],
        "vorname": ["vorname", "vornamen", "givenname", "givennames"],
        "geburtsdatum": ["geburtsdatum", "dateofbirth", "geborenam", "dob"],
        "geburtsort": ["geburtsort", "placeofbirth", "geborenin"],
        "staatsangehoerigkeit": ["staatsangehoerigkeit", "staatsangehörigkeit", "nationality", "nationalitaet"],
        "gueltig_bis": ["gueltigbis", "gültigbis", "dateofexpiry", "expiry", "validuntil"],
        "ausstellungsdatum": ["ausstellungsdatum", "dateofissue", "ausgestelltam"],
        "ausstellende_behoerde": ["ausstellendebehoerde", "ausstellendebehörde", "authority", "behoerde", "behörde"],
    },
}

# Fields whose values are monetary amounts (parsed to float).
_AMOUNT_FIELDS = {
    "zu_versteuerndes_einkommen", "festgesetzte_steuer", "ermässigung",
    "vorauszahlungen", "kirchensteuer", "solidaritätszuschlag",
    "brutto_lohn", "netto_lohn", "rentenversicherung", "krankenversicherung",
    "arbeitslosenversicherung", "pflegeversicherung",
}
_DATE_FIELDS = {"geburtsdatum", "gueltig_bis", "ausstellungsdatum", "einzugsdatum"}

# Value-shape patterns (used both to validate values and for label-free fallback).
_RE_DATE = re.compile(r"\b(0[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])\.((?:19|20)\d{2})\b")
_RE_PLZ = re.compile(r"\b(\d{5})\b")
_RE_STEUER_ID = re.compile(r"\b(\d{11})\b")
_RE_IBAN = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b")
_RE_AMOUNT = re.compile(r"(\d{1,3}(?:\.\d{3})+(?:,\d{2})?|\d+(?:,\d{2})?)\s*(?:€|EUR)?")
_RE_STEUERKLASSE = re.compile(r"\b(I{1,3}|IV|V|VI)\b")


def _norm(text: str) -> str:
    """Lowercase + keep alphanumerics only (for robust label matching)."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def parse_german_amount(text: str) -> Optional[float]:
    """Parse a German-formatted amount ('5.200,00 €') into a float."""
    match = _RE_AMOUNT.search(text)
    if not match:
        return None
    raw = match.group(1)
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        # No decimal comma: dots could be thousands separators.
        if raw.count(".") >= 1 and len(raw.split(".")[-1]) == 3:
            raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class SchemaFieldMapper:
    """Map OCR lines to canonical schema fields for a document type."""

    SAME_ROW_Y_TOL = 0.020          # vertical center tolerance for "same row"
    BELOW_MAX_CENTER_DELTA = 0.080  # max center-to-center distance for "below"
    BELOW_X_TOL = 0.080             # horizontal alignment tolerance for "below"

    SRC_CONFIDENCE = {
        "inline": 1.0,
        "spatial-right": 0.9,
        "spatial-below": 0.82,
        "pattern": 0.72,
    }

    def map(self, extraction: Any, document_type: DocumentType) -> dict[str, MappedField]:
        """Return canonical field name → :class:`MappedField`.

        Args:
            extraction: object exposing ``.lines`` (each with ``.text``,
                ``.confidence``, ``.bbox``) — e.g. a ``SuryaExtraction``.
            document_type: classified document type (selects the label set).
        """
        labels = _LABELS.get(document_type, {})
        lines = [ln for ln in (getattr(extraction, "lines", []) or [])
                 if str(getattr(ln, "text", "") or "").strip()]

        mapped: dict[str, MappedField] = {}
        consumed: set[int] = set()  # line indices already used as a value

        # Pass 1 — inline "Label: Value" on a single line.
        for idx, line in enumerate(lines):
            text = " ".join(str(line.text).split())
            key, value = self._split_inline(text)
            if not key or not value:
                continue
            canonical = self._match_label(key, labels, mapped)
            if not canonical:
                continue
            self._assign(mapped, canonical, value, line, "inline")
            consumed.add(idx)

        # Pass 2 — spatial association for labels with no inline value.
        for idx, line in enumerate(lines):
            if idx in consumed:
                continue
            text = " ".join(str(line.text).split())
            # A bare label line (no inline value).
            key, value = self._split_inline(text)
            label_text = key if (key and not value) else (text if not value else "")
            if not label_text:
                continue
            canonical = self._match_label(label_text, labels, mapped)
            if not canonical:
                continue
            value_idx = self._find_value_line(idx, lines, consumed, labels)
            if value_idx is None:
                continue
            vline = lines[value_idx]
            vtext = " ".join(str(vline.text).split())
            same_row = self._is_same_row(line, vline)
            self._assign(mapped, canonical, vtext, vline,
                         "spatial-right" if same_row else "spatial-below")
            consumed.add(value_idx)

        # Pass 3 — label-free value-shape fallback for still-missing key fields.
        self._pattern_fallback(lines, consumed, labels, mapped, document_type)

        # Pass 3b — card-specific: the surname on a Personalausweis is printed
        # WITHOUT a "Familienname:" label. Take the prominent uppercase name in
        # the data column (right of the photo) if familienname is still missing.
        if document_type == DocumentType.PERSONALAUSWEIS and "familienname" not in mapped:
            self._infer_card_surname(lines, consumed, mapped)

        # Pass 4 — preserve unmapped lines as low-confidence review items.
        for idx, line in enumerate(lines):
            if idx in consumed:
                continue
            text = " ".join(str(line.text).split())
            name = f"_line_{idx + 1:04d}"
            mapped[name] = MappedField(
                field_name=name,
                value=text,
                confidence=min(self._conf(line) * 0.5, 0.49),
                raw_text=text,
                bbox=getattr(line, "bbox", None),
                source="unmapped",
            )

        return mapped

    # -- assignment helpers --------------------------------------------------

    @staticmethod
    def _conf(line: Any) -> float:
        """Bounded OCR confidence for a line (default 0.85 when absent)."""
        try:
            return max(0.0, min(float(getattr(line, "confidence", 0.85)), 1.0))
        except (TypeError, ValueError):
            return 0.85

    def _assign(self, mapped: dict[str, MappedField], canonical: str,
                raw_value: str, line: Any, source: str) -> None:
        value = self._coerce_value(canonical, raw_value)
        if value is None:
            return
        confidence = min(self._conf(line) * self.SRC_CONFIDENCE.get(source, 0.7), 1.0)
        # Keep the highest-confidence reading if the field was already seen.
        existing = mapped.get(canonical)
        if existing is not None and existing.confidence >= confidence:
            return
        mapped[canonical] = MappedField(
            field_name=canonical,
            value=value,
            confidence=confidence,
            raw_text=raw_value,
            bbox=getattr(line, "bbox", None),
            source=source,
        )

    def _coerce_value(self, canonical: str, raw_value: str) -> Any:
        # Strip OCR markup tags (Surya emits <b>/<br>) before interpreting.
        raw_value = re.sub(r"</?b>|<br\s*/?>", " ", raw_value)
        raw_value = " ".join(raw_value.split()).strip()
        if not raw_value:
            return None
        if canonical in _AMOUNT_FIELDS:
            return parse_german_amount(raw_value)
        if canonical in _DATE_FIELDS:
            # Tolerate OCR fragmentation ("01.09. .2030" / "01.09..2030" → date).
            cleaned = re.sub(r"\.{2,}", ".", raw_value.replace(" ", ""))
            m = _RE_DATE.search(cleaned)
            return m.group(0) if m else None
        if canonical == "postleitzahl":
            m = _RE_PLZ.search(raw_value)
            return m.group(1) if m else None
        if canonical == "steueridentifikationsnummer":
            digits = re.sub(r"\D", "", raw_value)
            return digits if len(digits) == 11 else (raw_value or None)
        if canonical == "dokumentnummer":
            # 9 alphanumerics; OCR often appends the trailing check digit.
            cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_value).upper()
            match = re.match(r"[A-Z0-9]{9}", cleaned)
            return match.group(0) if match else (cleaned or None)
        if canonical == "steuerklasse":
            m = _RE_STEUERKLASSE.search(raw_value.upper())
            return m.group(1) if m else None
        if canonical == "veranlagungszeitraum":
            # Forms often print the full period ("01.01.2024 - 31.12.2024");
            # the schema field is the assessment *year*.
            m = re.search(r"(19|20)\d{2}", raw_value)
            return m.group(0) if m else raw_value
        return raw_value

    # -- label matching ------------------------------------------------------

    @staticmethod
    def _split_inline(text: str) -> tuple[str, str]:
        for sep in (":", "=", "\t"):
            if sep in text:
                left, right = text.split(sep, 1)
                return left.strip(), right.strip()
        return text.strip(), ""

    @staticmethod
    def _match_label(label: str, labels: dict[str, list[str]],
                     already: dict[str, MappedField]) -> Optional[str]:
        norm = _norm(label)
        if not norm:
            return None
        best: Optional[str] = None
        best_len = 0
        for canonical, synonyms in labels.items():
            for syn in synonyms:
                syn_n = _norm(syn)
                if not syn_n:
                    continue
                # Substring matching only for synonyms of length >= 4; shorter
                # ones (e.g. "ag", "nr", "rv") require an exact match, otherwise
                # they spuriously hit words like "SolidaritätszuschlAG".
                if len(syn_n) < 4:
                    hit = norm == syn_n
                else:
                    hit = syn_n in norm or (len(norm) >= 4 and norm in syn_n)
                if hit and len(syn_n) > best_len:
                    best, best_len = canonical, len(syn_n)
        # Don't overwrite an already-confident inline assignment with a weaker label.
        if best in already and already[best].source == "inline":
            return None
        return best

    # -- spatial geometry ----------------------------------------------------

    def _is_same_row(self, a: Any, b: Any) -> bool:
        ba, bb = getattr(a, "bbox", None), getattr(b, "bbox", None)
        if not ba or not bb:
            return False
        ca = (ba["y1"] + ba["y2"]) / 2
        return abs(ca - (bb["y1"] + bb["y2"]) / 2) <= self.SAME_ROW_Y_TOL or (
            bb["y1"] <= ca <= bb["y2"]
        )

    def _find_value_line(self, label_idx: int, lines: list[Any],
                         consumed: set[int], labels: dict[str, list[str]]) -> Optional[int]:
        label = lines[label_idx]
        lb = getattr(label, "bbox", None)
        if not lb:
            return None
        label_cy = (lb["y1"] + lb["y2"]) / 2

        right_best: Optional[tuple[float, int]] = None
        below_best: Optional[tuple[float, int]] = None
        for idx, line in enumerate(lines):
            if idx == label_idx or idx in consumed:
                continue
            b = getattr(line, "bbox", None)
            if not b:
                continue
            text = " ".join(str(line.text).split())
            # Skip lines that are themselves recognised labels.
            if self._looks_like_label(text, labels):
                continue
            # Bold lines are section / column headers (e.g. "Arbeitnehmer"),
            # not field values — skip them so two-column forms don't grab the
            # adjacent column's header.
            if "<b>" in str(line.text):
                continue
            cy = (b["y1"] + b["y2"]) / 2
            # Same-row, to the right.
            if abs(cy - label_cy) <= self.SAME_ROW_Y_TOL and b["x1"] >= lb["x1"]:
                dist = b["x1"] - lb["x2"]
                if dist >= -0.02 and (right_best is None or dist < right_best[0]):
                    right_best = (dist, idx)
            # Below: center lower than the label's, ranked by center distance
            # (not the top-to-bottom gap) so slightly-overlapping OCR boxes —
            # common on ID cards — still associate to the nearest value.
            if cy > label_cy and abs(b["x1"] - lb["x1"]) <= self.BELOW_X_TOL:
                delta = cy - label_cy
                if delta <= self.BELOW_MAX_CENTER_DELTA and (
                    below_best is None or delta < below_best[0]
                ):
                    below_best = (delta, idx)
        if right_best is not None:
            return right_best[1]
        if below_best is not None:
            return below_best[1]
        return None

    @staticmethod
    def _looks_like_label(text: str, labels: dict[str, list[str]]) -> bool:
        norm = _norm(text)
        if not norm:
            return False
        for synonyms in labels.values():
            for syn in synonyms:
                syn_n = _norm(syn)
                if syn_n and len(syn_n) >= 3 and syn_n in norm and len(norm) <= len(syn_n) + 4:
                    return True
        return False

    # -- pattern fallback ----------------------------------------------------

    def _pattern_fallback(self, lines: list[Any], consumed: set[int],
                          labels: dict[str, list[str]], mapped: dict[str, MappedField],
                          document_type: DocumentType) -> None:
        wants = set(labels.keys())
        for idx, line in enumerate(lines):
            if idx in consumed:
                continue
            text = " ".join(str(line.text).split())

            if "steueridentifikationsnummer" in wants and "steueridentifikationsnummer" not in mapped:
                m = _RE_STEUER_ID.search(text.replace(" ", ""))
                if m:
                    self._assign(mapped, "steueridentifikationsnummer", m.group(1), line, "pattern")
                    consumed.add(idx)
                    continue
            if "postleitzahl" in wants and "postleitzahl" not in mapped:
                m = _RE_PLZ.search(text)
                # Avoid grabbing a 5-run inside a longer number.
                if m and not re.search(r"\d{6,}", text):
                    self._assign(mapped, "postleitzahl", m.group(1), line, "pattern")
                    consumed.add(idx)
                    continue
            if "geburtsdatum" in wants and "geburtsdatum" not in mapped:
                m = _RE_DATE.search(text)
                if m:
                    self._assign(mapped, "geburtsdatum", m.group(0), line, "pattern")
                    consumed.add(idx)
                    continue

    def _infer_card_surname(
        self, lines: list[Any], consumed: set[int], mapped: dict[str, MappedField]
    ) -> None:
        """Assign the unlabeled, uppercase surname on an ID card to familienname.

        Picks the topmost unconsumed, uppercase-dominant alphabetic line in the
        data column (x1 >= 0.30, i.e. right of the photo) that is not itself a
        recognised label.
        """
        labels = _LABELS[DocumentType.PERSONALAUSWEIS]
        best_idx: Optional[int] = None
        best_y = 1.0
        for idx, line in enumerate(lines):
            if idx in consumed:
                continue
            bbox = getattr(line, "bbox", None)
            if not bbox or bbox.get("x1", 0.0) < 0.30:
                continue
            text = " ".join(str(line.text).split())
            stripped = re.sub(r"[^A-Za-zÄÖÜäöüß\s\-]", "", text).strip()
            letters = [c for c in stripped if c.isalpha()]
            if not (2 <= len(stripped) <= 40) or not letters:
                continue
            if sum(1 for c in letters if c.isupper()) < max(2, len(letters) - 1):
                continue
            if self._looks_like_label(text, labels):
                continue
            if bbox["y1"] < best_y:
                best_y, best_idx = bbox["y1"], idx
        if best_idx is None:
            return
        line = lines[best_idx]
        value = " ".join(str(line.text).split())
        mapped["familienname"] = MappedField(
            field_name="familienname",
            value=value,
            confidence=min(self._conf(line) * 0.85, 1.0),
            raw_text=value,
            bbox=getattr(line, "bbox", None),
            source="card-surname",
        )
        consumed.add(best_idx)
