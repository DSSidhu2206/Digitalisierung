"""
Realistic, checksum-valid field data for synthetic German documents.

Every generated value is *correct by construction*: Steuer-IDs pass the official
ISO 7064 MOD 11,10 check (the same validator used in production), IBANs pass
mod-97, and German names/streets/cities come from ``Faker(de_DE)``. Because we
generate the values, every rendered document comes with perfect ground-truth
labels — the whole point of synthetic training data.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from faker import Faker

from app.models.enums import DocumentType
from app.validators.checksums import validate_iban, validate_steuer_id

_NATIONALITIES = ["deutsch", "deutsch", "deutsch", "türkisch", "polnisch",
                  "italienisch", "französisch", "österreichisch", "griechisch"]
_AUTHORITIES = ["Bürgeramt", "Bürgerbüro", "Einwohnermeldeamt", "Meldebehörde"]


# ---------------------------------------------------------------------------
# Primitive valid-value generators
# ---------------------------------------------------------------------------

def _mod11_10_check(payload: list[int]) -> int:
    """ISO 7064 MOD 11,10 check digit over the first ten digits."""
    product = 10
    for digit in payload:
        s = (digit + product) % 10
        if s == 0:
            s = 10
        product = (s * 2) % 11
    return (11 - product) % 10


def valid_steuer_id(rng: random.Random) -> str:
    """An 11-digit Steuer-ID that passes ``validate_steuer_id``."""
    for _ in range(1000):
        distinct = rng.sample(range(10), 9)          # 9 distinct digits
        payload = distinct + [rng.choice(distinct)]  # one digit repeats → 10 digits
        rng.shuffle(payload)
        if payload[0] == 0:
            continue
        candidate = "".join(map(str, payload + [_mod11_10_check(payload)]))
        if validate_steuer_id(candidate):
            return candidate
    raise RuntimeError("could not generate a valid Steuer-ID")


def valid_iban_de(rng: random.Random) -> str:
    """A valid German IBAN (mod-97)."""
    bban = "".join(str(rng.randint(0, 9)) for _ in range(18))  # 8 BLZ + 10 account
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in bban + "DE00")
    check = 98 - (int(numeric) % 97)
    iban = f"DE{check:02d}{bban}"
    return iban if validate_iban(iban) else valid_iban_de(rng)


def valid_dokumentnummer(rng: random.Random) -> str:
    """9-character German ID document number (letter + 8 alphanumerics)."""
    alphabet = "ABCDEFGHJKLMNPRTVWXYZ0123456789"
    return rng.choice("CFGHJKLMNPRTVWXYZ") + "".join(rng.choice(alphabet) for _ in range(8))


def valid_plz(f: Faker, rng: random.Random) -> str:
    """A plausible 5-digit German PLZ (never the non-existent ``00xxx`` range)."""
    for _ in range(20):
        plz = str(f.postcode())
        if len(plz) == 5 and plz.isdigit() and plz[:2] != "00":
            return plz
    # Fallback: first digit 1-9 (skips the rare 01-09 range, but always valid).
    return f"{rng.randint(1, 9)}" + "".join(str(rng.randint(0, 9)) for _ in range(4))


def german_date(rng: random.Random, start_year: int, end_year: int) -> str:
    """A random valid date in ``DD.MM.YYYY``."""
    year = rng.randint(start_year, end_year)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)  # always valid
    return f"{day:02d}.{month:02d}.{year}"


def german_amount(value: float) -> str:
    """Format a number the German way: ``45236`` → ``45.236,00``."""
    s = f"{value:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Per-document-type field generators (keys match the strict Pydantic schemas)
# ---------------------------------------------------------------------------

def _meldebescheinigung(f: Faker, rng: random.Random) -> dict[str, str]:
    return {
        "familienname": f.last_name(),
        "vorname": f.first_name(),
        "geburtsdatum": german_date(rng, 1950, 2006),
        "geburtsort": f.city(),
        "staatsangehoerigkeit": rng.choice(_NATIONALITIES),
        "strasse": f.street_name(),
        "hausnummer": str(f.building_number()),
        "postleitzahl": valid_plz(f, rng),
        "wohnort": f.city(),
        "einzugsdatum": german_date(rng, 2008, 2024),
    }


def _steuerbescheid(f: Faker, rng: random.Random) -> dict[str, str]:
    brutto = rng.randint(24000, 95000)
    zve = int(brutto * rng.uniform(0.74, 0.93))
    steuer = int(zve * rng.uniform(0.14, 0.32))
    return {
        "steueridentifikationsnummer": valid_steuer_id(rng),
        "veranlagungszeitraum": str(rng.randint(2017, 2024)),
        "zu_versteuerndes_einkommen": german_amount(zve),
        "festgesetzte_steuer": german_amount(steuer),
        "solidaritätszuschlag": german_amount(round(steuer * 0.055, 2)),
    }


def _gehaltsausweis(f: Faker, rng: random.Random) -> dict[str, str]:
    brutto = rng.randint(2400, 8500)
    netto = int(brutto * rng.uniform(0.57, 0.71))
    return {
        "arbeitgeber": f.company(),
        "brutto_lohn": german_amount(brutto),
        "netto_lohn": german_amount(netto),
        "abrechnungszeitraum": f"{rng.randint(1, 12):02d}.{rng.randint(2020, 2024)}",
        "steuerklasse": rng.choice(["I", "II", "III", "IV", "V", "VI"]),
    }


def _personalausweis(f: Faker, rng: random.Random) -> dict[str, str]:
    return {
        "dokumentnummer": valid_dokumentnummer(rng),
        "familienname": f.last_name(),
        "vorname": f.first_name(),
        "geburtsdatum": german_date(rng, 1955, 2006),
        "geburtsort": f.city(),
        "staatsangehoerigkeit": "DEUTSCH",
        "gueltig_bis": german_date(rng, 2027, 2035),
        "ausstellungsdatum": german_date(rng, 2017, 2024),
        "ausstellende_behoerde": rng.choice(_AUTHORITIES) + " " + f.city(),
    }


_GENERATORS: dict[DocumentType, Callable[[Faker, random.Random], dict[str, str]]] = {
    DocumentType.MELDEBESCHEINIGUNG: _meldebescheinigung,
    DocumentType.STEUERBESCHEID: _steuerbescheid,
    DocumentType.GEHALTSAUSWEIS: _gehaltsausweis,
    DocumentType.PERSONALAUSWEIS: _personalausweis,
}


def generate_fields(document_type: DocumentType, f: Faker, rng: random.Random) -> dict[str, Any]:
    """Return a dict of valid field values for *document_type*."""
    return _GENERATORS[document_type](f, rng)
