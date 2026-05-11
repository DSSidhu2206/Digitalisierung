"""
Strict Pydantic schemas for bureaucratic document types.

Each schema enforces the expected field names, data types, and
syntactic constraints (regex patterns, value ranges) for a specific
document category.  All schemas participate in the ``DocumentSchema``
union so that downstream code can work with any document type
uniformly.

Anti-Fabrication Constraints (Section 12):
  - Every field can be wrapped in a :class:`FieldResult` with
    ``status = UNRESOLVED`` when extraction fails.
  - No field is ever auto-completed or extrapolated.
  - Cut-off text is captured exactly as seen.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Meldebescheinigung (Residence Registration Certificate)
# ---------------------------------------------------------------------------

class MeldebescheinigungSchema(BaseModel):
    """Anmeldung / Meldebescheinigung — strict schema.

    Represents a residence registration certificate containing
    personal data and the registered address.  All fields are mandatory
    at the schema level; extraction failures are handled by wrapping
    values in :class:`FieldResult` with ``status = UNRESOLVED``.
    """

    model_config = ConfigDict(title="Meldebescheinigung")

    familienname: str = Field(
        ..., description="Family name / Nachname"
    )
    vorname: str = Field(
        ..., description="First name / Vorname"
    )
    geburtsdatum: date = Field(
        ..., description="Date of birth in DD.MM.YYYY format"
    )
    geburtsort: str = Field(
        ..., description="Place of birth (city)"
    )
    staatsangehoerigkeit: str = Field(
        ..., description="Nationality"
    )
    strasse: str = Field(
        ..., description="Street name of registered address"
    )
    hausnummer: str = Field(
        ..., description="House number"
    )
    postleitzahl: str = Field(
        ..., pattern=r"^\d{5}$", description="5-digit postal code"
    )
    wohnort: str = Field(
        ..., description="City / municipality of residence"
    )
    einzugsdatum: Optional[date] = Field(
        None, description="Move-in date (if available)"
    )
    vorherige_anschrift: Optional[str] = Field(
        None, description="Previous address (if moved)"
    )


# ---------------------------------------------------------------------------
# Steuerbescheid (Tax Assessment Notice)
# ---------------------------------------------------------------------------

class SteuerbescheidSchema(BaseModel):
    """Steuerbescheid — strict schema.

    Represents a tax assessment notice from the Finanzamt.
    Contains the tax identification number, assessment period, and
    monetary amounts for taxable income and assessed tax.
    """

    model_config = ConfigDict(title="Steuerbescheid")

    steueridentifikationsnummer: str = Field(
        ...,
        pattern=r"^\d{11}$",
        description="Steuer-ID (11 digits, last digit is checksum)",
    )
    veranlagungszeitraum: str = Field(
        ..., description="Tax assessment period, e.g. '2023'"
    )
    zu_versteuerndes_einkommen: float = Field(
        ..., ge=0.0, description="Taxable income in EUR"
    )
    festgesetzte_steuer: float = Field(
        ..., ge=0.0, description="Assessed tax amount in EUR"
    )
    ermässigung: Optional[float] = Field(
        None, ge=0.0, description="Tax reduction / credit (if any)"
    )
    vorauszahlungen: Optional[float] = Field(
        None, ge=0.0, description="Advance tax payments already made"
    )
    kirchensteuer: Optional[float] = Field(
        None, ge=0.0, description="Church tax (Kirchensteuer) amount"
    )
    solidaritätszuschlag: Optional[float] = Field(
        None, ge=0.0, description="Solidarity surcharge (Soli) amount"
    )


# ---------------------------------------------------------------------------
# Gehaltsausweis (Salary Statement)
# ---------------------------------------------------------------------------

class GehaltsausweisSchema(BaseModel):
    """Gehaltsausweis / Lohnabrechnung — strict schema.

    Represents a salary statement containing employer
    information, gross/net pay, and the pay period.
    """

    model_config = ConfigDict(title="Gehaltsausweis")

    arbeitgeber: str = Field(
        ..., description="Employer name"
    )
    brutto_lohn: float = Field(
        ..., ge=0.0, description="Gross salary amount in EUR"
    )
    netto_lohn: float = Field(
        ..., ge=0.0, description="Net salary amount in EUR"
    )
    abrechnungszeitraum: str = Field(
        ..., description="Pay period, e.g. '01.2024' or 'Januar 2024'"
    )
    steuerklasse: Optional[str] = Field(
        None, description="Tax class (I, II, III, IV, V, VI)"
    )
    steueridentifikationsnummer: Optional[str] = Field(
        None,
        pattern=r"^\d{11}$",
        description="Employee Steuer-ID (11 digits)",
    )
    sozialversicherungsnummer: Optional[str] = Field(
        None, description="Social security number"
    )
    rentenversicherung: Optional[float] = Field(
        None, ge=0.0, description="Pension insurance contribution"
    )
    krankenversicherung: Optional[float] = Field(
        None, ge=0.0, description="Health insurance contribution"
    )
    arbeitslosenversicherung: Optional[float] = Field(
        None, ge=0.0, description="Unemployment insurance contribution"
    )
    pflegeversicherung: Optional[float] = Field(
        None, ge=0.0, description="Long-term care insurance contribution"
    )


# ---------------------------------------------------------------------------
# Personalausweis (Identity Card)
# ---------------------------------------------------------------------------

class PersonalausweisSchema(BaseModel):
    """Personalausweis — strict schema.

    Represents a national identity card (nPA).  Contains
    biometric document data as defined by ICAO 9303.
    """

    model_config = ConfigDict(title="Personalausweis")

    dokumentnummer: str = Field(
        ...,
        pattern=r"^[A-Z0-9]{9}$",
        description="Document number (9 alphanumeric characters)",
    )
    familienname: str = Field(
        ..., description="Family name"
    )
    vorname: str = Field(
        ..., description="First name(s)"
    )
    geburtsdatum: date = Field(
        ..., description="Date of birth in DD.MM.YYYY format"
    )
    geburtsort: str = Field(
        ..., description="Place of birth"
    )
    staatsangehoerigkeit: str = Field(
        ..., description="Nationality"
    )
    gueltig_bis: date = Field(
        ..., description="Expiry date in DD.MM.YYYY format"
    )
    ausstellungsdatum: Optional[date] = Field(
        None, description="Issue date"
    )
    ausstellende_behoerde: Optional[str] = Field(
        None, description="Issuing authority"
    )


# ---------------------------------------------------------------------------
# Union type for all document schemas
# ---------------------------------------------------------------------------

DocumentSchema = Union[
    MeldebescheinigungSchema,
    SteuerbescheidSchema,
    GehaltsausweisSchema,
    PersonalausweisSchema,
]
"""Discriminated union of all document schema types.

Use this type annotation when a function or endpoint must accept any
of the four supported document schemas.  The Pydantic schema for each
variant is available via the ``model_config.title`` attribute:
  - ``Meldebescheinigung``
  - ``Steuerbescheid``
  - ``Gehaltsausweis``
  - ``Personalausweis``
"""

# Registry mapping document types to their schema classes
DOCUMENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "Meldebescheinigung": MeldebescheinigungSchema,
    "Steuerbescheid": SteuerbescheidSchema,
    "Gehaltsausweis": GehaltsausweisSchema,
    "Personalausweis": PersonalausweisSchema,
}


def get_schema_for_document_type(document_type: str) -> type[BaseModel] | None:
    """Get the Pydantic schema class for a given document type string.

    Args:
        document_type: The document type name (e.g. "Meldebescheinigung")

    Returns:
        The schema class or None if not found.
    """
    return DOCUMENT_SCHEMAS.get(document_type)
