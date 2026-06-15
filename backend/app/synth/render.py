"""
PIL renderers for synthetic German documents.

Each renderer draws one document type from a field dict. Fonts, sizes, spacing
and small layout choices are randomised (domain randomisation) so a model
trained on these does not overfit a single pixel-perfect template.
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from app.models.enums import DocumentType

_FONT_DIR = "/System/Library/Fonts/Supplemental"
_SANS = [f"{_FONT_DIR}/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
_SERIF = [f"{_FONT_DIR}/Times New Roman.ttf"]
_MONO = [f"{_FONT_DIR}/Courier New.ttf"]
_FALLBACK = _SANS[0]


@lru_cache(maxsize=256)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(_FALLBACK, size)


class _Style:
    """Randomised per-document drawing style."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.body = rng.choice(_SANS + _SERIF)
        self.mono = _MONO[0]
        self.base = rng.randint(20, 26)
        self.ink = rng.choice([(15, 15, 15), (25, 25, 30), (10, 10, 10)])
        self.label_ink = rng.choice([(60, 60, 60), (30, 30, 30), (0, 0, 90)])
        self.margin = rng.randint(60, 110)
        self.row_gap = rng.randint(12, 26)

    def f(self, scale: float = 1.0, mono: bool = False) -> ImageFont.FreeTypeFont:
        return _font(self.mono if mono else self.body, max(10, int(self.base * scale)))


def _canvas(rng: random.Random) -> Image.Image:
    w = rng.randint(960, 1060)
    h = rng.randint(1320, 1500)
    shade = rng.randint(248, 255)
    return Image.new("RGB", (w, h), (shade, shade, shade))


def _row(d: ImageDraw.ImageDraw, st: _Style, x: int, y: int, label: str, value: str,
         *, mono: bool = False, gap: int = 14) -> int:
    """Draw an inline ``Label: value`` row; return the next y."""
    lf = st.f(0.9)
    vf = st.f(1.0, mono=mono)
    d.text((x, y), f"{label}:", font=lf, fill=st.label_ink)
    lw = d.textlength(f"{label}:", font=lf)
    d.text((x + lw + gap, y), value, font=vf, fill=st.ink)
    return y + int(st.base * 1.05) + st.row_gap


def _heading(d: ImageDraw.ImageDraw, st: _Style, x: int, y: int, text: str, scale: float = 1.7) -> int:
    f = st.f(scale)
    d.text((x, y), text, font=f, fill=st.ink)
    return y + int(st.base * scale) + st.row_gap


# ---------------------------------------------------------------------------
# Per-type renderers
# ---------------------------------------------------------------------------

def _render_meldebescheinigung(fields: dict, st: _Style, rng: random.Random) -> Image.Image:
    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    x, y = st.margin, st.margin
    y = _heading(d, st, x, y, "MELDEBESCHEINIGUNG", 1.8)
    d.text((x, y), rng.choice(["Bürgeramt", "Meldebehörde", "Einwohnermeldeamt"]),
           font=st.f(0.85), fill=st.label_ink)
    y += int(st.base * 1.6)
    d.line([(x, y), (img.width - st.margin, y)], fill=(180, 180, 180), width=1)
    y += st.row_gap
    order = ["familienname", "vorname", "geburtsdatum", "geburtsort", "staatsangehoerigkeit"]
    labels = {"familienname": "Familienname", "vorname": "Vorname", "geburtsdatum": "Geburtsdatum",
              "geburtsort": "Geburtsort", "staatsangehoerigkeit": "Staatsangehörigkeit"}
    for k in order:
        y = _row(d, st, x, y, labels[k], fields[k])
    # Address block (street + house number on one line)
    lf = st.f(0.9)
    d.text((x, y), "Straße:", font=lf, fill=st.label_ink)
    d.text((x + d.textlength("Straße:", font=lf) + 14, y), fields["strasse"], font=st.f(1.0), fill=st.ink)
    d.text((x + 520, y), "Hausnummer:", font=lf, fill=st.label_ink)
    d.text((x + 520 + d.textlength("Hausnummer:", font=lf) + 12, y), fields["hausnummer"], font=st.f(1.0), fill=st.ink)
    y += int(st.base * 1.05) + st.row_gap
    y = _row(d, st, x, y, "Postleitzahl", fields["postleitzahl"])
    y = _row(d, st, x, y, "Wohnort", fields["wohnort"])
    y = _row(d, st, x, y, "Einzugsdatum", fields["einzugsdatum"])
    return img


def _render_steuerbescheid(fields: dict, st: _Style, rng: random.Random) -> Image.Image:
    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    x, y = st.margin, st.margin
    d.text((x, y), "FINANZAMT " + rng.choice(["BERLIN-MITTE", "MÜNCHEN", "HAMBURG", "KÖLN"]),
           font=st.f(1.0), fill=st.ink)
    sid_label = st.f(0.75)
    d.text((img.width - st.margin - 320, y), "Steueridentifikationsnummer", font=sid_label, fill=st.label_ink)
    d.text((img.width - st.margin - 320, y + int(st.base * 0.95)),
           fields["steueridentifikationsnummer"], font=st.f(1.0, mono=True), fill=st.ink)
    y += int(st.base * 2.6)
    y = _heading(d, st, x, y, f"Steuerbescheid {fields['veranlagungszeitraum']}", 1.6)
    d.line([(x, y), (img.width - st.margin, y)], fill=(170, 170, 170), width=1)
    y += st.row_gap + 6

    def amount_row(label: str, value: str, yy: int) -> int:
        d.text((x, yy), label, font=st.f(0.95), fill=st.label_ink)
        vf = st.f(1.0, mono=True)
        d.text((img.width - st.margin - d.textlength(value + " €", font=vf), yy),
               value + " €", font=vf, fill=st.ink)
        return yy + int(st.base * 1.05) + st.row_gap

    y = _row(d, st, x, y, "Veranlagungszeitraum", fields["veranlagungszeitraum"])
    y += 6
    y = amount_row("zu versteuerndes Einkommen", fields["zu_versteuerndes_einkommen"], y)
    y = amount_row("festgesetzte Einkommensteuer", fields["festgesetzte_steuer"], y)
    y = amount_row("Solidaritätszuschlag", fields["solidaritätszuschlag"], y)
    return img


def _render_gehaltsausweis(fields: dict, st: _Style, rng: random.Random) -> Image.Image:
    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    x, y = st.margin, st.margin
    d.text((x, y), fields["arbeitgeber"], font=st.f(1.3), fill=st.ink)
    y += int(st.base * 2.0)
    y = _heading(d, st, x, y, f"Gehaltsausweis {fields['abrechnungszeitraum']}", 1.3)
    d.line([(x, y), (img.width - st.margin, y)], fill=(170, 170, 170), width=1)
    y += st.row_gap + 6
    # Employer / employee columns
    rx = x + 470
    d.text((x, y), "Arbeitgeber", font=st.f(0.95), fill=st.label_ink)
    d.text((rx, y), "Steuerklasse", font=st.f(0.95), fill=st.label_ink)
    y += int(st.base * 1.1)
    d.text((x, y), fields["arbeitgeber"], font=st.f(1.0), fill=st.ink)
    d.text((rx, y), fields["steuerklasse"], font=st.f(1.0), fill=st.ink)
    y += int(st.base * 1.9)
    d.line([(x, y), (img.width - st.margin, y)], fill=(200, 200, 200), width=1)
    y += st.row_gap

    def amount_row(label: str, value: str, yy: int, bold: bool = False) -> int:
        d.text((x, yy), label, font=st.f(1.05 if bold else 0.95), fill=st.ink if bold else st.label_ink)
        vf = st.f(1.05 if bold else 1.0, mono=True)
        d.text((img.width - st.margin - d.textlength(value + " €", font=vf), yy),
               value + " €", font=vf, fill=st.ink)
        return yy + int(st.base * 1.1) + st.row_gap

    y = amount_row("Gesamt-Brutto", fields["brutto_lohn"], y, bold=True)
    y += 8
    y = amount_row("Auszahlung / Nettolohn", fields["netto_lohn"], y, bold=True)
    return img


def _render_personalausweis(fields: dict, st: _Style, rng: random.Random) -> Image.Image:
    img = _canvas(rng)
    d = ImageDraw.Draw(img)
    x, y = st.margin, st.margin
    d.text((x, y), "BUNDESREPUBLIK DEUTSCHLAND", font=st.f(1.2), fill=st.ink)
    y += int(st.base * 1.5)
    d.text((x, y), "PERSONALAUSWEIS", font=st.f(1.0), fill=st.label_ink)
    # Photo placeholder
    photo = (x, y + 50, x + 250, y + 380)
    d.rectangle(photo, fill=(205, 205, 205), outline=(120, 120, 120))
    d.text((x + 60, y + 200), "FOTO", font=st.f(0.9), fill=(140, 140, 140))
    # Data column (right of photo), label above value
    cx = photo[2] + 60
    cy = y + 50
    rows = [
        ("Familienname", fields["familienname"]),
        ("Vorname", fields["vorname"]),
        ("Geburtsdatum", fields["geburtsdatum"]),
        ("Geburtsort", fields["geburtsort"]),
        ("Staatsangehörigkeit", fields["staatsangehoerigkeit"]),
        ("Dokumentnummer", fields["dokumentnummer"]),
        ("Gültig bis", fields["gueltig_bis"]),
    ]
    for label, value in rows:
        d.text((cx, cy), f"{label}:", font=st.f(0.78), fill=st.label_ink)
        cy += int(st.base * 0.95)
        mono = label == "Dokumentnummer"
        d.text((cx, cy), value, font=st.f(1.0, mono=mono), fill=st.ink)
        cy += int(st.base * 1.4)
    return img


_RENDERERS = {
    DocumentType.MELDEBESCHEINIGUNG: _render_meldebescheinigung,
    DocumentType.STEUERBESCHEID: _render_steuerbescheid,
    DocumentType.GEHALTSAUSWEIS: _render_gehaltsausweis,
    DocumentType.PERSONALAUSWEIS: _render_personalausweis,
}


def render(document_type: DocumentType, fields: dict, rng: random.Random) -> Image.Image:
    """Render a clean document image (before augmentation)."""
    return _RENDERERS[document_type](fields, _Style(rng), rng)
