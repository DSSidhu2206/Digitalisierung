"""
Diverse PIL renderers for synthetic German documents.

Heavy domain randomisation so a model can't overfit a template:
  * layout *mode* per doc (inline "Label: value", stacked label-above-value,
    or right-aligned table);
  * randomised fonts, sizes, spacing, accent colours and backgrounds
    (plain / cream / grey / faint form lines);
  * varied label phrasing ("Familienname" / "Name" / "Nachname" …);
  * document furniture — official stamps, signatures, barcodes, logos,
    reference numbers, separators, footers — each added probabilistically.

Content is stylised (drawn with primitives), but together with the capture
degradations in ``augment.py`` this gives wide, transfer-useful variety.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.models.enums import DocumentType

_FD = "/System/Library/Fonts/Supplemental"
_SANS = [f"{_FD}/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
_SERIF = [f"{_FD}/Times New Roman.ttf"]
_MONO = [f"{_FD}/Courier New.ttf"]
_FALLBACK = _SANS[0]

# Label phrasing variants per canonical field (the *value* is the training
# target, so varying the label only helps generalisation).
_LABELS = {
    "familienname": ["Familienname", "Name", "Nachname"],
    "vorname": ["Vorname", "Vornamen", "Rufname"],
    "geburtsdatum": ["Geburtsdatum", "geboren am", "Geb.-Datum"],
    "geburtsort": ["Geburtsort", "geboren in"],
    "staatsangehoerigkeit": ["Staatsangehörigkeit", "Nationalität"],
    "strasse": ["Straße", "Strasse", "Anschrift"],
    "hausnummer": ["Hausnummer", "Haus-Nr.", "Nr."],
    "postleitzahl": ["Postleitzahl", "PLZ"],
    "wohnort": ["Wohnort", "Ort", "Stadt"],
    "einzugsdatum": ["Einzugsdatum", "Einzug am", "Zuzugsdatum"],
    "steueridentifikationsnummer": ["Steueridentifikationsnummer", "Steuer-ID", "IdNr."],
    "veranlagungszeitraum": ["Veranlagungszeitraum", "Steuerjahr", "Zeitraum"],
    "zu_versteuerndes_einkommen": ["zu versteuerndes Einkommen", "zvE", "versteuerndes Einkommen"],
    "festgesetzte_steuer": ["festgesetzte Einkommensteuer", "festgesetzte Steuer", "Einkommensteuer"],
    "solidaritätszuschlag": ["Solidaritätszuschlag", "Soli", "SolZ"],
    "arbeitgeber": ["Arbeitgeber", "Firma", "Unternehmen"],
    "brutto_lohn": ["Gesamt-Brutto", "Bruttolohn", "Brutto"],
    "netto_lohn": ["Nettolohn", "Auszahlungsbetrag", "Netto"],
    "abrechnungszeitraum": ["Abrechnungszeitraum", "Abrechnungsmonat", "Monat"],
    "steuerklasse": ["Steuerklasse", "St.-Klasse", "StKl"],
    "dokumentnummer": ["Dokumentnummer", "Ausweisnummer", "Dokument-Nr."],
    "gueltig_bis": ["Gültig bis", "gültig bis", "Ablaufdatum"],
    "ausstellende_behoerde": ["Ausstellende Behörde", "Behörde"],
}


@lru_cache(maxsize=512)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(_FALLBACK, size)


class _Style:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.body = rng.choice(_SANS + _SANS + _SERIF)   # weight sans
        self.head = rng.choice(_SANS + _SERIF)
        self.mono = _MONO[0]
        self.base = rng.randint(19, 27)
        self.ink = rng.choice([(15, 15, 15), (25, 25, 30), (8, 8, 8), (35, 35, 40)])
        self.accent = rng.choice([(0, 0, 100), (10, 10, 10), (110, 20, 20), (20, 60, 110)])
        self.label_ink = rng.choice([(70, 70, 75), (40, 40, 40), self.accent])
        self.margin = rng.randint(55, 120)
        self.row_gap = rng.randint(10, 26)
        self.mode = rng.choice(["inline", "inline", "stacked", "table"])
        self.bg = rng.choice(["white", "white", "cream", "grey", "lines"])

    def f(self, scale: float = 1.0, mono: bool = False, head: bool = False) -> ImageFont.FreeTypeFont:
        path = self.mono if mono else (self.head if head else self.body)
        return _font(path, max(10, int(self.base * scale)))


def _label(st: _Style, key: str, default: str) -> str:
    return st.rng.choice(_LABELS.get(key, [default]))


def _canvas(st: _Style, rng: random.Random) -> Image.Image:
    w, h = rng.randint(940, 1080), rng.randint(1300, 1520)
    if st.bg == "cream":
        base = (rng.randint(248, 254), rng.randint(244, 251), rng.randint(232, 244))
    elif st.bg == "grey":
        s = rng.randint(238, 248)
        base = (s, s, s)
    else:
        s = rng.randint(250, 255)
        base = (s, s, s)
    img = Image.new("RGB", (w, h), base)
    if st.bg == "lines":
        d = ImageDraw.Draw(img)
        gap = rng.randint(44, 70)
        for y in range(st.margin + 120, h - st.margin, gap):
            d.line([(st.margin, y), (w - st.margin, y)], fill=(220, 222, 228), width=1)
    return img


# --------------------------------------------------------------------------
# Document furniture
# --------------------------------------------------------------------------

def _stamp(d: ImageDraw.ImageDraw, st: _Style, cx: int, cy: int, rng: random.Random) -> None:
    r = rng.randint(55, 90)
    col = rng.choice([(20, 40, 130), (130, 30, 30), (40, 40, 40)])
    overlay_box = [cx - r, cy - r, cx + r, cy + r]
    d.ellipse(overlay_box, outline=col, width=rng.randint(2, 4))
    d.ellipse([cx - r + 10, cy - r + 10, cx + r - 10, cy + r - 10], outline=col, width=1)
    f = _font(st.head, rng.randint(13, 17))
    text = rng.choice(["AMTLICH", "GEPRÜFT", "BEGLAUBIGT", "EINGEGANGEN"])
    tw = d.textlength(text, font=f)
    d.text((cx - tw / 2, cy - 8), text, font=f, fill=col)


def _signature(d: ImageDraw.ImageDraw, x: int, y: int, rng: random.Random) -> None:
    col = rng.choice([(20, 30, 110), (15, 15, 15)])
    px, py = x, y
    for _ in range(rng.randint(14, 26)):
        nx = px + rng.randint(6, 26)
        ny = y + rng.randint(-18, 18)
        d.line([(px, py), (nx, ny)], fill=col, width=rng.randint(1, 3))
        px, py = nx, ny


def _barcode(d: ImageDraw.ImageDraw, x: int, y: int, rng: random.Random) -> None:
    px = x
    for _ in range(rng.randint(28, 50)):
        w = rng.choice([1, 1, 2, 3])
        if rng.random() < 0.55:
            d.rectangle([px, y, px + w, y + rng.randint(34, 56)], fill=(10, 10, 10))
        px += w + rng.choice([1, 1, 2])


def _qrish(d: ImageDraw.ImageDraw, x: int, y: int, rng: random.Random) -> None:
    n, cell = 11, rng.randint(4, 6)
    for i in range(n):
        for j in range(n):
            if rng.random() < 0.5:
                d.rectangle([x + i * cell, y + j * cell, x + (i + 1) * cell, y + (j + 1) * cell], fill=(10, 10, 10))


def _logo(d: ImageDraw.ImageDraw, st: _Style, x: int, y: int, rng: random.Random) -> None:
    col = st.accent
    shape = rng.choice(["circle", "square", "shield"])
    s = rng.randint(34, 54)
    if shape == "circle":
        d.ellipse([x, y, x + s, y + s], outline=col, width=3)
    elif shape == "square":
        d.rectangle([x, y, x + s, y + s], outline=col, width=3)
    else:
        d.polygon([(x + s / 2, y), (x + s, y + s * 0.35), (x + s / 2, y + s), (x, y + s * 0.35)], outline=col, width=3)
    d.text((x + s + 10, y + s / 4), rng.choice(["BRD", "DE", "AMT", "FA"]), font=st.f(1.1, head=True), fill=col)


def _footer(d: ImageDraw.ImageDraw, st: _Style, w: int, h: int, rng: random.Random) -> None:
    f = _font(st.body, max(10, int(st.base * 0.62)))
    text = rng.choice([
        "Dieses Dokument wurde maschinell erstellt und ist ohne Unterschrift gültig.",
        "Seite 1 von 1   ·   Aktenzeichen: " + str(rng.randint(100000, 999999)),
        "Erstellt am " + f"{rng.randint(1,28):02d}.{rng.randint(1,12):02d}.{rng.randint(2020,2024)}",
    ])
    d.text((st.margin, h - st.margin + rng.randint(-10, 20)), text, font=f, fill=(120, 120, 125))


def _sep(d: ImageDraw.ImageDraw, st: _Style, x: int, y: int, w: int) -> None:
    d.line([(x, y), (w - st.margin, y)], fill=(175, 178, 185), width=1)


# --------------------------------------------------------------------------
# Row drawing (mode-aware)
# --------------------------------------------------------------------------

def _draw_rows(d: ImageDraw.ImageDraw, st: _Style, x: int, y: int, rows, *, mono_keys=()) -> int:
    for key, label, value in rows:
        mono = key in mono_keys
        if st.mode == "stacked":
            d.text((x, y), f"{label}:", font=st.f(0.78), fill=st.label_ink)
            y += int(st.base * 0.95)
            d.text((x, y), str(value), font=st.f(1.0, mono=mono), fill=st.ink)
            y += int(st.base * 1.15) + st.row_gap
        else:  # inline
            lf, vf = st.f(0.9), st.f(1.0, mono=mono)
            d.text((x, y), f"{label}:", font=lf, fill=st.label_ink)
            lw = d.textlength(f"{label}:", font=lf)
            d.text((x + lw + st.rng.randint(10, 22), y), str(value), font=vf, fill=st.ink)
            y += int(st.base * 1.05) + st.row_gap
    return y


def _draw_table(d: ImageDraw.ImageDraw, st: _Style, x: int, y: int, w: int, rows, *, mono_keys=(), euro=False) -> int:
    for key, label, value in rows:
        d.text((x, y), label, font=st.f(0.95), fill=st.label_ink)
        vf = st.f(1.0, mono=(key in mono_keys))
        txt = f"{value} €" if euro else str(value)
        d.text((w - st.margin - d.textlength(txt, font=vf), y), txt, font=vf, fill=st.ink)
        y += int(st.base * 1.1) + st.row_gap
    return y


def _add_furniture(img: Image.Image, st: _Style, rng: random.Random) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    if rng.random() < 0.45:
        _stamp(d, st, rng.randint(int(w * 0.55), w - st.margin), rng.randint(int(h * 0.45), h - st.margin), rng)
    if rng.random() < 0.40:
        _signature(d, rng.randint(st.margin, int(w * 0.4)), rng.randint(int(h * 0.6), h - st.margin), rng)
        d.text((st.margin, rng.randint(int(h * 0.6), h - st.margin) + 20), "Unterschrift", font=st.f(0.6), fill=(130, 130, 130))
    if rng.random() < 0.30:
        _barcode(d, rng.randint(st.margin, int(w * 0.5)), h - st.margin - rng.randint(60, 120), rng)
    if rng.random() < 0.20:
        _qrish(d, w - st.margin - 80, st.margin, rng)
    if rng.random() < 0.55:
        _footer(d, st, w, h, rng)
    if rng.random() < 0.5:
        d.text((st.margin, st.margin - rng.randint(0, 28) if st.margin > 30 else 4),
               "Az.: " + "/".join(str(rng.randint(10, 9999)) for _ in range(3)), font=st.f(0.6), fill=(120, 120, 125))


# --------------------------------------------------------------------------
# Per-type renderers
# --------------------------------------------------------------------------

def _header(d, st, x, y, title, rng, subtitle=None):
    if st.rng.random() < 0.5:
        _logo(d, st, x, y, rng)
        x += 80
    d.text((x, y), title, font=st.f(rng.uniform(1.5, 1.9), head=True), fill=st.ink)
    y += int(st.base * 2.0)
    if subtitle:
        d.text((st.margin, y), subtitle, font=st.f(0.85), fill=st.label_ink)
        y += int(st.base * 1.4)
    return y


def _render_meldebescheinigung(fields, st, rng):
    img = _canvas(st, rng)
    d = ImageDraw.Draw(img)
    x = st.margin
    y = _header(d, st, x, st.margin, rng.choice(["MELDEBESCHEINIGUNG", "Meldebescheinigung", "ANMELDEBESTÄTIGUNG"]), rng,
                rng.choice(["Bürgeramt", "Meldebehörde", "Einwohnermeldeamt", None]))
    _sep(d, st, x, y, img.width); y += st.row_gap
    keys = ["familienname", "vorname", "geburtsdatum", "geburtsort", "staatsangehoerigkeit",
            "strasse", "hausnummer", "postleitzahl", "wohnort", "einzugsdatum"]
    rows = [(k, _label(st, k, k), fields[k]) for k in keys if k in fields]
    _draw_rows(d, st, x, y, rows)
    _add_furniture(img, st, rng)
    return img


def _render_steuerbescheid(fields, st, rng):
    img = _canvas(st, rng)
    d = ImageDraw.Draw(img)
    x = st.margin
    d.text((x, st.margin), "Finanzamt " + rng.choice(["Berlin-Mitte", "München", "Hamburg", "Köln", "Frankfurt"]),
           font=st.f(1.0, head=True), fill=st.ink)
    sx = img.width - st.margin - 330
    d.text((sx, st.margin), _label(st, "steueridentifikationsnummer", "Steuer-ID"), font=st.f(0.7), fill=st.label_ink)
    d.text((sx, st.margin + int(st.base * 0.95)), fields["steueridentifikationsnummer"], font=st.f(1.0, mono=True), fill=st.ink)
    y = _header(d, st, x, st.margin + int(st.base * 2.4), f"Steuerbescheid {fields['veranlagungszeitraum']}", rng)
    _sep(d, st, x, y, img.width); y += st.row_gap + 4
    y = _draw_rows(d, st, x, y, [("veranlagungszeitraum", _label(st, "veranlagungszeitraum", "Zeitraum"), fields["veranlagungszeitraum"])])
    y += 6
    amounts = [(k, _label(st, k, k), fields[k]) for k in
               ["zu_versteuerndes_einkommen", "festgesetzte_steuer", "solidaritätszuschlag"] if k in fields]
    _draw_table(d, st, x, y, img.width, amounts, euro=True)
    _add_furniture(img, st, rng)
    return img


def _render_gehaltsausweis(fields, st, rng):
    img = _canvas(st, rng)
    d = ImageDraw.Draw(img)
    x = st.margin
    d.text((x, st.margin), fields["arbeitgeber"], font=st.f(1.3, head=True), fill=st.ink)
    y = _header(d, st, x, st.margin + int(st.base * 1.9),
                f"{rng.choice(['Gehaltsausweis', 'Lohnabrechnung', 'Verdienstabrechnung'])} {fields['abrechnungszeitraum']}", rng)
    _sep(d, st, x, y, img.width); y += st.row_gap
    info = [("arbeitgeber", _label(st, "arbeitgeber", "Arbeitgeber"), fields["arbeitgeber"]),
            ("steuerklasse", _label(st, "steuerklasse", "Steuerklasse"), fields["steuerklasse"])]
    y = _draw_rows(d, st, x, y, info)
    y += 8
    _sep(d, st, x, y, img.width); y += st.row_gap
    amounts = [("brutto_lohn", _label(st, "brutto_lohn", "Gesamt-Brutto"), fields["brutto_lohn"]),
               ("netto_lohn", _label(st, "netto_lohn", "Nettolohn"), fields["netto_lohn"])]
    _draw_table(d, st, x, y, img.width, amounts, euro=True)
    _add_furniture(img, st, rng)
    return img


def _render_personalausweis(fields, st, rng):
    img = _canvas(st, rng)
    d = ImageDraw.Draw(img)
    x = st.margin
    d.text((x, st.margin), "BUNDESREPUBLIK DEUTSCHLAND", font=st.f(1.2, head=True), fill=st.ink)
    d.text((x, st.margin + int(st.base * 1.5)), rng.choice(["PERSONALAUSWEIS", "Personalausweis"]),
           font=st.f(1.0), fill=st.label_ink)
    photo = (x, st.margin + 70, x + rng.randint(220, 270), st.margin + 70 + rng.randint(300, 360))
    d.rectangle(photo, fill=(rng.randint(195, 215),) * 3, outline=(120, 120, 120))
    d.text((photo[0] + 60, (photo[1] + photo[3]) // 2), "FOTO", font=st.f(0.9), fill=(140, 140, 140))
    cx, cy = photo[2] + rng.randint(45, 70), photo[1]
    keys = ["familienname", "vorname", "geburtsdatum", "geburtsort", "staatsangehoerigkeit", "dokumentnummer", "gueltig_bis"]
    for k in keys:
        d.text((cx, cy), f"{_label(st, k, k)}:", font=st.f(0.74), fill=st.label_ink)
        cy += int(st.base * 0.92)
        d.text((cx, cy), str(fields[k]), font=st.f(1.0, mono=(k == "dokumentnummer")), fill=st.ink)
        cy += int(st.base * 1.35)
    if rng.random() < 0.5:
        _barcode(d, cx, cy + 14, rng)
    _add_furniture(img, st, rng)
    return img


_RENDERERS = {
    DocumentType.MELDEBESCHEINIGUNG: _render_meldebescheinigung,
    DocumentType.STEUERBESCHEID: _render_steuerbescheid,
    DocumentType.GEHALTSAUSWEIS: _render_gehaltsausweis,
    DocumentType.PERSONALAUSWEIS: _render_personalausweis,
}


def render(document_type: DocumentType, fields: dict, rng: random.Random) -> Image.Image:
    """Render a clean (pre-augmentation) document image."""
    return _RENDERERS[document_type](fields, _Style(rng), rng)
