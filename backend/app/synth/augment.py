"""
Realistic degradations for synthetic documents.

The clean renders are too pristine to train a model that transfers to real
scans, so we corrupt them the way real capture does. Every document gets a
*baseline* of paper texture + lighting + noise + slight blur + JPEG (so nothing
is ever pristine), plus a random mix of heavier effects: perspective warp,
rotation/skew, ink bleed/fade, coffee stains, photocopy thresholding, vignette.
This domain randomisation is what lets a model trained on synthetic data work on
the real held-out documents. All randomness is seeded for reproducibility.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

_PAPER = (250, 250, 247)  # off-white fill for empty areas (rotation/warp)


def _np_rng(rng: random.Random) -> np.random.RandomState:
    return np.random.RandomState(rng.randint(0, 2**31 - 1))


# --- baseline (applied to every document) ----------------------------------

def _paper_texture(img: Image.Image, rng: random.Random) -> Image.Image:
    """Subtle non-uniform paper tint + low-frequency texture."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    nrng = _np_rng(rng)
    coarse = nrng.normal(0.0, rng.uniform(3.0, 9.0), (max(2, h // 40), max(2, w // 40), 3))
    tex = Image.fromarray(np.clip(128 + coarse, 0, 255).astype(np.uint8)).resize((w, h))
    tex = tex.filter(ImageFilter.GaussianBlur(7))
    arr += (np.asarray(tex).astype(np.float32) - 128.0) * 0.7
    arr += nrng.uniform(-6.0, 6.0, 3)  # global warm/cool tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _lighting_gradient(img: Image.Image, rng: random.Random) -> Image.Image:
    """Uneven lighting / soft shadow across one axis."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    a, b = sorted([rng.uniform(0.72, 1.0), rng.uniform(1.0, 1.30)])
    lo, hi = (a, b) if rng.random() < 0.5 else (b, a)
    if rng.random() < 0.5:
        ramp = np.linspace(lo, hi, w)[None, :, None]
    else:
        ramp = np.linspace(lo, hi, h)[:, None, None]
    return Image.fromarray(np.clip(arr * ramp, 0, 255).astype(np.uint8))


def _noise(img: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr += _np_rng(rng).normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --- heavier, occasional effects -------------------------------------------

def _perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    """Phone-photo-style perspective warp."""
    w, h = img.size
    m = rng.uniform(0.015, 0.06)

    def j() -> float:
        return rng.uniform(-m, m)

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(w * j(), h * j()), (w * (1 + j()), h * j()),
           (w * (1 + j()), h * (1 + j())), (w * j(), h * (1 + j()))]
    rows, rhs = [], []
    for (ox, oy), (ix, iy) in zip(dst, src):
        rows += [[ox, oy, 1, 0, 0, 0, -ox * ix, -oy * ix], [0, 0, 0, ox, oy, 1, -ox * iy, -oy * iy]]
        rhs += [ix, iy]
    try:
        coeffs = np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float)).tolist()
        return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=_PAPER)
    except Exception:
        return img


def _ink(img: Image.Image, rng: random.Random) -> Image.Image:
    """Ink bleed (thicken) or faded print (thin)."""
    return img.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))


def _coffee_stain(img: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = rng.randint(0, img.width), rng.randint(0, img.height)
    r = rng.randint(50, 190)
    fill = (rng.randint(95, 150), rng.randint(60, 100), rng.randint(20, 55), rng.randint(35, 90))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(90, 55, 25, 120), width=rng.randint(4, 11))
    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 6.0)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _photocopy(img: Image.Image, rng: random.Random) -> Image.Image:
    gray = ImageEnhance.Contrast(img.convert("L")).enhance(rng.uniform(1.3, 2.2))
    arr = np.asarray(gray).astype(np.float32)
    arr += _np_rng(rng).normal(0.0, rng.uniform(6.0, 18.0), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")


def _vignette(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1.0 - rng.uniform(0.15, 0.40) * np.clip(dist - 0.6, 0, None), 0.5, 1.0)
    return Image.fromarray(np.clip(arr * mask[:, :, None], 0, 255).astype(np.uint8))


# --- pipeline ---------------------------------------------------------------

def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply baseline + a random mix of heavier degradations."""
    img = img.convert("RGB")
    img = _paper_texture(img, rng)                                   # baseline
    if rng.random() < 0.40:
        img = _perspective(img, rng)
    angle = rng.uniform(-5.0, 5.0) if rng.random() < 0.85 else rng.choice([-90, 90, 180])
    img = img.rotate(angle, expand=True, fillcolor=_PAPER, resample=Image.BICUBIC)
    if rng.random() < 0.60:
        img = _lighting_gradient(img, rng)
    if rng.random() < 0.35:
        img = _ink(img, rng)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.74, 1.16))  # baseline
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.80, 1.30))
    if rng.random() < 0.28:
        img = _coffee_stain(img, rng)
    if rng.random() < 0.50:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    img = _noise(img, rng, sigma=rng.uniform(5.0, 20.0))             # baseline
    if rng.random() < 0.28:
        img = _photocopy(img, rng)
    if rng.random() < 0.40:
        img = _vignette(img, rng)
    return _jpeg(img, rng.randint(30, 80))                           # baseline
