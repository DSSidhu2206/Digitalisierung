# Datasets & evaluation methodology

This project fine-tunes a domain-specialised extraction model. The methodology is
deliberately **train on synthetic, evaluate on real** — the setup that actually
demonstrates generalisation (and avoids the most common portfolio mistake of
testing on data the model effectively memorised).

## Why synthetic training data

OCR / field-extraction training needs **image → text/fields** pairs. The
RVL-CDIP corpus we have (36 GB, 400k images) is a document *classification*
dataset — it has category labels but **no text transcriptions or field labels**,
so it cannot train an extraction model. Instead we *generate* labelled data:
because we synthesise the field values, every image ships with perfect ground
truth, for free, at any scale.

## 1. Synthetic training set (generated — gitignored)

`scripts/generate_synthetic_docs.py` → `backend/app/synth/` produces degraded
German documents paired with exact labels:

- **Valid-by-construction data**: Steuer-IDs pass the production ISO 7064
  validator, IBANs pass mod-97, names/streets/cities from `Faker(de_DE)`.
- **Randomised layouts** (fonts, sizes, spacing) so the model doesn't overfit a
  single template — *domain randomisation*.
- **Realistic degradation** (skew, sensor noise, blur, coffee stains, photocopy
  thresholding, JPEG artifacts) so it transfers to real scans.
- **Output**: `images/<type>_<id>.png` + `manifest.{train,val}.jsonl`, one
  `{"image", "document_type", "fields"}` record per line (Donut/TrOCR-ready).

```bash
python scripts/generate_synthetic_docs.py --n 500      # 500 per type → 2,000 docs
```

The four document types (Meldebescheinigung, Steuerbescheid, Gehaltsausweis,
Personalausweis) match the strict Pydantic schemas in `document_schemas.py`.

## 2. Real held-out test set (committed)

`test_documents/` (4 real, degraded German documents — coffee stain, skew, dark
photocopy, handwritten note) + `test_documents/ground_truth.json` (transcribed by
hand). **This is never trained on.** It is the only set used to report accuracy.

## The split, and why it's credible

| | Source | Used for | Seen in training? |
|---|---|---|---|
| Synthetic | generated, thousands/type | train + val | yes (train/val only) |
| `test_documents/` | real scans, 4 docs | **final eval only** | **no** |

Reported numbers come exclusively from the real hold-out via
`scripts/eval_german_documents.py` (OCR pipeline) and `scripts/eval_donut.py`
(the fine-tuned model), using identical metric code (CER / WER / exact-match / F1).

## Outcome

The experiment was run and **measured honestly — see
[training/RESULTS.md](training/RESULTS.md).** Short version: the OCR pipeline
reaches **90%** exact-match on the real hold-out, while the synthetic-trained
Donut reaches **40%** — the synthetic→real gap is too large for an OCR-free model
to beat general OCR on synthetic data alone. The methodology did its job: it
surfaced a real negative result instead of a synthetic-validation mirage.

> Next: expand the real hold-out beyond 4 documents to tighten the confidence
> interval; if revisiting the model, close the synthetic→real gap (more realistic
> data) before adding epochs — see RESULTS.md.
