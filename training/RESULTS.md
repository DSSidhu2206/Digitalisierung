# Results — does a synthetic-trained specialist beat general OCR?

**Research question.** Can an OCR-free model (Donut), fine-tuned *only* on
synthetic German documents, beat general-purpose OCR pipelines on **real,
degraded** German documents?

**Answer: no — not with this setup.** On the real held-out set the
synthetic-trained Donut reaches **40% exact-match**, well behind the OCR
pipelines at **90% / 85%**. This is a negative result, reported honestly, with
the analysis of *why* — and it is the more useful outcome than a flattering one,
because the methodology was built specifically so the number can be trusted.

## Headline

| System | Approach | Exact-match ↑ | Notes |
|---|---|---|---|
| Surya OCR → layout map → validation | General OCR + symbolic layout/validation | **90%** | production pipeline |
| Apple Vision → layout map | General OCR on the Apple Neural Engine | **85%** | ~23× faster, fully on-device |
| **Donut (synthetic fine-tune)** | OCR-free, image → JSON directly | **40%** | this experiment |

Donut detail: exact-match **40.0%**, mean CER **0.628**, mean WER **0.750**,
field precision **68.2%**, recall **75.0%**, F1 **71.4%**.

Every number comes from the **same harness on the same real hold-out**
(`test_documents/`, 4 documents / 20 labelled fields) that was **never seen in
training** — OCR via `eval_german_documents.py`, Donut via `eval_donut.py`,
identical metric code.

## What was built to test the hypothesis

- **A synthetic data engine** (`backend/app/synth/`, `scripts/generate_synthetic_docs.py`)
  — 12,000 diverse, degraded, photo-realistic German documents, *valid by
  construction* (Steuer-IDs pass the ISO 7064 checksum, IBANs pass mod-97),
  with randomised layouts/fonts and realistic degradation (skew, noise, coffee
  stains, photocopy thresholding, JPEG artifacts).
- **A cloud fine-tune of `donut-base`** (Kaggle T4, 3 epochs over a 6,000-doc
  subset; notebook in `training/train_donut.ipynb`). Final training loss ~1.0.
- **A train-on-synthetic / eval-on-real harness** using the *same* CER / WER /
  exact-match metrics as the OCR baselines, so the comparison is apples-to-apples.

## The result in detail

Per field, Donut on the four real documents:

```
=== 01_meldebescheinigung (coffee stain) ===        4 / 9 correct
  OK familienname  Müller     OK geburtsdatum 15.03.1985   OK geburtsort Hamburg
  OK postleitzahl  10115      XX vorname ''   XX strasse '' XX hausnummer 3≠42
  XX wohnort 'Allaanintreme xon Burgeramt'   XX staatsangehoerigkeit ''
=== 02_steuerbescheid (skewed) ===                  1 / 4 correct
  OK veranlagungszeitraum 2024
  XX steueridentifikationsnummer 04452397687 ≠ ...667   (one digit)
  XX zu_versteuerndes_einkommen 56.400,00 ≠ 45236,00
  XX festgesetzte_steuer '1.200,00 8.947,00'   (merged two numbers)
=== 03_personalausweis (photocopy) ===              3 / 5 correct
  OK vorname Hans  OK geburtsdatum 15.03.1985  OK gueltig_bis 01.09.2030
  XX dokumentnummer T22000129D (+D)   XX familienname MULLER (umlaut lost)
=== 04_gehaltsausweis (handwritten note) ===        0 / 2 correct
  XX arbeitgeber ''   XX brutto_lohn ''
```

It **clearly learned** — a model trained on *zero real data* read real scans
and returned correct `Müller`, `15.03.1985`, `Hamburg`, `10115`, `T22000129`,
`01.09.2030`. That is a real signal that the synthetic pipeline transfers. It
just isn't competitive with mature OCR.

## Why it lost — analysis

1. **The synthetic→real distribution gap is real.** Domain randomisation
   narrows it but doesn't close it: a coffee-stained, skewed, photocopied real
   form is not drawn from the generator's distribution, and the model fit
   synthetic characteristics that don't fully transfer.
2. **OCR-free means it had to learn to *read* from scratch** — on 6,000
   synthetic images. Surya and Apple Vision bring reading ability trained on
   *enormous* real-world text corpora. You cannot out-read a general OCR engine
   with a synthetic-only dataset of this size.
3. **The failure modes confirm it:** dropped fields (`vorname`, `arbeitgeber`,
   `brutto_lohn` came back empty → recall loss), **merged** values
   (`festgesetzte_steuer` = two numbers), and **hallucination** on the most
   degraded regions (`wohnort` = `'Allaanintreme xon Burgeramt'`).

## What would be needed to close the gap

- More, and more *realistic*, synthetic data — matched to the actual layouts and
  fonts of real German forms; train on the full 10,800 docs, more epochs, higher
  input resolution.
- Likely a **hybrid**: keep a general OCR engine for *reading* and learn only the
  *structure* — which is, in effect, what the 90% production pipeline already is.
- Honest expectation: 40% → 90% on a synthetic-only OCR-free model is a hard
  climb with uncertain payoff. The OCR pipeline is the pragmatic production
  choice, and this experiment is the evidence for that decision.

## What this demonstrates

- **Experimental design** and the discipline to evaluate on *real* held-out data
  rather than the synthetic validation loss the model could memorise.
- A full ML pipeline built end-to-end: data generation → cloud training →
  evaluation, including the unglamorous engineering (T4 OOM debugging, tokenizer
  reconstruction, reproducible eval).
- **Honest reporting of a negative result** — which, for the production decision
  and for the science, is worth more than a cherry-picked win.

## Reproduce

```bash
pip install sentencepiece                     # tokenizer dep for Donut
python scripts/eval_donut.py                   # Donut on the real hold-out (40%)
python scripts/eval_german_documents.py        # the OCR pipeline baseline (90%)
```
