#!/usr/bin/env python3
"""
German-document extraction accuracy harness — CER / WER + field accuracy.

This is the missing "prove it actually works" piece. It runs the *real*
pipeline (Surya OCR → layout mapping → symbolic validation) over the labelled
images in ``test_documents/`` and reports, per field and in aggregate:

  * field precision / recall / F1  (did we extract the right fields?)
  * exact-match accuracy           (did the value match exactly?)
  * mean CER / WER                 (how close was the value, character/word-wise?)

Ground truth lives in ``test_documents/ground_truth.json`` — fill in the true
values for each image (a template is generated on first run). Fields left blank
are skipped, so you can label incrementally.

Usage::

    python scripts/eval_german_documents.py                 # run the eval
    python scripts/eval_german_documents.py --self-test     # test the metrics
    python scripts/eval_german_documents.py --make-template # (re)write template

No values are ever invented — unlabelled fields simply do not count.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND = PROJECT_ROOT / "backend"
TEST_DOCS = PROJECT_ROOT / "test_documents"
GROUND_TRUTH = TEST_DOCS / "ground_truth.json"


# ---------------------------------------------------------------------------
# Metrics (pure Python — no dependencies)
# ---------------------------------------------------------------------------

def levenshtein(a: list, b: list) -> int:
    """Edit distance between two sequences (lists of chars or words)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    """Character Error Rate = edit distance / truth length."""
    truth = truth or ""
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(list(pred or ""), list(truth)) / len(truth)


def wer(pred: str, truth: str) -> float:
    """Word Error Rate = word edit distance / truth word count."""
    tw = (truth or "").split()
    if not tw:
        return 0.0 if not (pred or "").split() else 1.0
    return levenshtein((pred or "").split(), tw) / len(tw)


def _norm(value: Any) -> str:
    return " ".join(str(value if value is not None else "").strip().split())


_AMOUNT_FIELDS = {
    "zu_versteuerndes_einkommen", "festgesetzte_steuer", "ermässigung",
    "vorauszahlungen", "kirchensteuer", "solidaritätszuschlag",
    "brutto_lohn", "netto_lohn", "rentenversicherung", "krankenversicherung",
    "arbeitslosenversicherung", "pflegeversicherung",
}


def _to_float(text: str) -> Optional[float]:
    """Parse a German/plain numeric string to float (for amount comparison)."""
    import re
    s = (text or "").strip().lower().replace("€", "").replace("eur", "").strip()
    if not re.fullmatch(r"[\d.,]+", s or ""):
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") >= 1 and len(s.split(".")[-1]) == 3:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Ground-truth template
# ---------------------------------------------------------------------------

_TEMPLATE_FIELDS = {
    "Meldebescheinigung": [
        "familienname", "vorname", "geburtsdatum", "geburtsort",
        "staatsangehoerigkeit", "strasse", "hausnummer", "postleitzahl", "wohnort",
    ],
    "Steuerbescheid": [
        "steueridentifikationsnummer", "veranlagungszeitraum",
        "zu_versteuerndes_einkommen", "festgesetzte_steuer",
    ],
    "Gehaltsausweis": ["arbeitgeber", "brutto_lohn", "netto_lohn", "abrechnungszeitraum"],
    "Personalausweis": [
        "dokumentnummer", "familienname", "vorname", "geburtsdatum",
        "geburtsort", "staatsangehoerigkeit", "gueltig_bis",
    ],
}

# Best-effort doc-type guess from the bundled filenames.
_FILENAME_DOCTYPE = {
    "meldebescheinigung": "Meldebescheinigung",
    "steuerbescheid": "Steuerbescheid",
    "gehaltsausweis": "Gehaltsausweis",
    "personalausweis": "Personalausweis",
}


def make_template() -> dict:
    template: dict = {
        "_README": (
            "Fill in the true field values for each image, then run "
            "`python scripts/eval_german_documents.py`. Leave a field as \"\" "
            "to skip it. Do NOT guess — only label what the document actually says."
        )
    }
    for image in sorted(TEST_DOCS.glob("*.png")):
        name = image.name.lower()
        doc_type = next((dt for key, dt in _FILENAME_DOCTYPE.items() if key in name), "Unbekannt")
        fields = _TEMPLATE_FIELDS.get(doc_type, [])
        template[image.name] = {
            "document_type": doc_type,
            "fields": {f: "" for f in fields},
        }
    return template


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def _extract(pipeline: Any, image_path: Path) -> dict[str, Any]:
    result = await pipeline.process(str(image_path))
    return {name: field.value for name, field in result.fields.items()}


def evaluate(ground_truth: dict) -> None:
    sys.path.insert(0, str(BACKEND))
    try:
        from app.pipeline.orchestrator import ExtractionPipeline
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: could not import the pipeline ({exc}).")
        print("Run `bash setup_mac.sh` first so Surya / dependencies are installed.")
        sys.exit(2)

    pipeline = ExtractionPipeline()

    tp = fp = fn = 0          # field presence
    exact = compared = 0      # exact-match accuracy over labelled fields
    cer_sum = wer_sum = 0.0

    for filename, spec in ground_truth.items():
        if filename.startswith("_"):
            continue
        image_path = TEST_DOCS / filename
        if not image_path.is_file():
            print(f"  (skip {filename}: not found)")
            continue
        labelled = {k: v for k, v in spec.get("fields", {}).items() if _norm(v)}
        if not labelled:
            print(f"  (skip {filename}: no labelled fields yet)")
            continue

        predicted = asyncio.run(_extract(pipeline, image_path))
        print(f"\n=== {filename} ({spec.get('document_type', '?')}) ===")
        for field, truth in labelled.items():
            pred = _norm(predicted.get(field))
            truth_n = _norm(truth)
            compared += 1
            if pred:
                tp += 1
            else:
                fn += 1
            if field in _AMOUNT_FIELDS:
                pf, tf = _to_float(pred), _to_float(truth_n)
                is_exact = pf is not None and tf is not None and abs(pf - tf) < 0.005
                cmp_pred = f"{pf:.2f}" if pf is not None else pred
                cmp_truth = f"{tf:.2f}" if tf is not None else truth_n
            else:
                is_exact = pred == truth_n
                cmp_pred, cmp_truth = pred, truth_n
            field_cer = cer(cmp_pred, cmp_truth)
            field_wer = wer(cmp_pred, cmp_truth)
            cer_sum += field_cer
            wer_sum += field_wer
            exact += int(is_exact)
            mark = "OK " if is_exact else "XX "
            print(f"  {mark}{field:30} pred={pred!r:24} truth={truth_n!r:24} CER={field_cer:.2f}")

        # Predicted canonical fields not in ground truth → false positives.
        for field in predicted:
            if field.startswith("_"):
                continue
            if field not in labelled and _norm(predicted.get(field)):
                fp += 1

    print("\n" + "=" * 60)
    if compared == 0:
        print("No labelled fields evaluated. Fill in ground_truth.json first.")
        return
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("AGGREGATE")
    print(f"  fields compared      : {compared}")
    print(f"  exact-match accuracy : {exact / compared:.1%}")
    print(f"  mean CER             : {cer_sum / compared:.3f}")
    print(f"  mean WER             : {wer_sum / compared:.3f}")
    print(f"  field precision      : {precision:.1%}")
    print(f"  field recall         : {recall:.1%}")
    print(f"  field F1             : {f1:.1%}")
    print("=" * 60)


def _self_test() -> None:
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert cer("", "") == 0.0
    assert cer("abc", "abc") == 0.0
    assert abs(cer("abc", "abd") - 1 / 3) < 1e-9
    assert wer("a b c", "a b c") == 0.0
    assert abs(wer("a x c", "a b c") - 1 / 3) < 1e-9
    print("metrics self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="test the metric functions and exit")
    parser.add_argument("--make-template", action="store_true", help="(re)write the ground-truth template")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.make_template or not GROUND_TRUTH.is_file():
        GROUND_TRUTH.write_text(json.dumps(make_template(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote template: {GROUND_TRUTH}")
        if args.make_template:
            return
        print("Fill in the true values, then re-run this script.")
        return

    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    evaluate(ground_truth)


if __name__ == "__main__":
    main()
