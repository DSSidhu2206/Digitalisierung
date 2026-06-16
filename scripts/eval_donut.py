#!/usr/bin/env python3
"""
Donut head-to-head eval on the REAL held-out documents.

Runs the fine-tuned OCR-free Donut model (image -> field JSON, no OCR step)
over ``test_documents/`` and reports the *same* CER / WER / exact-match metrics
as ``eval_german_documents.py``, so the result is directly comparable to the
OCR baselines (Surya ~90% / Apple Vision ~85%).

The model was trained at 1280x960 input, so we pin that here too — feeding a
different resolution at inference would not match what it learned.

    python scripts/eval_donut.py
    python scripts/eval_donut.py --model models/donut-german --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DOCS = PROJECT_ROOT / "test_documents"
GROUND_TRUTH = TEST_DOCS / "ground_truth.json"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the identical metrics + comparison helpers from the OCR eval so the
# numbers mean exactly the same thing.
from eval_german_documents import cer, wer, _norm, _AMOUNT_FIELDS, _to_float  # noqa: E402

TASK = "<s_docextract>"
INPUT_SIZE = {"height": 1280, "width": 960}   # must match training
MAX_LEN = 384


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_dir: str, device: str):
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    # The checkpoint's tokenizer files don't load standalone (the sentencepiece
    # vocab wasn't saved), so start from the base tokenizer — which has the full
    # vocab — and re-add the field tokens in the SAME id order training used, so
    # token ids line up with the model's resized embedding rows.
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()

    existing = set(processor.tokenizer.get_vocab())
    tok_json = json.loads((Path(model_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    extra = sorted(tok_json.get("added_tokens", []), key=lambda t: t["id"])
    new_tokens = [t["content"] for t in extra if t["content"] not in existing]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    processor.image_processor.size = INPUT_SIZE

    emb_rows = model.decoder.get_input_embeddings().weight.shape[0]
    print(f"  tokenizer={len(processor.tokenizer)}  model_emb={emb_rows}  re-added={len(new_tokens)}")
    assert len(processor.tokenizer) == emb_rows, "tokenizer/model vocab mismatch"

    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK)
    return processor, model


def predict(processor, model, image_path: Path, device: str) -> dict:
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values,
            max_length=MAX_LEN,
            decoder_start_token_id=model.config.decoder_start_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    seq = processor.batch_decode(out)[0]
    for tok in (TASK, processor.tokenizer.eos_token, processor.tokenizer.pad_token, "<s>"):
        if tok:
            seq = seq.replace(tok, "")
    parsed = processor.token2json(seq)
    # Unwrap a possible {"docextract": {...}} wrapper from the task token.
    if isinstance(parsed, dict) and len(parsed) == 1:
        only = next(iter(parsed.values()))
        if isinstance(only, dict):
            parsed = only
    return parsed if isinstance(parsed, dict) else {}


def evaluate(processor, model, device: str, ground_truth: dict) -> None:
    tp = fp = fn = 0
    exact = compared = 0
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
            continue

        t0 = time.time()
        predicted = predict(processor, model, image_path, device)
        dt = time.time() - t0
        print(f"\n=== {filename} ({spec.get('document_type', '?')})  [{dt:.1f}s] ===")

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
                cmp_pred, cmp_truth = pred.casefold(), truth_n.casefold()
                is_exact = cmp_pred == cmp_truth
            field_cer = cer(cmp_pred, cmp_truth)
            field_wer = wer(cmp_pred, cmp_truth)
            cer_sum += field_cer
            wer_sum += field_wer
            exact += int(is_exact)
            mark = "OK " if is_exact else "XX "
            print(f"  {mark}{field:30} pred={pred!r:24} truth={truth_n!r:24} CER={field_cer:.2f}")

        for field in predicted:
            if str(field).startswith("_"):
                continue
            if field not in labelled and _norm(predicted.get(field)):
                fp += 1

    print("\n" + "=" * 60)
    if compared == 0:
        print("No labelled fields evaluated.")
        return
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"DONUT (OCR-free)  —  device={device}")
    print(f"  fields compared      : {compared}")
    print(f"  exact-match accuracy : {exact / compared:.1%}")
    print(f"  mean CER             : {cer_sum / compared:.3f}")
    print(f"  mean WER             : {wer_sum / compared:.3f}")
    print(f"  field precision      : {precision:.1%}")
    print(f"  field recall         : {recall:.1%}")
    print(f"  field F1             : {f1:.1%}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/donut-german")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    args = parser.parse_args()

    if not GROUND_TRUTH.is_file():
        print("No ground_truth.json — run eval_german_documents.py --make-template first.")
        sys.exit(2)

    device = pick_device(args.device)
    model_dir = str((PROJECT_ROOT / args.model).resolve())
    print(f"Loading {model_dir} on {device} ...")
    t0 = time.time()
    processor, model = load_model(model_dir, device)
    print(f"loaded in {time.time() - t0:.1f}s")

    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    evaluate(processor, model, device, ground_truth)


if __name__ == "__main__":
    main()
