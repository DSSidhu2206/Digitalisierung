#!/usr/bin/env python3
"""
Benchmark real document extraction engines against real image data.

This script intentionally separates:
  - measured OCR/KIE accuracy on datasets with human ground truth
  - measured throughput on local RVL-CDIP images without ground-truth text
  - measured learning-memory ingestion stats

It does not create synthetic ground truth and does not report OCR accuracy for
RVL-CDIP unless a manifest with transcripts is supplied.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SROIE_FIELDS = ("company", "date", "address", "total")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ExtractionOutput:
    text: str
    fields: dict[str, str]
    raw: str


class BenchmarkEngine:
    name = "base"
    model_id = ""
    structured_fields = False

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def extract_one(self, image_path: Path) -> ExtractionOutput:
        raise NotImplementedError

    def extract_batch(self, image_paths: list[Path]) -> list[ExtractionOutput]:
        return [self.extract_one(path) for path in image_paths]


class SuryaEngine(BenchmarkEngine):
    name = "surya"
    model_id = "surya-ocr"
    structured_fields = False

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.extractor: Any = None

    def load(self) -> None:
        from app.vision.surya_extractor import SuryaDocumentExtractor

        self.extractor = SuryaDocumentExtractor(enable_layout=False, allow_fallback=False)
        self.extractor.load()
        self.model_id = f"surya-ocr:{getattr(self.extractor, 'device', 'unknown')}"

    def unload(self) -> None:
        if self.extractor is not None:
            self.extractor.unload()

    def extract_batch(self, image_paths: list[Path]) -> list[ExtractionOutput]:
        results = self.extractor.extract_batch(image_paths, with_layout=False)
        return [
            ExtractionOutput(text=result.text or "", fields={}, raw=result.text or "")
            for result in results
        ]

    def extract_one(self, image_path: Path) -> ExtractionOutput:
        return self.extract_batch([image_path])[0]


class MlxVlmEngine(BenchmarkEngine):
    name = "mlx-vlm"
    structured_fields = True

    def __init__(self, model_id: str, max_tokens: int) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.model: Any = None
        self.processor: Any = None
        self.config: Any = None
        self.generate: Any = None
        self.apply_chat_template: Any = None

    def load(self) -> None:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(self.model_id)
        self.config = load_config(self.model_id)
        self.generate = generate
        self.apply_chat_template = apply_chat_template

    def extract_one(self, image_path: Path) -> ExtractionOutput:
        prompt = (
            "Read this receipt/document image. Return strict JSON only with keys "
            "full_text, company, date, address, total. Use empty strings for "
            "fields that are not visible. full_text must contain all visible "
            "text in reading order."
        )
        formatted = self.apply_chat_template(
            self.processor,
            self.config,
            prompt,
            num_images=1,
        )
        raw = self.generate(
            self.model,
            self.processor,
            formatted,
            [str(image_path)],
            max_tokens=self.max_tokens,
            temperature=0.0,
            verbose=False,
        )
        if hasattr(raw, "text"):
            raw = raw.text
        if not isinstance(raw, str):
            raw = str(raw)
        parsed = parse_jsonish(raw)
        fields = {field: str(parsed.get(field, "") or "") for field in SROIE_FIELDS}
        text = str(parsed.get("full_text", "") or raw)
        return ExtractionOutput(text=text, fields=fields, raw=raw)


class LightOnEngine(BenchmarkEngine):
    name = "lightonocr"
    model_id = "lightonai/LightOnOCR-2-1B"
    structured_fields = False

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.device = "cpu"
        self.dtype: Any = None
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float32 if self.device == "mps" else torch.bfloat16
        self.processor = LightOnOcrProcessor.from_pretrained(self.model_id)
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.model_id = f"{self.model_id}:{self.device}"

    def extract_one(self, image_path: Path) -> ExtractionOutput:
        import torch

        prompt = (
            "Return strict JSON only with keys full_text, company, date, address, total. "
            "full_text must contain all visible text in reading order."
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if hasattr(value, "is_floating_point") and value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        raw = self.processor.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_jsonish(raw)
        fields = {field: str(parsed.get(field, "") or "") for field in SROIE_FIELDS}
        text = str(parsed.get("full_text", "") or raw)
        return ExtractionOutput(text=text, fields=fields, raw=raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark document extraction models.")
    parser.add_argument(
        "--engine",
        choices=["surya", "mlx-vlm", "lightonocr", "learning", "prepare-sroie"],
        required=True,
    )
    parser.add_argument("--model", default="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sroie-limit", type=int, default=8)
    parser.add_argument("--rvl-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--rvl-dir", default="documents/rvl_cdip")
    parser.add_argument("--rvl-manifest", default="documents/rvl_cdip_manifest.json")
    parser.add_argument(
        "--sroie-gold-jsonl",
        default="",
        help="Optional pre-exported SROIE gold JSONL with image_path/entities/words.",
    )
    parser.add_argument("--learning-train-limit", type=int, default=64)
    parser.add_argument("--skip-sroie", action="store_true")
    parser.add_argument("--skip-rvl", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "requested_model": args.model,
        "environment": environment_snapshot(),
        "notes": [],
        "sroie_accuracy": None,
        "rvl_throughput": None,
        "learning_ingestion": None,
    }

    if args.engine == "learning":
        report["learning_ingestion"] = run_learning_ingestion(args, run_dir)
        report["notes"].append(
            "Learning ingestion measures retrieval-memory growth, not neural weight fine-tuning."
        )
        write_json(run_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return 0

    if args.engine == "prepare-sroie":
        records = load_sroie_records(args.sroie_limit, run_dir)
        report["sroie_gold_jsonl"] = str(run_dir / "sroie_gold.jsonl")
        report["sroie_samples"] = len(records)
        report["notes"].append("Prepared real SROIE images and human ground truth for shared model benchmarks.")
        write_json(run_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return 0

    engine = build_engine(args)
    load_started = time.perf_counter()
    engine.load()
    report["model_id"] = engine.model_id
    report["model_load_seconds"] = round(time.perf_counter() - load_started, 4)

    try:
        if not args.skip_sroie:
            sroie_records = load_sroie_records(args.sroie_limit, run_dir, args.sroie_gold_jsonl)
            report["sroie_accuracy"] = run_sroie_accuracy(engine, sroie_records, run_dir, args.batch_size)
        if not args.skip_rvl:
            rvl_paths = load_rvl_paths(args.rvl_manifest, args.rvl_dir, args.rvl_limit)
            report["rvl_throughput"] = run_rvl_throughput(engine, rvl_paths, run_dir, args.batch_size)
            report["notes"].append(
                "RVL-CDIP local manifest has no human text transcripts, so OCR accuracy is not measurable on RVL in this run."
            )
    finally:
        engine.unload()

    report["total_seconds"] = round(time.perf_counter() - started, 4)
    write_json(run_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def build_engine(args: argparse.Namespace) -> BenchmarkEngine:
    if args.engine == "surya":
        return SuryaEngine(batch_size=args.batch_size)
    if args.engine == "mlx-vlm":
        return MlxVlmEngine(args.model, max_tokens=args.max_tokens)
    if args.engine == "lightonocr":
        return LightOnEngine(max_tokens=args.max_tokens)
    raise ValueError(f"Unsupported engine: {args.engine}")


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "logs" / "benchmarks" / f"{stamp}-{args.engine}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["torch_mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:
        snapshot["torch"] = f"unavailable: {exc}"
    try:
        import mlx.core as mx

        arr = mx.array([1.0, 2.0])
        snapshot["mlx_available"] = bool(float(mx.sum(arr)) == 3.0)
    except Exception as exc:
        snapshot["mlx_available"] = False
        snapshot["mlx_error"] = str(exc)
    return snapshot


def load_sroie_records(limit: int, run_dir: Path, gold_jsonl: str = "") -> list[dict[str, Any]]:
    if gold_jsonl:
        records = []
        with Path(gold_jsonl).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= limit:
                    break
        return records

    from datasets import load_dataset

    dataset = load_dataset("jsdnrs/ICDAR2019-SROIE", split=f"test[:{limit}]")
    image_dir = run_dir / "sroie_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        key = str(item["key"])
        image_path = image_dir / f"{index:04d}_{key}.jpg"
        image = item["image"].convert("RGB")
        image.save(image_path)
        records.append(
            {
                "key": key,
                "image_path": str(image_path),
                "entities": item["entities"],
                "words": item["words"],
            }
        )
    write_jsonl(run_dir / "sroie_gold.jsonl", records)
    return records


def run_sroie_accuracy(
    engine: BenchmarkEngine,
    records: list[dict[str, Any]],
    run_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    latencies: list[float] = []
    text_scores: list[float] = []
    field_exact_scores: list[float] = []
    field_fuzzy_scores: list[float] = []
    errors = 0

    for batch in chunks(records, batch_size if isinstance(engine, SuryaEngine) else 1):
        paths = [Path(record["image_path"]) for record in batch]
        started = time.perf_counter()
        try:
            extracted = engine.extract_batch(paths)
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            extracted = [ExtractionOutput(text="", fields={}, raw=f"ERROR: {exc}")]
            errors += len(batch)
        per_item_latency = elapsed / max(len(batch), 1)
        for record, result in zip(batch, extracted):
            gold_text = "\n".join(str(word) for word in record.get("words", []))
            text_similarity = similarity(result.text, gold_text)
            field_metrics = compare_fields(result.fields, record.get("entities", {}))
            text_scores.append(text_similarity)
            field_exact_scores.append(field_metrics["exact_match_rate"])
            field_fuzzy_scores.append(field_metrics["mean_similarity"])
            latencies.append(per_item_latency)
            outputs.append(
                {
                    "key": record["key"],
                    "latency_seconds": per_item_latency,
                    "text_similarity": text_similarity,
                    "field_metrics": field_metrics,
                    "gold_entities": record.get("entities", {}),
                    "predicted_fields": result.fields,
                    "raw_output": result.raw,
                    "text_chars": len(result.text),
                }
            )

    write_jsonl(run_dir / f"{engine.name}_sroie_outputs.jsonl", outputs)
    return {
        "dataset": "jsdnrs/ICDAR2019-SROIE:test",
        "samples": len(records),
        "errors": errors,
        "text_similarity_mean": mean(text_scores),
        "field_exact_match_mean": mean(field_exact_scores) if engine.structured_fields else None,
        "field_similarity_mean": mean(field_fuzzy_scores) if engine.structured_fields else None,
        "structured_fields_supported": engine.structured_fields,
        "latency_seconds_mean": mean(latencies),
        "latency_seconds_p50": percentile(latencies, 50),
        "latency_seconds_p95": percentile(latencies, 95),
        "images_per_second": round(len(records) / sum(latencies), 4) if latencies and sum(latencies) else 0.0,
    }


def load_rvl_paths(manifest_path: str, image_dir: str, limit: int) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    base = Path(image_dir)
    paths: list[Path] = []
    for item in manifest:
        path = base / item["filename"]
        if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def run_rvl_throughput(
    engine: BenchmarkEngine,
    paths: list[Path],
    run_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    batch_latencies: list[float] = []
    item_latencies: list[float] = []
    char_counts: list[int] = []
    errors = 0
    started_all = time.perf_counter()

    for batch in chunks(paths, batch_size if isinstance(engine, SuryaEngine) else 1):
        started = time.perf_counter()
        try:
            extracted = engine.extract_batch(batch)
            elapsed = time.perf_counter() - started
        except Exception as exc:
            elapsed = time.perf_counter() - started
            extracted = [ExtractionOutput(text="", fields={}, raw=f"ERROR: {exc}") for _ in batch]
            errors += len(batch)
        batch_latencies.append(elapsed)
        per_item = elapsed / max(len(batch), 1)
        for path, result in zip(batch, extracted):
            item_latencies.append(per_item)
            char_counts.append(len(result.text))
            outputs.append(
                {
                    "filename": path.name,
                    "latency_seconds": per_item,
                    "text_chars": len(result.text),
                    "raw_output": result.raw[:4000],
                }
            )

    elapsed_all = time.perf_counter() - started_all
    write_jsonl(run_dir / f"{engine.name}_rvl_outputs.jsonl", outputs)
    return {
        "dataset": "local documents/rvl_cdip",
        "samples": len(paths),
        "errors": errors,
        "elapsed_seconds": round(elapsed_all, 4),
        "images_per_second": round(len(paths) / elapsed_all, 4) if elapsed_all else 0.0,
        "latency_seconds_mean": mean(item_latencies),
        "latency_seconds_p50": percentile(item_latencies, 50),
        "latency_seconds_p95": percentile(item_latencies, 95),
        "mean_extracted_chars": mean(char_counts),
        "nonempty_text_rate": round(sum(1 for count in char_counts if count > 0) / len(char_counts), 4)
        if char_counts
        else 0.0,
    }


def run_learning_ingestion(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from app.database.image_learning_store import ImageLearningStore, LearnedImageRecord
    from scripts.ingest_learning_images import (
        build_record_metadata,
        file_sha256,
        load_manifest,
    )
    from app.vision.quality_gate import QualityGate

    manifest = load_manifest(Path(args.rvl_manifest))
    store_dir = run_dir / "learning_store"
    store = ImageLearningStore(persist_dir=str(store_dir))
    quality_gate = QualityGate()
    records: list[LearnedImageRecord] = []
    started = time.perf_counter()
    scanned = 0
    skipped = 0
    for item in manifest:
        if scanned >= args.learning_train_limit:
            break
        image_path = Path(args.rvl_dir) / item["filename"]
        if not image_path.exists():
            skipped += 1
            continue
        image_id = file_sha256(image_path)
        candidate = build_record_metadata(image_path, item, quality_gate, image_id=image_id)
        scanned += 1
        if candidate is None:
            skipped += 1
            continue
        records.append(
            LearnedImageRecord(
                image_id=candidate["image_id"],
                image_path=str(candidate["image_path"]),
                filename=candidate["filename"],
                source_split=candidate.get("source_split", ""),
                label_name=candidate.get("label_name", ""),
                label_index=candidate.get("label_index"),
                width=candidate.get("width"),
                height=candidate.get("height"),
                file_size=candidate.get("file_size"),
                sha256=candidate.get("sha256", ""),
                quality_score=candidate.get("quality_score", 0.0),
                form_type=candidate.get("form_type", "Unbekannt"),
                orientation=candidate.get("orientation", "correct"),
                ocr_text="",
                visual_hash=candidate.get("visual_hash", ""),
            )
        )
    stored = store.add_records(records)
    stats = store.stats()
    retrieved = store.retrieve_similar("quality=0.5 orientation=correct", limit=3)
    store.close()
    elapsed = time.perf_counter() - started
    return {
        "dataset": "local documents/rvl_cdip",
        "samples_scanned": scanned,
        "records_prepared": len(records),
        "records_stored": stored,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 4),
        "records_per_second": round(stored / elapsed, 4) if elapsed else 0.0,
        "store_stats": stats,
        "retrieval_smoke_count": len(retrieved),
        "accuracy_measured": False,
        "accuracy_note": (
            "No OCR transcripts or labels exist in the local RVL manifest, so this "
            "learning test measures memory ingestion and retrieval availability only."
        ),
    }


def compare_fields(predicted: dict[str, str], gold: dict[str, str]) -> dict[str, Any]:
    exact = []
    sims = []
    per_field: dict[str, Any] = {}
    for field in SROIE_FIELDS:
        pred = predicted.get(field, "")
        expected = str(gold.get(field, "") or "")
        pred_norm = normalize_text(pred)
        expected_norm = normalize_text(expected)
        is_exact = bool(expected_norm) and pred_norm == expected_norm
        sim = similarity(pred, expected)
        exact.append(1.0 if is_exact else 0.0)
        sims.append(sim)
        per_field[field] = {
            "predicted": pred,
            "expected": expected,
            "exact": is_exact,
            "similarity": sim,
        }
    return {
        "exact_match_rate": mean(exact),
        "mean_similarity": mean(sims),
        "fields": per_field,
    }


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    braced = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if braced:
        candidates.insert(0, braced.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<\|LOC_\d+\|>", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9./:& -]+", " ", value.lower())).strip()


def similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return round(SequenceMatcher(None, a_norm, b_norm).ratio(), 4)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else 0.0


def percentile(values: Iterable[float], pct: int) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 4)
    index = (len(values) - 1) * (pct / 100)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return round(values[lower] * (1 - fraction) + values[upper] * fraction, 4)


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    size = max(1, size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
