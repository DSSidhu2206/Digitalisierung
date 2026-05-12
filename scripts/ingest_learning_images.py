#!/usr/bin/env python3
"""
Ingest images into the persistent image-learning corpus.

This is designed for long-running RVL-CDIP feeds.  It is resumable, commits in
batches, and stores OCR/quality/label signals in SQLite + ChromaDB so future
extractions can retrieve similar dirty-image examples.

Examples:
    PYTHONPATH=backend python scripts/ingest_learning_images.py --limit 1000
    PYTHONPATH=backend python scripts/ingest_learning_images.py --image-dir documents/rvl_cdip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database.image_learning_store import ImageLearningStore, LearnedImageRecord
from app.vision.surya_extractor import SuryaDocumentExtractor
from app.vision.quality_gate import QualityGate

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest document images for learning.")
    parser.add_argument("--image-dir", default="documents/rvl_cdip")
    parser.add_argument("--manifest", default="documents/rvl_cdip_manifest.json")
    parser.add_argument("--persist-dir", default="./chroma_data")
    parser.add_argument("--progress", default="documents/image_learning_progress.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=1.0,
        help="Minimum seconds between progress heartbeat writes while processing images.",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR and store metadata only.")
    parser.add_argument(
        "--allow-ocr-fallback",
        action="store_true",
        help="Allow pytesseract fallback if Surya cannot load. Off by default.",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=int(os.getenv("OCR_BATCH_SIZE", "8")),
        help="Number of images to send through Surya OCR per batch.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="Stop after this many seconds; useful for scheduled multi-day runs.",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=0,
        help="Stop when this many consecutive images add no new learning records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir)
    manifest_path = Path(args.manifest)
    progress_path = Path(args.progress)

    manifest = load_manifest(manifest_path)
    progress = load_json(progress_path, {"processed": 0, "stored": 0, "skipped": 0})
    store = ImageLearningStore(persist_dir=args.persist_dir)
    quality_gate = QualityGate()
    ocr_engine = (
        None
        if args.no_ocr
        else SuryaDocumentExtractor(
            enable_layout=False,
            allow_fallback=args.allow_ocr_fallback,
        )
    )

    batch: list[LearnedImageRecord] = []
    started = time.perf_counter()
    processed = int(progress.get("resume_index", progress.get("processed", 0)) or 0)
    run_start_processed = processed
    stored_total = int(progress.get("stored", 0))
    skipped = int(progress.get("skipped", 0))
    limit = args.limit
    no_new_records = 0
    total_images = len(manifest) if manifest else count_images(image_dir)
    last_heartbeat = 0.0
    pending_batch = 0
    pending_ocr: list[dict[str, Any]] = []
    heartbeat = ProgressHeartbeat(
        progress_path,
        started=started,
        run_start_processed=run_start_processed,
        interval=args.heartbeat_seconds,
    )

    heartbeat.update(
        processed,
        stored_total,
        skipped,
        total_images=total_images,
        pending_batch=pending_batch,
        state="running",
        last_event="started",
        write_now=True,
    )
    heartbeat.start()
    exit_code = 0
    final_state = "idle"
    final_event = "finished"

    try:
        if ocr_engine is not None:
            ocr_engine.load()
            print(f"OCR engine loaded: {ocr_engine.engine}", flush=True)
            if ocr_engine.engine != "surya" and not args.allow_ocr_fallback:
                raise RuntimeError("Surya OCR is required but did not load.")

        for index, item in enumerate(iter_manifest_or_files(image_dir, manifest), start=1):
            if args.max_seconds and time.perf_counter() - started >= args.max_seconds:
                print(f"Stopping: reached --max-seconds={args.max_seconds}.")
                break
            if index <= processed:
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1

            image_path = image_dir / item["filename"]
            last_heartbeat = maybe_save_progress(
                progress_path,
                processed,
                stored_total,
                skipped,
                started,
                last_heartbeat,
                args.heartbeat_seconds,
                heartbeat=heartbeat,
                run_start_processed=run_start_processed,
                total_images=total_images,
                current_file=item["filename"],
                current_index=index,
                pending_batch=pending_batch,
                state="running",
                last_event="processing",
                force=True,
            )
            if not image_path.exists():
                skipped += 1
                last_heartbeat = maybe_save_progress(
                    progress_path,
                    processed,
                    stored_total,
                    skipped,
                    started,
                    last_heartbeat,
                    args.heartbeat_seconds,
                    heartbeat=heartbeat,
                    run_start_processed=run_start_processed,
                    total_images=total_images,
                    current_file=item["filename"],
                    current_index=index,
                    pending_batch=pending_batch,
                    state="running",
                    last_event="missing_file",
                    force=True,
                )
                continue

            image_id = file_sha256(image_path)
            if store.has_image(image_id):
                skipped += 1
                no_new_records += 1
                last_heartbeat = maybe_save_progress(
                    progress_path,
                    processed,
                    stored_total,
                    skipped,
                    started,
                    last_heartbeat,
                    args.heartbeat_seconds,
                    heartbeat=heartbeat,
                    run_start_processed=run_start_processed,
                    total_images=total_images,
                    current_file=item["filename"],
                    current_index=index,
                    pending_batch=pending_batch,
                    state="running",
                    last_event="already_indexed",
                    force=True,
                )
                if args.plateau_window and no_new_records >= args.plateau_window:
                    print(f"Stopping: no new learning records in last {no_new_records} images.")
                    break
                continue

            candidate = build_record_metadata(
                image_path,
                item,
                quality_gate,
                image_id=image_id,
            )
            if candidate is None:
                skipped += 1
                no_new_records += 1
                last_heartbeat = maybe_save_progress(
                    progress_path,
                    processed,
                    stored_total,
                    skipped,
                    started,
                    last_heartbeat,
                    args.heartbeat_seconds,
                    heartbeat=heartbeat,
                    run_start_processed=run_start_processed,
                    total_images=total_images,
                    current_file=item["filename"],
                    current_index=index,
                    pending_batch=pending_batch,
                    state="running",
                    last_event="skipped",
                    force=True,
                )
                continue

            pending_ocr.append(candidate)
            pending_batch = len(batch) + len(pending_ocr)
            last_heartbeat = maybe_save_progress(
                progress_path,
                processed,
                stored_total,
                skipped,
                started,
                last_heartbeat,
                args.heartbeat_seconds,
                heartbeat=heartbeat,
                run_start_processed=run_start_processed,
                total_images=total_images,
                current_file=item["filename"],
                current_index=index,
                pending_batch=pending_batch,
                state="running",
                last_event="queued_for_ocr",
            )

            if len(pending_ocr) >= max(args.ocr_batch_size, 1):
                last_heartbeat = maybe_save_progress(
                    progress_path,
                    processed,
                    stored_total,
                    skipped,
                    started,
                    last_heartbeat,
                    args.heartbeat_seconds,
                    heartbeat=heartbeat,
                    run_start_processed=run_start_processed,
                    total_images=total_images,
                    current_file=item["filename"],
                    current_index=index,
                    pending_batch=len(batch) + len(pending_ocr),
                    state="running",
                    last_event="ocr_batch_started",
                    force=True,
                )
                new_records = build_records_from_candidates(
                    pending_ocr,
                    skip_ocr=args.no_ocr,
                    ocr_engine=ocr_engine,
                )
                skipped += len(pending_ocr) - len(new_records)
                no_new_records = 0 if new_records else no_new_records + len(pending_ocr)
                pending_ocr.clear()
                batch.extend(new_records)
                pending_batch = len(batch)
                if batch:
                    stored_total += flush_batch(store, batch)
                    batch.clear()
                    pending_batch = 0
                last_heartbeat = maybe_save_progress(
                    progress_path,
                    processed,
                    stored_total,
                    skipped,
                    started,
                    last_heartbeat,
                    args.heartbeat_seconds,
                    heartbeat=heartbeat,
                    run_start_processed=run_start_processed,
                    total_images=total_images,
                    current_file=item["filename"],
                    current_index=index,
                    pending_batch=pending_batch,
                    state="running",
                    last_event="indexed_ocr_batch",
                    force=True,
                )

            if len(batch) >= args.batch_size:
                stored_total += flush_batch(store, batch)
                batch.clear()
                pending_batch = 0
                last_heartbeat = maybe_save_progress(
                    progress_path,
                    processed,
                    stored_total,
                    skipped,
                    started,
                    last_heartbeat,
                    args.heartbeat_seconds,
                    heartbeat=heartbeat,
                    run_start_processed=run_start_processed,
                    total_images=total_images,
                    current_file=item["filename"],
                    current_index=index,
                    pending_batch=pending_batch,
                    state="running",
                    last_event="indexed_batch",
                    force=True,
                )

        if pending_ocr:
            last_heartbeat = maybe_save_progress(
                progress_path,
                processed,
                stored_total,
                skipped,
                started,
                last_heartbeat,
                args.heartbeat_seconds,
                heartbeat=heartbeat,
                run_start_processed=run_start_processed,
                total_images=total_images,
                pending_batch=len(batch) + len(pending_ocr),
                state="running",
                last_event="ocr_batch_started",
                force=True,
            )
            new_records = build_records_from_candidates(
                pending_ocr,
                skip_ocr=args.no_ocr,
                ocr_engine=ocr_engine,
            )
            skipped += len(pending_ocr) - len(new_records)
            pending_ocr.clear()
            batch.extend(new_records)
            pending_batch = len(batch)

        if batch:
            stored_total += flush_batch(store, batch)
            batch.clear()
            pending_batch = 0

    except Exception:
        exit_code = 1
        final_state = "error"
        final_event = "failed"
        traceback.print_exc()
    finally:
        if ocr_engine is not None:
            ocr_engine.unload()
        final_pending = max(pending_batch, len(batch) + len(pending_ocr))
        heartbeat.stop(
            processed,
            stored_total,
            skipped,
            total_images=total_images,
            pending_batch=final_pending,
            state=final_state,
            last_event=final_event,
        )
        store.close()

    elapsed = time.perf_counter() - started
    print(
        f"Processed={processed} stored={stored_total} skipped={skipped} "
        f"elapsed={elapsed:.1f}s"
    )
    return exit_code


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ProgressHeartbeat:
    """Write lightweight progress updates while a slow image is still running."""

    def __init__(
        self,
        path: Path,
        *,
        started: float,
        run_start_processed: int,
        interval: float,
    ) -> None:
        self.path = path
        self.started = started
        self.run_start_processed = run_start_processed
        self.interval = max(interval, 0.25)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot: dict[str, Any] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="progress-heartbeat", daemon=True)
        self._thread.start()

    def update(
        self,
        processed: int,
        stored: int,
        skipped: int,
        *,
        total_images: int = 0,
        current_file: str = "",
        current_index: Optional[int] = None,
        pending_batch: int = 0,
        state: str = "running",
        last_event: str = "progress",
        write_now: bool = False,
    ) -> None:
        with self._lock:
            self._snapshot = {
                "processed": processed,
                "stored": stored,
                "skipped": skipped,
                "total_images": total_images,
                "current_file": current_file,
                "current_index": current_index,
                "pending_batch": pending_batch,
                "state": state,
                "last_event": last_event,
            }
        if write_now:
            self.write()

    def stop(
        self,
        processed: int,
        stored: int,
        skipped: int,
        *,
        total_images: int = 0,
        current_file: str = "",
        current_index: Optional[int] = None,
        pending_batch: int = 0,
        state: str = "idle",
        last_event: str = "finished",
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.update(
            processed,
            stored,
            skipped,
            total_images=total_images,
            current_file=current_file,
            current_index=current_index,
            pending_batch=pending_batch,
            state=state,
            last_event=last_event,
            write_now=True,
        )

    def write(self) -> None:
        with self._lock:
            snapshot = dict(self._snapshot)
        if not snapshot:
            return
        save_progress(
            self.path,
            snapshot["processed"],
            snapshot["stored"],
            snapshot["skipped"],
            self.started,
            run_start_processed=self.run_start_processed,
            total_images=snapshot.get("total_images", 0),
            current_file=snapshot.get("current_file", ""),
            current_index=snapshot.get("current_index"),
            pending_batch=snapshot.get("pending_batch", 0),
            state=snapshot.get("state", "running"),
            last_event=snapshot.get("last_event", "progress"),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.write()


def save_progress(
    path: Path,
    processed: int,
    stored: int,
    skipped: int,
    started: float,
    *,
    run_start_processed: int = 0,
    total_images: int = 0,
    current_file: str = "",
    current_index: Optional[int] = None,
    pending_batch: int = 0,
    state: str = "running",
    last_event: str = "progress",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    elapsed = max(time.perf_counter() - started, 0.001)
    run_processed = max(processed - run_start_processed, 0)
    payload = {
        "state": state,
        "processed": processed,
        "resume_index": max(processed - pending_batch, 0) if pending_batch else processed,
        "stored": stored,
        "skipped": skipped,
        "total_images": total_images,
        "current_file": current_file,
        "current_index": current_index,
        "pending_batch": pending_batch,
        "last_event": last_event,
        "run_processed": run_processed,
        "images_per_second": round(run_processed / elapsed, 4),
        "elapsed_seconds": round(elapsed, 3),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at_epoch": time.time(),
    }
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def maybe_save_progress(
    path: Path,
    processed: int,
    stored: int,
    skipped: int,
    started: float,
    last_heartbeat: float,
    heartbeat_seconds: float,
    *,
    force: bool = False,
    heartbeat: Optional[ProgressHeartbeat] = None,
    run_start_processed: int = 0,
    total_images: int = 0,
    current_file: str = "",
    current_index: Optional[int] = None,
    pending_batch: int = 0,
    state: str = "running",
    last_event: str = "progress",
) -> float:
    now = time.perf_counter()
    if force or now - last_heartbeat >= heartbeat_seconds:
        if heartbeat is not None:
            heartbeat.update(
                processed,
                stored,
                skipped,
                total_images=total_images,
                current_file=current_file,
                current_index=current_index,
                pending_batch=pending_batch,
                state=state,
                last_event=last_event,
                write_now=force,
            )
        else:
            save_progress(
                path,
                processed,
                stored,
                skipped,
                started,
                run_start_processed=run_start_processed,
                total_images=total_images,
                current_file=current_file,
                current_index=current_index,
                pending_batch=pending_batch,
                state=state,
                last_event=last_event,
            )
        return now
    return last_heartbeat


def iter_manifest_or_files(
    image_dir: Path,
    manifest: list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    if manifest:
        yield from manifest
        return
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield {"filename": path.name}


def count_images(image_dir: Path) -> int:
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def build_record_metadata(
    image_path: Path,
    item: dict[str, Any],
    quality_gate: QualityGate,
    image_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build all record metadata that does not require OCR."""
    try:
        sha256 = image_id or file_sha256(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
            file_size = image_path.stat().st_size

        try:
            assessment = quality_gate.assess(str(image_path))
            quality_score = assessment.legibility_score
            form_type = assessment.form_type.value
            orientation = assessment.orientation
        except Exception:
            quality_score = 0.0
            form_type = "Unbekannt"
            orientation = "correct"

        label_index = item.get("label_index")
        return {
            "image_id": sha256,
            "image_path": image_path,
            "filename": image_path.name,
            "source_split": str(item.get("source_split") or ""),
            "label_name": str(item.get("label_name") or ""),
            "label_index": int(label_index) if label_index is not None else None,
            "width": width,
            "height": height,
            "file_size": file_size,
            "sha256": sha256,
            "quality_score": quality_score,
            "form_type": form_type,
            "orientation": orientation,
            "visual_hash": compute_dhash(image_path),
        }
    except Exception as exc:
        print(f"[skip] {image_path}: {exc}")
        return None


def build_records_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    skip_ocr: bool,
    ocr_engine: Optional[SuryaDocumentExtractor] = None,
) -> list[LearnedImageRecord]:
    """Materialize learned records, batching Surya OCR work when available."""
    if not candidates:
        return []

    if skip_ocr:
        texts = [""] * len(candidates)
    elif ocr_engine is not None:
        try:
            extractions = ocr_engine.extract_batch(
                [candidate["image_path"] for candidate in candidates],
                with_layout=False,
            )
            texts = [extraction.text[:8000] for extraction in extractions]
        except Exception as exc:
            if getattr(ocr_engine, "device", "") != "mps" or len(candidates) <= 1:
                raise
            print(
                f"[mps-batch-retry] batch of {len(candidates)} failed; "
                f"retrying one image at a time: {exc}",
                flush=True,
            )
            texts = extract_mps_single_image_texts(candidates, ocr_engine)
    else:
        texts = [run_ocr(candidate["image_path"]) for candidate in candidates]

    records: list[LearnedImageRecord] = []
    for candidate, ocr_text in zip(candidates, texts):
        if ocr_text is None:
            continue
        try:
            records.append(record_from_metadata(candidate, ocr_text))
        except Exception as exc:
            print(f"[skip] {candidate.get('image_path')}: {exc}")
    return records


def extract_mps_single_image_texts(
    candidates: list[dict[str, Any]],
    ocr_engine: SuryaDocumentExtractor,
) -> list[Optional[str]]:
    """Retry a failed MPS batch as single-image Surya calls."""
    texts: list[Optional[str]] = []
    previous_batch = os.environ.get("RECOGNITION_BATCH_SIZE")
    os.environ["RECOGNITION_BATCH_SIZE"] = "1"
    try:
        for candidate in candidates:
            path = candidate["image_path"]
            try:
                extraction = ocr_engine.extract_batch([path], with_layout=False)[0]
                texts.append(extraction.text[:8000])
            except Exception as exc:
                print(f"[skip] {path}: Surya MPS single-image retry failed: {exc}")
                texts.append(None)
    finally:
        if previous_batch is None:
            os.environ.pop("RECOGNITION_BATCH_SIZE", None)
        else:
            os.environ["RECOGNITION_BATCH_SIZE"] = previous_batch
    return texts


def record_from_metadata(candidate: dict[str, Any], ocr_text: str) -> LearnedImageRecord:
    image_path = Path(candidate["image_path"])
    return LearnedImageRecord(
        image_id=str(candidate["image_id"]),
        image_path=str(image_path),
        filename=str(candidate["filename"]),
        source_split=str(candidate.get("source_split") or ""),
        label_name=str(candidate.get("label_name") or ""),
        label_index=candidate.get("label_index"),
        width=candidate.get("width"),
        height=candidate.get("height"),
        file_size=candidate.get("file_size"),
        sha256=str(candidate.get("sha256") or ""),
        quality_score=float(candidate.get("quality_score") or 0.0),
        form_type=str(candidate.get("form_type") or "Unbekannt"),
        orientation=str(candidate.get("orientation") or "correct"),
        ocr_text=ocr_text,
        visual_hash=str(candidate.get("visual_hash") or ""),
    )


def build_record(
    image_path: Path,
    item: dict[str, Any],
    quality_gate: QualityGate,
    skip_ocr: bool,
    ocr_engine: Optional[SuryaDocumentExtractor] = None,
    image_id: Optional[str] = None,
) -> Optional[LearnedImageRecord]:
    candidate = build_record_metadata(
        image_path,
        item,
        quality_gate,
        image_id=image_id,
    )
    if candidate is None:
        return None
    ocr_text = "" if skip_ocr else run_ocr(image_path, ocr_engine=ocr_engine)
    return record_from_metadata(candidate, ocr_text)


def run_ocr(image_path: Path, ocr_engine: Optional[SuryaDocumentExtractor] = None) -> str:
    if ocr_engine is not None:
        return ocr_engine.extract_text(image_path)[:8000]
    try:
        import pytesseract

        with Image.open(image_path) as img:
            try:
                return pytesseract.image_to_string(img, lang="deu+eng")[:8000]
            except Exception:
                return pytesseract.image_to_string(img, lang="eng")[:8000]
    except Exception:
        return ""


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_dhash(path: Path) -> str:
    """Return a compact perceptual hash for near-duplicate visual retrieval."""
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((9, 8))
            if hasattr(img, "get_flattened_data"):
                pixels = list(img.get_flattened_data())
            else:
                pixels = list(img.getdata())
        bits = []
        for row in range(8):
            offset = row * 9
            for col in range(8):
                bits.append(1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return f"{value:016x}"
    except Exception:
        return ""


def flush_batch(store: ImageLearningStore, batch: list[LearnedImageRecord]) -> int:
    stored = store.add_records(batch)
    print(f"stored batch: {stored}")
    return stored


if __name__ == "__main__":
    raise SystemExit(main())
