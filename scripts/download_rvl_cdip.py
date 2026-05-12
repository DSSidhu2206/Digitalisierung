#!/usr/bin/env python3
"""
Download exactly 50,000 valid images from the RVL-CDIP dataset
and place them into ./documents/rvl_cdip/ for use in image/document
cleansing projects.

Usage:
    source .venv/bin/activate
    python scripts/download_rvl_cdip.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
from datasets import ClassLabel, Features, Image as HFImage, load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────
TARGET_COUNT = 50_000
OUTPUT_DIR = Path("documents/rvl_cdip")
MANIFEST_PATH = Path("documents/rvl_cdip_manifest.json")
PROGRESS_PATH = Path("documents/rvl_cdip_progress.json")
GITIGNORE_PATH = Path(".gitignore")

CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]

FEATURES = Features(
    {
        "image": HFImage(decode=False),
        "label": ClassLabel(names=CLASSES),
    }
)

SPLITS = ["train", "test", "validation"]
SAVE_EVERY = 50


# ── Helpers ────────────────────────────────────────────────────────────────
def atomic_json(path: Path, data: Any) -> None:
    """Write JSON atomically so an interrupted run never leaves a half file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def load_json(path: Path, default: Any) -> Any:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def validate_image(
    filepath: Path,
    split: str,
    filename: str,
    label_index: int | None = None,
) -> dict | None:
    """
    Validate an image file with Pillow + OpenCV.
    Returns metadata dict on success, None on failure (and deletes the file).
    """
    try:
        size = filepath.stat().st_size
        if size == 0:
            filepath.unlink(missing_ok=True)
            return None

        # Pillow validation (supports TIFF natively)
        with Image.open(filepath) as img:
            img.load()
            width, height = img.size

        # OpenCV validation
        img_cv = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img_cv is None or img_cv.size == 0:
            filepath.unlink(missing_ok=True)
            return None

        # SHA-256 hash
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha.update(chunk)

        return {
            "filename": filename,
            "source_split": split,
            "label_index": label_index,
            "label_name": CLASSES[label_index] if label_index is not None else None,
            "image_dimensions": [width, height],
            "file_size": size,
            "hash": sha.hexdigest(),
        }

    except (UnidentifiedImageError, OSError, Exception):
        filepath.unlink(missing_ok=True)
        return None


def update_gitignore() -> None:
    """Ensure .gitignore excludes the downloaded images and TIFF files."""
    entries = []
    if GITIGNORE_PATH.exists():
        with open(GITIGNORE_PATH, "r", encoding="utf-8") as f:
            entries = f.read().splitlines()

    needed = [
        "",
        "# --- RVL-CDIP dataset ---",
        "documents/rvl_cdip/",
        "*.tif",
        "*.tiff",
    ]
    changed = False
    for entry in needed:
        if entry not in entries:
            entries.append(entry)
            changed = True

    if changed:
        with open(GITIGNORE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(entries) + "\n")
        print("Updated .gitignore")


def clean_tmp_files() -> None:
    """Remove stale .tmp files from previous interrupted runs."""
    for tmp in OUTPUT_DIR.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            pass


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    start_time = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_tmp_files()
    update_gitignore()

    # Load resume state
    progress = load_json(PROGRESS_PATH, {})
    manifest: list[dict] = load_json(MANIFEST_PATH, [])

    valid_files: set[str] = set(progress.get("valid_files", []))
    skipped_corrupted: int = progress.get("skipped_corrupted", 0)
    total_downloaded: int = progress.get("total_downloaded", 0)

    # Reconcile disk state with progress
    existing_on_disk = {p.name for p in OUTPUT_DIR.iterdir() if p.is_file()}
    valid_files &= existing_on_disk
    manifest = [m for m in manifest if m["filename"] in valid_files]

    valid_count = len(valid_files)
    print(f"Resuming with {valid_count} valid images already on disk.")
    if valid_count >= TARGET_COUNT:
        print(f"Target of {TARGET_COUNT} already reached — nothing to do.")
        _print_summary(total_downloaded, skipped_corrupted, valid_count, start_time)
        return 0

    max_workers = min(16, (os.cpu_count() or 4) * 2)
    futures: dict[Any, tuple[str, str]] = {}

    pbar = tqdm(
        total=TARGET_COUNT,
        initial=valid_count,
        desc="Valid images",
        unit="img",
        ncols=80,
    )

    interrupted = False

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for split in SPLITS:
                if valid_count >= TARGET_COUNT:
                    break

                print(f"\nStreaming split: {split}")
                ds = load_dataset(
                    "aharley/rvl_cdip",
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                    features=FEATURES,
                )

                for sample in ds:
                    # Harvest any done futures first so valid_count is current
                    done = [f for f in list(futures.keys()) if f.done()]
                    for fut in done:
                        if valid_count >= TARGET_COUNT:
                            break
                        fname, _ = futures.pop(fut)
                        result = fut.result()
                        if result:
                            valid_files.add(fname)
                            manifest.append(result)
                            valid_count += 1
                            pbar.update(1)
                            if valid_count % SAVE_EVERY == 0:
                                _save_state(
                                    valid_files, valid_count, skipped_corrupted,
                                    total_downloaded, manifest,
                                )
                        else:
                            skipped_corrupted += 1

                    if valid_count >= TARGET_COUNT:
                        break

                    total_downloaded += 1

                    img_data = sample["image"]
                    if isinstance(img_data, dict):
                        raw_bytes: bytes = img_data["bytes"]
                        original_path: str = img_data["path"]
                    else:
                        # Fallback if the dataset decoded the image anyway
                        buf = io.BytesIO()
                        img_data.save(buf, format="TIFF")
                        raw_bytes = buf.getvalue()
                        original_path = f"images/unknown/{total_downloaded}.tif"

                    filename = os.path.basename(original_path)

                    # Already known-good — skip entirely
                    if filename in valid_files:
                        continue

                    filepath = OUTPUT_DIR / filename

                    # File exists but isn't in our progress — validate synchronously
                    if filepath.exists():
                        result = validate_image(
                            filepath,
                            split,
                            filename,
                            sample.get("label"),
                        )
                        if result:
                            valid_files.add(filename)
                            manifest.append(result)
                            valid_count += 1
                            pbar.update(1)
                            if valid_count % SAVE_EVERY == 0:
                                _save_state(
                                    valid_files, valid_count, skipped_corrupted,
                                    total_downloaded, manifest,
                                )
                        else:
                            skipped_corrupted += 1
                        continue

                    # Write bytes to disk
                    try:
                        with open(filepath, "wb") as f:
                            f.write(raw_bytes)
                    except Exception as exc:
                        print(f"\n[ERROR] Failed to write {filename}: {exc}")
                        skipped_corrupted += 1
                        continue

                    # Submit background validation
                    fut = executor.submit(
                        validate_image,
                        filepath,
                        split,
                        filename,
                        sample.get("label"),
                    )
                    futures[fut] = (filename, split)

            # Drain remaining futures
            if futures:
                print(f"\nWaiting for {len(futures)} remaining validations...")
                for fut in as_completed(futures):
                    if valid_count >= TARGET_COUNT:
                        break
                    fname, _ = futures.pop(fut)
                    result = fut.result()
                    if result:
                        valid_files.add(fname)
                        manifest.append(result)
                        valid_count += 1
                        pbar.update(1)
                    else:
                        skipped_corrupted += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
        interrupted = True
    finally:
        pbar.close()

        # Cancel & clean up any futures that were never counted
        for fut, (fname, _) in list(futures.items()):
            if fname not in valid_files:
                try:
                    fut.cancel()
                except Exception:
                    pass
                (OUTPUT_DIR / fname).unlink(missing_ok=True)

        _save_state(
            valid_files, valid_count, skipped_corrupted,
            total_downloaded, manifest,
        )

    _print_summary(total_downloaded, skipped_corrupted, valid_count, start_time)

    if not interrupted and valid_count < TARGET_COUNT:
        print(
            f"WARNING: Exhausted dataset with only {valid_count} valid images."
        )
        return 1

    return 0


def _save_state(
    valid_files: set[str],
    valid_count: int,
    skipped_corrupted: int,
    total_downloaded: int,
    manifest: list[dict],
) -> None:
    atomic_json(
        PROGRESS_PATH,
        {
            "valid_count": valid_count,
            "valid_files": sorted(valid_files),
            "skipped_corrupted": skipped_corrupted,
            "total_downloaded": total_downloaded,
        },
    )
    atomic_json(MANIFEST_PATH, manifest)


def _print_summary(
    total_downloaded: int,
    skipped_corrupted: int,
    valid_count: int,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    print(f"\n{'=' * 50}")
    print(f"Total downloaded: {total_downloaded}")
    print(f"Skipped corrupted: {skipped_corrupted}")
    print(f"Total valid saved: {valid_count}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    sys.exit(main())
