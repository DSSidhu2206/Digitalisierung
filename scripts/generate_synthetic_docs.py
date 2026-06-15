#!/usr/bin/env python3
"""
Generate a synthetic German-document dataset for fine-tuning an extraction model.

Each sample is a degraded, photo-realistic document image paired with its exact
field labels (perfect ground truth, because we generated the values). Training
uses this synthetic set; evaluation uses the *real* held-out ``test_documents/``.

Generation is parallelised across CPU cores, so tens of thousands of diverse
documents render in a few minutes.

Usage:
    python scripts/generate_synthetic_docs.py --n 3000          # 3000 per type → 12k
    python scripts/generate_synthetic_docs.py --n 5 --out data/synthetic_sample
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.models.enums import DocumentType  # noqa: E402

DOC_TYPES = [
    DocumentType.MELDEBESCHEINIGUNG,
    DocumentType.STEUERBESCHEID,
    DocumentType.GEHALTSAUSWEIS,
    DocumentType.PERSONALAUSWEIS,
]

_FAKER = None  # per-process, created in the pool initializer


def _init_worker() -> None:
    global _FAKER
    from faker import Faker

    _FAKER = Faker("de_DE")


def _downscale(img, max_dim: int):
    longest = max(img.size)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return img.resize((int(img.width * scale), int(img.height * scale)))


def _make_one(spec: tuple) -> dict:
    doc_value, index, seed, out_str, max_dim = spec
    from app.synth.augment import augment
    from app.synth.field_data import generate_fields
    from app.synth.render import render

    doc_type = DocumentType(doc_value)
    rng = random.Random(seed)
    _FAKER.seed_instance(seed)
    fields = generate_fields(doc_type, _FAKER, rng)
    img = _downscale(augment(render(doc_type, fields, rng), rng), max_dim)
    name = f"{doc_type.name.lower()}_{index:05d}.jpg"
    img.save(Path(out_str) / "images" / name, format="JPEG", quality=92)
    return {"image": f"images/{name}", "document_type": doc_value, "fields": fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000, help="samples per document type")
    parser.add_argument("--out", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    out = PROJECT_ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)

    master = random.Random(args.seed)
    specs = [
        (dt.value, i, master.randint(0, 2**31 - 1), str(out), args.max_dim)
        for dt in DOC_TYPES
        for i in range(args.n)
    ]
    total = len(specs)
    print(f"Generating {total} documents ({args.n}/type) with {args.workers} workers ...")

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for done, rec in enumerate(ex.map(_make_one, specs, chunksize=8), 1):
            records.append(rec)
            if done % 500 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)

    master.shuffle(records)
    cut = int(len(records) * args.val_split)
    _write_jsonl(out / "manifest.val.jsonl", records[:cut])
    _write_jsonl(out / "manifest.train.jsonl", records[cut:])
    print(f"Done: {len(records) - cut} train + {cut} val → {out}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
