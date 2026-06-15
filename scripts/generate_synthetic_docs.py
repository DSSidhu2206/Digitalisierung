#!/usr/bin/env python3
"""
Generate a synthetic German-document dataset for fine-tuning an extraction model.

Each sample is a degraded document image paired with its exact field labels
(perfect ground truth, because we generated the values). Training uses this
synthetic set; evaluation uses the *real* held-out ``test_documents/``.

Usage:
    python scripts/generate_synthetic_docs.py --n 500            # 500 per type
    python scripts/generate_synthetic_docs.py --n 5 --out data/synthetic_sample

Output (under --out):
    images/<type>_<id>.png
    manifest.train.jsonl   # {"image", "document_type", "fields"} per line
    manifest.val.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from faker import Faker  # noqa: E402

from app.models.enums import DocumentType  # noqa: E402
from app.synth.augment import augment  # noqa: E402
from app.synth.field_data import generate_fields  # noqa: E402
from app.synth.render import render  # noqa: E402

DOC_TYPES = [
    DocumentType.MELDEBESCHEINIGUNG,
    DocumentType.STEUERBESCHEID,
    DocumentType.GEHALTSAUSWEIS,
    DocumentType.PERSONALAUSWEIS,
]


def _downscale(img, max_dim: int):
    longest = max(img.size)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    return img.resize((int(img.width * scale), int(img.height * scale)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="samples per document type")
    parser.add_argument("--out", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-dim", type=int, default=1024)
    args = parser.parse_args()

    out = PROJECT_ROOT / args.out
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    faker = Faker("de_DE")
    faker.seed_instance(args.seed)
    master = random.Random(args.seed)

    train, val = [], []
    for doc_type in DOC_TYPES:
        for i in range(args.n):
            rng = random.Random(master.randint(0, 2**31 - 1))
            fields = generate_fields(doc_type, faker, rng)
            image = _downscale(augment(render(doc_type, fields, rng), rng), args.max_dim)

            name = f"{doc_type.name.lower()}_{i:05d}.png"
            image.save(images_dir / name)
            record = {
                "image": f"images/{name}",
                "document_type": doc_type.value,
                "fields": fields,
            }
            (val if rng.random() < args.val_split else train).append(record)

    _write_jsonl(out / "manifest.train.jsonl", train)
    _write_jsonl(out / "manifest.val.jsonl", val)

    print(f"Wrote {len(train)} train + {len(val)} val samples to {out}")
    print(f"  types: {[d.value for d in DOC_TYPES]}  ({args.n} each)")
    print(f"  images: {images_dir}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
