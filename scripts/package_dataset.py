#!/usr/bin/env python3
"""
Package the synthetic dataset into a single zip for upload to Kaggle.

The images are already JPEG (compressed), so this stores them with light
compression and preserves the layout (`images/...` + the manifests at the zip
root) so Kaggle extracts straight to ``/kaggle/input/<slug>/``.

    python scripts/package_dataset.py            # -> data/german-docs-synthetic.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="data/german-docs-synthetic.zip")
    args = parser.parse_args()

    data = PROJECT_ROOT / args.data
    out = PROJECT_ROOT / args.out
    if not data.is_dir():
        raise SystemExit(f"dataset dir not found: {data} (run generate_synthetic_docs.py first)")

    files = sorted(p for p in data.rglob("*") if p.is_file())
    print(f"Zipping {len(files)} files: {data} -> {out}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, path in enumerate(files, 1):
            zf.write(path, path.relative_to(data))
            if i % 2000 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")
    size_gb = out.stat().st_size / 1e9
    print(f"Done: {out} ({size_gb:.2f} GB)")
    print("Upload at https://www.kaggle.com/datasets → New Dataset → upload this zip "
          "(Kaggle auto-extracts). Then attach it to the training notebook.")


if __name__ == "__main__":
    main()
