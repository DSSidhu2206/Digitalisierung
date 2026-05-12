#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs documents

PYTHONPATH=backend .venv/bin/python scripts/ingest_learning_images.py \
  --image-dir documents/rvl_cdip \
  --batch-size "${BATCH_SIZE:-64}" \
  --heartbeat-seconds "${HEARTBEAT_SECONDS:-1}" \
  --plateau-window "${PLATEAU_WINDOW:-50000}" \
  --max-seconds "${MAX_SECONDS:-3300}" \
  --progress documents/image_learning_progress.json \
  >> logs/image_learning_ingest.log 2>&1
