#!/bin/bash
# Simple server starter (assumes setup already done)
cd "$(dirname "$0")"
if [ ! -f "frontend/out/dashboard/index.html" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "Frontend is not built and npm is not available. Run npm install && npm run build in frontend/."
    exit 1
  fi
  (cd frontend && npm install && npm run build) || exit 1
fi
cd backend
source ../.venv/bin/activate 2>/dev/null || { echo "No .venv found. Run: bash start.sh"; exit 1; }
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
