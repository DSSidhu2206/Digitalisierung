#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { echo "No .venv found. Run: bash setup_mac.sh"; exit 1; }

# Ensure runtime directories exist (same as start.sh)
mkdir -p uploads logs chroma_data

if [ ! -f frontend/out/dashboard/index.html ]; then
  (cd frontend && npm install && npm run build) || exit 1
fi

cd backend
export DEBUG=true
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
