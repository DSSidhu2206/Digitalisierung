#!/bin/bash
# Digitalisierung — One-Command Launcher
# Usage: bash start.sh
# Does everything: setup (if needed), start server, open dashboard

set -e

cd "$(dirname "$0")"
DEST="$(pwd)"
VENV="$DEST/.venv"
MODEL_DIR="$DEST/backend/models"

colors() { RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'; }
colors

info()  { echo -e "${BLUE}[digitalisierung]${NC} $*"; }
ok()    { echo -e "${GREEN}[digitalisierung]${NC} $*"; }
warn()  { echo -e "${YELLOW}[digitalisierung]${NC} $*"; }
fail()  { echo -e "${RED}[digitalisierung]${NC} $*"; exit 1; }

# ── Phase 0: Check Python ───────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found. Install from python.org or run: brew install python3"
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MINOR" -lt 11 ]; then
    fail "Python 3.11+ required, found $PY_VER"
fi
ok "Python $PY_VER"

# ── Phase 1: Create venv (if missing) ───────────────────────────────────────
if [ ! -d "$VENV" ]; then
    info "Creating virtual environment..."
    python3 -m venv "$VENV"
    ok "Virtual environment created"
fi

source "$VENV/bin/activate"

# ── Phase 2: Install dependencies (if missing) ──────────────────────────────
if ! python3 -c "import fastapi" 2>/dev/null; then
    info "Installing Python packages (first time, ~3 minutes)..."
    pip install --quiet --upgrade pip setuptools wheel

    if [ "$PY_MINOR" -ge 13 ]; then
        # Python 3.13: pre-built wheels only
        pip install --only-binary :all: fastapi uvicorn pydantic pydantic-settings
        pip install --only-binary :all: pillow numpy psutil
        pip install instructor python-multipart python-dotenv
        pip install chromadb sentence-transformers
        pip install mlx mlx-vlm
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python || warn "llama-cpp-python build skipped (will use stub)"
    else
        pip install -r "$DEST/backend/requirements.txt"
    fi
    ok "Dependencies installed"
else
    ok "Dependencies already installed"
fi

# ── Phase 3: Download model (if missing) ────────────────────────────────────
GGUF="$MODEL_DIR/llama-3.1-8b-instruct.Q4_K_M.gguf"
if [ ! -f "$GGUF" ]; then
    info "Downloading Llama-3.1-8B model (~5 GB, one-time)..."
    mkdir -p "$MODEL_DIR"
    curl -L --progress-bar "https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct.Q4_K_M.gguf" -o "$GGUF"
    ok "Model downloaded"
else
    ok "Model already downloaded"
fi

# ── Phase 4: Create runtime dirs ────────────────────────────────────────────
mkdir -p "$DEST/backend/chroma_data" "$DEST/backend/logs" "$DEST/backend/uploads"

# ── Phase 4b: Build frontend (if missing) ───────────────────────────────────
if [ ! -f "$DEST/frontend/out/dashboard/index.html" ]; then
    if ! command -v npm &>/dev/null; then
        fail "Node.js/npm is required to build the dashboard. Install Node.js, then rerun: bash start.sh"
    fi
    info "Installing frontend packages..."
    (cd "$DEST/frontend" && npm install)
    info "Building TypeScript dashboard..."
    (cd "$DEST/frontend" && npm run build)
    ok "Frontend built"
else
    ok "Frontend already built"
fi

# ── Phase 5: Create .env ────────────────────────────────────────────────────
if [ ! -f "$DEST/.env" ]; then
    cat > "$DEST/.env" << 'EOF'
DEBUG=false
VLM_MODEL_ID=mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
LLM_MODEL_PATH=backend/models/llama-3.1-8b-instruct.Q4_K_M.gguf
CHROMA_PERSIST_DIR=./backend/chroma_data
AUDIT_LOG_PATH=./backend/logs/audit.jsonl
UPLOAD_DIR=./backend/uploads
VLM_TEMPERATURE=0.0
LLM_TEMPERATURE=0.0
EOF
fi

# ── Phase 6: Start server ───────────────────────────────────────────────────
ok "Starting server on http://localhost:8000/dashboard"
cd "$DEST/backend"

# Open dashboard in background (served by the backend to avoid cross-origin issues)
(sleep 2 && open "http://localhost:8000/dashboard") &

# Start uvicorn
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
