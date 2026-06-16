#!/bin/bash
# Digitalisierung — macOS Setup Script
# Compatible with Python 3.11, 3.12, 3.13

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${SCRIPT_DIR}"
VENV_PATH="${DEST}/.venv"
MODEL_DIR="${DEST}/backend/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
fail()  { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Check platform
# ---------------------------------------------------------------------------

check_platform() {
    info "Checking platform ..."

    if [[ "$(uname -s)" != "Darwin" ]]; then
        fail "This script requires macOS. Detected: $(uname -s)"
    fi
    ok "Running on macOS $(sw_vers -productVersion 2>/dev/null || echo 'unknown version')"

    local arch
    arch="$(uname -m)"
    if [[ "$arch" != "arm64" ]]; then
        warn "Expected Apple Silicon (arm64), found: $arch"
        warn "MLX acceleration may not be available."
    else
        ok "Apple Silicon detected ($arch)"
    fi
}

# ---------------------------------------------------------------------------
# 2. Check Python
# ---------------------------------------------------------------------------

check_python() {
    info "Checking Python version ..."

    if ! command -v python3 &>/dev/null; then
        fail "python3 not found. Install from https://www.python.org/downloads/"
    fi

    local py_major py_minor
    py_major="$(python3 -c 'import sys; print(sys.version_info.major)')"
    py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)')"

    if [[ "$py_major" -lt 3 ]] || [[ "$py_minor" -lt 11 ]]; then
        fail "Python 3.11+ required, found ${py_major}.${py_minor}"
    fi

    ok "Python ${py_major}.${py_minor} found"

    if [[ "$py_minor" -ge 13 ]]; then
        warn "Python 3.13+ detected — using pre-built wheels only (no source builds)"
    fi
}

# ---------------------------------------------------------------------------
# 3. Create virtual environment
# ---------------------------------------------------------------------------

create_venv() {
    info "Creating virtual environment ..."

    if [[ -d "$VENV_PATH" ]]; then
        warn "Removing existing .venv"
        rm -rf "$VENV_PATH"
    fi

    python3 -m venv "$VENV_PATH"
    ok "Virtual environment created"
}

# ---------------------------------------------------------------------------
# 4. Install packages (wheel-only for Python 3.13)
# ---------------------------------------------------------------------------

install_packages() {
    info "Installing Python dependencies ..."

    source "${VENV_PATH}/bin/activate"

    local py_minor
    py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)')"

    # Upgrade pip/setuptools/wheel FIRST (critical for Python 3.13)
    info "Upgrading pip, setuptools, wheel ..."
    pip install --upgrade pip setuptools wheel

    if [[ "$py_minor" -ge 13 ]]; then
        # Python 3.13+: force pre-built wheels, never compile from source
        info "Python 3.13 mode: installing from pre-built wheels only ..."

        info "  -> fastapi, uvicorn"
        pip install --only-binary :all: fastapi uvicorn

        info "  -> pydantic, pydantic-settings"
        pip install --only-binary :all: pydantic pydantic-settings

        info "  -> pillow, numpy, psutil"
        pip install --only-binary :all: pillow numpy psutil

        info "  -> instructor, python-multipart, python-dotenv"
        pip install --only-binary :all: instructor python-multipart python-dotenv

        # chromadb and sentence-transformers — try wheels first, fallback to source
        info "  -> chromadb"
        pip install chromadb || pip install --only-binary :all: chromadb

        info "  -> sentence-transformers"
        pip install sentence-transformers || pip install --only-binary :all: sentence-transformers

        # macOS-only packages (MLX, llama.cpp)
        info "  -> mlx, mlx-vlm (Metal GPU)"
        pip install mlx mlx-vlm

        info "  -> llama-cpp-python (Metal GPU)"
        # May need cmake/libomp from Homebrew
        if command -v brew &>/dev/null; then
            brew list cmake &>/dev/null || brew install cmake 2>/dev/null || true
            brew list libomp &>/dev/null || brew install libomp 2>/dev/null || true
        fi
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python || {
            warn "llama-cpp-python build failed — will use mock fallback"
            warn "Install Xcode Command Line Tools: xcode-select --install"
        }
    else
        # Python 3.11/3.12: install from requirements.txt normally
        info "Installing from requirements.txt ..."
        pip install -r "${DEST}/backend/requirements.txt"
    fi

    ok "Dependencies installed"
}

# ---------------------------------------------------------------------------
# 5. Download model weights
# ---------------------------------------------------------------------------

# Verify a downloaded GGUF: magic bytes + optional pinned SHA-256.
verify_gguf() {
    local f="$1"
    [[ -f "$f" ]] || { info "Model file missing: $f"; return 1; }
    local magic
    magic="$(head -c 4 "$f" 2>/dev/null)"
    if [[ "$magic" != "GGUF" ]]; then
        info "File is not a valid GGUF (got '${magic}') — truncated or HTML error response?"
        return 1
    fi
    if [[ -n "${LLM_MODEL_SHA256:-}" ]]; then
        local got
        got="$(shasum -a 256 "$f" | awk '{print $1}')"
        if [[ "$got" != "$LLM_MODEL_SHA256" ]]; then
            info "SHA-256 mismatch: expected ${LLM_MODEL_SHA256}, got ${got}"
            return 1
        fi
        ok "Model SHA-256 verified"
    else
        info "No LLM_MODEL_SHA256 pinned; verified GGUF header only (set it in .env to pin)"
    fi
    return 0
}

download_models() {
    info "Downloading model weights ..."

    mkdir -p "$MODEL_DIR"

    local gguf_file="${MODEL_DIR}/llama-3.1-8b-instruct.Q4_K_M.gguf"
    local gguf_url="https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct.Q4_K_M.gguf"

    if [[ -f "$gguf_file" ]]; then
        local size_mb
        size_mb="$(du -m "$gguf_file" | cut -f1)"
        ok "LLM model already downloaded (${size_mb} MB)"
    else
        info "Downloading Llama-3.1-8B-Instruct Q4_K_M (~5 GB) ..."
        info "URL: $gguf_url"
        if command -v curl &>/dev/null; then
            curl -L --fail --progress-bar "$gguf_url" -o "$gguf_file" \
                || { rm -f "$gguf_file"; fail "Model download failed (HTTP error)"; }
        elif command -v wget &>/dev/null; then
            wget --progress=bar:force "$gguf_url" -O "$gguf_file" \
                || { rm -f "$gguf_file"; fail "Model download failed (HTTP error)"; }
        else
            fail "curl or wget required to download models"
        fi
        if ! verify_gguf "$gguf_file"; then
            rm -f "$gguf_file"
            fail "Downloaded model failed integrity check (corrupt or wrong file)"
        fi
        ok "LLM model downloaded and verified"
    fi

    info "VLM model (Llama-3.2-11B-Vision) downloads automatically on first run"
}

# ---------------------------------------------------------------------------
# 6. Create .env
# ---------------------------------------------------------------------------

create_env() {
    info "Creating .env configuration ..."

    local env_file="${DEST}/.env"
    [[ -f "$env_file" ]] && { ok ".env already exists"; return; }

    cat > "$env_file" << 'EOF'
DEBUG=false
VLM_MODEL_ID=mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
LLM_MODEL_PATH=backend/models/llama-3.1-8b-instruct.Q4_K_M.gguf
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_N_THREADS=8
LLM_N_BATCH=512
LLM_N_CTX=8192
RAM_THRESHOLD=0.80
MAX_CONCURRENT_REQUESTS=1
LEGIBILITY_THRESHOLD=0.70
CONFIDENCE_THRESHOLD=0.70
CHROMA_PERSIST_DIR=./chroma_data
MAX_FEW_SHOT_CORRECTIONS=5
AUDIT_LOG_PATH=./logs/audit.jsonl
AUDIT_LOG_ROTATE_SIZE_MB=100
AUDIT_LOG_MAX_ARCHIVES=10
VLM_TEMPERATURE=0.0
LLM_TEMPERATURE=0.0
UPLOAD_DIR=./uploads
EOF

    ok ".env created"
}

# ---------------------------------------------------------------------------
# 7. Create runtime directories
# ---------------------------------------------------------------------------

make_scripts_executable() {
    info "Making scripts executable ..."
    chmod +x "$DEST/start_server.sh" "$DEST/start_dev.sh" "$DEST/test.sh" 2>/dev/null || true
    ok "Scripts ready"
}

create_directories() {
    info "Creating runtime directories ..."
    mkdir -p "${DEST}/uploads"
    mkdir -p "${DEST}/logs"
    mkdir -p "${DEST}/chroma_data"
    info "Making scripts executable ..."
    chmod +x "${DEST}/start_server.sh" "${DEST}/start_dev.sh" "${DEST}/test.sh" 2>/dev/null || true
    ok "Runtime directories ready"
}

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------

print_summary() {
    echo ""
    echo "====================================================================="
    echo "  Digitalisierung — Setup Complete"
    echo "====================================================================="
    echo ""
    echo "  Virtual environment : ${VENV_PATH}"
    echo "  Models directory    : ${MODEL_DIR}"
    echo "  Configuration       : ${DEST}/.env"
    echo ""
    echo "  Start the server:"
    echo "    cd ${DEST}"
    echo "    ./start_server.sh"
    echo ""
    echo "  API docs: http://localhost:8000/docs"
    echo "====================================================================="
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo ""
    echo "====================================================================="
    echo "  Digitalisierung — macOS Setup"
    echo "====================================================================="
    echo ""

    check_platform
    check_python
    create_venv
    install_packages
    download_models
    create_env
    create_directories
    print_summary
}

main "$@"
