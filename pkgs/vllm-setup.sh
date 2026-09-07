#!/usr/bin/env bash
set -euo pipefail

C_CYAN="\033[1;36m"
C_GREEN="\033[1;32m"
C_YELLOW="\033[1;33m"
C_RED="\033[1;31m"
C_RESET="\033[0m"

say()  { printf "%b==>%b %s\n" "$C_CYAN" "$C_RESET" "$*"; }
warn() { printf "%b[warn]%b %s\n" "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf "%b[error]%b %s\n" "$C_RED" "$C_RESET" "$*" >&2; exit 1; }

VLLM_DIR="${VLLM_HOME:-$HOME/.local/share/nix-vllm}"
VENV="$VLLM_DIR/venv"
PY_VER="${PYTHON_VERSION:-3.12}"

say "Initializing vLLM environment for NixOS (Blackwell / CUDA 13.0+)..."

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  say "Detected GPU: $GPU_NAME (Driver: $DRIVER_VER)"
else
  warn "nvidia-smi not detected in PATH. Ensure NVIDIA kernel modules and userspace drivers are active."
fi

mkdir -p "$VLLM_DIR"

if [ ! -d "$VENV" ] || [ "${FORCE_REINSTALL:-0}" = "1" ]; then
  say "Creating Python $PY_VER virtual environment at $VENV..."
  uv venv "$VENV" --python "$PY_VER" --clear
fi

say "Installing official vLLM nightly binary wheels with CUDA 13.0 + FlashInfer..."
uv pip install --python "$VENV" \
  --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl \
  vllm \
  flashinfer-cubin \
  flashinfer-jit-cache \
  fastsafetensors \
  sentencepiece \
  tiktoken \
  huggingface-hub \
  setuptools

say "Applying optimization & bugfix patches for Qwen3.8-Flash-Next / Qwen4Exp on Blackwell..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP_DIR="$("$VENV/bin/python3" -c "import site; print(site.getsitepackages()[0])")"

"$VENV/bin/python3" "$SCRIPT_DIR/src/apply_patches.py" "$SP_DIR"

say "Verifying vLLM installation..."
VER_INFO="$("$VENV/bin/python3" -c "import vllm, torch; print(f'vLLM {vllm.__version__} | PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')")"
say "Status: $VER_INFO"

printf "\n%b[SUCCESS] vLLM installation and patching completed!%b\n" "$C_GREEN" "$C_RESET"
echo "Virtualenv: $VENV"
echo "Executable: $VENV/bin/vllm"
echo ""
echo "Next steps:"
echo "  1. Download or prepare the model checkpoint (int4 AutoRound + int8 lm_head + fp8 side layers)"
echo "  2. Run the server: nix run .#serve -- --model-dir /path/to/model"
echo "     Or directly:    vllm-serve --model-dir /path/to/model"
