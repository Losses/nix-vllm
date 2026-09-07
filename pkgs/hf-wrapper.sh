#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${VLLM_HOME:-$HOME/.local/share/nix-vllm}"
VENV="$VLLM_DIR/venv"
CMD="${VLLM_HF_CMD:-$(basename "$0")}"

if [ "$CMD" = "hf-wrapper.sh" ] || [ -z "$CMD" ]; then
    CMD="hf"
fi

if [ -x "$VENV/bin/$CMD" ]; then
    exec "$VENV/bin/$CMD" "$@"
elif [ -x "$VENV/bin/huggingface-cli" ]; then
    exec "$VENV/bin/huggingface-cli" "$@"
elif [ -x "$VENV/bin/hf" ]; then
    exec "$VENV/bin/hf" "$@"
else
    echo "Error: Hugging Face CLI not found in $VENV. Run vllm-setup first." >&2
    exit 1
fi
