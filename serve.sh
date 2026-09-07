#!/usr/bin/env bash
# Entrypoint launcher for vLLM on NixOS / Blackwell
# Supports running via nix flake (vllm-serve) or docker (serve-docker.sh)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${MODE:-fhs}"  # "fhs" (native nix FHS via binary wheels) or "docker"
export MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid}"
export PORT="${PORT:-8000}"
export SERVED_NAME="${SERVED_NAME:-qwen}"
export SEQS="${SEQS:-8}"
export MTP="${MTP:-3}"
export PREFIX_CACHE="${PREFIX_CACHE:-1}"
export GPU_MEM="${GPU_MEM:-0.85}"
export VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}"

if [ "$MODE" = "docker" ]; then
  exec ./scripts/serve-docker.sh "$@"
else
  # Native Nix FHS runner
  if command -v vllm-serve >/dev/null 2>&1; then
    exec vllm-serve "$@"
  else
    exec nix run .#serve -- "$@"
  fi
fi
