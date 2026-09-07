#!/usr/bin/env bash
# Native launcher for vLLM on NixOS (FHS environment)
set -euo pipefail
cd "$(dirname "$0")"

export FLASHINFER_DISABLE_VERSION_CHECK=1

if command -v vllm-serve >/dev/null 2>&1; then
  exec vllm-serve "$@"
else
  exec nix run .#serve -- "$@"
fi
