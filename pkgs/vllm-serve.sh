#!/usr/bin/env bash
# Server launcher for Qwen3.8-Flash-Next / Qwen4Exp on Blackwell / NixOS
set -euo pipefail

VLLM_DIR="${VLLM_HOME:-$HOME/.local/share/nix-vllm}"
VENV="$VLLM_DIR/venv"
SETUP_CMD="${VLLM_SETUP_CMD:-vllm-setup}"

MODEL_DIR="${MODEL_DIR:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SERVED_NAME="${SERVED_NAME:-qwen}"
CTX="${CTX:-131072}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.90}"
MTP="${MTP:-3}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
EXTRA="${EXTRA:-}"

# Parse CLI arguments if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "Usage: vllm-serve [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --model, --model-dir <path|repo>  Local directory or Hugging Face ID (e.g. nvidia/Qwen3.8-Flash-Next-NVFP4)"
      echo "  --port <port>                     Port to listen on (default: 8000)"
      echo "  --host <host>                     Host to bind (default: 0.0.0.0)"
      echo "  --served-name <name>              API model name (default: qwen)"
      echo "  --ctx <len>                       Max context length (default: 262144)"
      echo "  --seqs <num>                      Max concurrent sequences (default: 8)"
      echo "  --gpu-mem <ratio>                 GPU memory utilization (default: 0.85)"
      echo "  --kv-cache-dtype <dtype>          KV cache data type: fp8, auto, bfloat16 (default: fp8)"
      echo "  --kv-bytes <bytes>                Explicit KV cache size (e.g. 20g)"
      echo "  --mtp <num>                       Number of MTP speculative tokens (default: 3, 0 to disable)"
      echo "  --ple-cpu-offload <0|1>           Offload PLE n-gram table to host RAM (default: 1)"
      echo "  --prefix-cache <0|1>              Enable prefix caching (default: 1)"
      exit 0
      ;;
    --model-dir|--model)
      MODEL_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --ctx|--max-model-len)
      CTX="$2"
      shift 2
      ;;
    --seqs|--max-num-seqs)
      SEQS="$2"
      shift 2
      ;;
    --gpu-mem|--gpu-memory-utilization)
      GPU_MEM="$2"
      shift 2
      ;;
    --kv-cache-dtype)
      KV_CACHE_DTYPE="$2"
      shift 2
      ;;
    --kv-bytes|--kv-cache-memory-bytes)
      KV_BYTES="$2"
      shift 2
      ;;
    --mtp)
      MTP="$2"
      shift 2
      ;;
    --served-name|--served-model-name)
      SERVED_NAME="$2"
      shift 2
      ;;
    --ple-cpu-offload)
      VLLM_PLE_CPU_OFFLOAD="$2"
      shift 2
      ;;
    --prefix-cache)
      PREFIX_CACHE="$2"
      shift 2
      ;;
    *)
      EXTRA="$EXTRA $1"
      shift
      ;;
  esac
done

if [ ! -d "$VENV" ] || [ ! -x "$VENV/bin/vllm" ]; then
  echo "==> vLLM venv not found at $VENV. Automatically running setup..."
  if command -v "$SETUP_CMD" >/dev/null 2>&1; then
    "$SETUP_CMD"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "$SCRIPT_DIR/vllm-setup.sh"
  fi
fi

if [ -z "$MODEL_DIR" ]; then
  echo "Error: MODEL_DIR is not specified!" >&2
  echo "Usage: vllm-serve --model <local-path-or-hf-repo-id>" >&2
  echo "Example: vllm-serve --model nvidia/Qwen3.8-Flash-Next-NVFP4" >&2
  exit 1
fi

# Memory Sizing
[ -n "${KV_BYTES:-}" ] && EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

# Piecewise CUDA graphs splitting ops (supporting both Qwen4Exp and Qwen3_8FlashNext)
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# FlashInfer Autotune
AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = "1" ] && AT_ARG=

# Speculative Decoding (MTP)
SPEC=()
if [ "$MTP" != "0" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
fi

# Prefix Caching
PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-0}" = "1" ] && PC_ARG=--enable-prefix-caching

# Never-evict pin
PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ] && [ "${PREFIX_CACHE:-0}" = "1" ]; then
  PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
           --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
fi

# Runtime optimization environment variables
export VLLM_FP8_HYBRID="${FP8_HYBRID:-1}"
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}"
export VLLM_HIT_DEBUG="${HIT_DEBUG:-0}"
export VLLM_STEP_PROFILE="${STEP_PROFILE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo ">> Starting vLLM server on ${HOST}:${PORT}"
echo ">> Model: $MODEL_DIR (served as: $SERVED_NAME)"
echo ">> Context: $CTX, MTP: $MTP, Seqs: $SEQS, GPU Mem: $GPU_MEM"
echo ">> PLE CPU Offload: $VLLM_PLE_CPU_OFFLOAD (1 = pinned Host RAM offload)"

exec "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --load-format "$LOAD_FORMAT" \
  --max-model-len "$CTX" \
  --max-num-seqs "$SEQS" \
  --gpu-memory-utilization "$GPU_MEM" \
  $PC_ARG \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  $CC \
  $AT_ARG \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" \
  "${PIN_ARG[@]}" \
  "${SPEC[@]}" \
  $EXTRA
