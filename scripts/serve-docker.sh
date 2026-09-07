#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next / Qwen4Exp in patched Docker container on Blackwell
set -euo pipefail

NAME="${NAME:-qwen-vllm}"
IMAGE="${IMAGE:-qwen-vllm-blackwell}"
MODEL_DIR="${MODEL_DIR:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-3}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
SERVED_NAME="${SERVED_NAME:-qwen}"
EXTRA="${EXTRA:-}"

# Parse CLI arguments if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
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
    *)
      EXTRA="$EXTRA $1"
      shift
      ;;
  esac
done

if [ -z "$MODEL_DIR" ]; then
  echo "Error: MODEL_DIR is required!" >&2
  echo "Usage: $0 --model-dir /path/to/checkpoint" >&2
  exit 1
fi

[ -n "${KV_BYTES:-}" ] && EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

# Piecewise CUDA graph splitting ops (both Qwen4Exp and Qwen3_8FlashNext)
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = "1" ] && AT_ARG=

SPEC=()
if [ "$MTP" != "0" ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
fi

PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-0}" = "1" ] && PC_ARG=--enable-prefix-caching

PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ] && [ "${PREFIX_CACHE:-0}" = "1" ]; then
  PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
           --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --restart unless-stopped \
  --device=nvidia.com/gpu=all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_FP8_HYBRID="${FP8_HYBRID:-1}" \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}" \
  -e VLLM_HIT_DEBUG="${HIT_DEBUG:-0}" \
  -e VLLM_STEP_PROFILE="${STEP_PROFILE:-0}" \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  "$IMAGE" \
  /model --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port 8000 --load-format "$LOAD_FORMAT" \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    $PC_ARG --enable-chunked-prefill --max-num-batched-tokens 8192 \
    $CC \
    $AT_ARG \
    --kv-cache-dtype auto \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER" \
    "${PIN_ARG[@]}" "${SPEC[@]}" \
    $EXTRA

echo ">> $NAME started on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM)"
echo ">> Follow logs with: docker logs -f $NAME"
