#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-localhost}"
MODEL="${MODEL:-qwen}"
URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "==> Sending smoke test request to $URL (model: $MODEL)..."
curl -s "$URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Write a haiku about running Qwen on Blackwell GPU.\"}
    ],
    \"max_tokens\": 128,
    \"temperature\": 0.7
  }" | python3 -m json.tool || {
    echo "Request failed! Ensure vLLM server is running and listening on port $PORT." >&2
    exit 1
  }

echo -e "\n==> Smoke test passed!"
