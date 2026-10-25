#!/bin/sh
# Qwen3.8-27B Q4_K_M, no speculative decoding.
# Rope task, full answer, 2026-10-16: 5.3 tok/s.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
