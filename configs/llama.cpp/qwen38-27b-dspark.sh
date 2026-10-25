#!/bin/sh
# DSpark Q8_0 draft. Only the 512-token slice was measured (2026-10-16).
# Depth 7 was the fastest of 2..8 at 18.0 tok/s. Depth 8 dropped to 12.3.
# The full ~1370-token answer was not run.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/home/vlad/models/Qwen3.8-27B-DSpark/Qwen3.8-27B-DSpark-Q8_0.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  --spec-draft-model "$DRAFT" \
  --spec-type draft-dspark \
  --spec-draft-n-max "${N_MAX:-7}" \
  --spec-draft-ngl 99 \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
