#!/bin/sh
# Qwen3.6-35B-A3B UD-Q4_K_M plus the 0.5B DFlash2 draft, Q8_0.
# Block size 8, so n-max 7 is the trained depth.
# 2026-10-25: 23.8 GiB VRAM, 41.9 tok/s on the 512-token rope slice.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/home/vlad/models/Qwen3.6-35B-A3B-DFlash2/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  --spec-draft-model "$DRAFT" \
  --spec-type draft-dflash \
  --spec-draft-n-max "${N_MAX:-7}" \
  --spec-draft-ngl 99 \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --threads "${THREADS:-16}" \
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
