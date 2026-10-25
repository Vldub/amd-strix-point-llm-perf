#!/bin/sh
# Qwen3.6-35B-A3B, Unsloth UD-Q4_K_M (20.6 GiB).
# Measured 2026-10-25: 22.3 GiB VRAM, 26.9 tok/s on the 512-token rope slice.
# UD-IQ4_XS (16.5 GiB) was the same speed on that slice. Neither file has MTP layers.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --threads "${THREADS:-16}" \
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
