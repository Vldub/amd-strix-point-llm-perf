#!/bin/sh
# Nemotron 3.5 Bolt 74B-A9B, expert-pruned to the 16 GB build (REAP50, IQ2_XXS).
# Card settings: one slot, full GPU offload, context 32768.
# Reasoning stays controllable per request (the card's eval used reasoning off).
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Nemotron-3.5-Bolt-74B-A9B-16GB/Nemotron-3.5-Bolt-74B-A9B-16GB-REAP50-IQ2_XXS.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --threads "${THREADS:-16}" \
  --jinja \
  --reasoning-format deepseek \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
