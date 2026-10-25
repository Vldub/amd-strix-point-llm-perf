#!/bin/sh
# Built-in MTP heads of Qwen3.8-27B. No extra draft file.
# Depth 5 was the peak on the rope task (14.1 tok/s full answer).
# Depth 7 was slower: acceptance fell to 68%.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  --spec-type draft-mtp \
  --spec-draft-n-max "${N_MAX:-5}" \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
