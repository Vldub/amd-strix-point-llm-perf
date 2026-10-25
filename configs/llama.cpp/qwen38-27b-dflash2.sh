#!/bin/sh
# Current best profile on this box.
# DFlash2 Q4_K_M draft, depth 7 (trained block is 8: 1 anchor + 7 drafts).
# Rope task: ~16.7 tok/s on the full ~1370-token answer (2026-10-16),
# 19.3 tok/s on the first 512 tokens that day, 20.2 tok/s on 2026-10-25.
# Asking for n-max 14 is clamped to 7 by this build.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-/home/vlad/models/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
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
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
