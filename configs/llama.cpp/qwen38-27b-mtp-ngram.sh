#!/bin/sh
# Flags from KyaniteLabs/qwen38-27b-strix-halo, adapted to this box.
# That recipe targets Strix Halo (gfx1151, ROCm/HIP, 64 GiB+). Here it ran
# on Vulkan llama.cpp b11685 with the local Q4_K_M, context 32768.
# First-exposure rope-512 on 2026-10-25: 10.6 tok/s, 50% draft acceptance.
# A second identical generation jumped to 56 tok/s because ngram replayed it.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$HERE/env.sh"
MODEL="${MODEL:-/home/vlad/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf}"
exec "$LLAMA_DIR/llama-server" \
  -m "$MODEL" \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max "${N_MAX:-12}" \
  --spec-ngram-mod-n-min "${NGRAM_MIN:-24}" \
  -ngl 99 \
  -fa on \
  -c 32768 \
  --parallel 1 \
  --threads "${THREADS:-16}" \
  --jinja \
  --host 127.0.0.1 \
  --port "${PORT:-8080}"
