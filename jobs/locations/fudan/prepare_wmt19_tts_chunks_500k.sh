#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cd "$WORKSPACE_ROOT"

: "${WMT19_TTS_CHUNKS_ROOT:=$STATIC_HOME/datasets/wmt19_tts_chunks_500k}"
: "${WMT19_TTS_CHUNK_SIZE:=10000}"
: "${WMT19_TTS_CHUNK_COUNT:=50}"

for chunk_index in $(seq 0 "$((WMT19_TTS_CHUNK_COUNT - 1))"); do
  offset="$((chunk_index * WMT19_TTS_CHUNK_SIZE))"
  chunk_dir="$(printf "%s/chunk-%06d" "$WMT19_TTS_CHUNKS_ROOT" "$chunk_index")"
  if [[ -f "$chunk_dir/base/.ready" ]]; then
    continue
  fi
  "$WMT19_TTS_PYTHON" scripts/prepare_wmt19_tts.py \
    --root "$chunk_dir" \
    --offset "$offset" \
    --limit "$WMT19_TTS_CHUNK_SIZE" \
    --tts-batch-size 6 \
    --tts-max-new-tokens 8192 \
    --cleanup-work \
    "$@"
done
