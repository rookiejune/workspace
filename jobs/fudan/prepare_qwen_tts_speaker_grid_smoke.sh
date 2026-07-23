#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export HF_HOME="${HF_HOME:-${STATIC_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export ANYTRAIN_HOME="${ANYTRAIN_HOME:-${STATIC_HOME}/.anytrain}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SMOKE_ROOT="${QWEN_TTS_SPEAKER_GRID_SMOKE_ROOT:-${DYNAMIC_HOME}/debug/qwen_tts_speaker_grid_smoke}"

cd "$WORKSPACE_ROOT"

"$WORKSPACE_PYTHON" scripts/prepare_qwen_tts_speaker_grid.py --output-dir "${SMOKE_ROOT}/store" --export-dir "${SMOKE_ROOT}/wavs" --speaker-id vivian --speaker-id ryan --batch-size 2 --devices "${QWEN_TTS_DEVICES:-auto}" --overwrite "$@"
