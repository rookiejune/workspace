#!/usr/bin/env bash
set -euo pipefail

FUDAN_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LOCATION=fudan
export WORKSPACE_PYTHON="${WORKSPACE_PYTHON:-/home/zhuyin/anaconda3/envs/py312/bin/python}"
source "$FUDAN_JOBS_DIR/../env.sh"

: "${WMT19_TTS_CHUNKS_ROOT:=/mnt/pami201/zhuyin/datasets/wmt19_tts_chunks_500k}"
export WMT19_TTS_PYTHON="${WMT19_TTS_PYTHON:-$WORKSPACE_PYTHON}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
