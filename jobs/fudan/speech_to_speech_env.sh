#!/usr/bin/env bash
set -euo pipefail

FUDAN_SPEECH_TO_SPEECH_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LOCATION="${LOCATION:-fudan}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if [[ -z "${WORKSPACE_PYTHON:-}" && -x /home/zhuyin/anaconda3/envs/py312/bin/python ]]; then
  export WORKSPACE_PYTHON=/home/zhuyin/anaconda3/envs/py312/bin/python
fi

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${FUDAN_SPEECH_TO_SPEECH_JOBS_DIR}/../.." && pwd)}"
export REPOS_ROOT="${REPOS_ROOT:-$(cd -- "${WORKSPACE_ROOT}/.." && pwd)}"
export SPEECH_TO_SPEECH_ROOT="${SPEECH_TO_SPEECH_ROOT:-${REPOS_ROOT}/speech-to-speech}"

source "${SPEECH_TO_SPEECH_ROOT}/jobs/env.sh"

export HF_HOME="${HF_HOME:-${STATIC_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export ANYTRAIN_HOME="${ANYTRAIN_HOME:-${STATIC_HOME}/.anytrain}"

FDU_DATA_ARGS=()

fdu_oracle_data_args() {
  FDU_DATA_ARGS=()
  if [[ -n "${SPEECH_TO_SPEECH_ORACLE_DATA_ROOT:-}" ]]; then
    FDU_DATA_ARGS=("codec_oracle.data.root=${SPEECH_TO_SPEECH_ORACLE_DATA_ROOT}")
  fi
}

fdu_p0_data_args() {
  local key="${1:-data.root}"
  FDU_DATA_ARGS=()
  if [[ -n "${SPEECH_TO_SPEECH_P0_DATA_ROOT:-}" ]]; then
    FDU_DATA_ARGS=("${key}=${SPEECH_TO_SPEECH_P0_DATA_ROOT}")
  fi
}

fdu_stage_data_args() {
  local key="$1"
  FDU_DATA_ARGS=()
  if [[ -n "${SPEECH_TO_SPEECH_STAGE_DATA_ROOT:-}" ]]; then
    FDU_DATA_ARGS=("${key}=${SPEECH_TO_SPEECH_STAGE_DATA_ROOT}")
  fi
}

fdu_qwen_root() {
  local env_name="${1:-SPEECH_TO_SPEECH_STAGE_QWEN_ROOT}"
  local override="${!env_name-}"
  local default_qwen_root="${HF_HUB_CACHE}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
  if [[ -z "${override}" && -d "${default_qwen_root}" ]]; then
    printf '%s\n' "${default_qwen_root}"
  else
    printf '%s\n' "${override:-Qwen/Qwen3-0.6B}"
  fi
}
