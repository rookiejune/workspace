#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "$WORKSPACE_JOBS_DIR/.." && pwd)}"
export REPOS_ROOT="${REPOS_ROOT:-$(cd -- "$WORKSPACE_ROOT/.." && pwd)}"

export LOCATION="${LOCATION:-fudan}"

case "$LOCATION" in
  fudan)
    WORKSPACE_DEFAULT_STATIC_HOME=/mnt/pami202/zhuyin
    WORKSPACE_DEFAULT_DYNAMIC_HOME=/mnt/pami202/zhuyin/dynamic
    ;;
  hz)
    WORKSPACE_DEFAULT_STATIC_HOME=/nfs/yin.zhu
    WORKSPACE_DEFAULT_DYNAMIC_HOME=/yin.zhu
    ;;
  *)
    echo "unsupported LOCATION: $LOCATION" >&2
    exit 1
    ;;
esac

export STATIC_HOME="${STATIC_HOME:-$WORKSPACE_DEFAULT_STATIC_HOME}"
export DYNAMIC_HOME="${DYNAMIC_HOME:-$WORKSPACE_DEFAULT_DYNAMIC_HOME}"

export ANYDATASET_HOME="${ANYDATASET_HOME:-$STATIC_HOME/anydataset}"
export ANYTRAIN_HOME="${ANYTRAIN_HOME:-$STATIC_HOME}"
export BPE_CACHE_DIR="${BPE_CACHE_DIR:-$STATIC_HOME/bpe}"
export HF_HOME="${HF_HOME:-$STATIC_HOME/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$STATIC_HOME/torch}"
export ANYTRAIN_WHISPER_ROOT="${ANYTRAIN_WHISPER_ROOT:-$STATIC_HOME/whisper}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -x /home/zhuyin/anaconda3/envs/py312/bin/python ]]; then
  WORKSPACE_DEFAULT_PYTHON=/home/zhuyin/anaconda3/envs/py312/bin/python
elif [[ -x /home/zhuyin/miniconda3/envs/py312/bin/python ]]; then
  WORKSPACE_DEFAULT_PYTHON=/home/zhuyin/miniconda3/envs/py312/bin/python
elif [[ -x /Users/zhuyin/miniconda3/envs/torch2.12/bin/python ]]; then
  WORKSPACE_DEFAULT_PYTHON=/Users/zhuyin/miniconda3/envs/torch2.12/bin/python
else
  WORKSPACE_DEFAULT_PYTHON=python
fi

export PYTHONPATH="${PYTHONPATH:-$WORKSPACE_ROOT/src:$REPOS_ROOT/third_party/anydataset/src:$REPOS_ROOT/third_party/anytrain/src}"
export WORKSPACE_PYTHON="${WORKSPACE_PYTHON:-$WORKSPACE_DEFAULT_PYTHON}"
