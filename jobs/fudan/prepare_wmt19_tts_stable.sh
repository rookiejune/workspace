#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cd "$WORKSPACE_ROOT"

"${STABLE_CODEC_PYTHON:-$WORKSPACE_PYTHON}" scripts/prepare_wmt19_tts_codec_view.py stable \
  --posthoc-bottleneck 1x46656_400bps \
  "$@"
