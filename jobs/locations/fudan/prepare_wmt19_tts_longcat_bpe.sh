#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cd "$WORKSPACE_ROOT"

"$WORKSPACE_PYTHON" scripts/prepare_wmt19_tts_bpe.py "$@"
