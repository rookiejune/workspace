#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cd "$WORKSPACE_ROOT"

"${UNICODEC_PYTHON:-python}" scripts/prepare_wmt19_tts_codec.py unicodec "$@"
