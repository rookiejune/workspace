#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cd "$WORKSPACE_ROOT"

"$WORKSPACE_PYTHON" scripts/filter_wmt19_tts.py speech-translation "$@"
