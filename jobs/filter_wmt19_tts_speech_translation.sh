#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cd "$WORKSPACE_ROOT"

python scripts/filter_wmt19_tts_speech_translation.py "$@"
