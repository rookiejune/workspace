#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "$WORKSPACE_JOBS_DIR/.." && pwd)}"
export REPOS_ROOT="${REPOS_ROOT:-$(cd -- "$WORKSPACE_ROOT/.." && pwd)}"

WORKSPACE_PYTHONPATH="$WORKSPACE_ROOT/src:$REPOS_ROOT/third_party/anydataset/src:$REPOS_ROOT/third_party/anytrain/src"
export PYTHONPATH="$WORKSPACE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
