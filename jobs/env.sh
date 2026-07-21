#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "$WORKSPACE_JOBS_DIR/.." && pwd)}"
export REPOS_ROOT="${REPOS_ROOT:-$(cd -- "$WORKSPACE_ROOT/.." && pwd)}"

WORKSPACE_PYTHONPATH="$WORKSPACE_ROOT/src:$REPOS_ROOT/third_party/anydataset/src:$REPOS_ROOT/third_party/anytrain/src"
export PYTHONPATH="$WORKSPACE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export WORKSPACE_PYTHON="${WORKSPACE_PYTHON:-python}"

eval "$("${WORKSPACE_PYTHON}" - <<'PY'
from __future__ import annotations

import shlex
import warnings

from zhuyin import env

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    values = {
        env.LOCATION_ENV: env.location().value,
        env.STATIC_HOME_ENV: str(env.static_home()),
        env.DYNAMIC_HOME_ENV: str(env.dynamic_home()),
    }

for name, value in values.items():
    print(f"export {name}={shlex.quote(value)}")
PY
)"
