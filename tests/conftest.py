from __future__ import annotations

import pytest


_WORKSPACE_ENV_NAMES = (
    "LOCATION",
    "STATIC_HOME",
    "DYNAMIC_HOME",
    "ANYDATASET_HOME",
    "ANYTRAIN_HOME",
    "BPE_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
    "ANYTRAIN_WHISPER_ROOT",
)


@pytest.fixture(autouse=True)
def clean_workspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _WORKSPACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
