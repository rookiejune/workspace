from __future__ import annotations

import os
from pathlib import Path

import pytest

from zhuyin import env


def test_static_home_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)

    with pytest.raises(ValueError, match=env.STATIC_HOME_ENV):
        env.static_home()


def test_dynamic_home_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.raises(ValueError, match=env.DYNAMIC_HOME_ENV):
        env.dynamic_home()


def test_path_helpers_use_static_and_dynamic_homes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/data/dynamic")

    assert env.dataset_dir("common_voice") == Path("/data/static/datasets/common_voice")
    assert env.train_dir("anycodec") == Path("/data/dynamic/train/anycodec")
    assert env.models_dir() == Path("/data/static/models")
    assert env.debug_dir() == Path("/data/dynamic/debug")
    assert env.debug_dir("workspace") == Path("/data/dynamic/debug/workspace")


def test_configure_environment_respects_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.ANYDATASET_HOME_ENV, "/custom/anydataset")
    monkeypatch.setenv(env.HF_HOME_ENV, "/custom/hf")

    env.configure_environment()

    assert os.environ[env.ANYDATASET_HOME_ENV] == "/custom/anydataset"
    assert os.environ[env.HF_HOME_ENV] == "/custom/hf"


def test_configure_environment_rejects_empty_derived_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.ANYDATASET_HOME_ENV, "")

    with pytest.raises(ValueError, match=env.ANYDATASET_HOME_ENV):
        env.configure_environment()
