from __future__ import annotations

import os
from pathlib import Path

import pytest

from zhuyin import env


def _mock_existing_paths(
    monkeypatch: pytest.MonkeyPatch,
    paths: set[str],
) -> None:
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in paths)


def test_location_defaults_to_fudan_when_only_mnt_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt"})

    assert env.location() is env.Location.FUDAN
    assert env.LOCATION_ENV not in os.environ


def test_location_defaults_to_hz_when_nfs_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt", "/nfs/yin.zhu"})

    assert env.location() is env.Location.HZ
    assert env.LOCATION_ENV not in os.environ


def test_location_defaults_to_us_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt", "/nfs/yin.zhu", "/share5_video"})

    assert env.location() is env.Location.US
    assert env.LOCATION_ENV not in os.environ


def test_location_uses_fudan_fallback_without_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, set())

    assert env.location() is env.Location.FUDAN
    assert env.LOCATION_ENV not in os.environ


def test_location_accepts_hz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)

    assert env.location() is env.Location.HZ


def test_location_accepts_us(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.US.value)

    assert env.location() is env.Location.US


def test_location_rejects_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, "")

    with pytest.raises(ValueError, match=env.LOCATION_ENV):
        env.location()


def test_location_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, "unknown")

    with pytest.raises(ValueError, match=env.LOCATION_ENV):
        env.location()


def test_static_home_requires_context_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)

    with pytest.raises(RuntimeError, match=env.STATIC_HOME_ENV):
        env.static_home()

    assert env.STATIC_HOME_ENV not in os.environ


def test_dynamic_home_requires_context_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.raises(RuntimeError, match=env.DYNAMIC_HOME_ENV):
        env.dynamic_home()

    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_context_defaults_to_location_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.Location.HZ.value), env.context():
        assert env.static_home() == env.HzEnv.static_home
        assert env.dynamic_home() == env.HzEnv.dynamic_home

    assert os.environ[env.LOCATION_ENV] == env.Location.HZ.value
    assert env.STATIC_HOME_ENV not in os.environ
    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_context_defaults_to_us_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.US.value)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.Location.US.value), env.context():
        assert env.static_home() == env.UsEnv.static_home
        assert env.dynamic_home() == env.UsEnv.dynamic_home

    assert os.environ[env.LOCATION_ENV] == env.Location.US.value
    assert env.STATIC_HOME_ENV not in os.environ
    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_explicit_homes_override_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/custom/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/custom/dynamic")

    assert env.static_home() == Path("/custom/static")
    assert env.dynamic_home() == Path("/custom/dynamic")


def test_homes_reject_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "")

    with pytest.raises(ValueError, match=env.STATIC_HOME_ENV):
        env.static_home()
    with pytest.raises(ValueError, match=env.DYNAMIC_HOME_ENV):
        env.dynamic_home()


def test_context_applies_fudan_default_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)
    _mock_existing_paths(monkeypatch, set())

    with pytest.warns(RuntimeWarning), env.context():
        assert env.location() is env.Location.FUDAN
        assert env.static_home() == Path("/mnt/pami202/zhuyin")
        assert env.dynamic_home() == Path("/mnt/pami202/zhuyin/dynamic")
        assert env.datasets_home() == Path("/mnt/pami202/zhuyin/datasets")

    assert env.LOCATION_ENV not in os.environ
    assert env.STATIC_HOME_ENV not in os.environ
    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_context_restores_previous_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/old/static")

    with env.context(
        LOCATION=env.Location.US.value,
        STATIC_HOME="/data/static",
        DYNAMIC_HOME="/data/dynamic",
    ):
        assert env.static_home() == Path("/data/static")
        assert os.environ[env.LOCATION_ENV] == env.Location.US.value
        assert os.environ[env.STATIC_HOME_ENV] == "/data/static"
        assert os.environ[env.DYNAMIC_HOME_ENV] == "/data/dynamic"

    assert os.environ[env.STATIC_HOME_ENV] == "/old/static"
    assert env.LOCATION_ENV not in os.environ
    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_context_applies_workspace_environment_and_unsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/data/dynamic")

    with env.context():
        assert env.static_home() == Path("/data/static")
        assert os.environ[env.LOCATION_ENV] == env.Location.HZ.value
        assert os.environ[env.STATIC_HOME_ENV] == "/data/static"
        assert os.environ[env.DYNAMIC_HOME_ENV] == "/data/dynamic"

    assert os.environ[env.STATIC_HOME_ENV] == "/data/static"
    assert os.environ[env.DYNAMIC_HOME_ENV] == "/data/dynamic"


def test_context_none_override_re_resolves_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.FUDAN.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/old/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/old/dynamic")

    with (
        pytest.warns(RuntimeWarning, match=env.STATIC_HOME_ENV),
        env.context(STATIC_HOME=None, CUSTOM_ENV="value"),
    ):
        assert env.static_home() == env.FudanEnv.static_home
        assert os.environ[env.STATIC_HOME_ENV] == str(env.FudanEnv.static_home)
        assert os.environ["CUSTOM_ENV"] == "value"

    assert os.environ[env.STATIC_HOME_ENV] == "/old/static"
    assert os.environ[env.DYNAMIC_HOME_ENV] == "/old/dynamic"
    assert "CUSTOM_ENV" not in os.environ


def test_context_applies_non_workspace_overrides() -> None:
    with env.context(
        {
            env.LOCATION_ENV: env.Location.FUDAN.value,
            env.STATIC_HOME_ENV: "/data/static",
            env.DYNAMIC_HOME_ENV: "/data/dynamic",
            "HF_HOME": "/custom/hf",
            "CUSTOM_ENV": "/custom/value",
        }
    ):
        assert env.static_home() == Path("/data/static")
        assert env.dynamic_home() == Path("/data/dynamic")
        assert os.environ["HF_HOME"] == "/custom/hf"
        assert os.environ["CUSTOM_ENV"] == "/custom/value"

    assert env.STATIC_HOME_ENV not in os.environ
    assert "HF_HOME" not in os.environ
    assert "CUSTOM_ENV" not in os.environ


def test_context_rejects_empty_non_workspace_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = {
        env.LOCATION_ENV: env.Location.FUDAN.value,
        env.STATIC_HOME_ENV: "/data/static",
        env.DYNAMIC_HOME_ENV: "/data/dynamic",
        "CUSTOM_ENV": "",
    }

    with pytest.raises(ValueError, match="CUSTOM_ENV"), env.context(environ):
        pass
