from __future__ import annotations

import os
from pathlib import Path

import pytest

from zhuyin import _locations, env
from zhuyin._locations import fudan


def _mock_existing_paths(
    monkeypatch: pytest.MonkeyPatch,
    paths: set[str],
) -> None:
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in paths)


def test_location_defaults_to_fudan_when_mnt_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt"})

    assert env.location() is env.Location.FUDAN
    assert env.LOCATION_ENV not in os.environ


def test_location_uses_fudan_fallback_without_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, set())

    assert env.location() is env.Location.FUDAN
    assert env.LOCATION_ENV not in os.environ


def test_fudan_location_profile_matches_location_file() -> None:
    assert env.Location.FUDAN.static_home == fudan.STATIC_HOME
    assert env.Location.FUDAN.dynamic_home == fudan.DYNAMIC_HOME


@pytest.mark.parametrize(
    "item",
    [env.Location.FUDAN, env.Location.HZ, env.Location.US],
)
def test_location_accepts_known_values(
    monkeypatch: pytest.MonkeyPatch,
    item: env.Location,
) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, item.value)

    assert env.location() is item


def test_location_rejects_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, "")

    with pytest.raises(ValueError, match=env.LOCATION_ENV):
        env.location()


def test_location_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, "unknown")

    with pytest.raises(ValueError, match=env.LOCATION_ENV):
        env.location()


def test_static_home_uses_fudan_default_without_mutating_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    _mock_existing_paths(monkeypatch, set())

    with pytest.warns(RuntimeWarning, match=env.STATIC_HOME_ENV):
        assert env.static_home() == fudan.STATIC_HOME

    assert env.STATIC_HOME_ENV not in os.environ


def test_dynamic_home_uses_fudan_default_without_mutating_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt"})

    with pytest.warns(RuntimeWarning, match=env.DYNAMIC_HOME_ENV):
        assert env.dynamic_home() == fudan.DYNAMIC_HOME

    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_explicit_homes_override_fudan_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.FUDAN.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/custom/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/custom/dynamic")

    assert env.static_home() == Path("/custom/static")
    assert env.dynamic_home() == Path("/custom/dynamic")


@pytest.mark.parametrize("item", [env.Location.HZ, env.Location.US])
def test_explicit_homes_allow_unimplemented_locations(
    monkeypatch: pytest.MonkeyPatch,
    item: env.Location,
) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, item.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/custom/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/custom/dynamic")

    assert env.static_home() == Path("/custom/static")
    assert env.dynamic_home() == Path("/custom/dynamic")


@pytest.mark.parametrize("item", [env.Location.HZ, env.Location.US])
def test_context_accepts_explicit_homes_for_unimplemented_locations(
    item: env.Location,
) -> None:
    with env.context(
        LOCATION=item.value,
        STATIC_HOME="/custom/static",
        DYNAMIC_HOME="/custom/dynamic",
    ):
        assert env.location() is item
        assert env.static_home() == Path("/custom/static")
        assert env.dynamic_home() == Path("/custom/dynamic")


@pytest.mark.parametrize("item", [env.Location.HZ, env.Location.US])
def test_unimplemented_locations_require_explicit_homes(
    monkeypatch: pytest.MonkeyPatch,
    item: env.Location,
) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, item.value)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.raises(NotImplementedError, match=item.value):
        env.static_home()
    with pytest.raises(NotImplementedError, match=item.value):
        env.dynamic_home()


@pytest.mark.parametrize("item", [env.Location.HZ, env.Location.US])
def test_location_properties_raise_for_unimplemented_defaults(item: env.Location) -> None:
    with pytest.raises(NotImplementedError, match=item.value):
        _ = item.static_home
    with pytest.raises(NotImplementedError, match=item.value):
        _ = item.dynamic_home


@pytest.mark.parametrize("name", ["hz", "us"])
def test_location_profile_raises_for_unimplemented_locations(name: str) -> None:
    match = name.upper() if name == "us" else "Hangzhou"
    with pytest.raises(NotImplementedError, match=match):
        _locations.profile(name)


def test_location_registry_only_returns_implemented_profiles() -> None:
    profiles = _locations.implemented_profiles()

    assert tuple(profiles) == (env.Location.FUDAN.value,)
    assert profiles[env.Location.FUDAN.value] == {
        "static_home": fudan.STATIC_HOME,
        "dynamic_home": fudan.DYNAMIC_HOME,
    }


def test_location_markers_only_include_implemented_markers() -> None:
    assert _locations.markers() == ((fudan.MARKER, env.Location.FUDAN.value),)


def test_train_home_uses_dynamic_train_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/custom/dynamic")

    assert env.train_home() == Path("/custom/dynamic/train")
    assert env.train_home("speech-to-speech") == Path(
        "/custom/dynamic/train/speech-to-speech"
    )


def test_train_home_rejects_escaping_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/custom/dynamic")

    for project in ("", ".", "/tmp/run", "../run", "project/../run"):
        with pytest.raises(ValueError, match="project"):
            env.train_home(project)


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
        assert env.static_home() == fudan.STATIC_HOME
        assert env.dynamic_home() == fudan.DYNAMIC_HOME
        assert env.datasets_home() == fudan.STATIC_HOME / "datasets"

    assert env.LOCATION_ENV not in os.environ
    assert env.STATIC_HOME_ENV not in os.environ
    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_context_restores_previous_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/old/static")

    with env.context(
        LOCATION=env.Location.FUDAN.value,
        STATIC_HOME="/data/static",
        DYNAMIC_HOME="/data/dynamic",
    ):
        assert env.static_home() == Path("/data/static")
        assert os.environ[env.LOCATION_ENV] == env.Location.FUDAN.value
        assert os.environ[env.STATIC_HOME_ENV] == "/data/static"
        assert os.environ[env.DYNAMIC_HOME_ENV] == "/data/dynamic"

    assert os.environ[env.STATIC_HOME_ENV] == "/old/static"
    assert env.LOCATION_ENV not in os.environ
    assert env.DYNAMIC_HOME_ENV not in os.environ


def test_context_applies_workspace_environment_and_unsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.FUDAN.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/data/dynamic")

    with env.context():
        assert env.static_home() == Path("/data/static")
        assert os.environ[env.LOCATION_ENV] == env.Location.FUDAN.value
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
        assert env.static_home() == fudan.STATIC_HOME
        assert os.environ[env.STATIC_HOME_ENV] == str(fudan.STATIC_HOME)
        assert os.environ["CUSTOM_ENV"] == "value"

    assert os.environ[env.STATIC_HOME_ENV] == "/old/static"
    assert os.environ[env.DYNAMIC_HOME_ENV] == "/old/dynamic"
    assert "CUSTOM_ENV" not in os.environ


def test_context_applies_non_workspace_overrides() -> None:
    with env.context(
        LOCATION=env.Location.FUDAN.value,
        STATIC_HOME="/data/static",
        DYNAMIC_HOME="/data/dynamic",
        HF_HOME="/custom/hf",
        CUSTOM_ENV="/custom/value",
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
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.FUDAN.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/data/dynamic")

    with pytest.raises(ValueError, match="CUSTOM_ENV"), env.context(CUSTOM_ENV=""):
        pass
