"""Resolve workspace paths from temporary workspace environment contexts."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

from . import default
from .base import (
    DYNAMIC_HOME_ENV,
    LOCATION_ENV,
    STATIC_HOME_ENV,
    WORKSPACE_ENV_NAMES,
    BaseEnv,
    EnvValue,
    Location,
)
from .fudan import FudanEnv
from .hz import HzEnv
from .us import UsEnv


@contextmanager
def context(
    overrides: Mapping[str, EnvValue] | None = None,
    **kwargs: EnvValue,
) -> Iterator[None]:
    """Temporarily apply resolved workspace environment variables."""

    merged = _merged_overrides(overrides, kwargs)
    source = _environ_with_overrides(os.environ, merged)
    updates = _workspace_environment(source)
    unset_names: set[str] = set()
    for name, value in merged.items():
        if value is None:
            if name not in WORKSPACE_ENV_NAMES:
                unset_names.add(name)
            continue
        if name not in WORKSPACE_ENV_NAMES:
            updates[name] = _env_string(name, value)

    previous = {name: os.environ.get(name) for name in set(updates) | unset_names}
    try:
        for name in unset_names:
            os.environ.pop(name, None)
        os.environ.update(updates)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def location(environ: Mapping[str, str] | None = None) -> Location:
    """Return the configured workspace location."""

    source = os.environ if environ is None else environ
    value = source.get(LOCATION_ENV)
    if value is None:
        return default.location()
    if not value:
        raise ValueError(f"{LOCATION_ENV} must not be empty.")
    try:
        return Location(value)
    except ValueError as e:
        choices = ", ".join(item.value for item in Location)
        raise ValueError(f"{LOCATION_ENV} must be one of: {choices}.") from e


def profile(environ: Mapping[str, str] | None = None) -> type[BaseEnv]:
    """Return the workspace profile class selected by `LOCATION`."""

    return default.profile(location(environ))


def dynamic_home() -> Path:
    """Return the configured dynamic workspace home."""

    return _required_env_path(DYNAMIC_HOME_ENV)


def static_home() -> Path:
    """Return the configured static workspace home."""

    return _required_env_path(STATIC_HOME_ENV)


def datasets_home() -> Path:
    """Return the root containing prepared datasets."""

    return static_home() / "datasets"


def _workspace_environment(environ: Mapping[str, str]) -> dict[str, str]:
    selected = profile(environ)
    return {
        LOCATION_ENV: selected.location.value,
        STATIC_HOME_ENV: str(_home_or_default(STATIC_HOME_ENV, environ, selected)),
        DYNAMIC_HOME_ENV: str(_home_or_default(DYNAMIC_HOME_ENV, environ, selected)),
    }


def _merged_overrides(
    overrides: Mapping[str, EnvValue] | None,
    kwargs: Mapping[str, EnvValue],
) -> dict[str, EnvValue]:
    merged = {} if overrides is None else dict(overrides)
    merged.update(kwargs)
    return merged


def _environ_with_overrides(
    environ: Mapping[str, str],
    overrides: Mapping[str, EnvValue],
) -> dict[str, str]:
    output = dict(environ)
    for name, value in overrides.items():
        if value is None:
            output.pop(name, None)
            continue
        output[name] = _env_string(name, value)
    return output


def _env_string(name: str, value: str | PathLike[str]) -> str:
    if isinstance(value, PathLike):
        return str(Path(value).expanduser())
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return str(value)


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is not set; use `with zhuyin.env.context():`.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return Path(value).expanduser()


def _home_or_default(
    name: str,
    environ: Mapping[str, str],
    default_profile: type[BaseEnv],
) -> Path:
    value = environ.get(name)
    if value is not None:
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return Path(value).expanduser()
    return _default_home(name, default_profile)


def _default_home(name: str, default_profile: type[BaseEnv]) -> Path:
    value = default_profile.home(name)
    warnings.warn(
        f"{name} is not set; using {default_profile.location.value} default {value}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return value


__all__ = [
    "DYNAMIC_HOME_ENV",
    "LOCATION_ENV",
    "STATIC_HOME_ENV",
    "BaseEnv",
    "EnvValue",
    "FudanEnv",
    "HzEnv",
    "Location",
    "UsEnv",
    "context",
    "datasets_home",
    "dynamic_home",
    "location",
    "profile",
    "static_home",
]
