"""Resolve workspace machine locations, physical roots and environment injection.

This module owns the known `Location` values, each location's workspace homes
and the filesystem markers used for detection. Path helpers resolve environment
values or location defaults without modifying the process environment.
`context()` temporarily injects the same resolved values for code that requires
environment variables.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import auto
from os import PathLike
from pathlib import Path
from typing import Union

from ._compat import StrEnum

LOCATION_ENV = "LOCATION"
STATIC_HOME_ENV = "STATIC_HOME"
DYNAMIC_HOME_ENV = "DYNAMIC_HOME"

WORKSPACE_ENV_NAMES = (
    LOCATION_ENV,
    STATIC_HOME_ENV,
    DYNAMIC_HOME_ENV,
)
EnvValue = Union[str, PathLike[str], None]


class Location(StrEnum):
    """Known machine locations."""

    FUDAN = auto()
    HZ = auto()
    US = auto()

    @property
    def static_home(self) -> Path:
        return _HOMES[self][STATIC_HOME_ENV]

    @property
    def dynamic_home(self) -> Path:
        return _HOMES[self][DYNAMIC_HOME_ENV]


_HOMES: dict[Location, dict[str, Path]] = {
    Location.FUDAN: {
        STATIC_HOME_ENV: Path("/mnt/pami202/zhuyin"),
        DYNAMIC_HOME_ENV: Path("/mnt/pami202/zhuyin/dynamic"),
    },
    Location.HZ: {
        STATIC_HOME_ENV: Path("/nfs/yin.zhu"),
        DYNAMIC_HOME_ENV: Path("/yin.zhu"),
    },
    Location.US: {
        STATIC_HOME_ENV: Path("/zoomai/colddata/video/yin.zhu"),
        DYNAMIC_HOME_ENV: Path("/share5_video/users/yin.zhu"),
    },
}

# Detection markers checked in order; FUDAN is also the fallback.
_MARKERS = (
    (Path("/share5_video"), Location.US),
    (Path("/nfs/yin.zhu"), Location.HZ),
    (Path("/mnt"), Location.FUDAN),
)

@contextmanager
def context(**overrides: EnvValue) -> Iterator[None]:
    """Temporarily apply resolved workspace environment variables.

    Each workspace variable resolves in one fixed order: explicit keyword
    override, current process environment, then the default of the resolved
    `Location` (with a `RuntimeWarning`). Extra keywords are injected as
    plain environment variables; `None` unsets a non-workspace variable and
    re-resolves a workspace variable from the remaining sources.
    """

    source = dict(os.environ)
    unset_names: set[str] = set()
    for name, value in overrides.items():
        if value is None:
            source.pop(name, None)
            if name not in WORKSPACE_ENV_NAMES:
                unset_names.add(name)
        else:
            source[name] = _env_string(name, value)

    resolved = _location(source)
    updates = {LOCATION_ENV: resolved.value}
    for name in (STATIC_HOME_ENV, DYNAMIC_HOME_ENV):
        updates[name] = str(_home(name, source, resolved))
    for name, value in overrides.items():
        if name in WORKSPACE_ENV_NAMES or value is None:
            continue
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


def location() -> Location:
    """Return the configured workspace location."""

    return _location(os.environ)


def dynamic_home() -> Path:
    """Resolve the dynamic workspace home without modifying the environment."""

    return _resolved_home(DYNAMIC_HOME_ENV)


def static_home() -> Path:
    """Resolve the static workspace home without modifying the environment."""

    return _resolved_home(STATIC_HOME_ENV)


def datasets_home() -> Path:
    """Return the root containing prepared datasets."""

    return static_home() / "datasets"


def _resolved_home(name: str) -> Path:
    source = os.environ
    return _home(name, source, _location(source))


def _location(environ: Mapping[str, str]) -> Location:
    value = environ.get(LOCATION_ENV)
    if value is None:
        return _detected_location()
    if not value:
        raise ValueError(f"{LOCATION_ENV} must not be empty.")
    try:
        return Location(value)
    except ValueError as e:
        choices = ", ".join(item.value for item in Location)
        raise ValueError(f"{LOCATION_ENV} must be one of: {choices}.") from e


def _detected_location() -> Location:
    for marker, item in _MARKERS:
        if marker.exists():
            return item
    return Location.FUDAN


def _home(name: str, environ: Mapping[str, str], resolved: Location) -> Path:
    value = environ.get(name)
    if value is not None:
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return Path(value).expanduser()
    default = _HOMES[resolved][name]
    warnings.warn(
        f"{name} is not set; using {resolved.value} default {default}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return default


def _env_string(name: str, value: str | PathLike[str]) -> str:
    if isinstance(value, PathLike):
        return str(Path(value).expanduser())
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return str(value)


__all__ = [
    "DYNAMIC_HOME_ENV",
    "LOCATION_ENV",
    "STATIC_HOME_ENV",
    "EnvValue",
    "Location",
    "context",
    "datasets_home",
    "dynamic_home",
    "location",
    "static_home",
]
