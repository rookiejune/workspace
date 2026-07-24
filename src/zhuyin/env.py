"""Resolve workspace machine locations, physical roots and environment injection.

This module owns the public `Location` values and loads each location's homes
and filesystem markers from private location profile modules. Path helpers
resolve environment values or location defaults without modifying the process
environment. `context()` temporarily injects the same resolved values for code
that requires environment variables.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from enum import auto
from os import PathLike
from pathlib import Path
from typing import Union

from ._compat import StrEnum
from ._locations import DEFAULT_LOCATION, implemented_profiles, markers, names

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
        return _default_home(STATIC_HOME_ENV, self)

    @property
    def dynamic_home(self) -> Path:
        return _default_home(DYNAMIC_HOME_ENV, self)


_HOMES: dict[Location, dict[str, Path]] = {
    Location(name): {
        STATIC_HOME_ENV: profile["static_home"],
        DYNAMIC_HOME_ENV: profile["dynamic_home"],
    }
    for name, profile in implemented_profiles().items()
}

# Detection markers checked in order; FUDAN is also the fallback.
_MARKERS = tuple((marker, Location(name)) for marker, name in markers())


@contextmanager
def context(**overrides: EnvValue) -> Generator[None]:
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
            _ = source.pop(name, None)
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
            _ = os.environ.pop(name, None)
        os.environ.update(updates)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                _ = os.environ.pop(name, None)
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


def train_home(project: str | PathLike[str] | None = None) -> Path:
    """Return the workspace training output root.

    `project` is an optional relative child directory under
    ``dynamic_home()/train``. It is validated here so experiment repos can rely
    on one workspace path contract instead of maintaining project-specific
    training-root environment fallbacks.
    """

    root = dynamic_home() / "train"
    if project is None:
        return root
    subdir = Path(project)
    if subdir == Path(".") or subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("project must be a non-empty relative path without '..'.")
    return root / subdir


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
        choices = ", ".join(names())
        raise ValueError(f"{LOCATION_ENV} must be one of: {choices}.") from e


def _detected_location() -> Location:
    for marker, item in _MARKERS:
        if marker.exists():
            return item
    return Location(DEFAULT_LOCATION)


def _home(name: str, environ: Mapping[str, str], resolved: Location) -> Path:
    value = environ.get(name)
    if value is not None:
        if not value:
            raise ValueError(f"{name} must not be empty.")
        return Path(value).expanduser()
    default = _default_home(name, resolved)
    warnings.warn(
        f"{name} is not set; using {resolved.value} default {default}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return default


def _default_home(name: str, resolved: Location) -> Path:
    defaults = _HOMES.get(resolved)
    if defaults is None:
        raise NotImplementedError(
            f"{resolved.value} location defaults are not implemented; set {name} explicitly or add the corresponding location profile."
        )
    return defaults[name]


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
    "train_home",
]
