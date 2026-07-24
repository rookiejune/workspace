"""Workspace location profile registry.

The registry owns the stable top-level location API and loads concrete
location modules lazily. Each location module exposes the same contract, while
unimplemented profiles raise from their own module instead of being silently
filtered at import time.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

from ._types import LocationModule, LocationProfile

DEFAULT_LOCATION = "fudan"

_MODULES = {
    "fudan": "zhuyin._locations.fudan",
    "hz": "zhuyin._locations.hz",
    "us": "zhuyin._locations.us",
}


def names() -> tuple[str, ...]:
    """Return known location names in registry order."""

    return tuple(_MODULES)


def profile(name: str) -> LocationProfile:
    """Return one location profile, raising if that profile is not implemented."""

    return _module(name).profile()


def markers() -> tuple[tuple[Path, str], ...]:
    """Return detection markers for all modules that expose markers."""

    items: list[tuple[Path, str]] = []
    for name in names():
        module = _module(name)
        items.extend((marker, name) for marker in module.markers())
    return tuple(items)


def implemented_profiles() -> dict[str, LocationProfile]:
    """Return only profiles with concrete default homes."""

    profiles: dict[str, LocationProfile] = {}
    for name in names():
        try:
            profiles[name] = profile(name)
        except NotImplementedError:
            continue
    return profiles


def _module(name: str) -> LocationModule:
    try:
        module_path = _MODULES[name]
    except KeyError as e:
        choices = ", ".join(names())
        raise ValueError(f"location must be one of: {choices}.") from e
    module = cast(object, import_module(module_path))
    return cast(LocationModule, module)


__all__ = [
    "DEFAULT_LOCATION",
    "LocationModule",
    "LocationProfile",
    "implemented_profiles",
    "markers",
    "names",
    "profile",
]
