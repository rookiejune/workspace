"""Fudan workspace location defaults."""

from __future__ import annotations

from pathlib import Path

from ._types import LocationProfile

LOCATION = "fudan"
STATIC_HOME = Path("/mnt/pami202/zhuyin")
DYNAMIC_HOME = Path("/mnt/pami202/zhuyin/dynamic")
MARKER = Path("/mnt")


def profile() -> LocationProfile:
    """Return Fudan workspace roots."""

    return {
        "static_home": STATIC_HOME,
        "dynamic_home": DYNAMIC_HOME,
    }


def markers() -> tuple[Path, ...]:
    """Return filesystem markers for Fudan auto-detection."""

    return (MARKER,)


__all__ = [
    "DYNAMIC_HOME",
    "LOCATION",
    "MARKER",
    "STATIC_HOME",
    "markers",
    "profile",
]
