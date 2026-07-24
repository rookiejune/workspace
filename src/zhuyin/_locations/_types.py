"""Shared contracts for workspace location modules."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict


class LocationProfile(TypedDict):
    """Physical workspace roots for one location."""

    static_home: Path
    dynamic_home: Path


class LocationModule(Protocol):
    """Public shape implemented by each location module."""

    LOCATION: str

    def profile(self) -> LocationProfile:
        """Return default workspace roots for this location."""
        ...

    def markers(self) -> tuple[Path, ...]:
        """Return filesystem markers used to detect this location."""
        ...


__all__ = ["LocationModule", "LocationProfile"]
