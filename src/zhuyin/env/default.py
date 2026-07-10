"""Default workspace profile detection."""

from __future__ import annotations

from pathlib import Path

from .base import BaseEnv, Location
from .fudan import FudanEnv
from .hz import HzEnv
from .us import UsEnv

US_MARKER = Path("/share5_video")
HZ_MARKER = HzEnv.static_home
FUDAN_MARKER = Path("/mnt")

PROFILES: dict[Location, type[BaseEnv]] = {
    Location.FUDAN: FudanEnv,
    Location.HZ: HzEnv,
    Location.US: UsEnv,
}


def location() -> Location:
    """Detect the current workspace location from filesystem markers."""

    if US_MARKER.exists():
        return Location.US
    if HZ_MARKER.exists():
        return Location.HZ
    if FUDAN_MARKER.exists():
        return Location.FUDAN
    return Location.FUDAN


def profile(location: Location) -> type[BaseEnv]:
    """Return the workspace profile class for one location."""

    return PROFILES[location]
