"""Private workspace location profiles.

Only implemented locations expose physical homes and detection markers here.
Add another location module in the same shape when that location is ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from . import fudan


class LocationProfile(TypedDict):
    """Physical workspace roots for one location."""

    static_home: Path
    dynamic_home: Path


DEFAULT_LOCATION = fudan.LOCATION
LOCATION_PROFILES: dict[str, LocationProfile] = {
    fudan.LOCATION: {
        "static_home": fudan.STATIC_HOME,
        "dynamic_home": fudan.DYNAMIC_HOME,
    },
}

# Detection order is part of the implemented location contract.
DETECTION_MARKERS: tuple[tuple[Path, str], ...] = (
    (fudan.MARKER, fudan.LOCATION),
)


__all__ = [
    "DEFAULT_LOCATION",
    "DETECTION_MARKERS",
    "LOCATION_PROFILES",
    "LocationProfile",
]
