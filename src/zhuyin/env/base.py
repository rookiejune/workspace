"""Base workspace environment contract shared by all machine profiles."""

from __future__ import annotations

from enum import auto
from os import PathLike
from pathlib import Path
from typing import ClassVar, Union

from .._compat import StrEnum

LOCATION_ENV = "LOCATION"
DYNAMIC_HOME_ENV = "DYNAMIC_HOME"
STATIC_HOME_ENV = "STATIC_HOME"

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


class BaseEnv:
    """Workspace roots supplied by one machine profile."""

    location: ClassVar[Location]
    static_home: ClassVar[Path]
    dynamic_home: ClassVar[Path]

    @classmethod
    def home(cls, name: str) -> Path:
        """Return one configured workspace root by environment variable name."""

        if name == STATIC_HOME_ENV:
            return cls.static_home
        if name == DYNAMIC_HOME_ENV:
            return cls.dynamic_home
        raise ValueError(f"unsupported workspace home: {name}")
