"""Fudan workspace environment values."""

from __future__ import annotations

from pathlib import Path

from .base import BaseEnv, Location


class FudanEnv(BaseEnv):
    """Workspace roots on the Fudan shared machine."""

    location = Location.FUDAN
    static_home = Path("/mnt/pami202/zhuyin")
    dynamic_home = static_home / "dynamic"
