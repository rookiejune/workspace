"""US workspace environment values."""

from __future__ import annotations

from pathlib import Path

from .base import BaseEnv, Location


class UsEnv(BaseEnv):
    """Workspace roots on the US machine."""

    location = Location.US
    static_home = Path("/zoomai/colddata/video/yin.zhu")
    dynamic_home = Path("/share5_video/users/yin.zhu")
