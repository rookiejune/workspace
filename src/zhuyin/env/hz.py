"""Hangzhou workspace environment values."""

from __future__ import annotations

from pathlib import Path

from .base import BaseEnv, Location


class HzEnv(BaseEnv):
    """Workspace roots on the Hangzhou machine."""

    location = Location.HZ
    static_home = Path("/nfs/yin.zhu")
    dynamic_home = Path("/yin.zhu")
