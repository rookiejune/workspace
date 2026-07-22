"""Fudan workspace location defaults."""

from __future__ import annotations

from pathlib import Path

LOCATION = "fudan"
STATIC_HOME = Path("/mnt/pami202/zhuyin")
DYNAMIC_HOME = Path("/mnt/pami202/zhuyin/dynamic")
MARKER = Path("/mnt")


__all__ = ["DYNAMIC_HOME", "LOCATION", "MARKER", "STATIC_HOME"]
