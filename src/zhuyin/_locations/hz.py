"""Hangzhou location placeholder.

This module intentionally does not define a profile yet. When Hangzhou is
enabled, add the concrete homes and marker here, register it in
``zhuyin._locations``, and add the matching tests in the same change.
"""

from __future__ import annotations

from pathlib import Path

from ._types import LocationProfile

LOCATION = "hz"


def profile() -> LocationProfile:
    """Raise until the Hangzhou profile is implemented."""

    raise NotImplementedError("Hangzhou location profile is not implemented.")


def markers() -> tuple[Path, ...]:
    """Return no auto-detection markers until Hangzhou is implemented."""

    return ()


__all__ = ["LOCATION", "markers", "profile"]
