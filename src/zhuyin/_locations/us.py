"""US location placeholder.

This module intentionally does not define a profile yet. When US is enabled,
add the concrete homes and marker here, register it in ``zhuyin._locations``,
and add the matching tests in the same change.
"""

from __future__ import annotations


def profile() -> None:
    """Raise until the US profile is implemented."""

    raise NotImplementedError("US location profile is not implemented.")


__all__ = ["profile"]
