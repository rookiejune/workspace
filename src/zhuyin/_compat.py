"""Small compatibility helpers for supported Python versions."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Python 3.9-compatible subset of `enum.StrEnum`."""

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[object],
    ) -> str:
        return name.lower()

    def __str__(self) -> str:
        return self.value
