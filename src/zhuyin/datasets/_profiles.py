"""Map workspace locations to dataset physical loading profiles.

This private module keeps machine-specific physical dataset choices out of the
public dataset entrances. Entrances still expose logical objects; profiles only
select the default storage protocol and physical root used to build them.
"""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from ..env import HZ_HOME, Location, location

HZ_WMT19_TTS_EXPORT_ROOT = HZ_HOME / "train" / "text_to_speech" / "moss_tts_hz_export"
HZ_WMT19_TTS_LONGCAT_ROOT = HZ_HOME / "datasets" / "wmt19_tts_longcat_codes_text_cleaned"


class WMT19TTSProfile(StrEnum):
    """Physical profiles that resolve to the logical WMT19 TTS dataset."""

    STORE = auto()
    HZ_EXPORT = auto()

    @classmethod
    def resolve(
        cls,
        value: Self | str | None,
    ) -> Self:
        if value is not None:
            return cls(value)
        return cls.default()

    @classmethod
    def default(cls) -> Self:
        if location() is Location.HZ:
            return cls.HZ_EXPORT
        return cls.STORE

    @property
    def default_root(self) -> Path:
        if self is WMT19TTSProfile.HZ_EXPORT:
            return HZ_WMT19_TTS_EXPORT_ROOT
        raise ValueError(f"{self.value} profile does not own a standalone root.")


class WMT19TTSLongCatProfile(StrEnum):
    """Physical profiles that resolve to the logical WMT19 TTS LongCat dataset."""

    STORE = auto()
    HZ_HF_DISK_CODES = auto()

    @classmethod
    def resolve(
        cls,
        value: Self | str | None,
    ) -> Self:
        if value is not None:
            return cls(value)
        return cls.default()

    @classmethod
    def default(cls) -> Self:
        if location() is Location.HZ:
            return cls.HZ_HF_DISK_CODES
        return cls.STORE

    @property
    def default_root(self) -> Path:
        if self is WMT19TTSLongCatProfile.HZ_HF_DISK_CODES:
            return HZ_WMT19_TTS_LONGCAT_ROOT
        raise ValueError(f"{self.value} profile does not own a standalone root.")
