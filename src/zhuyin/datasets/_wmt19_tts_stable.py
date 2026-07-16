"""Stable Codec quantizer presets and WMT19 store identities."""

from __future__ import annotations

from zhuyin._compat import StrEnum


class StableQuantizer(StrEnum):
    """Supported Stable Codec posthoc quantizer presets."""

    FSQ_1X46656_400BPS = "1x46656_400bps"
    FSQ_2X15625_700BPS = "2x15625_700bps"
    FSQ_4X729_1000BPS = "4x729_1000bps"


DEFAULT_STABLE_QUANTIZER = StableQuantizer.FSQ_1X46656_400BPS


def store_dir(
    quantizer: StableQuantizer | str = DEFAULT_STABLE_QUANTIZER,
) -> str:
    """Return the standard store directory for one Stable quantizer preset."""

    return f"stable-{StableQuantizer(quantizer).value}"


__all__ = ["DEFAULT_STABLE_QUANTIZER", "StableQuantizer", "store_dir"]
