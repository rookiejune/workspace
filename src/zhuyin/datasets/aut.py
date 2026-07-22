"""Load prepared Qwen2.5-Omni audio-tower targets for WMT19 TTS.

`prepared_aut()` returns a map-style dataset whose samples contain a 16 kHz
mono ``[1, time]`` waveform and its aligned, offline Qwen2.5-Omni audio
features. The loader accepts only the pinned teacher, Transformers runtime,
feature layout and timing schema defined by this module. It reads an immutable
prepared store and does not run the teacher, resample audio or recover omitted
text metadata.

An explicit ``root`` is the ``prepared_aut/wmt19_tts`` dataset root. Without
one, the root is resolved below ``zhuyin.env.datasets_home()``.
"""

from __future__ import annotations

from os import PathLike
from typing import TypedDict

from torch import Tensor
from torch.utils.data import Dataset

CHECKPOINT = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
TRANSFORMERS_VERSION = "5.14.1"
FEATURE_DIM = 3584
SAMPLE_RATE = 16_000


class Sample(TypedDict):
    """One validated waveform and aligned Qwen2.5-Omni AuT target."""

    sample_id: str
    audio_sha256: str
    waveform: Tensor
    waveform_length: Tensor
    sample_rate: int
    aut_features: Tensor
    aut_feature_mask: Tensor
    audio_placeholders: Tensor


def prepared_aut(
    *,
    root: str | PathLike[str] | None = None,
    split: str = "train",
    revision: str = DEFAULT_REVISION,
) -> Dataset[Sample]:
    """Load one split of the pinned WMT19 TTS prepared AuT store."""

    from zhuyin.datasets._aut_store import PreparedAuTStore

    return PreparedAuTStore(root=root, split=split, revision=revision)


__all__ = [
    "CHECKPOINT",
    "DEFAULT_REVISION",
    "FEATURE_DIM",
    "SAMPLE_RATE",
    "TRANSFORMERS_VERSION",
    "Sample",
    "prepared_aut",
]
