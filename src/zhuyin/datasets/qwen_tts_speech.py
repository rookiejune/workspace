"""Load prepared Qwen TTS speaker-grid stores as grouped logical datasets.

The materialization workflow lives in ``scripts/_qwen_tts_speech.py``. This
module only resolves a prepared store and returns the generic anydataset
speaker-axis wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset import GroupedSpeakerAudioDataset, TextRef
from anydataset.types import Modality, Role


def qwen_tts_speaker_grid(
    *,
    root: str | PathLike[str],
    speaker_ids: Sequence[str],
    split: str = "train",
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
    audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO),
) -> GroupedSpeakerAudioDataset:
    """Load one expanded Qwen TTS store grouped by source text index."""

    dataset = AnyDataset(
        Spec(
            source=Source.STORE,
            path=str(Path(root).expanduser()),
            split=split,
        )
    )
    return GroupedSpeakerAudioDataset(
        dataset,
        speaker_ids,
        text_ref=text_ref,
        audio_ref=audio_ref,
    )


__all__ = ["qwen_tts_speaker_grid"]
