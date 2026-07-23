"""Materialize text datasets into Qwen CustomVoice speech stores.

This module owns workspace-level assembly for Qwen TTS speech materialization:
it assigns speaker ids to text samples, builds a Qwen provider factory, and
uses anydataset.store.ModalityMaterializer to write a resumable store.

The reusable sample wrapper lives in anydataset.dataset. Qwen inference lives
behind anydataset.provider.QwenTTSProvider and anytrain.tts.qwen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset import (
    SpeakerCartesianDataset,
    SpeakerIdDataset,
    SpeakerMode,
    TextRef,
)
from anydataset.provider import QwenTTSProvider
from anydataset.store import ModalityMaterializer
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextReq,
    TextView,
)

DatasetFactory = Callable[[], Any]


@dataclass(frozen=True)
class SpeakerDatasetFactory:
    dataset_factory: DatasetFactory
    speaker_ids: tuple[str, ...]
    mode: SpeakerMode
    text_ref: TextRef

    def __call__(self) -> SpeakerIdDataset:
        return SpeakerIdDataset(
            self.dataset_factory(),
            self.speaker_ids,
            mode=self.mode,
            text_ref=self.text_ref,
        )


@dataclass(frozen=True)
class SpeakerGridDatasetFactory:
    dataset_factory: DatasetFactory
    speaker_ids: tuple[str, ...]
    text_ref: TextRef

    def __call__(self) -> SpeakerCartesianDataset:
        return SpeakerCartesianDataset(
            self.dataset_factory(),
            self.speaker_ids,
            text_ref=self.text_ref,
        )


@dataclass(frozen=True)
class GroupedQwenTTSSpeechDataset:
    dataset: Any
    speaker_ids: tuple[str, ...]
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT)
    audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO)

    def __post_init__(self) -> None:
        if not self.speaker_ids:
            raise ValueError("speaker_ids must not be empty.")
        if len(self.dataset) % len(self.speaker_ids) != 0:
            raise ValueError("flat TTS dataset length must be divisible by speaker count.")

    def __len__(self) -> int:
        return len(self.dataset) // len(self.speaker_ids)

    def __getitem__(self, index: int) -> Sample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer.")
        if index < 0:
            raise ValueError("index must be non-negative.")
        if index >= len(self):
            raise IndexError("grouped sample index out of range.")

        start = index * len(self.speaker_ids)
        samples = [self.dataset[start + offset] for offset in range(len(self.speaker_ids))]
        first_text = _text_item(samples[0][self.text_ref], self.text_ref)
        grouped: Sample = dict(samples[0])
        grouped[self.text_ref] = TextItem(
            views=first_text.views,
            meta={
                **first_text.meta,
                TextMeta.SOURCE_INDEX: index,
            },
        )
        waveforms: list[Any] = []
        lengths: list[int] = []
        sample_rate: int | None = None
        for offset, sample in enumerate(samples):
            speaker_id = self.speaker_ids[offset]
            text_item = _text_item(sample[self.text_ref], self.text_ref)
            source_index = text_item.meta.get(TextMeta.SOURCE_INDEX)
            if source_index != index:
                raise ValueError(
                    f"flat sample {start + offset} has source index {source_index!r}; "
                    f"expected {index!r}."
                )
            actual_speaker = text_item.views.get(TextView.SPEAKERS)
            if actual_speaker != speaker_id:
                raise ValueError(
                    f"flat sample {start + offset} has speaker {actual_speaker!r}; "
                    f"expected {speaker_id!r}."
                )
            audio_item = _audio_item(sample[self.audio_ref], self.audio_ref)
            waveform, current_sample_rate = _waveform(audio_item.views[AudioView.WAVEFORM])
            if sample_rate is None:
                sample_rate = current_sample_rate
            elif current_sample_rate != sample_rate:
                raise ValueError("grouped speaker waveforms must share one sample rate.")
            waveforms.append(waveform)
            lengths.append(int(waveform.shape[-1]))
        grouped[self.audio_ref] = AudioItem(
            views={
                AudioView.WAVEFORM: (_stack_waveforms(waveforms), sample_rate),
                AudioView.SPEAKERS: self.speaker_ids,
                AudioView.SPEAKER_LENGTHS: _speaker_lengths(lengths),
            },
            meta={
                AudioMeta.DURATION: max(lengths) / sample_rate,
                AudioMeta.SPEAKER_ID: self.speaker_ids,
            },
        )
        return grouped


@dataclass(frozen=True)
class QwenProviderFactory:
    model: str | Path | None = None
    default_language: str = "Auto"
    default_instruct: str | None = None
    runtime_kwargs: Mapping[str, object] | None = None
    load_options: Mapping[str, object] = field(default_factory=dict)

    def __call__(self, device: str) -> QwenTTSProvider:
        load_options = dict(self.load_options)
        if "device_map" not in load_options:
            load_options["device_map"] = device
        return QwenTTSProvider(
            self.model,
            default_language=self.default_language,
            default_instruct=self.default_instruct,
            runtime_kwargs=self.runtime_kwargs,
            **load_options,
        )


def materialize_qwen_tts_speech(
    *,
    text_dataset_factory: DatasetFactory,
    speaker_ids: Sequence[str],
    output_dir: str | Path,
    speaker_mode: SpeakerMode = "aligned",
    split: str | None = None,
    model: str | Path | None = None,
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
    default_language: str = "Auto",
    default_instruct: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
    load_options: Mapping[str, object] | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    devices: Any = "auto",
) -> Path:
    """Write a Qwen CustomVoice speech store aligned with a text dataset."""

    dataset_factory = SpeakerDatasetFactory(
        text_dataset_factory,
        tuple(speaker_ids),
        speaker_mode,
        text_ref,
    )
    provider_factory = QwenProviderFactory(
        model=model,
        default_language=default_language,
        default_instruct=default_instruct,
        runtime_kwargs=runtime_kwargs,
        load_options=dict(load_options or {}),
    )
    return ModalityMaterializer(
        output_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        keep_schema={
            text_ref: TextReq(
                views=frozenset({TextView.TEXT, TextView.SPEAKERS}),
            )
        },
    ).write(
        dataset_factory=dataset_factory,
        provider_factory=provider_factory,
        devices=devices,
    )


def materialize_qwen_tts_speaker_grid(
    *,
    text_dataset_factory: DatasetFactory,
    speaker_ids: Sequence[str],
    output_dir: str | Path,
    split: str | None = None,
    model: str | Path | None = None,
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
    default_language: str = "Auto",
    default_instruct: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
    load_options: Mapping[str, object] | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    devices: Any = "auto",
) -> Path:
    """Write an expanded Qwen speaker grid store for grouped text-level reads."""

    dataset_factory = SpeakerGridDatasetFactory(
        text_dataset_factory,
        tuple(speaker_ids),
        text_ref,
    )
    provider_factory = QwenProviderFactory(
        model=model,
        default_language=default_language,
        default_instruct=default_instruct,
        runtime_kwargs=runtime_kwargs,
        load_options=dict(load_options or {}),
    )
    return ModalityMaterializer(
        output_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        keep_schema={
            text_ref: TextReq(
                views=frozenset({TextView.TEXT, TextView.SPEAKERS}),
                meta=frozenset({TextMeta.SOURCE_INDEX}),
            )
        },
    ).write(
        dataset_factory=dataset_factory,
        provider_factory=provider_factory,
        devices=devices,
    )


def qwen_tts_speaker_grid(
    *,
    root: str | Path,
    speaker_ids: Sequence[str],
    split: str = "train",
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
    audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO),
) -> GroupedQwenTTSSpeechDataset:
    """Load an expanded Qwen TTS store as grouped text-level samples."""

    return GroupedQwenTTSSpeechDataset(
        AnyDataset(Spec(source=Source.STORE, path=str(Path(root).expanduser()), split=split)),
        tuple(speaker_ids),
        text_ref=text_ref,
        audio_ref=audio_ref,
    )


def _text_item(value: object, ref: TextRef) -> TextItem:
    if not isinstance(value, TextItem):
        raise TypeError(f"{ref!r} must contain a TextItem.")
    return value


def _audio_item(value: object, ref: tuple[Role, Modality]) -> AudioItem:
    if not isinstance(value, AudioItem):
        raise TypeError(f"{ref!r} must contain an AudioItem.")
    return value


def _waveform(value: object):
    import torch

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
    waveform, sample_rate = value
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("AudioView.WAVEFORM sample rate must be an integer.")
    return waveform, sample_rate


def _stack_waveforms(waveforms: Sequence[Any]):
    import torch
    import torch.nn.functional as F

    if not waveforms:
        raise ValueError("speaker waveform list must not be empty.")
    max_length = max(int(waveform.shape[-1]) for waveform in waveforms)
    padded = [
        F.pad(waveform, (0, max_length - int(waveform.shape[-1])))
        for waveform in waveforms
    ]
    return torch.stack(padded, dim=0)


def _speaker_lengths(lengths: Sequence[int]):
    import torch

    return torch.tensor(tuple(lengths), dtype=torch.int64)
