"""Build Qwen TTS audio views from selected text references."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from anydataset.dataset import (
    SpeakerAssignment,
    SpeakerCartesianDataset,
    SpeakerIdDataset,
    TextRef,
)
from anydataset.provider import QwenTTSProvider
from anydataset.store import ModalityMaterializer
from anydataset.types import Modality, Role, Sample, TextItem, TextMeta, TextReq, TextView

DatasetFactory = Callable[[], Any]


class SelectedTextDataset:
    """Project a map-style dataset onto the text references used for TTS."""

    def __init__(self, dataset: Any, text_refs: tuple[TextRef, ...]) -> None:
        self.dataset = dataset
        self.text_refs = tuple(text_refs)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        sample = self.dataset[index]
        selected: dict[TextRef, TextItem] = {}
        for text_ref in self.text_refs:
            item = sample.get(text_ref)
            if not isinstance(item, TextItem):
                raise TypeError(f"{text_ref!r} must contain a TextItem.")
            selected[text_ref] = item
        return selected


class SpeakerDatasetFactory:
    """Picklable factory for multi-reference speaker assignment."""

    def __init__(
        self,
        dataset_factory: DatasetFactory,
        assignments: Mapping[TextRef, SpeakerAssignment],
    ) -> None:
        self.dataset_factory = dataset_factory
        self.assignments = assignments

    def __call__(self) -> SelectedTextDataset:
        dataset = SpeakerIdDataset(self.dataset_factory(), self.assignments)
        return SelectedTextDataset(dataset, tuple(dataset.assignments))


class SpeakerGridDatasetFactory:
    """Picklable factory for one text-reference speaker grid."""

    def __init__(
        self,
        dataset_factory: DatasetFactory,
        speaker_ids: Sequence[str],
        text_ref: TextRef,
    ) -> None:
        self.dataset_factory = dataset_factory
        self.speaker_ids = tuple(speaker_ids)
        self.text_ref = text_ref

    def __call__(self) -> SelectedTextDataset:
        dataset = SpeakerCartesianDataset(
            self.dataset_factory(),
            self.speaker_ids,
            text_ref=self.text_ref,
        )
        return SelectedTextDataset(dataset, (dataset.text_ref,))


class QwenProviderFactory:
    """Picklable Qwen CustomVoice provider configuration."""

    def __init__(
        self,
        *,
        model: str | Path | None = None,
        default_language: str = "Auto",
        default_instruct: str | None = None,
        runtime_kwargs: Mapping[str, object] | None = None,
        load_options: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model
        self.default_language = default_language
        self.default_instruct = default_instruct
        self.runtime_kwargs = None if runtime_kwargs is None else dict(runtime_kwargs)
        self.load_options = {} if load_options is None else dict(load_options)

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
    assignments: Mapping[TextRef, SpeakerAssignment],
    output_dir: str | Path,
    split: str | None = None,
    model: str | Path | None = None,
    default_language: str = "Auto",
    default_instruct: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
    load_options: Mapping[str, object] | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    devices: Any = "auto",
) -> Path:
    """Materialize audio for every explicitly assigned text reference."""

    return ModalityMaterializer(
        output_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        keep_schema=_text_schema(assignments),
    ).write(
        dataset_factory=SpeakerDatasetFactory(
            text_dataset_factory,
            assignments,
        ),
        provider_factory=QwenProviderFactory(
            model=model,
            default_language=default_language,
            default_instruct=default_instruct,
            runtime_kwargs=runtime_kwargs,
            load_options=load_options,
        ),
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
    """Materialize every speaker id for one selected text reference."""

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
        dataset_factory=SpeakerGridDatasetFactory(
            text_dataset_factory,
            speaker_ids,
            text_ref,
        ),
        provider_factory=QwenProviderFactory(
            model=model,
            default_language=default_language,
            default_instruct=default_instruct,
            runtime_kwargs=runtime_kwargs,
            load_options=load_options,
        ),
        devices=devices,
    )


def _text_schema(
    assignments: Mapping[TextRef, SpeakerAssignment],
) -> dict[TextRef, TextReq]:
    return {
        text_ref: TextReq(
            views=frozenset({TextView.TEXT, TextView.SPEAKERS}),
        )
        for text_ref in assignments
    }


__all__ = [
    "QwenProviderFactory",
    "SelectedTextDataset",
    "SpeakerDatasetFactory",
    "SpeakerGridDatasetFactory",
    "materialize_qwen_tts_speaker_grid",
    "materialize_qwen_tts_speech",
]
