from __future__ import annotations

import pytest
import torch
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)

from zhuyin.datasets.qwen_tts_speech import (
    GroupedQwenTTSSpeechDataset,
    SpeakerDatasetFactory,
    SpeakerGridDatasetFactory,
)


def test_speaker_dataset_factory_assigns_speaker_ids() -> None:
    def dataset_factory():
        return [
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"})}
        ]

    factory = SpeakerDatasetFactory(
        dataset_factory,
        ("Vivian",),
        "aligned",
        (Role.DEFAULT, Modality.TEXT),
    )

    item = factory()[0][Role.DEFAULT, Modality.TEXT]

    assert isinstance(item, TextItem)
    assert item.views[TextView.SPEAKERS] == "Vivian"


def test_speaker_dataset_factory_rejects_empty_speaker_ids() -> None:
    def dataset_factory():
        return [
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"})}
        ]

    with pytest.raises(ValueError, match="speaker_ids must not be empty"):
        SpeakerDatasetFactory(
            dataset_factory,
            (),
            "cycle",
            (Role.DEFAULT, Modality.TEXT),
        )


def test_speaker_grid_factory_expands_text_by_speaker() -> None:
    def dataset_factory():
        return [
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"})},
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "world"})},
        ]

    factory = SpeakerGridDatasetFactory(
        dataset_factory,
        ("Vivian", "Ryan"),
        (Role.DEFAULT, Modality.TEXT),
    )
    dataset = factory()

    assert len(dataset) == 4
    first = dataset[0][Role.DEFAULT, Modality.TEXT]
    second = dataset[1][Role.DEFAULT, Modality.TEXT]
    third = dataset[2][Role.DEFAULT, Modality.TEXT]
    assert isinstance(first, TextItem)
    assert isinstance(second, TextItem)
    assert isinstance(third, TextItem)
    assert first.views[TextView.TEXT] == "hello"
    assert first.views[TextView.SPEAKERS] == "Vivian"
    assert first.meta[TextMeta.SOURCE_INDEX] == 0
    assert second.views[TextView.SPEAKERS] == "Ryan"
    assert second.meta[TextMeta.SOURCE_INDEX] == 0
    assert third.views[TextView.TEXT] == "world"
    assert third.meta[TextMeta.SOURCE_INDEX] == 1


def test_grouped_qwen_tts_speech_dataset_groups_waveforms_by_text() -> None:
    flat = [
        _flat_sample("hello", "Vivian", 0, (torch.tensor([[1.0, 2.0]]), 24000)),
        _flat_sample("hello", "Ryan", 0, (torch.tensor([[3.0]]), 24000)),
        _flat_sample("world", "Vivian", 1, (torch.tensor([[4.0]]), 24000)),
        _flat_sample("world", "Ryan", 1, (torch.tensor([[5.0, 6.0]]), 24000)),
    ]

    grouped = GroupedQwenTTSSpeechDataset(flat, ("Vivian", "Ryan"))

    assert len(grouped) == 2
    first = grouped[0]
    text = first[Role.DEFAULT, Modality.TEXT]
    audio = first[Role.DEFAULT, Modality.AUDIO]
    assert isinstance(text, TextItem)
    assert isinstance(audio, AudioItem)
    assert text.views[TextView.TEXT] == "hello"
    assert text.meta[TextMeta.SOURCE_INDEX] == 0
    assert audio.meta[AudioMeta.SPEAKER_ID] == ("Vivian", "Ryan")
    waveform, sample_rate = audio.views[AudioView.WAVEFORM]
    assert sample_rate == 24000
    assert audio.views[AudioView.SPEAKERS] == ("Vivian", "Ryan")
    assert torch.equal(audio.views[AudioView.SPEAKER_LENGTHS], torch.tensor([2, 1]))
    assert torch.equal(
        waveform,
        torch.tensor([[[1.0, 2.0]], [[3.0, 0.0]]]),
    )


def test_grouped_qwen_tts_speech_dataset_rejects_invalid_sample_rate() -> None:
    flat = [
        _flat_sample("hello", "Vivian", 0, (torch.tensor([[1.0]]), 0)),
    ]
    grouped = GroupedQwenTTSSpeechDataset(flat, ("Vivian",))

    with pytest.raises(ValueError, match="sample rate must be positive"):
        grouped[0]


def test_grouped_qwen_tts_speech_dataset_rejects_non_2d_waveform() -> None:
    flat = [
        _flat_sample("hello", "Vivian", 0, (torch.tensor([1.0, 2.0]), 24000)),
    ]
    grouped = GroupedQwenTTSSpeechDataset(flat, ("Vivian",))

    with pytest.raises(ValueError, match=r"shape \[channel, time\]"):
        grouped[0]


def test_grouped_qwen_tts_speech_dataset_rejects_mismatched_channels() -> None:
    flat = [
        _flat_sample("hello", "Vivian", 0, (torch.ones(1, 2), 24000)),
        _flat_sample("hello", "Ryan", 0, (torch.ones(2, 2), 24000)),
    ]
    grouped = GroupedQwenTTSSpeechDataset(flat, ("Vivian", "Ryan"))

    with pytest.raises(ValueError, match="expected prefix shape"):
        grouped[0]


def _flat_sample(text: str, speaker_id: str, source_index: int, waveform):
    return {
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: text, TextView.SPEAKERS: speaker_id},
            meta={TextMeta.SOURCE_INDEX: source_index},
        ),
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: waveform},
            meta={AudioMeta.SPEAKER_ID: speaker_id},
        ),
    }
