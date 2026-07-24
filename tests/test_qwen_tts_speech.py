from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from anydataset.dataset import (
    GroupedSpeakerAudioDataset,
    SpeakerAssignment,
    SpeakerCartesianDataset,
    SpeakerIdDataset,
)
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)

from zhuyin.datasets import qwen_tts_speech as loader

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _qwen_tts_speech as workflow  # noqa: E402


def test_speaker_dataset_factory_assigns_multiple_text_references() -> None:
    source_ref = (Role.SOURCE, Modality.TEXT)
    target_ref = (Role.TARGET, Modality.TEXT)

    def dataset_factory():
        return [
            {
                source_ref: TextItem(views={TextView.TEXT: "你好"}),
                target_ref: TextItem(views={TextView.TEXT: "hello"}),
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: "unselected"}
                ),
            }
        ]

    factory = workflow.SpeakerDatasetFactory(
        dataset_factory,
        {
            source_ref: SpeakerAssignment(("Vivian",)),
            target_ref: SpeakerAssignment(("Ryan",)),
        },
    )

    dataset = factory()
    sample = dataset[0]

    assert isinstance(dataset.dataset, SpeakerIdDataset)
    assert set(sample) == {source_ref, target_ref}
    assert sample[source_ref].views[TextView.SPEAKERS] == "Vivian"
    assert sample[target_ref].views[TextView.SPEAKERS] == "Ryan"


def test_speaker_dataset_factory_rejects_empty_assignments() -> None:
    factory = workflow.SpeakerDatasetFactory(lambda: [], {})

    with pytest.raises(ValueError, match="assignments must be a non-empty mapping"):
        factory()


def test_speaker_grid_factory_expands_selected_text_reference() -> None:
    text_ref = (Role.DEFAULT, Modality.TEXT)

    def dataset_factory():
        return [
            {
                text_ref: TextItem(views={TextView.TEXT: "hello"}),
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: "unselected"}
                ),
            }
        ]

    dataset = workflow.SpeakerGridDatasetFactory(
        dataset_factory,
        ("Vivian", "Ryan"),
        text_ref,
    )()

    assert isinstance(dataset.dataset, SpeakerCartesianDataset)
    assert len(dataset) == 2
    assert set(dataset[0]) == {text_ref}
    assert dataset[0][text_ref].views[TextView.SPEAKERS] == "Vivian"
    assert dataset[1][text_ref].views[TextView.SPEAKERS] == "Ryan"
    assert dataset[0][text_ref].meta[TextMeta.SOURCE_INDEX] == 0
    assert dataset[1][text_ref].meta[TextMeta.SOURCE_INDEX] == 0


def test_materialize_qwen_tts_speech_builds_multi_ref_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_ref = (Role.SOURCE, Modality.TEXT)
    target_ref = (Role.TARGET, Modality.TEXT)
    calls: dict[str, Any] = {}

    class FakeMaterializer:
        def __init__(self, output_dir: Path, **kwargs: Any) -> None:
            calls["output_dir"] = output_dir
            calls["init"] = kwargs

        def write(self, **kwargs: Any) -> Path:
            calls["write"] = kwargs
            return tmp_path / "ready"

    monkeypatch.setattr(workflow, "ModalityMaterializer", FakeMaterializer)
    assignments = {
        source_ref: SpeakerAssignment(("Vivian",), mode="cycle"),
        target_ref: SpeakerAssignment(("Ryan",), mode="cycle"),
    }

    result = workflow.materialize_qwen_tts_speech(
        text_dataset_factory=lambda: [
            {
                source_ref: TextItem(views={TextView.TEXT: "你好"}),
                target_ref: TextItem(views={TextView.TEXT: "hello"}),
            }
        ],
        assignments=assignments,
        output_dir=tmp_path / "store",
        devices="cpu",
    )

    assert result == tmp_path / "ready"
    assert set(calls["init"]["keep_schema"]) == {source_ref, target_ref}
    dataset = calls["write"]["dataset_factory"]()
    assert dataset[0][source_ref].views[TextView.SPEAKERS] == "Vivian"
    assert dataset[0][target_ref].views[TextView.SPEAKERS] == "Ryan"
    assert isinstance(calls["write"]["provider_factory"], workflow.QwenProviderFactory)
    assert calls["write"]["devices"] == "cpu"


def test_materialize_qwen_tts_speaker_grid_keeps_source_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeMaterializer:
        def __init__(self, _output_dir: Path, **kwargs: Any) -> None:
            calls["init"] = kwargs

        def write(self, **kwargs: Any) -> Path:
            calls["write"] = kwargs
            return tmp_path / "ready"

    monkeypatch.setattr(workflow, "ModalityMaterializer", FakeMaterializer)
    text_ref = (Role.DEFAULT, Modality.TEXT)

    workflow.materialize_qwen_tts_speaker_grid(
        text_dataset_factory=lambda: [
            {text_ref: TextItem(views={TextView.TEXT: "hello"})}
        ],
        speaker_ids=("Vivian", "Ryan"),
        output_dir=tmp_path / "store",
    )

    requirement = calls["init"]["keep_schema"][text_ref]
    assert requirement.meta == frozenset({TextMeta.SOURCE_INDEX})
    dataset = calls["write"]["dataset_factory"]()
    assert dataset[0][text_ref].meta[TextMeta.SOURCE_INDEX] == 0
    assert dataset[1][text_ref].meta[TextMeta.SOURCE_INDEX] == 0


def test_qwen_tts_speaker_grid_loader_returns_generic_grouped_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flat = [{}, {}]
    calls: list[Any] = []

    def fake_dataset(spec: Any):
        calls.append(spec)
        return flat

    monkeypatch.setattr(loader, "AnyDataset", fake_dataset)

    dataset = loader.qwen_tts_speaker_grid(
        root=tmp_path,
        speaker_ids=("Vivian", "Ryan"),
        split="dev",
    )

    assert isinstance(dataset, GroupedSpeakerAudioDataset)
    assert dataset.dataset is flat
    assert calls[0].path == str(tmp_path)
    assert calls[0].split == "dev"
