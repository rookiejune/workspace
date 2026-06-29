from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from anydataset import AudioItem, AudioView, Modality, Role, Sample, TextItem, TextView

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import filter_wmt19_tts  # noqa: E402
import prepare_wmt19_tts  # noqa: E402
import prepare_wmt19_tts_longcat  # noqa: E402


def test_prepare_parser_does_not_accept_filter_options() -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts.parse_args(["--quality-device", "cpu"])


def test_prepare_parser_does_not_accept_longcat_options() -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts.parse_args(["--longcat-decoder", "16k_4codebooks"])


def test_target_role_text_sample_trims_source_reference() -> None:
    waveform = torch.arange(20).reshape(1, 20)
    sample: Sample = {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, 10)}
        ),
        (Role.SOURCE, Modality.TEXT): TextItem(views={TextView.TEXT: "source"}),
        (Role.TARGET, Modality.TEXT): TextItem(views={TextView.TEXT: "target"}),
    }

    output = prepare_wmt19_tts.role_text_sample(
        sample,
        Role.TARGET,
        reference_seconds=0.5,
    )

    reference = output[Role.SOURCE, Modality.AUDIO]
    clipped, sample_rate = reference.views[AudioView.WAVEFORM]
    assert sample_rate == 10
    assert torch.equal(clipped, waveform[:, :5])
    assert output[Role.TARGET, Modality.TEXT] == sample[Role.TARGET, Modality.TEXT]


def test_target_audio_store_name_tracks_reference_seconds() -> None:
    assert prepare_wmt19_tts.target_audio_store_name(8.0) == "target-audio-ref8s"
    assert prepare_wmt19_tts.target_audio_store_name(None) == "target-audio"


def test_longcat_prepare_parser_uses_batch_size() -> None:
    args = prepare_wmt19_tts_longcat.parse_args(["--batch-size", "8"])

    assert args.batch_size == 8


@pytest.mark.parametrize(
    "option",
    ["--longcat-batch-size", "--longcat-decoder", "--hf-endpoint"],
)
def test_longcat_prepare_parser_rejects_provider_options(option: str) -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts_longcat.parse_args([option, "value"])


def test_filter_uses_wmt19_tts_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(filter_wmt19_tts, "wmt19_tts", fake_wmt19_tts)
    root = tmp_path / "wmt19_tts"
    split = "train"
    factory = filter_wmt19_tts.StoreFactory(root, split)

    assert factory() == []
    assert calls == [{"dataset_dir": root, "split": split}]


def test_filter_default_factory_uses_default_wmt19_tts_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(filter_wmt19_tts, "wmt19_tts", fake_wmt19_tts)
    factory = filter_wmt19_tts.StoreFactory(None, "train")

    assert factory() == []
    assert calls == [{"split": "train"}]


def test_longcat_prepare_uses_wmt19_tts_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(prepare_wmt19_tts_longcat, "wmt19_tts", fake_wmt19_tts)
    split = "train"
    factory = prepare_wmt19_tts_longcat.TTSFactory(split)

    assert factory() == []
    assert calls == [{"split": split}]


def test_longcat_factory_uses_default_provider_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeProvider:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(prepare_wmt19_tts_longcat, "LongCatProvider", FakeProvider)

    provider = prepare_wmt19_tts_longcat.LongCatFactory()("cpu")

    assert isinstance(provider, FakeProvider)
    assert calls == [{"device": None}]
