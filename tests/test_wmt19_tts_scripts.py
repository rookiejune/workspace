from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample, TextItem, TextView

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import filter_wmt19_tts_speech  # noqa: E402
import filter_wmt19_tts_speech_translation  # noqa: E402
import filter_wmt19_tts_translation  # noqa: E402
import prepare_wmt19_tts  # noqa: E402
import prepare_wmt19_tts_longcat  # noqa: E402


def test_prepare_parser_does_not_accept_filter_options() -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts.parse_args(["--quality-device", "cpu"])


def test_prepare_parser_does_not_accept_longcat_options() -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts.parse_args(["--longcat-decoder", "16k_4codebooks"])


def test_prepare_parser_accepts_chunk_options() -> None:
    args = prepare_wmt19_tts.parse_args(
        ["--offset", "10000", "--limit", "2000", "--cleanup-work"]
    )

    assert args.offset == 10000
    assert args.limit == 2000
    assert args.cleanup_work is True


def test_limited_wmt19_samples_uses_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWMT19:
        @staticmethod
        def create(**_kwargs: Any) -> range:
            return range(10)

    class FakePreset:
        WMT19 = FakeWMT19

    monkeypatch.setattr(prepare_wmt19_tts, "Preset", FakePreset)

    samples = list(
        prepare_wmt19_tts.limited_wmt19_samples(
            split="train",
            source_lang="zh",
            target_lang="en",
            offset=3,
            limit=4,
        )
    )

    assert samples == [3, 4, 5, 6]


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


@pytest.mark.parametrize("option", ["--prefetch-factor", "--resume"])
def test_longcat_prepare_parser_rejects_compat_options(option: str) -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts_longcat.parse_args([option, "4"])


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

    monkeypatch.setattr(filter_wmt19_tts_speech, "wmt19_tts", fake_wmt19_tts)
    root = tmp_path / "wmt19_tts"
    split = "train"
    factory = filter_wmt19_tts_speech.StoreFactory(root, split)

    assert factory() == []
    assert calls == [
        {
            "dataset_dir": root,
            "profile": filter_wmt19_tts_speech.WMT19TTSProfile.STORE,
            "split": split,
        }
    ]


def test_filter_default_factory_uses_default_wmt19_tts_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(filter_wmt19_tts_speech, "wmt19_tts", fake_wmt19_tts)
    factory = filter_wmt19_tts_speech.StoreFactory(None, "train")

    assert factory() == []
    assert calls == [{"split": "train"}]


def test_translation_filter_uses_wmt19_tts_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(filter_wmt19_tts_translation, "wmt19_tts", fake_wmt19_tts)
    root = tmp_path / "wmt19_tts"
    split = "train"
    factory = filter_wmt19_tts_translation.StoreFactory(root, split)

    assert factory() == []
    assert calls == [
        {
            "dataset_dir": root,
            "profile": filter_wmt19_tts_translation.WMT19TTSProfile.STORE,
            "split": split,
        }
    ]


def test_translation_filter_default_factory_uses_default_wmt19_tts_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(filter_wmt19_tts_translation, "wmt19_tts", fake_wmt19_tts)
    factory = filter_wmt19_tts_translation.StoreFactory(None, "train")

    assert factory() == []
    assert calls == [{"split": "train"}]


def test_translation_filter_parser_defaults_to_clean_usable_cpu() -> None:
    args = filter_wmt19_tts_translation.parse_args([])

    assert args.filter_device == "cpu"
    assert args.selected_labels == ["clean", "usable"]
    assert args.filter_rule_name == "wmt19_zh_en_translation_quality_rules_v1"


def test_translation_filter_parser_rejects_prefetch_factor_alias() -> None:
    with pytest.raises(SystemExit):
        filter_wmt19_tts_translation.parse_args(["--prefetch-factor", "4"])


def test_translation_filter_factory_builds_translation_predicate() -> None:
    args = filter_wmt19_tts_translation.parse_args([])

    predicate = filter_wmt19_tts_translation.TranslationQualityFactory.from_args(args)()

    assert isinstance(predicate, filter_wmt19_tts_translation.TranslationQuality)


def test_speech_translation_filter_parser_defaults_to_translation_first() -> None:
    args = filter_wmt19_tts_speech_translation.parse_args([])

    assert args.order == (
        filter_wmt19_tts_speech_translation.Stage.TRANSLATION,
        filter_wmt19_tts_speech_translation.Stage.SPEECH,
    )
    assert args.translation_labels == ["clean", "usable"]
    assert args.translation_rule_name == "wmt19_zh_en_translation_quality_rules_v1"


def test_speech_translation_filter_parser_rejects_prefetch_factor_alias() -> None:
    with pytest.raises(SystemExit):
        filter_wmt19_tts_speech_translation.parse_args(
            ["--translation-prefetch-factor", "4"]
        )


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


def test_longcat_prepare_writes_view_materializer_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    wmt19_tts_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("STATIC_HOME", str(tmp_path / "static"))
    monkeypatch.setenv("DYNAMIC_HOME", str(tmp_path / "dynamic"))

    class FakeDataset:
        def merge(self, _other: object) -> object:
            raise AssertionError("LongCat prepare should not merge the base store.")

    class FakeViewMaterializer:
        def __init__(
            self,
            output_dir: Path,
            *,
            split: str,
            max_shard_samples: int,
            batch_size: int,
            num_workers: int,
            prefetch_factor: int | None,
            write_workers: int,
            write_prefetch: int | None,
        ) -> None:
            calls["init"] = {
                "output_dir": output_dir,
                "split": split,
                "max_shard_samples": max_shard_samples,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "write_workers": write_workers,
                "write_prefetch": write_prefetch,
            }

        def write(
            self,
            *,
            dataset_factory: Any,
            provider_factory: Any,
            devices: str,
        ) -> Path:
            calls["write"] = {
                "dataset": dataset_factory(),
                "provider_factory": provider_factory,
                "devices": devices,
            }
            return tmp_path / "longcat"

    def fake_wmt19_tts(**kwargs: Any) -> FakeDataset:
        wmt19_tts_calls.append(kwargs)
        return FakeDataset()

    monkeypatch.setattr(
        prepare_wmt19_tts_longcat,
        "ViewMaterializer",
        FakeViewMaterializer,
    )
    monkeypatch.setattr(prepare_wmt19_tts_longcat, "wmt19_tts", fake_wmt19_tts)
    monkeypatch.setattr(prepare_wmt19_tts_longcat, "is_ready_store", lambda _path: False)
    monkeypatch.setattr(
        prepare_wmt19_tts_longcat,
        "store_sample_count",
        lambda _path: 1,
    )

    args = prepare_wmt19_tts_longcat.parse_args(
        [
            "--root",
            str(tmp_path),
            "--split",
            "dev",
            "--devices",
            "cpu",
            "--max-shard-samples",
            "7",
            "--batch-size",
            "3",
            "--num-workers",
            "2",
            "--read-prefetch",
            "4",
            "--write-workers",
            "2",
            "--write-prefetch",
            "5",
        ]
    )
    prepare_wmt19_tts_longcat.configure_env(args)

    stage = prepare_wmt19_tts_longcat.write_longcat_store(args)

    assert calls["init"] == {
        "output_dir": tmp_path / "longcat",
        "split": "dev",
        "max_shard_samples": 7,
        "batch_size": 3,
        "num_workers": 2,
        "prefetch_factor": 4,
        "write_workers": 2,
        "write_prefetch": 5,
    }
    assert calls["write"]["devices"] == "cpu"
    assert isinstance(
        calls["write"]["provider_factory"],
        prepare_wmt19_tts_longcat.LongCatFactory,
    )
    assert wmt19_tts_calls == [
        {
            "dataset_dir": tmp_path,
            "profile": prepare_wmt19_tts_longcat.WMT19TTSProfile.STORE,
            "split": "dev",
        }
    ]
    assert stage.path == str(tmp_path / "longcat")
    assert stage.sample_count == 1


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


def test_prepare_is_ready_store_exposes_broken_ready_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / ".ready").write_text("ready\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        prepare_wmt19_tts.is_ready_store(store)
