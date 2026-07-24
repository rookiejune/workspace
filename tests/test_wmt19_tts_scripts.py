from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample, TextItem, TextView

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _filter_wmt19_tts_speech as speech_filter  # noqa: E402
import _filter_wmt19_tts_speech_translation as speech_translation_filter  # noqa: E402
import _filter_wmt19_tts_translation as translation_filter  # noqa: E402
import _prepare_wmt19_tts_codec as codec_prepare  # noqa: E402
import _prepare_wmt19_tts_longcat as longcat_prepare  # noqa: E402
import _wmt19_tts_store as wmt19_store  # noqa: E402
import filter_wmt19_tts  # noqa: E402
import prepare_wmt19_tts  # noqa: E402
import prepare_wmt19_tts_codec_view  # noqa: E402

tts_prepare = prepare_wmt19_tts


def test_prepare_parser_does_not_accept_filter_options() -> None:
    with pytest.raises(SystemExit):
        prepare_wmt19_tts.parse_args(["--quality-device", "cpu"])


def test_codec_view_entry_forwards_longcat_options() -> None:
    args = prepare_wmt19_tts_codec_view.parse_args(["longcat", "--batch-size", "8"])

    assert args.codec == "longcat"
    assert args.args == ["--batch-size", "8"]


def test_filter_entry_forwards_translation_options() -> None:
    args = filter_wmt19_tts.parse_args(["translation", "--selected-labels", "accept"])

    assert args.task == "translation"
    assert args.args == ["--selected-labels", "accept"]


def test_prepare_tts_parser_does_not_accept_longcat_options() -> None:
    with pytest.raises(SystemExit):
        tts_prepare.parse_args(["--longcat-decoder", "16k_4codebooks"])


def test_prepare_tts_parser_accepts_chunk_options() -> None:
    args = tts_prepare.parse_args(
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

    monkeypatch.setattr(tts_prepare, "Preset", FakePreset)

    samples = list(
        tts_prepare.limited_wmt19_samples(
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

    output = tts_prepare.role_text_sample(
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
    assert tts_prepare.target_audio_store_name(8.0) == "target-audio-ref8s"
    assert tts_prepare.target_audio_store_name(None) == "target-audio"


def test_longcat_prepare_parser_uses_batch_size() -> None:
    args = longcat_prepare.parse_args(["--batch-size", "8"])

    assert args.batch_size == 8


def test_dac_prepare_parser_uses_official_defaults() -> None:
    args = codec_prepare.parse_args(["dac"])

    assert args.devices == "auto"
    assert args.model_type == "44khz"
    assert args.model_bitrate == "8kbps"
    assert args.tag == "latest"
    assert args.n_quantizers is None
    assert args.local_files_only is False


def test_dac_prepare_parser_accepts_codec_configuration(tmp_path: Path) -> None:
    args = codec_prepare.parse_args(
        [
            "dac",
            "--dac-cache-dir",
            str(tmp_path),
            "--model-type",
            "24khz",
            "--model-bitrate",
            "8kbps",
            "--n-quantizers",
            "4",
            "--local-files-only",
        ]
    )

    assert args.dac_cache_dir == tmp_path
    assert args.model_type == "24khz"
    assert args.model_bitrate == "8kbps"
    assert args.n_quantizers == 4
    assert args.local_files_only is True


def test_stable_prepare_parser_uses_constrained_default() -> None:
    args = codec_prepare.parse_args(["stable"])

    assert args.posthoc_bottleneck.value == "1x46656_400bps"
    assert codec_prepare.prepare_config(args)["posthoc_bottleneck"] is (
        args.posthoc_bottleneck
    )
    assert codec_prepare.run_config(args)["posthoc_bottleneck"] == (
        "1x46656_400bps"
    )


def test_stable_prepare_parser_accepts_supported_posthoc_preset() -> None:
    args = codec_prepare.parse_args(
        ["stable", "--posthoc-bottleneck", "4x729_1000bps"]
    )

    assert args.posthoc_bottleneck.value == "4x729_1000bps"


def test_stable_prepare_parser_rejects_native_codes() -> None:
    with pytest.raises(SystemExit):
        codec_prepare.parse_args(
            ["stable", "--posthoc-bottleneck", "native"]
        )


def test_codec_prepare_parser_requires_codec() -> None:
    with pytest.raises(SystemExit):
        codec_prepare.parse_args([])


def test_codec_prepare_parser_rejects_other_codec_options() -> None:
    with pytest.raises(SystemExit):
        codec_prepare.parse_args(["unicodec", "--model-type", "44khz"])


def test_unicodec_prepare_dispatches_codec_configuration(tmp_path: Path) -> None:
    args = codec_prepare.parse_args(
        ["unicodec", "--unicodec-cache-dir", str(tmp_path), "--bandwidth-id", "2"]
    )

    config = codec_prepare.prepare_config(args)

    assert args.prepare is codec_prepare.prepare_unicodec
    assert config["cache_dir"] == tmp_path
    assert config["domain"] == "0"
    assert config["bandwidth_id"] == 2
    assert "unicodec_cache_dir" not in config


@pytest.mark.parametrize("option", ["--prefetch-factor", "--resume"])
def test_longcat_prepare_parser_rejects_compat_options(option: str) -> None:
    with pytest.raises(SystemExit):
        longcat_prepare.parse_args([option, "4"])


@pytest.mark.parametrize(
    "option",
    ["--longcat-batch-size", "--longcat-decoder", "--hf-endpoint"],
)
def test_longcat_prepare_parser_rejects_provider_options(option: str) -> None:
    with pytest.raises(SystemExit):
        longcat_prepare.parse_args([option, "value"])


def test_filter_uses_wmt19_tts_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_wmt19_tts(**kwargs: Any) -> list[object]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(wmt19_store, "wmt19_tts", fake_wmt19_tts)
    root = tmp_path / "wmt19_tts"
    split = "train"
    factory = speech_filter.StoreFactory(root, split)

    assert factory() == []
    assert calls == [
        {
            "root": root,
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

    monkeypatch.setattr(wmt19_store, "wmt19_tts", fake_wmt19_tts)
    factory = speech_filter.StoreFactory(None, "train")

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

    monkeypatch.setattr(wmt19_store, "wmt19_tts", fake_wmt19_tts)
    root = tmp_path / "wmt19_tts"
    split = "train"
    factory = translation_filter.StoreFactory(root, split)

    assert factory() == []
    assert calls == [
        {
            "root": root,
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

    monkeypatch.setattr(wmt19_store, "wmt19_tts", fake_wmt19_tts)
    factory = translation_filter.StoreFactory(None, "train")

    assert factory() == []
    assert calls == [{"split": "train"}]


def test_translation_filter_parser_defaults_to_accept_cpu() -> None:
    args = translation_filter.parse_args([])

    assert args.filter_device == "cpu"
    assert args.selected_labels == ["accept"]
    assert args.filter_rule_name == "wmt19_zh_en_translation_quality_rules_v1"


def test_translation_filter_parser_rejects_prefetch_factor_alias() -> None:
    with pytest.raises(SystemExit):
        translation_filter.parse_args(["--prefetch-factor", "4"])


def test_translation_filter_factory_builds_translation_predicate() -> None:
    args = translation_filter.parse_args([])

    predicate = translation_filter.TranslationQualityFactory.from_args(args)()

    assert isinstance(predicate, translation_filter.TranslationQuality)


def test_speech_translation_filter_parser_defaults_to_translation_first() -> None:
    args = speech_translation_filter.parse_args([])

    assert args.order == (
        speech_translation_filter.Stage.TRANSLATION,
        speech_translation_filter.Stage.SPEECH,
    )
    assert args.translation_labels == ["accept"]
    assert args.translation_rule_name == "wmt19_zh_en_translation_quality_rules_v1"


def test_speech_translation_filter_parser_rejects_prefetch_factor_alias() -> None:
    with pytest.raises(SystemExit):
        speech_translation_filter.parse_args(["--translation-prefetch-factor", "4"])


def test_prepare_is_ready_store_exposes_broken_ready_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / ".ready").write_text("ready\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        tts_prepare.is_ready_store(store)
