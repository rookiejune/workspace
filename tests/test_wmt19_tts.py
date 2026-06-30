from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from anydataset import AudioView, Modality, Role, Source, TextMeta, TextView

from zhuyin.datasets._profiles import WMT19TTSLongCatProfile, WMT19TTSProfile
from zhuyin.datasets import wmt19_tts as module


class BuiltDataset:
    def __init__(self, spec: Any, parse_fn: Any = None) -> None:
        self.spec = spec
        self.parse_fn = parse_fn


def test_wmt19_tts_uses_explicit_dataset_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", str(tmp_path / "static"))
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts(dataset_dir="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/base"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_uses_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts()

    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/base"


def test_wmt19_tts_hz_location_defaults_to_hz_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts(split="train")

    assert dataset.spec.source == "sharded_csv"
    assert dataset.spec.path == "/nfs/yin.zhu/train/text_to_speech/moss_tts_hz_export"
    assert dataset.spec.split == "train"
    assert dataset.parse_fn is module._parse_hz_tts_row


def test_wmt19_tts_explicit_profile_overrides_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts(profile=WMT19TTSProfile.STORE)

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/base"


def test_wmt19_tts_explicit_dataset_dir_overrides_current_profile_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts(dataset_dir="/data/wmt19", split="dev")

    assert dataset.spec.source == "sharded_csv"
    assert dataset.spec.path == "/data/wmt19"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_longcat_uses_longcat_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_longcat(dataset_dir="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/longcat"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_longcat_explicit_dataset_dir_overrides_current_profile_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_longcat(dataset_dir="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.HF_DISK
    assert dataset.spec.path == "/data/wmt19"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_longcat_hz_location_defaults_to_hf_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_longcat(split="train")

    assert dataset.spec.source == Source.HF_DISK
    assert dataset.spec.path == "/nfs/yin.zhu/datasets/wmt19_tts_longcat_codes_text_cleaned"
    assert dataset.spec.split == "train"
    assert dataset.parse_fn is module._parse_hz_longcat_row


def test_wmt19_tts_longcat_explicit_profile_overrides_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_longcat(profile=WMT19TTSLongCatProfile.STORE)

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/longcat"


def test_wmt19_tts_rejects_empty_static_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", "")

    with pytest.raises(ValueError, match="STATIC_HOME"):
        module.wmt19_tts()


def test_wmt19_tts_defaults_to_fudan_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STATIC_HOME", raising=False)
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    with pytest.warns(RuntimeWarning, match="STATIC_HOME"):
        dataset = module.wmt19_tts()

    assert dataset.spec.path == "/mnt/pami202/zhuyin/datasets/wmt19_tts/base"


def test_wmt19_tts_configures_derived_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_home = tmp_path / "static"
    monkeypatch.setenv("STATIC_HOME", str(static_home))
    monkeypatch.delenv("ANYDATASET_HOME", raising=False)
    monkeypatch.delenv("ANYTRAIN_HOME", raising=False)
    monkeypatch.delenv("BPE_CACHE_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.delenv("ANYTRAIN_WHISPER_ROOT", raising=False)
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    module.wmt19_tts()

    assert os.environ["ANYDATASET_HOME"] == str(static_home / "anydataset")
    assert os.environ["ANYTRAIN_HOME"] == str(static_home)
    assert os.environ["BPE_CACHE_DIR"] == str(static_home / "bpe")
    assert os.environ["HF_HOME"] == str(static_home / "huggingface")
    assert os.environ["HF_HUB_CACHE"] == str(static_home / "huggingface" / "hub")
    assert os.environ["HF_DATASETS_CACHE"] == str(static_home / "huggingface" / "datasets")
    assert os.environ["TORCH_HOME"] == str(static_home / "torch")
    assert os.environ["ANYTRAIN_WHISPER_ROOT"] == str(static_home / "whisper")


def test_parse_hz_tts_row_loads_logical_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_load_audio",
        lambda path: (path, 16_000),
    )

    sample = module._parse_hz_tts_row(
        {
            "source/audio": "/data/rank_0/source.wav",
            "source/text": "你好",
            "source/lang": "zh",
            "target/audio": "/data/rank_0/target.wav",
            "target/text": "hello",
            "target/lang": "en",
        }
    )

    source_audio = sample[Role.SOURCE, Modality.AUDIO]
    source_text = sample[Role.SOURCE, Modality.TEXT]
    target_audio = sample[Role.TARGET, Modality.AUDIO]
    target_text = sample[Role.TARGET, Modality.TEXT]

    assert source_audio.views[AudioView.WAVEFORM] == (
        "/data/train/shard_0/source.wav",
        16_000,
    )
    assert source_text.views[TextView.TEXT] == "你好"
    assert source_text.meta[TextMeta.LANG] == "zh"
    assert target_audio.views[AudioView.WAVEFORM] == (
        "/data/train/shard_0/target.wav",
        16_000,
    )
    assert target_text.views[TextView.TEXT] == "hello"
    assert target_text.meta[TextMeta.LANG] == "en"


def test_parse_hz_longcat_row_loads_logical_sample() -> None:
    sample = module._parse_hz_longcat_row(
        {
            "source_semantic_codes": [1, 2],
            "source_acoustic_codes": [[3, 4]],
            "source_text": "你好",
            "source_language": "zh",
            "target_semantic_codes": [5, 6],
            "target_acoustic_codes": [[7, 8]],
            "target_text": "hello",
            "target_language": "en",
        }
    )

    source_audio = sample[Role.SOURCE, Modality.AUDIO]
    source_text = sample[Role.SOURCE, Modality.TEXT]
    target_audio = sample[Role.TARGET, Modality.AUDIO]
    target_text = sample[Role.TARGET, Modality.TEXT]

    assert torch.equal(
        source_audio.views[AudioView.LONGCAT]["semantic_codes"],
        torch.tensor([1, 2]),
    )
    assert torch.equal(
        source_audio.views[AudioView.LONGCAT]["acoustic_codes"],
        torch.tensor([[3, 4]]),
    )
    assert source_text.views[TextView.TEXT] == "你好"
    assert source_text.meta[TextMeta.LANG] == "zh"
    assert torch.equal(
        target_audio.views[AudioView.LONGCAT]["semantic_codes"],
        torch.tensor([5, 6]),
    )
    assert torch.equal(
        target_audio.views[AudioView.LONGCAT]["acoustic_codes"],
        torch.tensor([[7, 8]]),
    )
    assert target_text.views[TextView.TEXT] == "hello"
    assert target_text.meta[TextMeta.LANG] == "en"
