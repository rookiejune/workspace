from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from anydataset.types import AudioMeta, AudioView, Modality, Role, Source, TextMeta, TextView

from zhuyin import env
from zhuyin.datasets import wmt19_tts as module


class BuiltDataset:
    def __init__(
        self,
        spec: Any,
        parse_fn: Any = None,
        transforms: Any = None,
    ) -> None:
        self.spec = spec
        self.parse_fn = parse_fn
        self.transforms = transforms
        self.merged: BuiltDataset | None = None

    def merge(self, dataset: BuiltDataset) -> BuiltDataset:
        self.merged = dataset
        return self


def test_wmt19_tts_uses_explicit_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", str(tmp_path / "static"))
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts(root="/data/wmt19", split="dev")

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


def test_wmt19_tts_dataset_root_uses_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")

    assert module.dataset_root() == Path("/data/static/datasets/wmt19_tts")


def test_wmt19_tts_explicit_root_selects_store_on_hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts(root="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/base"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_codec_longcat_uses_longcat_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(
        codec=module.Codec.LONGCAT,
        root="/data/wmt19",
        split="dev",
    )

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/longcat"
    assert dataset.spec.split == "dev"
    assert set(dataset.transforms) == {
        (Role.SOURCE, Modality.AUDIO),
        (Role.TARGET, Modality.AUDIO),
    }


def test_longcat_store_transform_accepts_anytrain_codes() -> None:
    item = module.AudioItem(
        views={AudioView.LONGCAT: torch.tensor([[1, 4], [2, 5]])},
        meta={AudioMeta.SPEAKER_ID: "test"},
    )

    transformed = module._longcat_item(item)

    assert transformed is item


def test_longcat_store_transform_rejects_legacy_codes() -> None:
    item = module.AudioItem(
        views={
            AudioView.LONGCAT: {
                "semantic_codes": torch.tensor([1, 2]),
                "acoustic_codes": torch.tensor([[3, 4]]),
            }
        }
    )

    with pytest.raises(TypeError, match="rematerialize"):
        module._longcat_item(item)


def test_wmt19_tts_codec_longcat_explicit_root_selects_store_on_hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(
        codec=module.Codec.LONGCAT,
        root="/data/wmt19",
        split="dev",
    )

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/longcat"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_codec_longcat_hz_location_defaults_to_hf_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.LONGCAT, split="train")

    assert dataset.spec.source == Source.HF_DISK
    assert dataset.spec.path == "/nfs/yin.zhu/datasets/wmt19_tts_longcat_codes_text_cleaned"
    assert dataset.spec.split == "train"
    assert dataset.parse_fn is module._parse_hz_longcat_row


def test_wmt19_tts_codec_stable_uses_stable_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.STABLE, split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == ("/data/static/datasets/wmt19_tts/stable-1x46656_400bps")
    assert dataset.spec.split == "dev"
    assert dataset.merged is None


def test_wmt19_tts_codec_dac_uses_store_on_hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.DAC, split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/dac"
    assert dataset.spec.split == "dev"
    assert dataset.merged is None


def test_wmt19_tts_codec_unicodec_uses_store_on_hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "hz")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.UNICODEC, split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/unicodec"
    assert dataset.spec.split == "dev"
    assert dataset.merged is None


def test_wmt19_tts_codec_stable_explicit_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(
        codec="stable",
        root=tmp_path / "wmt19",
        split="train",
    )

    assert dataset.spec.path == str(tmp_path / "wmt19" / "stable-1x46656_400bps")
    assert dataset.merged is None


def test_wmt19_tts_stable_uses_named_codec_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_stable(root="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/stable-1x46656_400bps"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_stable_selects_quantizer_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_stable(
        root="/data/wmt19",
        quantizer=module.StableQuantizer.FSQ_2X15625_700BPS,
    )

    assert dataset.spec.path == "/data/wmt19/stable-2x15625_700bps"


def test_wmt19_tts_dac_uses_named_codec_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_dac(root="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/dac"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_unicodec_uses_named_codec_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_unicodec(root="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/unicodec"
    assert dataset.spec.split == "dev"


def test_wmt19_tts_dataset_root_expands_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")

    assert module.dataset_root("~/wmt19") == Path("/home/tester/wmt19")


def test_wmt19_tts_rejects_empty_static_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", "")

    with pytest.raises(ValueError, match="STATIC_HOME"):
        module.wmt19_tts()


def test_wmt19_tts_defaults_to_fudan_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCATION", raising=False)
    monkeypatch.delenv("STATIC_HOME", raising=False)
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)
    monkeypatch.setattr(Path, "exists", lambda _path: False)

    with pytest.warns(RuntimeWarning, match=env.STATIC_HOME_ENV):
        dataset = module.wmt19_tts()

    assert dataset.spec.path == "/mnt/pami202/zhuyin/datasets/wmt19_tts/base"
    assert env.STATIC_HOME_ENV not in os.environ


def test_wmt19_tts_does_not_configure_derived_environment(
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

    assert "ANYDATASET_HOME" not in os.environ
    assert "ANYTRAIN_HOME" not in os.environ
    assert "BPE_CACHE_DIR" not in os.environ
    assert "HF_HOME" not in os.environ
    assert "HF_HUB_CACHE" not in os.environ
    assert "HF_DATASETS_CACHE" not in os.environ
    assert "TORCH_HOME" not in os.environ
    assert "ANYTRAIN_WHISPER_ROOT" not in os.environ


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
        source_audio.views[AudioView.LONGCAT],
        torch.tensor([[1, 3], [2, 4]]),
    )
    assert source_text.views[TextView.TEXT] == "你好"
    assert source_text.meta[TextMeta.LANG] == "zh"
    assert torch.equal(
        target_audio.views[AudioView.LONGCAT],
        torch.tensor([[5, 7], [6, 8]]),
    )
    assert target_text.views[TextView.TEXT] == "hello"
    assert target_text.meta[TextMeta.LANG] == "en"


def test_parse_hz_longcat_row_aligns_codes_to_shortest_length() -> None:
    sample = module._parse_hz_longcat_row(
        {
            "source_semantic_codes": [1, 2, 3],
            "source_acoustic_codes": [[4, 5]],
            "source_text": "你好",
            "source_language": "zh",
            "target_semantic_codes": [6, 7],
            "target_acoustic_codes": [[8, 9, 10]],
            "target_text": "hello",
            "target_language": "en",
        }
    )

    source_audio = sample[Role.SOURCE, Modality.AUDIO]
    target_audio = sample[Role.TARGET, Modality.AUDIO]

    assert torch.equal(
        source_audio.views[AudioView.LONGCAT],
        torch.tensor([[1, 4], [2, 5]]),
    )
    assert torch.equal(
        target_audio.views[AudioView.LONGCAT],
        torch.tensor([[6, 8], [7, 9]]),
    )
