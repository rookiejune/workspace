from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from anydataset.types import AudioMeta, AudioView, Modality, Role, Source

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
    monkeypatch.setenv("LOCATION", "fudan")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts()

    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/base"


def test_wmt19_tts_dataset_root_uses_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")

    assert module.dataset_root() == Path("/data/static/datasets/wmt19_tts")


def test_wmt19_tts_codec_longcat_uses_longcat_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "fudan")
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


def test_wmt19_tts_codec_longcat_fudan_default_uses_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "fudan")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.LONGCAT, split="train")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/longcat"
    assert dataset.spec.split == "train"
    assert dataset.merged is not None
    assert dataset.merged.spec.path == "/data/static/datasets/wmt19_tts/base"


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


def test_wmt19_tts_codec_stable_uses_stable_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "fudan")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.STABLE, split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == ("/data/static/datasets/wmt19_tts/stable-1x46656_400bps")
    assert dataset.spec.split == "dev"
    assert dataset.merged is None


def test_wmt19_tts_codec_dac_uses_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "fudan")
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_codec(codec=module.Codec.DAC, split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/static/datasets/wmt19_tts/dac"
    assert dataset.spec.split == "dev"
    assert dataset.merged is None


def test_wmt19_tts_codec_unicodec_uses_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCATION", "fudan")
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
    monkeypatch.setenv("LOCATION", "fudan")
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

    assert dataset.spec.path == str(env.Location.FUDAN.static_home / "datasets/wmt19_tts/base")
    assert env.STATIC_HOME_ENV not in os.environ


def test_wmt19_tts_does_not_configure_derived_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_home = tmp_path / "static"
    monkeypatch.setenv("LOCATION", "fudan")
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
