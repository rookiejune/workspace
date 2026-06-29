from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from anydataset import Source

from zhuyin.datasets import wmt19_tts as module


class BuiltDataset(NamedTuple):
    spec: Any


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


def test_wmt19_tts_longcat_uses_longcat_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    dataset = module.wmt19_tts_longcat(dataset_dir="/data/wmt19", split="dev")

    assert dataset.spec.source == Source.STORE
    assert dataset.spec.path == "/data/wmt19/longcat-delta"
    assert dataset.spec.split == "dev"


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
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(module, "AnyDataset", BuiltDataset)

    module.wmt19_tts()

    assert os.environ["ANYDATASET_HOME"] == str(static_home / "anydataset")
    assert os.environ["HF_HOME"] == str(static_home / "huggingface")
