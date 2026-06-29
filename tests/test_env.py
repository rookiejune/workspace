from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from zhuyin import env


def _mock_existing_paths(
    monkeypatch: pytest.MonkeyPatch,
    paths: set[str],
) -> None:
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in paths)


def test_location_defaults_to_fudan_when_only_mnt_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt"})

    assert env.location() is env.Location.FUDAN
    assert os.environ[env.LOCATION_ENV] == env.Location.FUDAN.value


def test_location_defaults_to_hz_when_nfs_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt", "/nfs/yin.zhu"})

    assert env.location() is env.Location.HZ
    assert os.environ[env.LOCATION_ENV] == env.Location.HZ.value


def test_location_defaults_to_us_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, {"/mnt", "/nfs/yin.zhu", "/share5_video"})

    assert env.location() is env.Location.US
    assert os.environ[env.LOCATION_ENV] == env.Location.US.value


def test_location_uses_fudan_fallback_without_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    _mock_existing_paths(monkeypatch, set())

    assert env.location() is env.Location.FUDAN
    assert os.environ[env.LOCATION_ENV] == env.Location.FUDAN.value


def test_location_accepts_hz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)

    assert env.location() is env.Location.HZ


def test_location_accepts_us(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.US.value)

    assert env.location() is env.Location.US


def test_location_rejects_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, "")

    with pytest.raises(ValueError, match=env.LOCATION_ENV):
        env.location()


def test_location_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, "unknown")

    with pytest.raises(ValueError, match=env.LOCATION_ENV):
        env.location()


def test_static_home_defaults_to_fudan_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.STATIC_HOME_ENV):
        assert env.static_home() == env.DEFAULT_STATIC_HOME

    assert os.environ[env.STATIC_HOME_ENV] == str(env.DEFAULT_STATIC_HOME)


def test_dynamic_home_defaults_to_fudan_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.LOCATION_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.DYNAMIC_HOME_ENV):
        assert env.dynamic_home() == env.DEFAULT_DYNAMIC_HOME

    assert os.environ[env.DYNAMIC_HOME_ENV] == str(env.DEFAULT_DYNAMIC_HOME)


def test_homes_default_to_location_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.Location.HZ.value):
        assert env.static_home() == env.HZ_HOME
    with pytest.warns(RuntimeWarning, match=env.Location.HZ.value):
        assert env.dynamic_home() == Path("/yin.zhu")

    assert os.environ[env.STATIC_HOME_ENV] == "/nfs/yin.zhu"
    assert os.environ[env.DYNAMIC_HOME_ENV] == "/yin.zhu"


def test_homes_default_to_us_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.US.value)
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.DYNAMIC_HOME_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.Location.US.value):
        assert env.static_home() == env.US_STATIC_HOME
    with pytest.warns(RuntimeWarning, match=env.Location.US.value):
        assert env.dynamic_home() == env.US_DYNAMIC_HOME

    assert os.environ[env.STATIC_HOME_ENV] == "/zoomai/colddata/video/yin.zhu"
    assert os.environ[env.DYNAMIC_HOME_ENV] == "/share5_video/users/yin.zhu"


def test_explicit_homes_override_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.LOCATION_ENV, env.Location.HZ.value)
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/custom/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/custom/dynamic")

    assert env.static_home() == Path("/custom/static")
    assert env.dynamic_home() == Path("/custom/dynamic")


def test_homes_reject_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "")

    with pytest.raises(ValueError, match=env.STATIC_HOME_ENV):
        env.static_home()
    with pytest.raises(ValueError, match=env.DYNAMIC_HOME_ENV):
        env.dynamic_home()


def test_path_helpers_use_static_and_dynamic_homes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.DYNAMIC_HOME_ENV, "/data/dynamic")
    monkeypatch.delenv(env.ANYTRAIN_HOME_ENV, raising=False)
    monkeypatch.delenv(env.BPE_CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(env.HF_HOME_ENV, raising=False)
    monkeypatch.delenv(env.HF_HUB_CACHE_ENV, raising=False)
    monkeypatch.delenv(env.HF_DATASETS_CACHE_ENV, raising=False)
    monkeypatch.delenv(env.TORCH_HOME_ENV, raising=False)
    monkeypatch.delenv(env.WHISPER_ROOT_ENV, raising=False)

    assert env.dataset_dir("common_voice") == Path("/data/static/datasets/common_voice")
    assert env.train_dir("anycodec") == Path("/data/dynamic/train/anycodec")
    assert env.models_dir() == Path("/data/static/models")
    assert env.anytrain_home() == Path("/data/static")
    assert env.bpe_cache_dir() == Path("/data/static/bpe")
    assert env.hf_home() == Path("/data/static/huggingface")
    assert env.hf_hub_cache() == Path("/data/static/huggingface/hub")
    assert env.hf_datasets_cache() == Path("/data/static/huggingface/datasets")
    assert env.torch_home() == Path("/data/static/torch")
    assert env.whisper_root() == Path("/data/static/whisper")
    assert env.debug_dir() == Path("/data/dynamic/debug")
    assert env.debug_dir("workspace") == Path("/data/dynamic/debug/workspace")


@pytest.mark.parametrize(
    ("name", "helper"),
    [
        (env.ANYTRAIN_HOME_ENV, env.anytrain_home),
        (env.BPE_CACHE_DIR_ENV, env.bpe_cache_dir),
        (env.HF_HOME_ENV, env.hf_home),
        (env.HF_HUB_CACHE_ENV, env.hf_hub_cache),
        (env.HF_DATASETS_CACHE_ENV, env.hf_datasets_cache),
        (env.TORCH_HOME_ENV, env.torch_home),
        (env.WHISPER_ROOT_ENV, env.whisper_root),
    ],
)
def test_static_cache_helpers_respect_explicit_values(
    name: str,
    helper: Callable[[], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(name, "/custom/cache")

    assert helper() == Path("/custom/cache")


def test_configure_environment_respects_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(env.ANYDATASET_HOME_ENV, "/custom/anydataset")
    monkeypatch.setenv(env.ANYTRAIN_HOME_ENV, "/custom/anytrain")
    monkeypatch.setenv(env.BPE_CACHE_DIR_ENV, "/custom/bpe")
    monkeypatch.setenv(env.HF_HOME_ENV, "/custom/hf")
    monkeypatch.setenv(env.HF_HUB_CACHE_ENV, "/custom/hub")
    monkeypatch.setenv(env.HF_DATASETS_CACHE_ENV, "/custom/datasets")
    monkeypatch.setenv(env.TORCH_HOME_ENV, "/custom/torch")
    monkeypatch.setenv(env.WHISPER_ROOT_ENV, "/custom/whisper")

    env.configure_environment()

    assert os.environ[env.ANYDATASET_HOME_ENV] == "/custom/anydataset"
    assert os.environ[env.ANYTRAIN_HOME_ENV] == "/custom/anytrain"
    assert os.environ[env.BPE_CACHE_DIR_ENV] == "/custom/bpe"
    assert os.environ[env.HF_HOME_ENV] == "/custom/hf"
    assert os.environ[env.HF_HUB_CACHE_ENV] == "/custom/hub"
    assert os.environ[env.HF_DATASETS_CACHE_ENV] == "/custom/datasets"
    assert os.environ[env.TORCH_HOME_ENV] == "/custom/torch"
    assert os.environ[env.WHISPER_ROOT_ENV] == "/custom/whisper"


def test_configure_environment_uses_fudan_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(env.STATIC_HOME_ENV, raising=False)
    monkeypatch.delenv(env.ANYDATASET_HOME_ENV, raising=False)
    monkeypatch.delenv(env.ANYTRAIN_HOME_ENV, raising=False)
    monkeypatch.delenv(env.BPE_CACHE_DIR_ENV, raising=False)
    monkeypatch.delenv(env.HF_HOME_ENV, raising=False)
    monkeypatch.delenv(env.HF_HUB_CACHE_ENV, raising=False)
    monkeypatch.delenv(env.HF_DATASETS_CACHE_ENV, raising=False)
    monkeypatch.delenv(env.TORCH_HOME_ENV, raising=False)
    monkeypatch.delenv(env.WHISPER_ROOT_ENV, raising=False)

    with pytest.warns(RuntimeWarning, match=env.STATIC_HOME_ENV):
        env.configure_environment()

    assert os.environ[env.ANYDATASET_HOME_ENV] == "/mnt/pami202/zhuyin/anydataset"
    assert os.environ[env.ANYTRAIN_HOME_ENV] == "/mnt/pami202/zhuyin"
    assert os.environ[env.BPE_CACHE_DIR_ENV] == "/mnt/pami202/zhuyin/bpe"
    assert os.environ[env.HF_HOME_ENV] == "/mnt/pami202/zhuyin/huggingface"
    assert os.environ[env.HF_HUB_CACHE_ENV] == "/mnt/pami202/zhuyin/huggingface/hub"
    assert os.environ[env.HF_DATASETS_CACHE_ENV] == "/mnt/pami202/zhuyin/huggingface/datasets"
    assert os.environ[env.TORCH_HOME_ENV] == "/mnt/pami202/zhuyin/torch"
    assert os.environ[env.WHISPER_ROOT_ENV] == "/mnt/pami202/zhuyin/whisper"


@pytest.mark.parametrize(
    "name",
    [
        env.ANYDATASET_HOME_ENV,
        env.ANYTRAIN_HOME_ENV,
        env.BPE_CACHE_DIR_ENV,
        env.HF_HOME_ENV,
        env.HF_HUB_CACHE_ENV,
        env.HF_DATASETS_CACHE_ENV,
        env.TORCH_HOME_ENV,
        env.WHISPER_ROOT_ENV,
    ],
)
def test_configure_environment_rejects_empty_derived_values(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env.STATIC_HOME_ENV, "/data/static")
    monkeypatch.setenv(name, "")

    with pytest.raises(ValueError, match=name):
        env.configure_environment()
