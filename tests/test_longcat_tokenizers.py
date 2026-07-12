from __future__ import annotations

from pathlib import Path

import pytest

from zhuyin.tokenizers import codec_bpe
from zhuyin.tokenizers._codec_bpe_artifact import artifact_name, artifact_path


def test_longcat_bpe_path_uses_static_bpe_root(monkeypatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.delenv("BPE_CACHE_DIR", raising=False)

    path = codec_bpe.codec_bpe_path()

    assert path == Path("/data/static/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192")


def test_longcat_bpe_path_respects_env_cache(monkeypatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.setenv("BPE_CACHE_DIR", "/cache")

    path = codec_bpe.codec_bpe_path()

    assert path == Path("/cache/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192")


def test_artifact_name_matches_speech_to_speech_runtime() -> None:
    assert (
        artifact_name(
            vocab_size=100_000,
            codebook_sizes=(8192,),
        )
        == "vocab_100k_minfreq_0_maxlen_none_codes_8192"
    )


def test_artifact_name_includes_non_default_bpe_trainer_params() -> None:
    assert (
        artifact_name(
            vocab_size=8192,
            min_frequency=2,
            max_token_length=32,
            codebook_sizes=(8192,),
        )
        == "vocab_8192_minfreq_2_maxlen_32_codes_8192"
    )


def test_longcat_bpe_path_respects_explicit_root_and_artifact() -> None:
    artifact = artifact_path(
        vocab_size=8192,
        min_frequency=2,
        max_token_length=32,
        sample_limit=1000,
    )
    path = codec_bpe.codec_bpe_path(root="/cache", artifact=artifact)

    assert path == Path("/cache/longcat/vocab_8192_minfreq_2_maxlen_32_codes_8192_samples_1000")


def test_longcat_bpe_path_rejects_absolute_artifact() -> None:
    with pytest.raises(ValueError, match="relative"):
        codec_bpe.codec_bpe_path(root="/cache", artifact="/artifact")
