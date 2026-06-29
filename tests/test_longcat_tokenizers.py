from __future__ import annotations

from pathlib import Path

from zhuyin.tokenizers import longcat


def test_longcat_bpe_path_uses_static_bpe_root(monkeypatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")
    monkeypatch.delenv("BPE_CACHE_DIR", raising=False)

    path = longcat.longcat_bpe_path()

    assert path == Path("/data/static/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192")


def test_artifact_name_matches_speech_to_speech_runtime() -> None:
    assert (
        longcat.artifact_name(
            vocab_size=100_000,
            codebook_sizes=(8192,),
        )
        == "vocab_100k_minfreq_0_maxlen_none_codes_8192"
    )


def test_artifact_name_includes_non_default_bpe_trainer_params() -> None:
    assert (
        longcat.artifact_name(
            vocab_size=8192,
            min_frequency=2,
            max_token_length=32,
            codebook_sizes=(8192,),
        )
        == "vocab_8192_minfreq_2_maxlen_32_codes_8192"
    )


def test_longcat_bpe_path_respects_explicit_cache_and_sample_limit() -> None:
    path = longcat.longcat_bpe_path(
        cache_dir="/cache",
        vocab_size=8192,
        min_frequency=2,
        max_token_length=32,
        sample_limit=1000,
    )

    assert path == Path(
        "/cache/longcat/vocab_8192_minfreq_2_maxlen_32_codes_8192_samples_1000"
    )
