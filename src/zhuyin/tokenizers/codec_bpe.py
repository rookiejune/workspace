"""Load prepared semantic BPE tokenizer artifacts.

This module resolves reusable BPE artifacts from the workspace static
cache. It exposes the artifact path for jobs that only need to pass a location,
and a lazy `CodecBPE` loader for callers that need the tokenizer object.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from zhuyin.env import static_home

if TYPE_CHECKING:
    from anytrain.tokenizer import CodecBPE

DEFAULT_CODEC_NAME = "longcat"
DEFAULT_VOCAB_SIZE = 100_000
DEFAULT_MIN_FREQUENCY = 0
DEFAULT_MAX_TOKEN_LENGTH = None
DEFAULT_CODEBOOK_SIZES = (8192,)


def codec_bpe_path(
    *,
    cache_dir: str | Path | None = None,
    codec_name: str = DEFAULT_CODEC_NAME,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_token_length: int | None = DEFAULT_MAX_TOKEN_LENGTH,
    codebook_sizes: tuple[int, ...] = DEFAULT_CODEBOOK_SIZES,
    sample_limit: int | None = None,
) -> Path:
    """Return the workspace path for one LongCat BPE artifact."""

    root = _cache_dir(cache_dir)
    path = (
        root
        / codec_name
        / artifact_name(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_token_length=max_token_length,
            codebook_sizes=codebook_sizes,
        )
    )
    if sample_limit is None:
        return path
    return path.with_name(f"{path.name}_samples_{sample_limit}")


def codec_bpe(
    *,
    cache_dir: str | Path | None = None,
    codec_name: str = DEFAULT_CODEC_NAME,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_token_length: int | None = DEFAULT_MAX_TOKEN_LENGTH,
    codebook_sizes: tuple[int, ...] = DEFAULT_CODEBOOK_SIZES,
    sample_limit: int | None = None,
) -> CodecBPE:
    """Load a prepared LongCat semantic BPE tokenizer."""

    from anytrain.tokenizer import CodecBPE

    return CodecBPE.from_pretrained(
        codec_bpe_path(
            cache_dir=cache_dir,
            codec_name=codec_name,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_token_length=max_token_length,
            codebook_sizes=codebook_sizes,
            sample_limit=sample_limit,
        )
    )


def artifact_name(
    *,
    vocab_size: int,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_token_length: int | None = DEFAULT_MAX_TOKEN_LENGTH,
    codebook_sizes: tuple[int, ...],
) -> str:
    """Return the LongCat BPE artifact directory name."""

    vocab = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    codebooks = "x".join(str(size) for size in codebook_sizes)
    maxlen = "none" if max_token_length is None else str(max_token_length)
    return f"vocab_{vocab}_minfreq_{min_frequency}_maxlen_{maxlen}_codes_{codebooks}"


def _cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    value = os.environ.get("BPE_CACHE_DIR")
    if value is None:
        return static_home() / "bpe"
    if not value:
        raise ValueError("BPE_CACHE_DIR must not be empty.")
    return Path(value).expanduser()
