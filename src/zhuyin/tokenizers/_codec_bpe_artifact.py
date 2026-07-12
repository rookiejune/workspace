"""Codec BPE training defaults and artifact naming."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CODEC_NAME = "longcat"
DEFAULT_VOCAB_SIZE = 100_000
DEFAULT_MIN_FREQUENCY = 0
DEFAULT_MAX_TOKEN_LENGTH = None
DEFAULT_CODEBOOK_SIZES = (8192,)


def artifact_name(
    *,
    vocab_size: int,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_token_length: int | None = DEFAULT_MAX_TOKEN_LENGTH,
    codebook_sizes: tuple[int, ...],
) -> str:
    """Return the directory name for one codec BPE training configuration."""

    vocab = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    codebooks = "x".join(str(size) for size in codebook_sizes)
    maxlen = "none" if max_token_length is None else str(max_token_length)
    return f"vocab_{vocab}_minfreq_{min_frequency}_maxlen_{maxlen}_codes_{codebooks}"


def artifact_path(
    *,
    codec_name: str = DEFAULT_CODEC_NAME,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    max_token_length: int | None = DEFAULT_MAX_TOKEN_LENGTH,
    codebook_sizes: tuple[int, ...] = DEFAULT_CODEBOOK_SIZES,
    sample_limit: int | None = None,
) -> Path:
    """Return the artifact path relative to the BPE root."""

    path = Path(codec_name) / artifact_name(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        codebook_sizes=codebook_sizes,
    )
    if sample_limit is None:
        return path
    return path.with_name(f"{path.name}_samples_{sample_limit}")


DEFAULT_ARTIFACT = artifact_path()
