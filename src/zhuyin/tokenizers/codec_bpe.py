"""Locate and load prepared semantic BPE tokenizer artifacts.

`codec_bpe_path()` resolves one stable artifact path from an explicit BPE root,
`BPE_CACHE_DIR`, or the workspace static home. `codec_bpe(path)` loads exactly
the path supplied by the caller and does not reinterpret training parameters.
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from zhuyin.env import static_home

from ._codec_bpe_artifact import DEFAULT_ARTIFACT

if TYPE_CHECKING:
    from anytrain.tokenizer import CodecBPE

BPE_CACHE_DIR_ENV = "BPE_CACHE_DIR"


def codec_bpe_path(
    *,
    root: str | PathLike[str] | None = None,
    artifact: str | PathLike[str] = DEFAULT_ARTIFACT,
) -> Path:
    """Resolve one prepared codec BPE artifact path."""

    relative = Path(artifact)
    if relative.is_absolute():
        raise ValueError("artifact must be relative to the BPE root.")
    return _root(root) / relative


def codec_bpe(path: str | PathLike[str]) -> CodecBPE:
    """Load a prepared codec BPE tokenizer from one artifact path."""

    from anytrain.tokenizer import CodecBPE

    return CodecBPE.from_pretrained(Path(path).expanduser())


def _root(root: str | PathLike[str] | None) -> Path:
    if root is not None:
        return Path(root).expanduser()

    value = os.environ.get(BPE_CACHE_DIR_ENV)
    if value is None:
        return static_home() / "bpe"
    if not value:
        raise ValueError(f"{BPE_CACHE_DIR_ENV} must not be empty.")
    return Path(value).expanduser()
