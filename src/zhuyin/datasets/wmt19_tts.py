"""Load prepared WMT19 zh-en TTS/LongCat datasets.

This module exposes the prepared WMT19 zh-en TTS/LongCat store as an
`anydataset.AnyDataset`. The default root is resolved from
`STATIC_HOME/datasets/wmt19-zh-en-tts-longcat-1000`. Callers may still pass one
explicit dataset directory for a one-off override.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from anydataset import AnyDataset, Source, Spec

from zhuyin.env import configure_environment, dataset_dir

DATASET_NAME = "wmt19-zh-en-tts-longcat-1000"
_STORE_DIR = "full-store"


def wmt19_tts(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the prepared WMT19 zh-en TTS/LongCat dataset.

    The returned object is an `anydataset.AnyDataset` over a store dataset. It
    is expected to contain source and target audio items with `AudioView.LONGCAT`
    views, each including `semantic_codes` and `acoustic_codes`.
    """

    configure_environment()
    root = _dataset_root(dataset_dir)
    return AnyDataset(_dataset_spec(dataset_dir=root, split=split))


def _dataset_spec(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> Spec:
    """Return the anydataset store spec for the current WMT19 TTS dataset."""

    return Spec(
        source=Source.STORE,
        path=str(_store_path(dataset_dir)),
        split=split,
    )


def _store_path(dataset_dir: str | PathLike[str] | None = None) -> Path:
    """Return the store directory inside the WMT19 TTS dataset root."""

    return _dataset_root(dataset_dir) / _STORE_DIR


def _dataset_root(value: str | PathLike[str] | None = None) -> Path:
    """Resolve the WMT19 TTS dataset root."""

    if value is not None:
        return Path(value).expanduser()

    return dataset_dir(DATASET_NAME)
