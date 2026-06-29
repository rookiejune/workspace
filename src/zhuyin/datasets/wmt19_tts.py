"""Load prepared WMT19 zh-en TTS datasets and LongCat views.

This module exposes the prepared WMT19 zh-en TTS store and its LongCat-derived
view as `anydataset.AnyDataset` objects. The default root is resolved from
`STATIC_HOME/datasets/wmt19_tts`. Callers may still pass one explicit dataset
directory for a one-off override.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from anydataset import AnyDataset, Source, Spec

from zhuyin.env import configure_environment, dataset_dir

WMT19_TTS = "wmt19_tts"
_TTS_STORE_DIR = "base"
_LONGCAT_STORE_DIR = "longcat-delta"


def wmt19_tts(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the prepared WMT19 zh-en TTS dataset.

    The returned object is an `anydataset.AnyDataset` over a store dataset. It
    is expected to contain source and target text items plus audio waveform
    views.
    """

    return _store_dataset(
        store_dir=_TTS_STORE_DIR,
        dataset_dir=dataset_dir,
        split=split,
    )


def wmt19_tts_longcat(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the prepared WMT19 zh-en TTS LongCat view dataset.

    This is the LongCat-only delta store consumed by speech-to-speech training.
    It contains source and target audio items with `AudioView.LONGCAT` views,
    each including `semantic_codes` and `acoustic_codes`.
    """

    return _store_dataset(
        store_dir=_LONGCAT_STORE_DIR,
        dataset_dir=dataset_dir,
        split=split,
    )


def _store_dataset(
    *,
    store_dir: str,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    configure_environment()
    root = _dataset_root(dataset_dir)
    return AnyDataset(
        _dataset_spec(store_dir=store_dir, dataset_dir=root, split=split)
    )


def _dataset_spec(
    *,
    store_dir: str,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> Spec:
    """Return the anydataset store spec for the current WMT19 TTS dataset."""

    return Spec(
        source=Source.STORE,
        path=str(_store_path(store_dir=store_dir, dataset_dir=dataset_dir)),
        split=split,
    )


def _store_path(
    *,
    store_dir: str,
    dataset_dir: str | PathLike[str] | None = None,
) -> Path:
    """Return the store directory inside the WMT19 TTS dataset root."""

    return _dataset_root(dataset_dir) / store_dir


def _dataset_root(value: str | PathLike[str] | None = None) -> Path:
    """Resolve the WMT19 TTS dataset root."""

    if value is not None:
        return Path(value).expanduser()

    return dataset_dir(WMT19_TTS)
