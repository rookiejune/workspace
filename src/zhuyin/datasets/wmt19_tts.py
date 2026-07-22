"""Load WMT19 zh-en TTS datasets and codec views as logical objects.

This module exposes WMT19 zh-en TTS and its codec-derived views as
`anydataset.AnyDataset` objects with a stable logical sample schema.

Passing `root=...` always reads a prepared standard store. Without an explicit
root, physical loading is currently implemented only for the Fudan profile and
uses the standard store under `datasets_home()`.
"""

from __future__ import annotations

from enum import auto
from os import PathLike
from pathlib import Path

from anydataset import AnyDataset
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
)

from zhuyin._compat import StrEnum
from zhuyin.datasets._wmt19_tts_stable import (
    DEFAULT_STABLE_QUANTIZER,
    StableQuantizer,
)
from zhuyin.datasets._wmt19_tts_stable import (
    store_dir as _stable_store_dir,
)
from zhuyin.env import Location, datasets_home, location

WMT19_TTS = "wmt19_tts"
_TTS_STORE_DIR = "base"


class Codec(StrEnum):
    """Logical codec view selected by `wmt19_tts_codec()`."""

    LONGCAT = auto()
    DAC = auto()
    STABLE = auto()
    UNICODEC = auto()


def wmt19_tts(
    *,
    root: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the logical WMT19 zh-en TTS dataset.

    The returned object contains source and target text items plus audio
    waveform views.
    """

    if root is not None:
        return _store_view(store_dir=_TTS_STORE_DIR, root=root, split=split)
    if location() is not Location.FUDAN:
        _raise_unimplemented_location("wmt19_tts")
    return _store_view(store_dir=_TTS_STORE_DIR, root=root, split=split)


def wmt19_tts_codec(
    *,
    codec: Codec | str = Codec.LONGCAT,
    root: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return one codec view over the logical WMT19 zh-en TTS dataset."""

    resolved_codec = Codec(codec)
    if root is None and location() is not Location.FUDAN:
        _raise_unimplemented_location("wmt19_tts_codec")
    store_dir = _stable_store_dir() if resolved_codec is Codec.STABLE else resolved_codec.value
    return _store_view(
        store_dir=store_dir,
        root=root,
        split=split,
        merge_base=resolved_codec is Codec.LONGCAT,
        transforms=_longcat_transforms() if resolved_codec is Codec.LONGCAT else None,
    )


def wmt19_tts_stable(
    *,
    root: str | PathLike[str] | None = None,
    split: str = "train",
    quantizer: StableQuantizer | str = DEFAULT_STABLE_QUANTIZER,
) -> AnyDataset:
    """Return one posthoc-quantized Stable Codec view of WMT19 zh-en TTS."""

    return _store_view(
        store_dir=_stable_store_dir(quantizer),
        root=root,
        split=split,
    )


def wmt19_tts_dac(
    *,
    root: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the Descript Audio Codec view of WMT19 zh-en TTS."""

    return wmt19_tts_codec(codec=Codec.DAC, root=root, split=split)


def wmt19_tts_unicodec(
    *,
    root: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the UniCodec view of WMT19 zh-en TTS."""

    return wmt19_tts_codec(codec=Codec.UNICODEC, root=root, split=split)


def _store_view(
    *,
    store_dir: str,
    root: str | PathLike[str] | None = None,
    split: str = "train",
    merge_base: bool = False,
    transforms=None,
) -> AnyDataset:
    view = _store_dataset(
        store_dir=store_dir,
        root=root,
        split=split,
        transforms=transforms,
    )
    if not merge_base:
        return view
    return view.merge(
        _store_dataset(store_dir=_TTS_STORE_DIR, root=root, split=split)
    )


def _store_dataset(
    *,
    store_dir: str,
    root: str | PathLike[str] | None = None,
    split: str = "train",
    transforms=None,
) -> AnyDataset:
    resolved = dataset_root(root)
    return AnyDataset(
        Spec(source=Source.STORE, path=str(resolved / store_dir), split=split),
        transforms=transforms,
    )


def dataset_root(root: str | PathLike[str] | None = None) -> Path:
    """Resolve the standard WMT19 TTS store root."""

    if root is not None:
        return Path(root).expanduser()
    return datasets_home() / WMT19_TTS


def _raise_unimplemented_location(loader: str) -> None:
    raise NotImplementedError(
        f"{loader} default loading is only implemented for Fudan; pass root=... "
        "or add an explicit source implementation for this location."
    )


def _longcat_transforms():
    return {
        (role, Modality.AUDIO): _longcat_item
        for role in (Role.SOURCE, Role.TARGET)
    }


def _longcat_item(item: AudioItem) -> AudioItem:
    import torch

    value = item.views[AudioView.LONGCAT]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "stored LongCat view must use the anytrain [frame, codebook] Tensor "
            "contract; rematerialize the longcat store."
        )
    if value.ndim != 2:
        raise ValueError("stored LongCat codes must have shape [frame, codebook].")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError("stored LongCat codes must contain integer ids.")
    return item
