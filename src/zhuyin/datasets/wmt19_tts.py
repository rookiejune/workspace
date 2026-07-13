"""Load WMT19 zh-en TTS datasets and codec views as logical objects.

This module exposes WMT19 zh-en TTS and its codec-derived views as
`anydataset.AnyDataset` objects with a stable logical sample schema.

Passing `root=...` always reads a prepared standard store. Without an explicit
root, physical loading follows the current location and requested logical view:
HZ uses its private TTS and LongCat exports, while other views use the standard
store under `datasets_home()`.
"""

from __future__ import annotations

from enum import auto
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anydataset import AnyDataset
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
    TextItem,
    TextMeta,
    TextView,
)

from zhuyin._compat import StrEnum
from zhuyin.env import Location, datasets_home, location

if TYPE_CHECKING:
    from torch import Tensor

WMT19_TTS = "wmt19_tts"
_TTS_STORE_DIR = "base"
_SHARDED_CSV_SOURCE = "sharded_csv"
_HZ_TTS_ROOT = Location.HZ.static_home / "train" / "text_to_speech" / "moss_tts_hz_export"
_HZ_LONGCAT_ROOT = (
    Location.HZ.static_home / "datasets" / "wmt19_tts_longcat_codes_text_cleaned"
)


class Codec(StrEnum):
    """Codec view selected by `wmt19_tts_codec()`; values match store dirs."""

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

    if root is not None or location() is not Location.HZ:
        return _store_view(store_dir=_TTS_STORE_DIR, root=root, split=split)
    return AnyDataset(
        Spec(
            source=_SHARDED_CSV_SOURCE,
            path=str(_HZ_TTS_ROOT),
            split=split,
        ),
        parse_fn=_parse_hz_tts_row,
    )


def wmt19_tts_codec(
    *,
    codec: Codec | str = Codec.LONGCAT,
    root: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return one codec view over the logical WMT19 zh-en TTS dataset."""

    resolved_codec = Codec(codec)
    if root is not None or resolved_codec is not Codec.LONGCAT or location() is not Location.HZ:
        return _store_view(
            store_dir=resolved_codec.value,
            root=root,
            split=split,
            merge_base=resolved_codec is Codec.LONGCAT,
            transforms=_longcat_transforms() if resolved_codec is Codec.LONGCAT else None,
        )
    return AnyDataset(
        Spec(source=Source.HF_DISK, path=str(_HZ_LONGCAT_ROOT), split=split),
        parse_fn=_parse_hz_longcat_row,
    )


def wmt19_tts_stable(
    *,
    root: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the Stable Codec view of WMT19 zh-en TTS."""

    return wmt19_tts_codec(codec=Codec.STABLE, root=root, split=split)


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


def _parse_hz_tts_row(row: dict[str, str]) -> dict[tuple[Role, Modality], Any]:
    try:
        return {
            (Role.SOURCE, Modality.AUDIO): AudioItem(
                views={
                    AudioView.WAVEFORM: _load_audio(
                        row["source/audio"].replace("rank", "train/shard")
                    ),
                },
            ),
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: row["source/text"]},
                meta={TextMeta.LANG: row["source/lang"]},
            ),
            (Role.TARGET, Modality.AUDIO): AudioItem(
                views={
                    AudioView.WAVEFORM: _load_audio(
                        row["target/audio"].replace("rank", "train/shard")
                    ),
                },
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: row["target/text"]},
                meta={TextMeta.LANG: row["target/lang"]},
            ),
        }
    except Exception as e:
        raise ValueError(f"invalid HZ WMT19 TTS row: {row}") from e


def _parse_hz_longcat_row(row: dict[str, Any]) -> dict[tuple[Role, Modality], Any]:
    return _parse_hz_longcat_side(row, Role.SOURCE, "source") | _parse_hz_longcat_side(
        row,
        Role.TARGET,
        "target",
    )


def _parse_hz_longcat_side(
    row: dict[str, Any],
    role: Role,
    prefix: str,
) -> dict[tuple[Role, Modality], Any]:
    try:
        codes = _longcat_codes(
            semantic_codes=_tensor(row[f"{prefix}_semantic_codes"]),
            acoustic_codes=_tensor(row[f"{prefix}_acoustic_codes"]),
        )
        return {
            (role, Modality.AUDIO): AudioItem(
                views={AudioView.LONGCAT: codes},
            ),
            (role, Modality.TEXT): TextItem(
                views={TextView.TEXT: row[f"{prefix}_text"]},
                meta={TextMeta.LANG: row[f"{prefix}_language"]},
            ),
        }
    except Exception as e:
        raise ValueError(f"invalid HZ WMT19 TTS LongCat row: {row}") from e


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


def _longcat_codes(
    *,
    semantic_codes: Tensor,
    acoustic_codes: Tensor,
) -> Tensor:
    import torch

    if semantic_codes.ndim != 1:
        raise ValueError("LongCat semantic codes must have shape [frame].")
    if acoustic_codes.ndim != 2:
        raise ValueError("LongCat acoustic codes must have shape [codebook, frame].")
    length = min(semantic_codes.shape[-1], acoustic_codes.shape[-1])
    return torch.cat(
        (
            semantic_codes[:length].unsqueeze(0),
            acoustic_codes[:, :length],
        ),
        dim=0,
    ).transpose(0, 1).contiguous()


def _load_audio(path: str) -> Any:
    import torchaudio

    return torchaudio.load(path)


def _tensor(value: Any) -> Tensor:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)
