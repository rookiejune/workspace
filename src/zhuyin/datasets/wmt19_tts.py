"""Load WMT19 zh-en TTS datasets and codec views as logical objects.

This module exposes WMT19 zh-en TTS and its codec-derived views as
`anydataset.AnyDataset` objects with a stable logical sample schema. Physical
storage can vary by workspace profile; callers may still pass one explicit
dataset directory for a one-off standard store override.
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
from zhuyin.datasets._profiles import (
    WMT19TTSLongCatProfile,
    WMT19TTSProfile,
)
from zhuyin.env import datasets_home

if TYPE_CHECKING:
    from torch import Tensor

WMT19_TTS = "wmt19_tts"
_TTS_STORE_DIR = "base"
_LONGCAT_STORE_DIR = "longcat"
_STABLE_STORE_DIR = "stable"
_SHARDED_CSV_SOURCE = "sharded_csv"


class Codec(StrEnum):
    """Codec view selected by `wmt19_tts_codec()`."""

    LONGCAT = auto()
    STABLE = auto()


def wmt19_tts(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    profile: WMT19TTSProfile | str | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the logical WMT19 zh-en TTS dataset.

    The returned object contains source and target text items plus audio
    waveform views. The default physical profile is selected by `LOCATION`
    unless `dataset_dir` or `profile` is explicit.
    """

    resolved = WMT19TTSProfile.resolve(profile)
    if resolved is WMT19TTSProfile.STORE:
        return _store_view(
            store_dir=_TTS_STORE_DIR,
            dataset_dir=dataset_dir,
            split=split,
        )
    if resolved is WMT19TTSProfile.HZ_EXPORT:
        root = _explicit_or_profile_root(dataset_dir, resolved.default_root)
        return AnyDataset(
            Spec(source=_SHARDED_CSV_SOURCE, path=str(root), split=split),
            parse_fn=_parse_hz_tts_row,
        )
    raise ValueError(f"unsupported WMT19 TTS profile: {resolved.value}")


def wmt19_tts_codec(
    *,
    codec: Codec | str = Codec.LONGCAT,
    dataset_dir: str | PathLike[str] | None = None,
    profile: WMT19TTSLongCatProfile | str | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return one codec view over the logical WMT19 zh-en TTS dataset."""

    resolved = Codec(codec)
    if resolved is Codec.LONGCAT:
        return _longcat_dataset(
            dataset_dir=dataset_dir,
            profile=profile,
            split=split,
        )
    if resolved is Codec.STABLE:
        if profile is not None:
            raise ValueError(f"codec={resolved.value} does not accept profile.")
        return _store_view(
            store_dir=_STABLE_STORE_DIR,
            dataset_dir=dataset_dir,
            split=split,
            merge_base=True,
        )
    raise ValueError(f"unsupported WMT19 TTS codec: {resolved.value}")


def _longcat_dataset(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    profile: WMT19TTSLongCatProfile | str | None = None,
    split: str = "train",
) -> AnyDataset:
    resolved = WMT19TTSLongCatProfile.resolve(profile)
    if resolved is WMT19TTSLongCatProfile.STORE:
        return _store_view(
            store_dir=_LONGCAT_STORE_DIR,
            dataset_dir=dataset_dir,
            split=split,
            merge_base=True,
        )
    if resolved is WMT19TTSLongCatProfile.HZ_HF_DISK_CODES:
        root = _explicit_or_profile_root(dataset_dir, resolved.default_root)
        return AnyDataset(
            Spec(source=Source.HF_DISK, path=str(root), split=split),
            parse_fn=_parse_hz_longcat_row,
        )
    raise ValueError(f"unsupported WMT19 TTS LongCat profile: {resolved.value}")


def _store_view(
    *,
    store_dir: str,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
    merge_base: bool = False,
) -> AnyDataset:
    view = _store_dataset(store_dir=store_dir, dataset_dir=dataset_dir, split=split)
    if not merge_base:
        return view
    return view.merge(
        _store_dataset(
            store_dir=_TTS_STORE_DIR,
            dataset_dir=dataset_dir,
            split=split,
        )
    )


def _store_dataset(
    *,
    store_dir: str,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    root = _dataset_root(dataset_dir)
    return AnyDataset(
        Spec(source=Source.STORE, path=str(root / store_dir), split=split),
    )


def _dataset_root(value: str | PathLike[str] | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser()
    return datasets_home() / WMT19_TTS


def _explicit_or_profile_root(
    value: str | PathLike[str] | None,
    default: Path,
) -> Path:
    if value is not None:
        return Path(value).expanduser()
    return default


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
        semantic_codes, acoustic_codes = _aligned_longcat_codes(
            semantic_codes=_tensor(row[f"{prefix}_semantic_codes"]),
            acoustic_codes=_tensor(row[f"{prefix}_acoustic_codes"]),
        )
        return {
            (role, Modality.AUDIO): AudioItem(
                views={
                    AudioView.LONGCAT: {
                        "semantic_codes": semantic_codes,
                        "acoustic_codes": acoustic_codes,
                    },
                },
            ),
            (role, Modality.TEXT): TextItem(
                views={TextView.TEXT: row[f"{prefix}_text"]},
                meta={TextMeta.LANG: row[f"{prefix}_language"]},
            ),
        }
    except Exception as e:
        raise ValueError(f"invalid HZ WMT19 TTS LongCat row: {row}") from e


def _aligned_longcat_codes(
    *,
    semantic_codes: Tensor,
    acoustic_codes: Tensor,
) -> tuple[Tensor, Tensor]:
    length = min(semantic_codes.shape[-1], acoustic_codes.shape[-1])
    return semantic_codes[..., :length], acoustic_codes[..., :length]


def _load_audio(path: str) -> Any:
    import torchaudio

    return torchaudio.load(path)


def _tensor(value: Any) -> Tensor:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)
