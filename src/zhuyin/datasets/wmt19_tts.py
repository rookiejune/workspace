"""Load WMT19 zh-en TTS datasets and LongCat views as logical objects.

This module exposes WMT19 zh-en TTS and its LongCat-derived view as
`anydataset.AnyDataset` objects with a stable logical sample schema. Physical
storage can vary by workspace profile; callers may still pass one explicit
dataset directory for a one-off standard store override.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anydataset import (
    AnyDataset,
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

from zhuyin.datasets._profiles import WMT19TTSLongCatProfile, WMT19TTSProfile
from zhuyin.env import configure_environment, dataset_dir

if TYPE_CHECKING:
    from torch import Tensor

WMT19_TTS = "wmt19_tts"
_TTS_STORE_DIR = "base"
_LONGCAT_STORE_DIR = "longcat"
_SHARDED_CSV_SOURCE = "sharded_csv"


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

    configure_environment()
    resolved = WMT19TTSProfile.resolve(profile)
    if resolved is WMT19TTSProfile.STORE:
        return _store_dataset(
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


def wmt19_tts_longcat(
    *,
    dataset_dir: str | PathLike[str] | None = None,
    profile: WMT19TTSLongCatProfile | str | None = None,
    split: str = "train",
) -> AnyDataset:
    """Return the logical WMT19 zh-en TTS LongCat view dataset.

    It contains source and target audio items with `AudioView.LONGCAT` views,
    each including `semantic_codes` and `acoustic_codes`. The default physical
    profile is selected by `LOCATION` unless `dataset_dir` or `profile` is
    explicit.
    """

    configure_environment()
    resolved = WMT19TTSLongCatProfile.resolve(profile)
    if resolved is WMT19TTSLongCatProfile.STORE:
        return _store_dataset(
            store_dir=_LONGCAT_STORE_DIR,
            dataset_dir=dataset_dir,
            split=split,
        )
    if resolved is WMT19TTSLongCatProfile.HZ_HF_DISK_CODES:
        root = _explicit_or_profile_root(dataset_dir, resolved.default_root)
        return AnyDataset(
            Spec(source=Source.HF_DISK, path=str(root), split=split),
            parse_fn=_parse_hz_longcat_row,
        )
    raise ValueError(f"unsupported WMT19 TTS LongCat profile: {resolved.value}")


def _store_dataset(
    *,
    store_dir: str,
    dataset_dir: str | PathLike[str] | None = None,
    split: str = "train",
) -> AnyDataset:
    root = _dataset_root(dataset_dir)
    return AnyDataset(_dataset_spec(store_dir=store_dir, dataset_dir=root, split=split))


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
        return {
            (role, Modality.AUDIO): AudioItem(
                views={
                    AudioView.LONGCAT: {
                        "semantic_codes": _tensor(row[f"{prefix}_semantic_codes"]),
                        "acoustic_codes": _tensor(row[f"{prefix}_acoustic_codes"]),
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


def _load_audio(path: str) -> Any:
    import torchaudio

    return torchaudio.load(path)


def _tensor(value: Any) -> Tensor:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)
