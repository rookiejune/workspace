"""Load prepared Common Voice datasets.

This module exposes the physical Common Voice dataset entrance used by local
speech experiments. Experiment-specific resources such as speaker vocabularies
or run directories stay outside this loader. Callers choose only the dataset
root and split; language, corpus version and TSV layout are inferred from the
Common Voice directory structure.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from zhuyin.env import configure_environment, dataset_dir

if TYPE_CHECKING:
    from anydataset.dataset import MultipleAnyDataset
    from anydataset.dataset.abc import IterableAnyDataset

    CommonVoice = IterableAnyDataset | MultipleAnyDataset
else:
    CommonVoice = Any


DEFAULT_SPLIT = "train"
DATASET_NAME = "common_voice"


class _Root(NamedTuple):
    """Resolved Common Voice root information."""

    path: Path
    language: str | None = None


def common_voice(
    *,
    root: str | PathLike[str] | None = None,
    split: str = DEFAULT_SPLIT,
) -> CommonVoice:
    """Load one Common Voice split."""

    from anydataset import Preset

    configure_environment()
    resolved = _root(root)
    return Preset.COMMON_VOICE.create(
        root=resolved.path,
        split=split,
        language=resolved.language,
    )


def dataset_root(root: str | PathLike[str] | None = None) -> Path:
    """Resolve the Common Voice root."""

    if root is not None:
        return _expand(root)

    return dataset_dir(DATASET_NAME)


def _root(root: str | PathLike[str] | None = None) -> _Root:
    path = dataset_root(root)
    return _Root(path=path, language=_language_from_root(path))


def _language_from_root(root: Path) -> str | None:
    if root.parent.name.startswith("cv-corpus-"):
        return root.name
    return None


def _expand(value: str | PathLike[str]) -> Path:
    return Path(value).expanduser()
