"""Load Common Voice datasets as anydataset objects.

This module exposes the physical Common Voice dataset entrance used by local
speech experiments. Experiment-specific resources such as speaker vocabularies
or run directories stay outside this loader. Common Voice metadata remains on
the anydataset sample items produced by the preset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from zhuyin.env import configure_environment, dataset_dir

if TYPE_CHECKING:
    from anydataset.dataset import MultipleAnyDataset
    from anydataset.dataset.abc import IterableAnyDataset
    from anydataset.types.item import Transforms

    CommonVoice = IterableAnyDataset | MultipleAnyDataset
else:
    CommonVoice = Any
    Transforms = Any


DEFAULT_LANGUAGE = "en"
DEFAULT_SPLIT = "train"
DATASET_NAME = "common_voice"


class Args(NamedTuple):
    """Arguments accepted by `common_voice`."""

    split: str = DEFAULT_SPLIT
    root: str | Path | None = None
    language: str = DEFAULT_LANGUAGE
    version: str | None = None


def common_voice(
    args: Args = Args(),
    *,
    transforms: Transforms | None = None,
    **load_options: Any,
) -> CommonVoice:
    """Load one Common Voice split."""

    from anydataset import Preset

    configure_environment()
    return Preset.COMMON_VOICE.create(
        split=args.split,
        root=dataset_root(args.root),
        language=args.language,
        version=args.version,
        transforms=transforms,
        **load_options,
    )


def dataset_root(root: str | Path | None = None) -> Path:
    """Resolve the Common Voice root."""

    if root is not None:
        return _expand(root)

    return dataset_dir(DATASET_NAME)


def _expand(value: str | Path) -> Path:
    return Path(value).expanduser()
