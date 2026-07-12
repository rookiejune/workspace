"""Standard WMT19 TTS store root and dataset factory."""

from __future__ import annotations

from pathlib import Path

from anydataset import AnyDataset

from zhuyin.datasets.wmt19_tts import dataset_root, wmt19_tts


class StoreFactory:
    """Picklable factory for one standard WMT19 TTS store split."""

    __slots__ = ("root", "split")

    def __init__(self, root: Path | None, split: str) -> None:
        self.root = root
        self.split = split

    def __call__(self) -> AnyDataset:
        if self.root is None:
            return wmt19_tts(split=self.split)
        return wmt19_tts(root=self.root, split=self.split)


def resolve_root(root: Path | None) -> Path:
    """Resolve a standard WMT19 TTS root for a script invocation."""

    return dataset_root(root).resolve()


__all__ = ["StoreFactory", "resolve_root"]
