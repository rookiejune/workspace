"""Store status and report IO shared by WMT19 TTS services."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any, TypedDict, cast

from anydataset.store.reader import read_store_manifest


class Stage(TypedDict):
    """Serializable summary for one materialization stage."""

    name: str
    path: str | None
    reused: bool
    seconds: float | None
    sample_count: int | None


def stage(
    name: str,
    path: Path,
    *,
    seconds: float,
    sample_count: int,
) -> Stage:
    return {
        "name": name,
        "path": str(path),
        "reused": False,
        "seconds": seconds,
        "sample_count": sample_count,
    }


def is_ready_store(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / ".ready").exists():
        return False
    read_store_manifest(path)
    return True


def ready_stage(name: str, path: Path) -> Stage:
    return {
        "name": name,
        "path": str(path),
        "reused": True,
        "seconds": None,
        "sample_count": store_sample_count(path),
    }


def store_sample_count(path: Path) -> int:
    return read_store_manifest(path).sample_count


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_metrics_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def preview_metrics(path: Path, *, limit: int) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, limit):
            output.append(cast(Mapping[str, Any], json.loads(line)))
    return output


__all__ = [
    "Stage",
    "is_ready_store",
    "preview_metrics",
    "ready_stage",
    "stage",
    "store_sample_count",
    "write_json",
    "write_metrics_jsonl",
]
