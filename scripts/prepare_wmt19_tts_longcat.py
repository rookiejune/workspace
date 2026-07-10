"""Build the WMT19 TTS LongCat store from the prepared TTS store."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anydataset import AnyDataset
from anydataset.provider.longcat import LongCatProvider
from anydataset.store import ViewMaterializer
from anydataset.store.reader import read_store_manifest

from zhuyin.datasets._profiles import WMT19TTSProfile
from zhuyin.datasets.wmt19_tts import WMT19_TTS, wmt19_tts
from zhuyin.env import context, datasets_home

LONGCAT_STORE_DIR = "longcat"


@dataclass(frozen=True)
class Stage:
    name: str
    path: str | None
    reused: bool
    seconds: float | None
    sample_count: int | None = None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        configure_env(args)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        stage = write_longcat_store(args)
        summary = {
            "config": run_config(args),
            "stage": asdict(stage),
            "seconds": time.perf_counter() - started_at,
        }
        write_json(args.reports_dir / "prepare_longcat_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def write_longcat_store(args: argparse.Namespace) -> Stage:
    if is_ready_store(args.longcat_store):
        return ready_stage("longcat", args.longcat_store)

    start = time.perf_counter()
    ViewMaterializer(
        args.longcat_store,
        split=args.split,
        max_shard_samples=args.max_shard_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.read_prefetch,
        write_workers=args.write_workers,
        write_prefetch=args.write_prefetch,
    ).write(
        dataset_factory=TTSFactory(args.split, args.root),
        provider_factory=LongCatFactory(),
        devices=args.devices,
    )
    return Stage(
        name="longcat",
        path=str(args.longcat_store),
        reused=False,
        seconds=time.perf_counter() - start,
        sample_count=store_sample_count(args.longcat_store),
    )


@dataclass(frozen=True)
class TTSFactory:
    split: str
    dataset_dir: Path | None = None

    def __call__(self) -> AnyDataset:
        if self.dataset_dir is None:
            return wmt19_tts(split=self.split)
        return wmt19_tts(
            dataset_dir=self.dataset_dir,
            profile=WMT19TTSProfile.STORE,
            split=self.split,
        )


@dataclass(frozen=True)
class LongCatFactory:
    def __call__(self, device: str) -> LongCatProvider:
        return LongCatProvider(
            device=device if device != "cpu" else None,
        )


def is_ready_store(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        read_store_manifest(path)
    except Exception:
        return False
    return True


def ready_stage(name: str, path: Path) -> Stage:
    return Stage(
        name=name,
        path=str(path),
        reused=True,
        seconds=None,
        sample_count=store_sample_count(path),
    )


def store_sample_count(path: Path) -> int:
    return read_store_manifest(path).sample_count


def configure_env(args: argparse.Namespace) -> None:
    args.root = datasets_home() / WMT19_TTS if args.root is None else args.root
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    args.longcat_store = args.root / LONGCAT_STORE_DIR
    args.reports_dir = args.root / "reports"


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "tts_store": str(args.root / "base"),
        "longcat_store": str(args.longcat_store),
        "split": args.split,
        "devices": args.devices,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "read_prefetch": args.read_prefetch,
        "write_workers": args.write_workers,
        "write_prefetch": args.write_prefetch,
        "resume": True,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LongCat views for the workspace WMT19 TTS dataset."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--read-prefetch", dest="read_prefetch", type=int)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--write-prefetch", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
