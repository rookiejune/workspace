"""Build the WMT19 TTS LongCat store from the prepared TTS store."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from anydataset.provider.longcat import LongCatProvider
from anydataset.store import ViewMaterializer

from zhuyin.datasets._wmt19_tts_io import (
    Stage,
    is_ready_store,
    ready_stage,
    store_sample_count,
    write_json,
)
from zhuyin.datasets._wmt19_tts_io import (
    stage as new_stage,
)
from zhuyin.datasets._wmt19_tts_store import StoreFactory, resolve_root
from zhuyin.env import context

LONGCAT_STORE_DIR = "longcat"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        configure_env(args)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        stage = write_longcat_store(args)
        summary = {
            "config": run_config(args),
            "stage": stage,
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
        dataset_factory=StoreFactory(args.root, args.split),
        provider_factory=LongCatFactory(),
        devices=args.devices,
    )
    return new_stage(
        "longcat",
        args.longcat_store,
        seconds=time.perf_counter() - start,
        sample_count=store_sample_count(args.longcat_store),
    )


class LongCatFactory:
    def __call__(self, device: str) -> LongCatProvider:
        return LongCatProvider(
            device=device if device != "cpu" else None,
        )


def configure_env(args: argparse.Namespace) -> None:
    args.root = resolve_root(args.root)
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
