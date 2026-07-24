"""Build the WMT19 TTS LongCat store from the prepared TTS store."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from _wmt19_tts_codec import prepare_longcat
from _wmt19_tts_io import write_json
from _wmt19_tts_store import resolve_root

from zhuyin.env import context


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        args.root = resolve_root(args.root)
        args.root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        stage = prepare_longcat(
            root=args.root,
            split=args.split,
            devices=args.devices,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            read_prefetch=args.read_prefetch,
            write_workers=args.write_workers,
            write_prefetch=args.write_prefetch,
        )
        summary = {
            "config": run_config(args),
            "stage": stage,
            "seconds": time.perf_counter() - started_at,
        }
        write_json(args.root / "reports" / "prepare_longcat_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "root": str(args.root),
        "tts_store": str(args.root / "base"),
        "longcat_store": str(args.root / "longcat"),
        "split": args.split,
        "devices": args.devices,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "read_prefetch": args.read_prefetch,
        "write_workers": args.write_workers,
        "write_prefetch": args.write_prefetch,
        "resume": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
