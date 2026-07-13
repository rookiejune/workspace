"""Build a text-preserving DAC view for the WMT19 TTS store."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from zhuyin.datasets._wmt19_tts_codec import prepare_dac
from zhuyin.datasets._wmt19_tts_io import write_json
from zhuyin.datasets._wmt19_tts_store import resolve_root
from zhuyin.env import context


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        args.root = resolve_root(args.root)
        args.root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        stage = prepare_dac(
            root=args.root,
            split=args.split,
            devices=args.devices,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            read_prefetch=args.read_prefetch,
            write_workers=args.write_workers,
            write_prefetch=args.write_prefetch,
            cache_dir=args.dac_cache_dir,
            model_type=args.model_type,
            model_bitrate=args.model_bitrate,
            tag=args.tag,
            n_quantizers=args.n_quantizers,
            local_files_only=args.local_files_only,
        )
        summary = {
            "config": run_config(args),
            "stage": stage,
            "seconds": time.perf_counter() - started_at,
        }
        write_json(args.root / "reports" / "prepare_dac_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "root": str(args.root),
        "split": args.split,
        "devices": args.devices,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "read_prefetch": args.read_prefetch,
        "write_workers": args.write_workers,
        "write_prefetch": args.write_prefetch,
        "dac_cache_dir": None if args.dac_cache_dir is None else str(args.dac_cache_dir),
        "model_type": args.model_type,
        "model_bitrate": args.model_bitrate,
        "tag": args.tag,
        "n_quantizers": args.n_quantizers,
        "local_files_only": args.local_files_only,
        "keep_text": True,
        "resume": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a text-preserving DAC view for WMT19 TTS."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--read-prefetch", type=int)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--write-prefetch", type=int)
    parser.add_argument("--dac-cache-dir", type=Path)
    parser.add_argument("--model-type", choices=("16khz", "24khz", "44khz"), default="44khz")
    parser.add_argument("--model-bitrate", choices=("8kbps", "16kbps"), default="8kbps")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--n-quantizers", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
