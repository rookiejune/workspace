"""Build a codec view from the prepared WMT19 TTS waveform store."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from zhuyin.datasets._wmt19_tts_codec import (
    prepare_dac,
    prepare_stable_codec,
    prepare_unicodec,
)
from zhuyin.datasets._wmt19_tts_io import Stage, write_json
from zhuyin.datasets._wmt19_tts_stable import (
    DEFAULT_STABLE_QUANTIZER,
    StableQuantizer,
)
from zhuyin.datasets._wmt19_tts_store import resolve_root
from zhuyin.env import context

Prepare = Callable[..., Stage]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        args.root = resolve_root(args.root)
        args.root.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        stage = args.prepare(**prepare_config(args))
        summary = {
            "config": run_config(args),
            "stage": stage,
            "seconds": time.perf_counter() - started_at,
        }
        write_json(
            args.root / "reports" / f"prepare_{stage['name']}_summary.json",
            summary,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "root": args.root,
        "split": args.split,
        "devices": args.devices,
        "max_shard_samples": args.max_shard_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "read_prefetch": args.read_prefetch,
        "write_workers": args.write_workers,
        "write_prefetch": args.write_prefetch,
    }
    config.update(args.codec_config(args))
    return config


def run_config(args: argparse.Namespace) -> dict[str, object]:
    config = prepare_config(args)
    config["root"] = str(args.root)
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, StableQuantizer):
            config[key] = value.value
    config.update(codec=args.codec, keep_text=True, resume=True)
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a codec view for WMT19 TTS."
    )
    subparsers = parser.add_subparsers(dest="codec", required=True)

    dac = codec_parser(subparsers, "dac", prepare_dac, dac_config)
    dac.add_argument("--dac-cache-dir", type=Path)
    dac.add_argument("--model-type", choices=("16khz", "24khz", "44khz"), default="44khz")
    dac.add_argument("--model-bitrate", choices=("8kbps", "16kbps"), default="8kbps")
    dac.add_argument("--tag", default="latest")
    dac.add_argument("--n-quantizers", type=int)
    dac.add_argument("--local-files-only", action="store_true")

    stable = codec_parser(
        subparsers, "stable", prepare_stable_codec, stable_config
    )
    stable.add_argument("--version", default="speech-16k")
    stable.add_argument("--pretrained-model")
    stable.add_argument(
        "--posthoc-bottleneck",
        type=StableQuantizer,
        choices=tuple(StableQuantizer),
        default=DEFAULT_STABLE_QUANTIZER,
    )
    stable.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )

    unicodec = codec_parser(
        subparsers, "unicodec", prepare_unicodec, unicodec_config
    )
    unicodec.add_argument("--unicodec-cache-dir", type=Path)
    unicodec.add_argument("--domain", choices=("0", "1", "2"), default="0")
    unicodec.add_argument("--bandwidth-id", type=int, default=0)
    unicodec.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def codec_parser(
    subparsers: Any,
    name: str,
    prepare: Prepare,
    codec_config: Callable[[argparse.Namespace], dict[str, object]],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.set_defaults(prepare=prepare, codec_config=codec_config)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--read-prefetch", type=int)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--write-prefetch", type=int)
    return parser


def dac_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "cache_dir": args.dac_cache_dir,
        "model_type": args.model_type,
        "model_bitrate": args.model_bitrate,
        "tag": args.tag,
        "n_quantizers": args.n_quantizers,
        "local_files_only": args.local_files_only,
    }


def stable_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "version": args.version,
        "pretrained_model": args.pretrained_model,
        "posthoc_bottleneck": args.posthoc_bottleneck,
        "normalize": args.normalize,
    }


def unicodec_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "cache_dir": args.unicodec_cache_dir,
        "domain": args.domain,
        "bandwidth_id": args.bandwidth_id,
        "local_files_only": args.local_files_only,
    }
