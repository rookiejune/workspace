"""Build the WMT19 TTS LongCat store from the prepared TTS store."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from anydataset import AnyDataset, ViewMaterializer
from anydataset.provider.longcat import LongCatProvider
from anydataset.store.reader import read_store_dataset

from zhuyin.datasets.wmt19_tts import WMT19_TTS, wmt19_tts
from zhuyin.env import configure_environment as configure_workspace_environment
from zhuyin.env import dataset_dir, hf_home

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
    with TemporaryDirectory(
        prefix=f".{WMT19_TTS}-longcat-",
        dir=str(args.root),
    ) as tmpdir:
        longcat_view_store = Path(tmpdir) / "longcat-view"
        ViewMaterializer(
            longcat_view_store,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.longcat_batch_size,
        ).write(
            dataset_factory=TTSFactory(args.split),
            provider_factory=LongCatFactory(args),
            devices=args.devices,
        )

        wmt19_tts(split=args.split).merge(
            store_dataset(longcat_view_store, args.split)
        ).write(
            args.longcat_store,
            dataset_id=args.dataset_id,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
        )
    return Stage(
        name="longcat",
        path=str(args.longcat_store),
        reused=False,
        seconds=time.perf_counter() - start,
        sample_count=len(read_store_dataset(args.longcat_store)),
    )


@dataclass(frozen=True)
class TTSFactory:
    split: str

    def __call__(self) -> AnyDataset:
        return wmt19_tts(split=self.split)


@dataclass(frozen=True)
class LongCatFactory:
    args: argparse.Namespace

    def __call__(self, device: str) -> LongCatProvider:
        return LongCatProvider(
            cache_dir=self.args.longcat_cache_dir,
            decoders=(self.args.longcat_decoder,),
            device=device if device != "cpu" else None,
            local_files_only=self.args.local_files_only,
        )


def store_dataset(path: Path, split: str) -> AnyDataset:
    from anydataset import Source, Spec

    return AnyDataset(Spec(source=Source.STORE, path=str(path), split=split))


def is_ready_store(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        read_store_dataset(path)
    except Exception:
        return False
    return True


def ready_stage(name: str, path: Path) -> Stage:
    store = read_store_dataset(path)
    return Stage(
        name=name,
        path=str(path),
        reused=True,
        seconds=None,
        sample_count=len(store),
    )


def configure_env(args: argparse.Namespace) -> None:
    configure_workspace_environment()
    args.root = dataset_dir(WMT19_TTS) if args.root is None else args.root
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    args.longcat_store = args.root / LONGCAT_STORE_DIR
    args.reports_dir = args.root / "reports"
    args.hf_home = hf_home()
    args.longcat_cache_dir = (
        args.hf_home / "longcat-audio-codec"
        if args.longcat_cache_dir is None
        else args.longcat_cache_dir
    )
    os.environ["HF_ENDPOINT"] = args.hf_endpoint


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "tts_store": str(args.root / "base"),
        "longcat_store": str(args.longcat_store),
        "split": args.split,
        "devices": args.devices,
        "longcat_batch_size": args.longcat_batch_size,
        "longcat_decoder": args.longcat_decoder,
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
    parser.add_argument("--dataset-id", default=WMT19_TTS)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--longcat-batch-size", type=int, default=1)
    parser.add_argument("--longcat-cache-dir", type=Path)
    parser.add_argument("--longcat-decoder", default="16k_4codebooks")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
