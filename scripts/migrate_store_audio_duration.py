"""Add AudioMeta.DURATION to an existing anydataset store manifest.

The migration reads audio payloads only to inspect their time-axis length. It
does not load codec models and does not rewrite payload shards.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from anydataset.store.manifest import SampleManifestEntry
from anydataset.store.manifestio import read_samples_manifest, write_samples_manifest
from anydataset.store.paths import samples_parquet_path
from anydataset.store.reader import read_store_dataset, read_store_views
from anydataset.types import AudioItem, AudioMeta, AudioView, Modality

CODEC_FRAME_RATES: Mapping[str, float] = {
    "longcat": 50.0,
    "stable": 25.0,
    "unicodec": 75.0,
}
CODEC_VIEWS: Mapping[str, AudioView] = {
    "longcat": AudioView.LONGCAT,
    "stable": AudioView.STABLE,
    "unicodec": AudioView.UNICODEC,
}


@dataclass(frozen=True)
class Config:
    store: Path
    view: AudioView
    frame_rate: float | None = None
    overwrite: bool = False
    backup: Path | None = None
    dry_run: bool = False
    log_every: int = 10_000


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    summary = migrate(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


def migrate(config: Config) -> dict[str, object]:
    store = config.store.expanduser().resolve()
    _validate_config(config, store)
    pending = _pending_audio_items(store, overwrite=config.overwrite)
    summary: dict[str, object] = {
        "store": str(store),
        "view": config.view.value,
        "frame_rate": config.frame_rate,
        "pending_audio_items": pending,
        "updated_audio_items": 0,
        "dry_run": config.dry_run,
        "backup": None,
    }
    if pending == 0 or config.dry_run:
        return summary

    views = tuple(
        view
        for view in read_store_views(store)
        if view[1] is Modality.AUDIO and view[2] is config.view
    )
    if not views:
        raise ValueError(
            f"Store {store} has no audio {config.view.value!r} view."
        )
    dataset = read_store_dataset(store, views=views)
    backup = config.backup
    if backup is not None:
        backup = backup.expanduser().resolve()
        if backup.exists():
            raise FileExistsError(backup)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(samples_parquet_path(store), backup)
        summary["backup"] = str(backup)

    stats = {"updated": 0}
    write_samples_manifest(
        store,
        _entries(dataset, store=store, config=config, stats=stats),
    )
    read_store_dataset(store, views=())
    summary["updated_audio_items"] = stats["updated"]
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add AudioMeta.DURATION to an anydataset store without rewriting "
            "payload shards."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    waveform = subparsers.add_parser(
        "waveform",
        help="derive seconds from waveform samples and sample rate",
    )
    _common_arguments(waveform)

    codec = subparsers.add_parser(
        "codec",
        help="derive seconds from codec frames and a constant frame rate",
    )
    _common_arguments(codec)
    codec.add_argument("--codec", choices=tuple(CODEC_VIEWS), required=True)
    codec.add_argument(
        "--frame-rate",
        type=_positive_number,
        help="override the registered codec frame rate",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    if args.no_backup and args.backup is not None:
        raise ValueError("--backup and --no-backup cannot be used together.")
    backup = None
    if not args.no_backup:
        backup = args.backup or _default_backup(args.store)
    if args.mode == "waveform":
        return Config(
            store=args.store,
            view=AudioView.WAVEFORM,
            overwrite=args.overwrite,
            backup=backup,
            dry_run=args.dry_run,
            log_every=args.log_every,
        )
    codec = cast(str, args.codec)
    return Config(
        store=args.store,
        view=CODEC_VIEWS[codec],
        frame_rate=args.frame_rate or CODEC_FRAME_RATES[codec],
        overwrite=args.overwrite,
        backup=backup,
        dry_run=args.dry_run,
        log_every=args.log_every,
    )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("store", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing duration metadata instead of only filling gaps",
    )
    parser.add_argument("--backup", type=Path, help="manifest backup path")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="rewrite the manifest without retaining a backup",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=_positive_int, default=10_000)


def _entries(
    dataset: Any,
    *,
    store: Path,
    config: Config,
    stats: dict[str, int],
) -> Iterator[SampleManifestEntry]:
    duration_key = AudioMeta.DURATION.value
    for position, entry in enumerate(read_samples_manifest(store)):
        sample = None
        items = []
        for ref, meta in entry.items:
            values = dict(meta)
            if ref[1] is Modality.AUDIO and (
                config.overwrite or duration_key not in values
            ):
                if sample is None:
                    sample = dataset[entry.sample_index]
                item = sample.get(ref)
                if not isinstance(item, AudioItem):
                    raise TypeError(
                        f"Sample {entry.sample_index} item {ref!r} must be an AudioItem."
                    )
                values[duration_key] = _duration(item, config)
                stats["updated"] += 1
            items.append((ref, values))
        if config.log_every and (position + 1) % config.log_every == 0:
            print(
                f"processed {position + 1} samples; updated {stats['updated']} audio items",
                file=sys.stderr,
            )
        yield SampleManifestEntry(
            sample_id=entry.sample_id,
            sample_index=entry.sample_index,
            items=tuple(items),
        )


def _duration(item: AudioItem, config: Config) -> float:
    value = item.views.get(config.view)
    if value is None:
        raise ValueError(f"Audio item is missing {config.view.value!r} view.")
    if config.view is AudioView.WAVEFORM:
        return _waveform_duration(value)
    if config.frame_rate is None:
        raise ValueError("codec migration requires frame_rate.")
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("codec view must have shape [frame, codebook].")
    return float(shape[0]) / config.frame_rate


def _waveform_duration(value: object) -> float:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("waveform view must be a (waveform, sample_rate) tuple.")
    waveform, sample_rate = value
    sample_rate = _positive_int(sample_rate, name="sample_rate")
    shape = getattr(waveform, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("waveform must expose a non-empty shape.")
    return float(shape[-1]) / float(sample_rate)


def _pending_audio_items(store: Path, *, overwrite: bool) -> int:
    duration_key = AudioMeta.DURATION.value
    return sum(
        1
        for entry in read_samples_manifest(store)
        for ref, meta in entry.items
        if ref[1] is Modality.AUDIO and (overwrite or duration_key not in meta)
    )


def _validate_config(config: Config, store: Path) -> None:
    if not store.is_dir():
        raise FileNotFoundError(store)
    if config.view is AudioView.WAVEFORM:
        if config.frame_rate is not None:
            raise ValueError("waveform migration must not set frame_rate.")
    elif config.frame_rate is None:
        raise ValueError("codec migration requires frame_rate.")
    else:
        _positive_number(config.frame_rate)
    _positive_int(config.log_every, name="log_every")


def _default_backup(store: Path) -> Path:
    return store / "samples.before-duration.parquet"


def _positive_number(value: object) -> float:
    if isinstance(value, str):
        try:
            output = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("value must be a number") from exc
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise argparse.ArgumentTypeError("value must be a number")
    else:
        output = float(value)
    if not math.isfinite(output) or output <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return output


def _positive_int(value: object, *, name: str = "value") -> int:
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer.") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise argparse.ArgumentTypeError(f"{name} must be an integer.")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive.")
    return value


if __name__ == "__main__":
    main()
