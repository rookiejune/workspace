"""Apply chained translation and speech quality filters to WMT19 TTS.

The script composes the reusable text-translation and speech-quality predicates
as cached anydataset filter stages. Reports live under the WMT19 TTS reports
root resolved by workspace environment settings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from enum import auto
from pathlib import Path
from typing import Any

if __package__:
    from . import _filter_wmt19_tts_speech as speech_filter
    from . import _filter_wmt19_tts_translation as translation_filter
else:
    import _filter_wmt19_tts_speech as speech_filter
    import _filter_wmt19_tts_translation as translation_filter
from anydataset import FilterRule
from anydataset.filter import FilteredDataset

from zhuyin._compat import StrEnum
from zhuyin.datasets._wmt19_tts_io import preview_metrics, write_json, write_metrics_jsonl
from zhuyin.datasets._wmt19_tts_store import StoreFactory, resolve_root
from zhuyin.env import context, static_home


class Stage(StrEnum):
    TRANSLATION = auto()
    SPEECH = auto()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        configure_env(args)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        stages = apply_filters(args)
        summary = {
            "config": run_config(args),
            "stages": stages,
            "final": stages[-1],
            "seconds": time.perf_counter() - started_at,
        }
        write_json(args.reports_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def apply_filters(args: argparse.Namespace) -> list[dict[str, Any]]:
    factory: Any = StoreFactory(
        args.root if args.root_explicit else None,
        args.split,
    )
    summaries: list[dict[str, Any]] = []
    for stage in args.order:
        if stage is Stage.TRANSLATION:
            started_at = time.perf_counter()
            result = apply_translation_stage(args, factory)
            summaries.append(
                translation_summary(
                    args,
                    result,
                    seconds=time.perf_counter() - started_at,
                )
            )
            factory = result.select_by(*args.translation_labels).dataset_factory
        if stage is Stage.SPEECH:
            started_at = time.perf_counter()
            result = apply_speech_stage(args, factory)
            summaries.append(
                speech_summary(
                    args,
                    result,
                    seconds=time.perf_counter() - started_at,
                )
            )
            factory = result.select_by("accept").dataset_factory
    return summaries


def apply_translation_stage(
    args: argparse.Namespace,
    dataset_factory: Any,
) -> FilteredDataset:
    rule = FilterRule(
        args.translation_rule_name,
        translation_filter.TranslationQualityFactory.from_args(args),
    )
    return rule.apply(
        dataset_factory=dataset_factory,
        metrics=True,
        device=args.translation_device,
        batch_size=args.translation_batch_size,
        num_workers=args.translation_num_workers,
        prefetch_factor=args.translation_read_prefetch,
        commit_samples=args.translation_commit_samples,
        max_shard_samples=args.max_shard_samples,
        write_workers=args.translation_write_workers,
        write_prefetch=args.translation_write_prefetch,
    )


def apply_speech_stage(
    args: argparse.Namespace,
    dataset_factory: Any,
) -> FilteredDataset:
    rule = FilterRule(
        args.speech_rule_name,
        speech_filter.SpeechQualityFactory.from_args(args),
    )
    return rule.apply(
        dataset_factory=dataset_factory,
        metrics=True,
        device=args.speech_filter_device,
        num_workers=args.speech_num_workers,
        prefetch_factor=args.speech_read_prefetch,
        commit_samples=args.speech_commit_samples,
        max_shard_samples=args.max_shard_samples,
        write_workers=args.speech_write_workers,
        write_prefetch=args.speech_write_prefetch,
    )


def translation_summary(
    args: argparse.Namespace,
    result: FilteredDataset,
    *,
    seconds: float,
) -> dict[str, Any]:
    metrics_report = args.reports_dir / "translation_quality_metrics.jsonl"
    write_metrics_jsonl(metrics_report, result.iter_metrics())
    return {
        "name": Stage.TRANSLATION.value,
        "rule_name": args.translation_rule_name,
        "seconds": seconds,
        "counts": dict(result.counts),
        "labels": list(result.labels),
        "selected_labels": list(args.translation_labels),
        "selected_count": sum(result.counts.get(label, 0) for label in args.translation_labels),
        "cache_path": str(result.cache_path),
        "metrics_path": None if result.metrics_path is None else str(result.metrics_path),
        "metrics_jsonl": str(metrics_report),
        "preview": preview_metrics(metrics_report, limit=args.preview_metrics),
    }


def speech_summary(
    args: argparse.Namespace,
    result: FilteredDataset,
    *,
    seconds: float,
) -> dict[str, Any]:
    metrics_report = args.reports_dir / "speech_quality_metrics.jsonl"
    write_metrics_jsonl(metrics_report, result.iter_metrics())
    return {
        "name": Stage.SPEECH.value,
        "rule_name": args.speech_rule_name,
        "seconds": seconds,
        "counts": dict(result.counts),
        "labels": list(result.labels),
        "selected_labels": ["accept"],
        "selected_count": result.counts.get("accept", 0),
        "cache_path": str(result.cache_path),
        "metrics_path": None if result.metrics_path is None else str(result.metrics_path),
        "metrics_jsonl": str(metrics_report),
        "preview": preview_metrics(metrics_report, limit=args.preview_metrics),
    }


def configure_env(args: argparse.Namespace) -> None:
    args.root_explicit = args.root is not None
    args.root = resolve_root(args.root)
    args.reports_dir = args.root / "reports" / "speech_translation"
    args.whisper_root = _env_path(
        "ANYTRAIN_WHISPER_ROOT",
        static_home() / "whisper",
    )
    if args.hf_endpoint is not None:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None:
        return default
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return Path(value).expanduser()


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "tts_store": str(args.root / "base"),
        "split": args.split,
        "order": [stage.value for stage in args.order],
        "translation": {
            "rule_name": args.translation_rule_name,
            "device": args.translation_device,
            "selected_labels": list(args.translation_labels),
            "batch_size": args.translation_batch_size,
            "num_workers": args.translation_num_workers,
            "read_prefetch": args.translation_read_prefetch,
            "write_workers": args.translation_write_workers,
            "write_prefetch": args.translation_write_prefetch,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "thresholds": {
                "review_min_ratio": args.review_min_ratio,
                "review_max_ratio": args.review_max_ratio,
                "reject_min_ratio": args.reject_min_ratio,
                "reject_max_ratio": args.reject_max_ratio,
                "min_identical_script_chars": args.min_identical_script_chars,
            },
        },
        "speech": {
            "rule_name": args.speech_rule_name,
            "filter_device": args.speech_filter_device,
            "quality_device": args.quality_device,
            "whisper_model": args.whisper_model,
            "num_workers": args.speech_num_workers,
            "read_prefetch": args.speech_read_prefetch,
            "write_workers": args.speech_write_workers,
            "write_prefetch": args.speech_write_prefetch,
            "thresholds": {
                "min_utmos": args.min_utmos,
                "max_wer": args.max_wer,
                "min_chrf": args.min_chrf,
                "max_seconds_per_text_unit": args.max_seconds_per_text_unit,
                "min_peak_amplitude": args.min_peak_amplitude,
                "min_bleu": args.min_bleu,
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter WMT19 zh-en TTS by chained translation and speech quality."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--order", type=parse_order, default=parse_order("translation,speech"))
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--preview-metrics", type=int, default=5)
    parser.add_argument("--hf-endpoint")

    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--translation-device", default="cpu")
    parser.add_argument(
        "--translation-rule-name",
        default="wmt19_zh_en_translation_quality_rules_v1",
    )
    parser.add_argument("--translation-labels", nargs="+", default=["accept"])
    parser.add_argument("--translation-commit-samples", type=int, default=100_000)
    parser.add_argument("--translation-batch-size", type=int, default=1)
    parser.add_argument("--translation-num-workers", type=int, default=0)
    parser.add_argument(
        "--translation-read-prefetch",
        dest="translation_read_prefetch",
        type=int,
    )
    parser.add_argument("--translation-write-workers", type=int, default=1)
    parser.add_argument("--translation-write-prefetch", type=int)
    parser.add_argument("--review-min-ratio", type=float, default=0.2)
    parser.add_argument("--review-max-ratio", type=float, default=6.0)
    parser.add_argument("--reject-min-ratio", type=float, default=0.05)
    parser.add_argument("--reject-max-ratio", type=float, default=20.0)
    parser.add_argument("--min-identical-script-chars", type=int, default=4)

    parser.add_argument("--speech-filter-device", default="auto")
    parser.add_argument("--quality-device")
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument("--min-utmos", type=float, default=2.8)
    parser.add_argument("--max-wer", type=float, default=None)
    parser.add_argument("--min-chrf", type=float, default=50.0)
    parser.add_argument("--max-seconds-per-text-unit", type=float, default=4.0)
    parser.add_argument("--min-peak-amplitude", type=float, default=0.05)
    parser.add_argument("--min-bleu", type=float, default=None)
    parser.add_argument(
        "--speech-rule-name",
        default="wmt19_zh_en_tts_speech_quality_v2_utmos28_chrf50_len4_peak005_zhsimp",
    )
    parser.add_argument("--speech-commit-samples", type=int, default=16)
    parser.add_argument("--speech-num-workers", type=int, default=1)
    parser.add_argument("--speech-read-prefetch", dest="speech_read_prefetch", type=int)
    parser.add_argument("--speech-write-workers", type=int, default=1)
    parser.add_argument("--speech-write-prefetch", type=int)
    return parser.parse_args(argv)


def parse_order(value: str) -> tuple[Stage, ...]:
    stages = tuple(Stage(item.strip()) for item in value.split(",") if item.strip())
    if stages not in (
        (Stage.TRANSLATION, Stage.SPEECH),
        (Stage.SPEECH, Stage.TRANSLATION),
    ):
        raise argparse.ArgumentTypeError(
            "order must be `translation,speech` or `speech,translation`."
        )
    return stages
