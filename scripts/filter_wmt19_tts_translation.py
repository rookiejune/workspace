"""Apply text-translation quality filtering to the prepared WMT19 TTS store.

The script consumes `wmt19_tts()`, builds cached translation-quality partitions,
writes audit metrics and summary reports, and leaves synthesis artifacts
untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anydataset import FilterRule
from anydataset.quality.translation import Predicate as TranslationQuality
from anydataset.quality.translation import Profile as TranslationQualityProfile

from zhuyin.datasets._wmt19_tts_io import preview_metrics, write_json, write_metrics_jsonl
from zhuyin.datasets._wmt19_tts_store import StoreFactory, resolve_root
from zhuyin.env import context


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        configure_env(args)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        filter_summary = apply_translation_filter(args)
        summary = {
            "config": run_config(args),
            "filter": filter_summary,
            "seconds": time.perf_counter() - started_at,
        }
        write_json(args.reports_dir / "translation_filter_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def apply_translation_filter(args: argparse.Namespace) -> dict[str, Any]:
    dataset_factory = StoreFactory(args.root if args.root_explicit else None, args.split)
    rule = FilterRule(args.filter_rule_name, TranslationQualityFactory.from_args(args))
    start = time.perf_counter()
    result = rule.apply(
        dataset_factory=dataset_factory,
        metrics=True,
        device=args.filter_device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.read_prefetch,
        commit_samples=args.filter_commit_samples,
        max_shard_samples=args.max_shard_samples,
        write_workers=args.write_workers,
        write_prefetch=args.write_prefetch,
    )
    metrics_report = args.reports_dir / "translation_quality_metrics.jsonl"
    write_metrics_jsonl(metrics_report, result.iter_metrics())
    return {
        "seconds": time.perf_counter() - start,
        "counts": dict(result.counts),
        "labels": list(result.labels),
        "selected_labels": list(args.selected_labels),
        "selected_count": sum(result.counts.get(label, 0) for label in args.selected_labels),
        "cache_path": str(result.cache_path),
        "metrics_path": None if result.metrics_path is None else str(result.metrics_path),
        "metrics_jsonl": str(metrics_report),
        "preview": preview_metrics(metrics_report, limit=args.preview_metrics),
    }


@dataclass(frozen=True)
class TranslationQualityFactory:
    source_lang: str
    target_lang: str
    min_chars: int
    review_min_ratio: float
    review_max_ratio: float
    reject_min_ratio: float
    reject_max_ratio: float
    min_script_ratio: float
    reject_script_ratio: float
    min_script_chars: int
    max_control_ratio: float
    max_repeated_run: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TranslationQualityFactory:
        return cls(
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            min_chars=args.min_chars,
            review_min_ratio=args.review_min_ratio,
            review_max_ratio=args.review_max_ratio,
            reject_min_ratio=args.reject_min_ratio,
            reject_max_ratio=args.reject_max_ratio,
            min_script_ratio=args.min_script_ratio,
            reject_script_ratio=args.reject_script_ratio,
            min_script_chars=args.min_script_chars,
            max_control_ratio=args.max_control_ratio,
            max_repeated_run=args.max_repeated_run,
        )

    def __call__(self) -> TranslationQuality:
        return TranslationQuality(
            TranslationQualityProfile(
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                min_chars=self.min_chars,
                review_min_ratio=self.review_min_ratio,
                review_max_ratio=self.review_max_ratio,
                reject_min_ratio=self.reject_min_ratio,
                reject_max_ratio=self.reject_max_ratio,
                min_script_ratio=self.min_script_ratio,
                reject_script_ratio=self.reject_script_ratio,
                min_script_chars=self.min_script_chars,
                max_control_ratio=self.max_control_ratio,
                max_repeated_run=self.max_repeated_run,
            )
        )


def configure_env(args: argparse.Namespace) -> None:
    args.root_explicit = args.root is not None
    args.root = resolve_root(args.root)
    args.reports_dir = args.root / "reports"


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "tts_store": str(args.root / "base"),
        "split": args.split,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "filter_rule_name": args.filter_rule_name,
        "filter_device": args.filter_device,
        "selected_labels": list(args.selected_labels),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "read_prefetch": args.read_prefetch,
        "write_workers": args.write_workers,
        "write_prefetch": args.write_prefetch,
        "thresholds": {
            "min_chars": args.min_chars,
            "review_min_ratio": args.review_min_ratio,
            "review_max_ratio": args.review_max_ratio,
            "reject_min_ratio": args.reject_min_ratio,
            "reject_max_ratio": args.reject_max_ratio,
            "min_script_ratio": args.min_script_ratio,
            "reject_script_ratio": args.reject_script_ratio,
            "min_script_chars": args.min_script_chars,
            "max_control_ratio": args.max_control_ratio,
            "max_repeated_run": args.max_repeated_run,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the prepared workspace WMT19 zh-en TTS dataset by translation quality."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--filter-device", default="cpu")
    parser.add_argument(
        "--filter-rule-name",
        default="wmt19_zh_en_translation_quality_rules_v1",
    )
    parser.add_argument(
        "--selected-labels",
        nargs="+",
        default=["clean", "usable"],
    )
    parser.add_argument("--filter-commit-samples", type=int, default=100_000)
    parser.add_argument("--preview-metrics", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--read-prefetch", dest="read_prefetch", type=int)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--write-prefetch", type=int)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--review-min-ratio", type=float, default=0.2)
    parser.add_argument("--review-max-ratio", type=float, default=6.0)
    parser.add_argument("--reject-min-ratio", type=float, default=0.05)
    parser.add_argument("--reject-max-ratio", type=float, default=20.0)
    parser.add_argument("--min-script-ratio", type=float, default=0.45)
    parser.add_argument("--reject-script-ratio", type=float, default=0.2)
    parser.add_argument("--min-script-chars", type=int, default=4)
    parser.add_argument("--max-control-ratio", type=float, default=0.02)
    parser.add_argument("--max-repeated-run", type=int, default=24)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
