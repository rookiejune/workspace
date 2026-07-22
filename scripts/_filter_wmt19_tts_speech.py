"""Apply speech-quality filtering to the prepared WMT19 TTS store.

The script consumes `wmt19_tts()`, writes filter metrics and summary reports,
and leaves synthesis artifacts untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anydataset import FilterRule
from anydataset.quality.speech import SpeechQuality, SpeechQualityProfile
from anytrain.evaluator.speech import SpeechEvaluator, UTMOSEvaluator, WhisperASREvaluator

from zhuyin.datasets._wmt19_tts_io import preview_metrics, write_json, write_metrics_jsonl
from zhuyin.datasets._wmt19_tts_store import StoreFactory, resolve_root
from zhuyin.env import context, static_home


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        configure_env(args)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        filter_summary = apply_speech_filter(args)
        summary = {
            "config": run_config(args),
            "filter": filter_summary,
            "seconds": time.perf_counter() - started_at,
        }
        write_json(args.reports_dir / "speech_filter_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def apply_speech_filter(args: argparse.Namespace) -> dict[str, Any]:
    dataset_factory = StoreFactory(args.root if args.root_explicit else None, args.split)
    rule = FilterRule(args.filter_rule_name, SpeechQualityFactory.from_args(args))
    start = time.perf_counter()
    result = rule.apply(
        dataset_factory=dataset_factory,
        metrics=True,
        device=args.filter_device,
        num_workers=args.num_workers,
        prefetch_factor=args.read_prefetch,
        commit_samples=args.filter_commit_samples,
        max_shard_samples=args.max_shard_samples,
        write_workers=args.write_workers,
        write_prefetch=args.write_prefetch,
    )
    metrics_report = args.reports_dir / "speech_quality_metrics.jsonl"
    write_metrics_jsonl(metrics_report, result.iter_metrics())
    return {
        "seconds": time.perf_counter() - start,
        "counts": dict(result.counts),
        "labels": list(result.labels),
        "accepted": result.counts.get("accept", 0),
        "rejected": result.counts.get("reject", 0),
        "cache_path": str(result.cache_path),
        "metrics_path": None if result.metrics_path is None else str(result.metrics_path),
        "metrics_jsonl": str(metrics_report),
        "preview": preview_metrics(metrics_report, limit=args.preview_metrics),
    }


@dataclass(frozen=True)
class SpeechQualityFactory:
    whisper_model: str
    quality_device: str | None
    whisper_root: Path
    min_utmos: float
    max_wer: float
    min_chrf: float
    max_seconds_per_text_unit: float
    min_peak_amplitude: float
    min_bleu: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SpeechQualityFactory:
        return cls(
            whisper_model=args.whisper_model,
            quality_device=args.quality_device,
            whisper_root=args.whisper_root,
            min_utmos=args.min_utmos,
            max_wer=args.max_wer,
            min_chrf=args.min_chrf,
            max_seconds_per_text_unit=args.max_seconds_per_text_unit,
            min_peak_amplitude=args.min_peak_amplitude,
            min_bleu=args.min_bleu,
        )

    def __call__(self) -> SpeechQuality:
        device = quality_device(self.quality_device)
        evaluator = SpeechEvaluator(
            asr=WhisperASREvaluator(
                model_name=self.whisper_model,
                device=device,
                download_root=self.whisper_root,
                decode_options={"temperature": 0.0},
            ),
            utmos=UTMOSEvaluator(
                device=device,
                backend_load_options={"trust_repo": True},
            ),
        )
        return SpeechQuality(
            profile=SpeechQualityProfile(
                min_utmos=self.min_utmos,
                max_wer=self.max_wer,
                min_chrf=self.min_chrf,
                max_seconds_per_text_unit=self.max_seconds_per_text_unit,
                min_peak_amplitude=self.min_peak_amplitude,
                min_bleu=self.min_bleu,
            ),
            evaluator=evaluator,
            decode_options={},
        )


def quality_device(override: str | None) -> str:
    if override is not None:
        return override
    device = os.environ.get("ANYDATASET_FILTER_DEVICE")
    if device is None:
        raise RuntimeError(
            "quality device is unset; pass --quality-device or run the filter through "
            "anydataset so ANYDATASET_FILTER_DEVICE is populated."
        )
    return device


def configure_env(args: argparse.Namespace) -> None:
    args.root_explicit = args.root is not None
    args.root = resolve_root(args.root)
    args.reports_dir = args.root / "reports"
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
        "filter_device": args.filter_device,
        "quality_device": args.quality_device,
        "whisper_model": args.whisper_model,
        "filter_rule_name": args.filter_rule_name,
        "num_workers": args.num_workers,
        "read_prefetch": args.read_prefetch,
        "write_workers": args.write_workers,
        "write_prefetch": args.write_prefetch,
        "thresholds": {
            "min_utmos": args.min_utmos,
            "max_wer": args.max_wer,
            "min_chrf": args.min_chrf,
            "max_seconds_per_text_unit": args.max_seconds_per_text_unit,
            "min_peak_amplitude": args.min_peak_amplitude,
            "min_bleu": args.min_bleu,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the prepared workspace WMT19 zh-en TTS dataset by speech quality."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--filter-device", default="auto")
    parser.add_argument("--quality-device")
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument("--min-utmos", type=float, default=2.8)
    parser.add_argument("--max-wer", type=float, default=None)
    parser.add_argument("--min-chrf", type=float, default=50.0)
    parser.add_argument("--max-seconds-per-text-unit", type=float, default=4.0)
    parser.add_argument("--min-peak-amplitude", type=float, default=0.05)
    parser.add_argument("--min-bleu", type=float, default=None)
    parser.add_argument(
        "--filter-rule-name",
        default="wmt19_zh_en_tts_speech_quality_v2_utmos28_chrf50_len4_peak005_zhsimp",
    )
    parser.add_argument("--filter-commit-samples", type=int, default=16)
    parser.add_argument("--preview-metrics", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--read-prefetch", dest="read_prefetch", type=int)
    parser.add_argument("--write-workers", type=int, default=1)
    parser.add_argument("--write-prefetch", type=int)
    parser.add_argument("--hf-endpoint")
    return parser.parse_args(argv)
