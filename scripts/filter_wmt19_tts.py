"""Apply speech-quality filtering to the prepared WMT19 TTS store.

The script consumes `wmt19_tts()`, writes filter metrics and summary reports,
and leaves synthesis artifacts untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

from anydataset import AnyDataset, FilterRule
from anydataset.quality.speech import Predicate as SpeechQuality
from anydataset.quality.speech import Profile as SpeechQualityProfile
from anytrain.evaluator.speech import SpeechEvaluator, UTMOSEvaluator, WhisperASREvaluator

from zhuyin.datasets.wmt19_tts import WMT19_TTS, wmt19_tts
from zhuyin.env import configure_environment as configure_workspace_environment
from zhuyin.env import dataset_dir, whisper_root


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_env(args)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    filter_summary = apply_speech_filter(args)
    summary = {
        "config": run_config(args),
        "filter": filter_summary,
        "seconds": time.perf_counter() - started_at,
    }
    write_json(args.reports_dir / "filter_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def apply_speech_filter(args: argparse.Namespace) -> dict[str, Any]:
    dataset_factory = StoreFactory(args.root if args.root_explicit else None, args.split)
    rule = FilterRule(args.filter_rule_name, SpeechQualityFactory.from_args(args))
    start = time.perf_counter()
    result = rule.apply(
        dataset_factory=dataset_factory,
        metrics=True,
        num_workers=args.num_workers,
        commit_samples=args.filter_commit_samples,
        max_shard_samples=args.max_shard_samples,
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
class StoreFactory:
    path: Path | None
    split: str

    def __call__(self) -> AnyDataset:
        if self.path is None:
            return wmt19_tts(split=self.split)
        return wmt19_tts(dataset_dir=self.path, split=self.split)


@dataclass(frozen=True)
class SpeechQualityFactory:
    whisper_model: str
    quality_device: str
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
        evaluator = SpeechEvaluator(
            asr=WhisperASREvaluator(
                model_name=self.whisper_model,
                device=self.quality_device,
                download_root=self.whisper_root,
                decode_options={"temperature": 0.0},
            ),
            utmos=UTMOSEvaluator(
                device=self.quality_device,
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


def write_metrics_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def preview_metrics(path: Path, *, limit: int) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in islice(handle, limit):
            output.append(cast(Mapping[str, Any], json.loads(line)))
    return output


def configure_env(args: argparse.Namespace) -> None:
    configure_workspace_environment()
    args.root_explicit = args.root is not None
    args.root = dataset_dir(WMT19_TTS) if args.root is None else args.root
    args.root = args.root.expanduser().resolve()
    args.reports_dir = args.root / "reports"
    args.whisper_root = whisper_root()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "tts_store": str(args.root / "base"),
        "split": args.split,
        "quality_device": args.quality_device,
        "whisper_model": args.whisper_model,
        "filter_rule_name": args.filter_rule_name,
        "num_workers": args.num_workers,
        "thresholds": {
            "min_utmos": args.min_utmos,
            "max_wer": args.max_wer,
            "min_chrf": args.min_chrf,
            "max_seconds_per_text_unit": args.max_seconds_per_text_unit,
            "min_peak_amplitude": args.min_peak_amplitude,
            "min_bleu": args.min_bleu,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the prepared workspace WMT19 zh-en TTS dataset."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--quality-device", default="cuda:0")
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
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
