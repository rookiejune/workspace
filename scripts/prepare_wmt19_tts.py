"""Build the prepared WMT19 TTS stores used by workspace dataset loaders.

The script turns WMT19 text samples into source/target TTS audio stores. LongCat
codec materialization and speech-quality filtering are separate jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

import torch
from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    DatasetWriter,
    Modality,
    ModalityMaterializer,
    MapStyleABC,
    Preset,
    Role,
    Sample,
    Source,
    Spec,
    TextItem,
    TextView,
)
from anydataset.provider.moss_tts import MossTTSProvider
from anydataset.store.reader import read_store_dataset
from anytrain.tts import TTSOptions

from zhuyin.env import configure_environment as configure_workspace_environment
from zhuyin.env import dataset_dir, hf_home

WMT19_TTS = "wmt19_tts"
TTS_STORE_DIR = "base"


@dataclass(frozen=True)
class Stage:
    name: str
    path: str | None
    reused: bool
    seconds: float | None
    sample_count: int | None = None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_env(args)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    stage = write_tts_store(args)
    summary = {
        "config": run_config(args),
        "stage": asdict(stage),
        "seconds": time.perf_counter() - started_at,
    }
    write_json(args.reports_dir / "prepare_tts_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def write_tts_store(args: argparse.Namespace) -> Stage:
    if is_ready_store(args.tts_store):
        return ready_stage("tts", args.tts_store)

    start = time.perf_counter()
    work = args.root / "work" / "base"
    text_store = work / "text"
    source_audio_store = work / "source-audio"
    target_audio_store = work / "target-audio"

    if not is_ready_store(text_store):
        DatasetWriter(
            text_store,
            dataset_id=args.dataset_id,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
        ).write(
            limited_wmt19_samples(
                split=args.split,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                limit=args.limit,
            )
        )

    if not is_ready_store(source_audio_store):
        ModalityMaterializer(
            source_audio_store,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.tts_batch_size,
        ).write(
            dataset_factory=RoleTextStoreFactory(text_store, args.split, Role.SOURCE),
            provider_factory=MossTTSFactory(args, reference_role=None),
            devices=args.devices,
            resume=True,
        )

    if not is_ready_store(target_audio_store):
        ModalityMaterializer(
            target_audio_store,
            split=args.split,
            max_shard_samples=args.max_shard_samples,
            batch_size=args.tts_batch_size,
        ).write(
            dataset_factory=RoleTextMergedStoreFactory(
                text_store,
                source_audio_store,
                args.split,
                Role.TARGET,
            ),
            provider_factory=MossTTSFactory(args, reference_role=Role.SOURCE),
            devices=args.devices,
            resume=True,
        )

    source_tts = store_dataset(text_store, args.split).merge(
        store_dataset(source_audio_store, args.split)
    )
    source_tts.merge(store_dataset(target_audio_store, args.split)).write(
        args.tts_store,
        dataset_id=args.dataset_id,
        split=args.split,
        max_shard_samples=args.max_shard_samples,
    )
    return Stage(
        name="tts",
        path=str(args.tts_store),
        reused=False,
        seconds=time.perf_counter() - start,
        sample_count=len(read_store_dataset(args.tts_store)),
    )


@dataclass(frozen=True)
class RoleTextStoreFactory:
    path: Path
    split: str
    role: Role

    def __call__(self) -> RoleTextDataset:
        return RoleTextDataset(store_dataset(self.path, self.split), self.role)


@dataclass(frozen=True)
class RoleTextMergedStoreFactory:
    text: Path
    audio: Path
    split: str
    role: Role

    def __call__(self) -> RoleTextDataset:
        dataset = store_dataset(self.text, self.split).merge(
            store_dataset(self.audio, self.split)
        )
        return RoleTextDataset(dataset, self.role)


@dataclass(frozen=True)
class RoleTextDataset:
    dataset: MapStyleABC
    role: Role

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return role_text_sample(self.dataset[index], self.role)

    def __iter__(self) -> Iterable[Sample]:
        for index in range(len(self)):
            yield self[index]


@dataclass(frozen=True)
class MossTTSFactory:
    args: argparse.Namespace
    reference_role: Role | None

    def __call__(self, device: str) -> MossTTSProvider:
        return MossTTSProvider(
            resolve_pretrained_source(self.args.moss_model, self.args),
            options=tts_options(self.args),
            reference_role=self.reference_role,
            cache_dir=self.args.hf_home,
            codec_model=resolve_pretrained_source(self.args.moss_codec_model, self.args),
            device=device if device != "cpu" else None,
            local_files_only=self.args.local_files_only,
            trust_remote_code=self.args.trust_remote_code,
            dtype=self.args.tts_dtype,
            attn_implementation=self.args.tts_attn_implementation,
            runtime_kwargs=tts_runtime_kwargs(self.args),
        )


def limited_wmt19_samples(
    *,
    split: str,
    source_lang: str,
    target_lang: str,
    limit: int,
) -> Iterable[Sample]:
    dataset = Preset.WMT19.create(
        split=split,
        source_lang=source_lang,
        target_lang=target_lang,
        streaming=True,
    )
    yield from islice(dataset, limit)


def role_text_dataset(dataset: Iterable[Sample], role: Role) -> Iterable[Sample]:
    for sample in dataset:
        yield role_text_sample(sample, role)


def role_text_sample(sample: Sample, role: Role) -> Sample:
    output: dict[tuple[Role, Modality], Any] = {
        (role, Modality.TEXT): sample[role, Modality.TEXT],
    }
    if role is Role.TARGET:
        output[Role.SOURCE, Modality.AUDIO] = sample[Role.SOURCE, Modality.AUDIO]
    return cast(Sample, output)


def store_dataset(path: Path, split: str) -> AnyDataset:
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


def tts_options(args: argparse.Namespace) -> TTSOptions:
    kwargs: dict[str, Any] = {
        "sample_rate": args.tts_sample_rate,
        "max_new_tokens": args.tts_max_new_tokens,
        "temperature": args.tts_temperature,
        "top_p": args.tts_top_p,
        "seed": args.tts_seed,
    }
    return TTSOptions(**{key: value for key, value in kwargs.items() if value is not None})


def tts_runtime_kwargs(args: argparse.Namespace) -> dict[str, object]:
    output: dict[str, object] = {"do_sample": args.tts_do_sample}
    return {key: value for key, value in output.items() if value is not None}


def resolve_pretrained_source(source: str, args: argparse.Namespace) -> str:
    path = Path(source).expanduser()
    if path.exists() or not args.local_files_only:
        return str(path) if path.exists() else source

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return source

    return snapshot_download(
        source,
        cache_dir=str(args.hf_home / "hub"),
        local_files_only=True,
    )


def inspect_sample(path: Path, split: str) -> dict[str, Any]:
    dataset = store_dataset(path, split)
    sample = dataset[0]
    output: dict[str, Any] = {}
    for role in (Role.SOURCE, Role.TARGET):
        text = cast(TextItem, sample[role, Modality.TEXT])
        audio = cast(AudioItem, sample[role, Modality.AUDIO])
        waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        output[role.value] = {
            "text": text.views[TextView.TEXT],
            "sample_rate": int(sample_rate),
            "waveform_shape": list(torch.as_tensor(waveform).shape),
        }
    return output


def configure_env(args: argparse.Namespace) -> None:
    configure_workspace_environment()
    args.root = dataset_dir(WMT19_TTS) if args.root is None else args.root
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    args.tts_store = args.root / TTS_STORE_DIR
    args.reports_dir = args.root / "reports"
    args.hf_home = hf_home()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "tts_store": str(args.tts_store),
        "split": args.split,
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "limit": args.limit,
        "devices": args.devices,
        "tts_batch_size": args.tts_batch_size,
        "moss_model": args.moss_model,
        "moss_codec_model": args.moss_codec_model,
        "first_sample": inspect_sample(args.tts_store, args.split)
        if is_ready_store(args.tts_store)
        else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the workspace WMT19 zh-en TTS dataset."
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-lang", default="zh")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--dataset-id", default=WMT19_TTS)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-shard-samples", type=int, default=100_000)
    parser.add_argument("--tts-batch-size", type=int, default=1)
    parser.add_argument("--moss-model", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    parser.add_argument("--moss-codec-model", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--tts-sample-rate", type=int, default=None)
    parser.add_argument("--tts-max-new-tokens", type=int, default=None)
    parser.add_argument("--tts-temperature", type=float, default=None)
    parser.add_argument("--tts-top-p", type=float, default=None)
    parser.add_argument("--tts-seed", type=int, default=None)
    parser.add_argument("--tts-do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tts-dtype", default="auto")
    parser.add_argument("--tts-attn-implementation", default="sdpa")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
