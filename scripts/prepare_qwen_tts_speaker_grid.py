"""Materialize a small Qwen TTS speaker-grid store and export grouped wavs."""

from __future__ import annotations

import argparse
import json
import shutil
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample, TextItem, TextView

from zhuyin.datasets.qwen_tts_speech import (
    materialize_qwen_tts_speaker_grid,
    qwen_tts_speaker_grid,
)
from zhuyin.env import context, dynamic_home

DEFAULT_TEXTS = (
    "你好，这是 Qwen TTS 多说话人数据集的第一个测试样本。",
    "第二个样本用于检查按文本聚合后的不同 speaker 波形。",
)
DEFAULT_SPEAKERS = ("vivian", "ryan")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    with context():
        if args.overwrite:
            _remove_if_exists(args.output_dir)
            _remove_if_exists(args.export_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.export_dir.mkdir(parents=True, exist_ok=True)
        speakers = tuple(args.speaker_id)
        materialize_qwen_tts_speaker_grid(
            text_dataset_factory=TextDatasetFactory(tuple(args.text)),
            speaker_ids=speakers,
            output_dir=args.output_dir,
            split=args.split,
            model=args.model,
            default_language=args.language,
            batch_size=args.batch_size,
            devices=args.devices,
            load_options=_load_options(args),
            runtime_kwargs=_runtime_kwargs(args),
        )
        manifest = export_grouped_wavs(
            root=args.output_dir,
            export_dir=args.export_dir,
            speaker_ids=speakers,
            split=args.split,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


@dataclass(frozen=True)
class TextDatasetFactory:
    texts: tuple[str, ...]

    def __call__(self) -> TextDataset:
        return TextDataset(self.texts)


@dataclass(frozen=True)
class TextDataset:
    texts: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Sample:
        return {
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: self.texts[index]}
            )
        }


def export_grouped_wavs(
    *,
    root: Path,
    export_dir: Path,
    speaker_ids: tuple[str, ...],
    split: str,
) -> dict[str, Any]:
    dataset = qwen_tts_speaker_grid(root=root, speaker_ids=speaker_ids, split=split)
    samples: list[dict[str, Any]] = []
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        text = _text_item(sample[Role.DEFAULT, Modality.TEXT]).views[TextView.TEXT]
        audio = _audio_item(sample[Role.DEFAULT, Modality.AUDIO])
        waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        speakers = tuple(audio.views[AudioView.SPEAKERS])
        lengths = audio.views[AudioView.SPEAKER_LENGTHS].tolist()
        exports: list[dict[str, Any]] = []
        for speaker_index, speaker_id in enumerate(speakers):
            length = int(lengths[speaker_index])
            speaker_waveform = waveform[speaker_index, :, :length]
            filename = f"sample{sample_index:04d}_{_safe_name(speaker_id)}.wav"
            path = export_dir / filename
            write_wav(path, speaker_waveform, int(sample_rate))
            exports.append(
                {
                    "speaker_id": speaker_id,
                    "path": str(path),
                    "sample_rate": int(sample_rate),
                    "samples": length,
                }
            )
        samples.append({"sample_index": sample_index, "text": text, "exports": exports})
    manifest = {
        "root": str(root),
        "export_dir": str(export_dir),
        "speaker_ids": list(speaker_ids),
        "samples": samples,
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    value = waveform.detach().to("cpu", dtype=torch.float32)
    if value.ndim != 2:
        raise ValueError("waveform must have shape [channel, time].")
    value = (value.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    frames = value.transpose(0, 1).contiguous().numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(value.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def _text_item(value: object) -> TextItem:
    if not isinstance(value, TextItem):
        raise TypeError("sample text entry must be a TextItem.")
    return value


def _audio_item(value: object) -> AudioItem:
    if not isinstance(value, AudioItem):
        raise TypeError("sample audio entry must be an AudioItem.")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", type=_non_empty_str, default=None)
    parser.add_argument("--speaker-id", action="append", type=_non_empty_str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None or args.export_dir is None:
        default_root = dynamic_home() / "debug" / "qwen_tts_speaker_grid_smoke"
        if args.output_dir is None:
            args.output_dir = default_root / "store"
        if args.export_dir is None:
            args.export_dir = default_root / "wavs"
    if args.text is None:
        args.text = list(DEFAULT_TEXTS)
    if args.speaker_id is None:
        args.speaker_id = list(DEFAULT_SPEAKERS)
    if not args.text:
        raise ValueError("at least one --text is required.")
    if not args.speaker_id:
        raise ValueError("at least one --speaker-id is required.")
    return args


def _load_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    if args.device_map is not None:
        options["device_map"] = args.device_map
    if args.dtype is not None:
        options["torch_dtype"] = _torch_dtype(args.dtype)
    if args.attn_implementation is not None:
        options["attn_implementation"] = args.attn_implementation
    return options


def _runtime_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    return kwargs


def _torch_dtype(value: str) -> torch.dtype:
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype: {value!r}")
    return dtype


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _non_empty_str(value: str) -> str:
    if value == "":
        raise argparse.ArgumentTypeError("value must be non-empty.")
    return value


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _remove_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    main()
