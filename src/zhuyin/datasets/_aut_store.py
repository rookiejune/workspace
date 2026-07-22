"""Strict reader for prepared WMT19 TTS audio-tower artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, cast

import torch
from torch.utils.data import Dataset

from zhuyin.datasets.aut import (
    CHECKPOINT,
    FEATURE_DIM,
    SAMPLE_RATE,
    TRANSFORMERS_VERSION,
    Sample,
)
from zhuyin.env import datasets_home

_SCHEMA_VERSION = 1
_DATASET = "wmt19_tts"
_MODEL_DIR = "qwen2_5-omni-7b"
_FAMILY = "qwen2_5_omni"
_FEATURE_NAME = "audio_tower_final_projected"
_TIMING_NAME = "qwen2_5_conv2_pool2_v1"
_MEL_RATE = 100
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MANIFEST_KEYS = {
    "schema_version",
    "sample_count",
    "dataset",
    "teacher",
    "timing",
    "audio",
    "storage",
    "fields",
}
_DATASET_KEYS = {"name", "split"}
_TEACHER_KEYS = {
    "family",
    "checkpoint",
    "revision",
    "transformers_version",
    "feature_name",
    "feature_dim",
    "feature_dtype",
}
_TIMING_KEYS = {
    "name",
    "mel_rate_hz",
    "frame_reference",
    "mel_length_rounding",
    "conv_stride",
    "pool_stride",
}
_AUDIO_KEYS = {"sample_rate", "channels", "dtype"}
_STORAGE_KEYS = {"format", "index"}
_FIELD_SCHEMA = {
    "waveform": "float32[1,time]",
    "waveform_length": "int64[]",
    "aut_features": f"float32[feature_frame,{FEATURE_DIM}]",
    "aut_feature_mask": "bool[feature_frame]",
    "audio_placeholders": "int64[]",
    "sample_id": "index:string",
    "audio_sha256": "index:sha256",
}
_PAYLOAD_KEYS = {
    "waveform",
    "waveform_length",
    "aut_features",
    "aut_feature_mask",
    "audio_placeholders",
}
_INDEX_KEYS = {"sample_id", "audio_sha256", "path"}


class _Record(TypedDict):
    sample_id: str
    audio_sha256: str
    path: Path


class PreparedAuTStore(Dataset[Sample]):
    """Map-style dataset backed by individually serialized tensor payloads."""

    def __init__(
        self,
        *,
        root: str | PathLike[str] | None,
        split: str,
        revision: str,
    ) -> None:
        _component(split, "split")
        _component(revision, "revision")
        dataset_root = (
            Path(root).expanduser()
            if root is not None
            else datasets_home() / "prepared_aut" / _DATASET
        )
        self.root = dataset_root / _MODEL_DIR / revision / split
        self.split = split
        self.revision = revision
        _require_file(self.root / ".ready", "prepared AuT ready marker")
        count = _manifest(self.root / "manifest.json", split=split, revision=revision)
        self._records = _index(self.root, count=count)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Sample:
        record = self._records[index]
        try:
            value = torch.load(record["path"], map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(
                f"failed to load prepared AuT sample {record['sample_id']!r} from {record['path']}"
            ) from error
        payload = _object(value, _PAYLOAD_KEYS, f"sample {record['sample_id']!r} payload")
        return _sample(payload, record)


def _manifest(path: Path, *, split: str, revision: str) -> int:
    value = _object(_json(path), _MANIFEST_KEYS, "prepared AuT manifest")
    _equal(value["schema_version"], _SCHEMA_VERSION, "manifest schema_version")
    count = _integer(value["sample_count"], "manifest sample_count", minimum=0)

    dataset = _object(value["dataset"], _DATASET_KEYS, "manifest dataset")
    _equal(dataset["name"], _DATASET, "manifest dataset.name")
    _equal(dataset["split"], split, "manifest dataset.split")

    teacher = _object(value["teacher"], _TEACHER_KEYS, "manifest teacher")
    expected_teacher: dict[str, object] = {
        "family": _FAMILY,
        "checkpoint": CHECKPOINT,
        "revision": revision,
        "transformers_version": TRANSFORMERS_VERSION,
        "feature_name": _FEATURE_NAME,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
    }
    _fixed(teacher, expected_teacher, "manifest teacher")

    timing = _object(value["timing"], _TIMING_KEYS, "manifest timing")
    expected_timing: dict[str, object] = {
        "name": _TIMING_NAME,
        "mel_rate_hz": _MEL_RATE,
        "frame_reference": "end",
        "mel_length_rounding": "ceil",
        "conv_stride": 2,
        "pool_stride": 2,
    }
    _fixed(timing, expected_timing, "manifest timing")

    audio = _object(value["audio"], _AUDIO_KEYS, "manifest audio")
    _fixed(
        audio,
        {"sample_rate": SAMPLE_RATE, "channels": 1, "dtype": "float32"},
        "manifest audio",
    )
    storage = _object(value["storage"], _STORAGE_KEYS, "manifest storage")
    _fixed(
        storage,
        {"format": "torch_weights_only_v1", "index": "samples.jsonl"},
        "manifest storage",
    )
    fields = _object(value["fields"], set(_FIELD_SCHEMA), "manifest fields")
    _fixed(fields, _FIELD_SCHEMA, "manifest fields")
    return count


def _index(root: Path, *, count: int) -> list[_Record]:
    samples = root / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(f"prepared AuT samples directory not found: {samples}")
    samples_root = samples.resolve()
    path = root / "samples.jsonl"
    _require_file(path, "prepared AuT sample index")
    records: list[_Record] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank prepared AuT index record at {path}:{line_number}")
            try:
                raw = json.loads(line, object_pairs_hook=_unique_object)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid prepared AuT index JSON at {path}:{line_number}: {error}"
                ) from error
            value = _object(raw, _INDEX_KEYS, f"prepared AuT index record {line_number}")
            sample_id = _string(value["sample_id"], f"index sample_id at line {line_number}")
            if not sample_id:
                raise ValueError(f"index sample_id at line {line_number} must not be empty")
            if sample_id in identifiers:
                raise ValueError(
                    f"duplicate prepared AuT sample_id {sample_id!r} at line {line_number}"
                )
            identifiers.add(sample_id)
            digest = _string(value["audio_sha256"], f"index audio_sha256 at line {line_number}")
            if _SHA256.fullmatch(digest) is None:
                raise ValueError(
                    f"index audio_sha256 at line {line_number} must be 64 lowercase hex characters"
                )
            sample_path = _payload_path(
                root,
                samples_root,
                value["path"],
                line_number=line_number,
            )
            records.append({"sample_id": sample_id, "audio_sha256": digest, "path": sample_path})
    if len(records) != count:
        raise ValueError(
            f"prepared AuT manifest sample_count is {count}, but index contains {len(records)} records"
        )
    return records


def _payload_path(
    root: Path,
    samples_root: Path,
    value: object,
    *,
    line_number: int,
) -> Path:
    text = _string(value, f"index path at line {line_number}")
    relative = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "samples"
        or relative.name in {"", ".", ".."}
        or relative.suffix != ".pt"
        or relative.as_posix() != text
    ):
        raise ValueError(
            f"index path at line {line_number} must be a direct samples/*.pt relative path"
        )
    path = root.joinpath(*relative.parts)
    _require_file(path, f"prepared AuT payload at index line {line_number}")
    resolved = path.resolve()
    try:
        resolved.relative_to(samples_root)
    except ValueError as error:
        raise ValueError(
            f"index path at line {line_number} escapes the samples directory"
        ) from error
    return resolved


def _sample(payload: Mapping[str, object], record: _Record) -> Sample:
    name = f"sample {record['sample_id']!r}"
    waveform = _tensor(payload["waveform"], f"{name} waveform")
    if waveform.dtype is not torch.float32:
        raise TypeError(f"{name} waveform must use torch.float32")
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError(f"{name} waveform must have shape [1, time] for mono audio")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{name} waveform must contain only finite values")

    waveform_length = _scalar_int64(payload["waveform_length"], f"{name} waveform_length")
    length = int(waveform_length.item())
    if length <= 0:
        raise ValueError(f"{name} waveform_length must be positive")
    if waveform.size(1) != length:
        raise ValueError(
            f"{name} waveform_length is {length}, but waveform contains {waveform.size(1)} frames"
        )

    features = _tensor(payload["aut_features"], f"{name} aut_features")
    if features.dtype is not torch.float32:
        raise TypeError(f"{name} aut_features must use torch.float32")
    if features.dim() != 2 or features.size(1) != FEATURE_DIM:
        raise ValueError(f"{name} aut_features must have shape [feature_frame, {FEATURE_DIM}]")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError(f"{name} aut_features must contain only finite values")

    mask = _tensor(payload["aut_feature_mask"], f"{name} aut_feature_mask")
    if mask.dtype is not torch.bool:
        raise TypeError(f"{name} aut_feature_mask must use torch.bool")
    if mask.dim() != 1 or mask.size(0) != features.size(0):
        raise ValueError(f"{name} aut_feature_mask must align with aut_features")
    if not bool(mask.all().item()):
        raise ValueError(f"{name} aut_feature_mask must contain one contiguous all-true frame span")

    placeholders = _scalar_int64(payload["audio_placeholders"], f"{name} audio_placeholders")
    placeholder_count = int(placeholders.item())
    if placeholder_count <= 0:
        raise ValueError(f"{name} audio_placeholders must be positive")
    valid_frames = int(mask.sum().item())
    expected_frames = _feature_frames(length)
    if expected_frames <= 0:
        raise ValueError(f"{name} Qwen2.5 timing must produce at least one AuT frame")
    if valid_frames != placeholder_count or placeholder_count != expected_frames:
        raise ValueError(
            f"{name} AuT frame count mismatch: mask={valid_frames}, "
            f"audio_placeholders={placeholder_count}, timing={expected_frames}"
        )

    digest = hashlib.sha256(waveform.detach().contiguous().numpy().tobytes(order="C")).hexdigest()
    if digest != record["audio_sha256"]:
        raise ValueError(
            f"{name} waveform SHA-256 mismatch: index={record['audio_sha256']}, payload={digest}"
        )
    return {
        "sample_id": record["sample_id"],
        "audio_sha256": record["audio_sha256"],
        "waveform": waveform,
        "waveform_length": waveform_length,
        "sample_rate": SAMPLE_RATE,
        "aut_features": features,
        "aut_feature_mask": mask,
        "audio_placeholders": placeholders,
    }


def _feature_frames(waveform_length: int) -> int:
    mel_length = (waveform_length * _MEL_RATE + SAMPLE_RATE - 1) // SAMPLE_RATE
    after_conv = (mel_length + 1) // 2
    return after_conv // 2


def _json(path: Path) -> object:
    _require_file(path, "prepared AuT manifest")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual, key=repr)
        extra = sorted(actual - keys, key=repr)
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}")
    return cast(dict[str, object], value)


def _fixed(value: Mapping[str, object], expected: Mapping[str, object], label: str) -> None:
    for key, target in expected.items():
        _equal(value[key], target, f"{label}.{key}")


def _equal(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{label} must be {expected!r}, got {value!r}")


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    result = cast(int, value)
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return cast(str, value)


def _tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a Tensor")
    return value


def _scalar_int64(value: object, label: str) -> torch.Tensor:
    tensor = _tensor(value, label)
    if tensor.dtype is not torch.int64 or tensor.dim() != 0:
        raise TypeError(f"{label} must be a scalar torch.int64 Tensor")
    return tensor


def _component(value: str, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be one non-empty path component")
    if label == "revision" and _REVISION.fullmatch(value) is None:
        raise ValueError("revision must be a 40-character lowercase hexadecimal commit hash")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


__all__ = ["PreparedAuTStore"]
