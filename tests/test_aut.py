from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from zhuyin.datasets.aut import (
    CHECKPOINT,
    DEFAULT_REVISION,
    FEATURE_DIM,
    SAMPLE_RATE,
    TRANSFORMERS_VERSION,
    prepared_aut,
)

_MODEL_DIR = "qwen2_5-omni-7b"


def _payload(length: int = 1_600) -> dict[str, torch.Tensor]:
    waveform = torch.linspace(-0.5, 0.5, length, dtype=torch.float32).unsqueeze(0)
    mel_length = (length * 100 + SAMPLE_RATE - 1) // SAMPLE_RATE
    feature_frames = ((mel_length + 1) // 2) // 2
    return {
        "waveform": waveform,
        "waveform_length": torch.tensor(length, dtype=torch.int64),
        "aut_features": torch.zeros(feature_frames, FEATURE_DIM, dtype=torch.float32),
        "aut_feature_mask": torch.ones(feature_frames, dtype=torch.bool),
        "audio_placeholders": torch.tensor(feature_frames, dtype=torch.int64),
    }


def _digest(waveform: torch.Tensor) -> str:
    return hashlib.sha256(waveform.contiguous().numpy().tobytes(order="C")).hexdigest()


def _manifest(*, split: str, revision: str, count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_count": count,
        "dataset": {"name": "wmt19_tts", "split": split},
        "teacher": {
            "family": "qwen2_5_omni",
            "checkpoint": CHECKPOINT,
            "revision": revision,
            "transformers_version": TRANSFORMERS_VERSION,
            "feature_name": "audio_tower_final_projected",
            "feature_dim": FEATURE_DIM,
            "feature_dtype": "float32",
        },
        "timing": {
            "name": "qwen2_5_conv2_pool2_v1",
            "mel_rate_hz": 100,
            "frame_reference": "end",
            "mel_length_rounding": "ceil",
            "conv_stride": 2,
            "pool_stride": 2,
        },
        "audio": {"sample_rate": SAMPLE_RATE, "channels": 1, "dtype": "float32"},
        "storage": {"format": "torch_weights_only_v1", "index": "samples.jsonl"},
        "fields": {
            "waveform": "float32[1,time]",
            "waveform_length": "int64[]",
            "aut_features": f"float32[feature_frame,{FEATURE_DIM}]",
            "aut_feature_mask": "bool[feature_frame]",
            "audio_placeholders": "int64[]",
            "sample_id": "index:string",
            "audio_sha256": "index:sha256",
        },
    }


def _store(
    root: Path,
    *,
    split: str = "train",
    revision: str = DEFAULT_REVISION,
    payloads: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> Path:
    values = [_payload()] if payloads is None else payloads
    path = root / _MODEL_DIR / revision / split
    samples = path / "samples"
    samples.mkdir(parents=True)
    generated: list[dict[str, Any]] = []
    for index, payload in enumerate(values):
        relative = f"samples/{index:06d}.pt"
        torch.save(payload, path / relative)
        generated.append(
            {
                "sample_id": f"sample-{index}",
                "audio_sha256": _digest(payload["waveform"]),
                "path": relative,
            }
        )
    (path / "manifest.json").write_text(
        json.dumps(
            _manifest(split=split, revision=revision, count=len(values))
            if manifest is None
            else manifest
        ),
        encoding="utf-8",
    )
    rows = records if records is not None else generated
    (path / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (path / ".ready").touch()
    return path


def test_prepared_aut_loads_valid_map_style_store(tmp_path: Path) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    _store(root)

    dataset = prepared_aut(root=root)
    sample = dataset[0]

    assert len(dataset) == 1
    assert set(sample) == {
        "sample_id",
        "audio_sha256",
        "waveform",
        "waveform_length",
        "sample_rate",
        "aut_features",
        "aut_feature_mask",
        "audio_placeholders",
    }
    assert sample["sample_id"] == "sample-0"
    assert sample["audio_sha256"] == _digest(sample["waveform"])
    assert sample["waveform"].shape == (1, 1_600)
    assert sample["waveform"].dtype is torch.float32
    assert sample["waveform_length"].shape == ()
    assert sample["waveform_length"].dtype is torch.int64
    assert sample["sample_rate"] == SAMPLE_RATE
    assert sample["aut_features"].shape == (2, FEATURE_DIM)
    assert sample["aut_feature_mask"].tolist() == [True, True]
    assert sample["audio_placeholders"].item() == 2


def test_prepared_aut_default_root_uses_static_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets" / "prepared_aut" / "wmt19_tts"
    _store(root, split="dev")
    monkeypatch.setenv("STATIC_HOME", str(tmp_path))

    dataset = prepared_aut(split="dev")

    assert dataset[0]["sample_id"] == "sample-0"


def test_prepared_aut_uses_weights_only_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zhuyin.datasets._aut_store as store_module

    root = tmp_path / "prepared_aut" / "wmt19_tts"
    _store(root)
    calls: list[dict[str, object]] = []
    load = torch.load

    def tracked(path: Path, **kwargs: object) -> object:
        calls.append(kwargs)
        return load(path, **kwargs)

    monkeypatch.setattr(store_module.torch, "load", tracked)

    prepared_aut(root=root)[0]

    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=True), "keys differ"),
        (lambda value: value["teacher"].update(checkpoint="other/model"), "checkpoint"),
        (
            lambda value: value["teacher"].update(transformers_version="5.15.0"),
            "transformers_version",
        ),
        (lambda value: value["teacher"].update(feature_dim=1024), "feature_dim"),
        (lambda value: value["teacher"].update(feature_name="audio_tower_output"), "feature_name"),
        (lambda value: value["timing"].update(name="qwen2_5"), "timing.name"),
        (lambda value: value["audio"].update(sample_rate=24_000), "audio.sample_rate"),
        (lambda value: value["storage"].update(format="torch_pt"), "storage.format"),
        (lambda value: value["fields"].update(path="string"), "fields keys differ"),
    ],
)
def test_prepared_aut_rejects_manifest_schema_changes(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    manifest = _manifest(split="train", revision=DEFAULT_REVISION, count=1)
    mutate(manifest)
    _store(root, manifest=manifest)

    with pytest.raises(ValueError, match=match):
        prepared_aut(root=root)


def test_prepared_aut_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    payloads = [_payload(), _payload()]
    digest = _digest(payloads[0]["waveform"])
    records = [
        {"sample_id": "same", "audio_sha256": digest, "path": "samples/000000.pt"},
        {"sample_id": "same", "audio_sha256": digest, "path": "samples/000001.pt"},
    ]
    _store(root, payloads=payloads, records=records)

    with pytest.raises(ValueError, match="duplicate prepared AuT sample_id"):
        prepared_aut(root=root)


@pytest.mark.parametrize(
    "path",
    ["../outside.pt", "samples/../outside.pt", "/samples/000000.pt", "samples/nested/000000.pt"],
)
def test_prepared_aut_rejects_index_path_traversal(tmp_path: Path, path: str) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    payload = _payload()
    records = [
        {"sample_id": "sample-0", "audio_sha256": _digest(payload["waveform"]), "path": path}
    ]
    _store(root, payloads=[payload], records=records)

    with pytest.raises(ValueError, match=r"samples/\*\.pt"):
        prepared_aut(root=root)


def test_prepared_aut_rejects_index_extra_keys_and_count_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "extra" / "wmt19_tts"
    payload = _payload()
    record = {
        "sample_id": "sample-0",
        "audio_sha256": _digest(payload["waveform"]),
        "path": "samples/000000.pt",
        "text": "must not be indexed",
    }
    _store(root, payloads=[payload], records=[record])
    with pytest.raises(ValueError, match="index record 1 keys differ"):
        prepared_aut(root=root)

    root = tmp_path / "count" / "wmt19_tts"
    _store(
        root,
        manifest=_manifest(split="train", revision=DEFAULT_REVISION, count=2),
    )
    with pytest.raises(ValueError, match="sample_count is 2"):
        prepared_aut(root=root)


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda value: value.update(text="not allowed"), ValueError, "payload keys differ"),
        (
            lambda value: value.update(waveform=value["waveform"].squeeze(0)),
            ValueError,
            r"\[1, time\]",
        ),
        (
            lambda value: value.update(waveform=value["waveform"].double()),
            TypeError,
            "torch.float32",
        ),
        (lambda value: value["waveform"].fill_(float("nan")), ValueError, "finite"),
        (
            lambda value: value.update(waveform_length=torch.tensor(1_599)),
            ValueError,
            "waveform_length",
        ),
        (
            lambda value: value.update(aut_features=value["aut_features"].half()),
            TypeError,
            "aut_features",
        ),
        (lambda value: value["aut_features"].fill_(float("inf")), ValueError, "finite"),
        (lambda value: value["aut_feature_mask"].fill_(False), ValueError, "all-true"),
        (lambda value: value.update(audio_placeholders=torch.tensor(1)), ValueError, "frame count"),
        (lambda value: value.update(audio_placeholders=torch.tensor(0)), ValueError, "positive"),
    ],
)
def test_prepared_aut_rejects_invalid_payloads(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    error: type[Exception],
    match: str,
) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    payload = _payload()
    mutate(payload)
    _store(root, payloads=[payload])

    with pytest.raises(error, match=match):
        prepared_aut(root=root)[0]


def test_prepared_aut_rejects_waveform_sha_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    payload = _payload()
    records = [{"sample_id": "sample-0", "audio_sha256": "0" * 64, "path": "samples/000000.pt"}]
    _store(root, payloads=[payload], records=records)

    with pytest.raises(ValueError, match="waveform SHA-256 mismatch"):
        prepared_aut(root=root)[0]


@pytest.mark.parametrize(("name", "value"), [("split", "../train"), ("revision", "a/b")])
def test_prepared_aut_rejects_path_components(tmp_path: Path, name: str, value: str) -> None:
    kwargs = {name: value, "root": tmp_path}

    with pytest.raises(ValueError, match="one non-empty path component"):
        prepared_aut(**kwargs)


@pytest.mark.parametrize("revision", ["a" * 39, "A" * 40, "main"])
def test_prepared_aut_requires_commit_revision(tmp_path: Path, revision: str) -> None:
    with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
        prepared_aut(root=tmp_path, revision=revision)


def test_prepared_aut_rejects_zero_frame_timing(tmp_path: Path) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    _store(root, payloads=[_payload(length=1)])

    with pytest.raises(ValueError, match="audio_placeholders must be positive"):
        prepared_aut(root=root)[0]


def test_prepared_aut_requires_ready_marker(tmp_path: Path) -> None:
    root = tmp_path / "prepared_aut" / "wmt19_tts"
    path = _store(root)
    (path / ".ready").unlink()

    with pytest.raises(FileNotFoundError, match="ready marker"):
        prepared_aut(root=root)
