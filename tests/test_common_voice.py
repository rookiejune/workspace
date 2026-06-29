from __future__ import annotations

import os
from pathlib import Path

import pytest
from anydataset import AudioMeta, Modality, Role

from zhuyin.datasets.common_voice import Args, common_voice, dataset_root


def test_common_voice_builds_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = tmp_path / "cv-corpus-24.0-2025-12-05" / "en"
    (corpus / "clips").mkdir(parents=True)
    (corpus / "train.tsv").write_text(
        "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("STATIC_HOME", str(tmp_path / "static"))

    dataset = common_voice(Args(root=tmp_path))

    assert dataset.spec.path == str(corpus)
    assert dataset.spec.split == "train"
    assert dataset.spec.version == "24.0-2025-12-05"


def test_common_voice_accepts_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = tmp_path / "cv-corpus-24.0-2025-12-05" / "en"
    (corpus / "clips").mkdir(parents=True)
    (corpus / "dev.tsv").write_text(
        "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("STATIC_HOME", str(tmp_path / "static"))

    dataset = common_voice(Args(root=tmp_path, split="dev"))

    assert dataset.spec.path == str(corpus)
    assert dataset.spec.split == "dev"


def test_dataset_root_prefers_explicit_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/env/static")

    assert dataset_root(tmp_path) == tmp_path


def test_dataset_root_uses_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATIC_HOME", "/data/static")

    assert dataset_root() == Path("/data/static/datasets/common_voice")


def test_dataset_root_requires_static_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STATIC_HOME", raising=False)

    with pytest.raises(ValueError, match="STATIC_HOME"):
        dataset_root()


def test_common_voice_sample_contains_speaker_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "cv-corpus-24.0-2025-12-05" / "en"
    (corpus / "clips").mkdir(parents=True)
    (corpus / "clips" / "sample.mp3").write_bytes(b"")
    (corpus / "train.tsv").write_text(
        "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
        "speaker-1\tsample.mp3\tsentence-1\tHello.\tgeneral\t"
        "2\t0\tthirties\tfemale\t\t\ten\t\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("STATIC_HOME", str(tmp_path / "static"))

    sample = next(iter(common_voice(Args(root=tmp_path))))
    audio = sample[(Role.DEFAULT, Modality.AUDIO)]

    assert audio.meta[AudioMeta.SPEAKER_ID] == "speaker-1"
    assert "client_id" not in audio.meta[AudioMeta.LABELS]


def test_common_voice_configures_derived_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "cv-corpus-24.0-2025-12-05" / "en"
    (corpus / "clips").mkdir(parents=True)
    (corpus / "train.tsv").write_text(
        "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n",
        encoding="utf-8",
    )
    static_home = tmp_path / "static"
    monkeypatch.setenv("STATIC_HOME", str(static_home))
    monkeypatch.delenv("ANYDATASET_HOME", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    common_voice(Args(root=tmp_path))

    assert os.environ["ANYDATASET_HOME"] == str(static_home / "anydataset")
    assert os.environ["HF_HOME"] == str(static_home / "huggingface")
