from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from anydataset.store import DatasetWriter
from anydataset.store.reader import read_store_dataset
from anydataset.types import AudioItem, AudioMeta, AudioView, Modality, Role

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import migrate_store_audio_duration as migration  # noqa: E402


@pytest.fixture(autouse=True)
def _anydataset_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "anydataset-home"))


def test_waveform_migration_adds_seconds_without_rewriting_payload(tmp_path: Path) -> None:
    store = tmp_path / "waveform"
    DatasetWriter(store, dataset_id="waveform").write(
        [
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (torch.zeros(1, 8), 4)}
                )
            }
        ]
    )
    payload = next((store / "default" / "audio" / "waveform" / "shards").iterdir())
    before = payload.read_bytes()

    summary = migration.migrate(
        migration.Config(
            store=store,
            view=AudioView.WAVEFORM,
            backup=store / "backup.parquet",
        )
    )

    audio = read_store_dataset(store)[0][Role.DEFAULT, Modality.AUDIO]
    assert audio.meta[AudioMeta.DURATION] == pytest.approx(2.0)
    assert payload.read_bytes() == before
    assert summary["updated_audio_items"] == 1
    assert (store / "backup.parquet").is_file()


def test_codec_migration_uses_frame_rate_without_loading_a_model(tmp_path: Path) -> None:
    store = tmp_path / "longcat"
    DatasetWriter(store, dataset_id="longcat").write(
        [
            {
                (Role.SOURCE, Modality.AUDIO): AudioItem(
                    views={AudioView.LONGCAT: torch.zeros(5, 2, dtype=torch.long)}
                ),
                (Role.TARGET, Modality.AUDIO): AudioItem(
                    views={AudioView.LONGCAT: torch.zeros(10, 2, dtype=torch.long)}
                ),
            }
        ]
    )

    summary = migration.migrate(
        migration.Config(
            store=store,
            view=AudioView.LONGCAT,
            frame_rate=5.0,
            backup=None,
        )
    )

    sample = read_store_dataset(store)[0]
    assert sample[Role.SOURCE, Modality.AUDIO].meta[AudioMeta.DURATION] == 1.0
    assert sample[Role.TARGET, Modality.AUDIO].meta[AudioMeta.DURATION] == 2.0
    assert summary["updated_audio_items"] == 2


def test_codec_parser_uses_registered_rates_and_allows_override(tmp_path: Path) -> None:
    stable = migration.config_from_args(
        migration.parse_args(["codec", str(tmp_path), "--codec", "stable"])
    )
    unicodec = migration.config_from_args(
        migration.parse_args(["codec", str(tmp_path), "--codec", "unicodec"])
    )
    overridden = migration.config_from_args(
        migration.parse_args(
            [
                "codec",
                str(tmp_path),
                "--codec",
                "stable",
                "--frame-rate",
                "30",
            ]
        )
    )

    assert stable.frame_rate == migration.CODEC_FRAME_RATES["stable"]
    assert unicodec.frame_rate == 75.0
    assert overridden.frame_rate == 30.0
