from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "prepare_qwen_tts_speaker_grid.py"


class _SpeakerGridScript(Protocol):
    dynamic_home: Callable[[], Path]

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace: ...


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_qwen_tts_speaker_grid", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


speaker_grid = cast(_SpeakerGridScript, cast(object, _load_script()))


def test_parser_does_not_resolve_default_root_when_output_dirs_are_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dynamic_home() -> Path:
        raise AssertionError("dynamic_home should not be called.")

    output_dir = tmp_path / "store"
    export_dir = tmp_path / "wavs"
    monkeypatch.setattr(speaker_grid, "dynamic_home", fail_dynamic_home)

    args = speaker_grid.parse_args(
        ["--output-dir", str(output_dir), "--export-dir", str(export_dir)]
    )

    assert args.output_dir == output_dir
    assert args.export_dir == export_dir
