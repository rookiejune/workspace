from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample, Source, Spec

from scripts import prepare_wmt19_tts_bpe as script


class FakeDataset:
    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = list(samples)
        self.spec = Spec(source=Source.STORE, path="/data/wmt19/longcat", split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def test_longcat_corpus_yields_source_and_target_sequences() -> None:
    dataset = FakeDataset([sample(source=[1, 2, 3], target=[4, 5])])

    corpus = list(script.corpus(dataset))

    assert corpus == [
        [[1], [2], [3]],
        [[4], [5]],
    ]


def test_corpus_factory_returns_replayable_corpus() -> None:
    dataset = FakeDataset([sample(source=[1, 2, 3], target=[4, 5])])
    factory = script.corpus_factory(dataset)

    assert callable(factory)
    assert list(factory()) == list(factory())


def test_run_writes_bpe_artifact_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset([sample(source=[1, 2, 1, 2], target=[1, 2])])
    calls: list[dict[str, object]] = []

    def fake_wmt19_tts_codec(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(script, "wmt19_tts_codec", fake_wmt19_tts_codec)

    summary = script.run(
        argparse.Namespace(
            root=None,
            split="train",
            bpe_root=tmp_path / "bpe",
            codec_name="longcat",
            vocab_size=100_000,
            min_frequency=0,
            max_token_length=None,
            codebook_sizes=[8192],
            sample_limit=None,
            overwrite=False,
            show_progress=False,
        )
    )

    artifact_dir = tmp_path / "bpe" / "longcat" / "vocab_100k_minfreq_0_maxlen_none_codes_8192"
    meta = json.loads((artifact_dir / script.META_FILE).read_text(encoding="utf-8"))

    assert calls == [{"codec": script.Codec.LONGCAT, "root": None, "split": "train"}]
    assert (artifact_dir / script.STATE_FILE).exists()
    assert (artifact_dir / script.EVAL_FILE).exists()
    assert meta["datasets"] == [dataset.spec.to_dict()]
    assert meta["min_frequency"] == 0
    assert meta["max_token_length"] is None
    assert summary["actual_vocab_size"] >= 2
    assert summary["artifact_dir"] == str(artifact_dir)
    assert "codec_bpe" not in summary
    assert "tokenizer" not in summary
    assert "meta" not in summary
    assert "reused" not in summary
    eval_stats = summary["eval"]
    assert eval_stats["encoded_tokens"] == 2
    assert eval_stats["num_sequences"] == 2
    assert eval_stats["original_frames"] == 6
    assert eval_stats["compression_factor"] == pytest.approx(3.0)
    assert eval_stats["compression_gain"] == pytest.approx(2 / 3)
    assert eval_stats["compression_ratio"] == pytest.approx(1 / 3)
    assert eval_stats["mean_encoded_length"] == pytest.approx(1.0)
    assert eval_stats["mean_original_length"] == pytest.approx(3.0)
    assert json.loads((artifact_dir / script.EVAL_FILE).read_text(encoding="utf-8")) == summary


def test_run_evaluates_existing_bpe_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset([sample(source=[1, 2, 1, 2], target=[1, 2])])
    monkeypatch.setattr(script, "wmt19_tts_codec", lambda **_: dataset)
    args = argparse.Namespace(
        root=None,
        split="train",
        bpe_root=tmp_path / "bpe",
        codec_name="longcat",
        vocab_size=100_000,
        min_frequency=0,
        max_token_length=None,
        codebook_sizes=[8192],
        sample_limit=None,
        overwrite=False,
        show_progress=False,
    )

    first = script.run(args)
    second = script.run(args)

    assert second == first
    assert "reused" not in second


def test_run_uses_explicit_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = FakeDataset([sample(source=[1, 2, 1, 2], target=[1, 2])])
    calls: list[dict[str, object]] = []

    def fake_wmt19_tts_codec(**kwargs: object) -> FakeDataset:
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(script, "wmt19_tts_codec", fake_wmt19_tts_codec)
    root = tmp_path / "wmt19_tts"

    script.run(
        argparse.Namespace(
            root=root,
            split="train",
            bpe_root=tmp_path / "bpe",
            codec_name="longcat",
            vocab_size=100_000,
            min_frequency=0,
            max_token_length=None,
            codebook_sizes=[8192],
            sample_limit=None,
            overwrite=False,
            show_progress=False,
        )
    )

    assert calls == [
        {
            "codec": script.Codec.LONGCAT,
            "root": root,
            "split": "train",
        }
    ]


def sample(*, source: Sequence[int], target: Sequence[int]) -> Sample:
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: torch.tensor(source).unsqueeze(-1)}
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: torch.tensor(target).unsqueeze(-1)}
        ),
    }
